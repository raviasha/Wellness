# Plan: Dynamic Goal-Driven Outcome Weighting

## TL;DR
Replace the 5 hardcoded goals (`Goal` enum) with a free-text goal system where users type goals like "pickleball tournament in 5 days" or "half marathon in 2 months". An LLM translates the free-text goal into structured outcome weights + exercise type preferences. The existing `compute_reward()` and action scoring pipeline already accepts weight dicts — we just need to generate them dynamically instead of looking them up from `GOAL_WEIGHTS`.

## Current Architecture
- `Goal` enum with 5 fixed values → `GOAL_WEIGHTS` dict in `payoff.py`
- `compute_reward(deltas, goal, biomarkers)` uses `GOAL_WEIGHTS[goal]` to weight 7 biomarker deltas
- Frontend: dropdown of 5 goals in `GarminDashboard.tsx`
- DB: `user_profile.goal` stores a string like "stress_management"
- Inference: goal string passed through `/api/recommendations?goal=...` → `get_coaching_recommendation()` → `_generate_llm_recommendation()` / NN / simulator

## Design

### Phase 1: LLM Goal Interpreter (Backend)

**New module**: `backend/goal_interpreter.py`
- Function `interpret_goal(free_text_goal: str, target_date: date | None) -> GoalProfile`
- Uses LLM (GPT-4o-mini) to translate free-text goals into:
  ```python
  @dataclass
  class GoalProfile:
      original_text: str                    # "Pickleball tournament in 5 days"
      outcome_weights: dict[str, float]     # 7 biomarker weights summing to ~1.0
      exercise_preferences: dict[str, float]  # exercise type relevance scores
      focus_summary: str                    # 1-line coaching focus for LLM prompts
      target_date: date | None              # explicit user-selected goal date
      days_to_target: int | None            # computed from today
      periodization_phase: str              # base_build | specific_build | sharpen | taper | event_week
  ```
- LLM prompt includes the 7 outcome names and their meanings, plus exercise type enums
- Cache the result in DB so we don't re-call LLM on every recommendation request

**Exercise preference example** for "pickleball tournament":
  ```json
  {
    "exercise_preferences": {
      "none": 0.0,
      "cardio": 0.15,
      "strength": 0.20,
      "flexibility": 0.25,
      "hiit": 0.10,
      "sport_specific": 0.30
    }
  }
  ```

### Phase 2: Wire Goal Weights into Reward Pipeline

1. **`wellness_env/payoff.py`** — Modify `compute_reward()` to accept `weights: dict[str, float]` directly instead of requiring a `Goal` enum. Keep backward compat by accepting either.

2. **`backend/inference_service.py`** — In `get_coaching_recommendation()`:
   - If user has a custom goal, load cached GoalProfile (or re-interpret if goal text changed)
   - Pass `outcome_weights` to `compute_reward()` instead of `Goal` enum
   - Include `exercise_preferences` in the LLM prompt context so it recommends sport-specific training
   - For NN path: NN still uses generic weights; overlay custom weights when scoring the NN output

3. **`backend/inference_service.py`** — In `_generate_llm_recommendation()`:
   - Include the parsed goal profile (weights + exercise preferences) in the system prompt
   - Tell the LLM: "This user wants to prepare for X — here are the most important biomarkers and exercise modalities"

### Phase 3: Database Schema

- Add column `custom_goal_text` (Text, nullable) to `user_profile` table
- Add column `custom_goal_target_date` (Date, nullable) to `user_profile` table
- Add column `custom_goal_profile` (Text/JSON, nullable) to `user_profile` table — caches the LLM-interpreted GoalProfile
- Add column `goal_updated_at` (DateTime) for recency and rescheduling logic
- Existing `goal` column becomes a fallback (used when custom_goal_text is null)

### Phase 4: API Endpoints

- `POST /api/user/goal` — Set a custom free-text goal with optional date
  - Body: `{ "goal_text": "Pickleball tournament", "target_date": "2026-05-10" }`
  - Validates date is today or later; computes `days_to_target`
  - Calls `interpret_goal(goal_text, target_date)`, stores text + date + profile in DB
  - Returns interpreted profile and current `periodization_phase` for transparency

- `GET /api/user/goal` — Get current goal (custom or preset) including target date + days remaining

- `DELETE /api/user/goal` — Clear custom goal, revert to preset dropdown

### Phase 5: Frontend

- **GarminDashboard.tsx** — Replace/augment the 5-option dropdown with a free-text input:
  - Text field: "What's your goal?" with placeholder examples
  - Date picker for target date (required for periodization to activate)
  - "Set Goal" button → POST `/api/user/goal`
  - Display the interpreted profile (weights, exercise plan, active phase) for transparency
  - Keep preset quick-select buttons as shortcuts that populate the text field
  - Show active goal + days remaining + phase badge prominently on the dashboard

- **Settings tab or inline** — Allow changing goal anytime

### Phase 6: Date-Based Backward Planning (Periodization)

- Add `get_periodization_phase(days_to_target: int) -> str` helper:
  - `> 42 days` → `base_build` (aerobic base + foundational strength + sleep consistency)
  - `21–42 days` → `specific_build` (goal-specific training split ramps up)
  - `7–20 days` → `sharpen` (higher specificity, controlled load, skill work)
  - `0–6 days` → `taper` / `event_week` (fatigue reduction, readiness optimization)
- Each phase adjusts:
  - Outcome weight multipliers (e.g., stress + body_battery boosted in taper)
  - Exercise type preferences (e.g., specificity up near event; heavy strength down in final days)
  - Daily load caps for intensity minutes and exercise duration
- Recommendation generation becomes date-aware:
  - If target date exists, compose recommendations from current phase constraints
  - If no date, default to ongoing progression without taper logic

### Phase 7: Exercise-Aware Recommendations

- In `_generate_llm_recommendation()`, include exercise preferences from GoalProfile
- The LLM system prompt should mention: "For this goal, prioritize: {exercise_preferences}"
- Example: "User wants to excel at pickleball. Recommend: 3x/week sport-specific play, 2x/week strength (legs/core), 1x/week flexibility, 1x/week cardio"

---

## Implementation Steps

### Phase A: Backend Goal Interpreter (steps 1–3)
1. Create `backend/goal_interpreter.py` with `GoalProfile` dataclass and `interpret_goal()` function using LLM
2. Add `custom_goal_text`, `custom_goal_target_date`, `custom_goal_profile`, `goal_updated_at` columns to `UserProfile` in `backend/database.py`
3. Add `POST /api/user/goal`, `GET /api/user/goal`, `DELETE /api/user/goal` endpoints in `app.py`

### Phase B: Wire into Recommendation Pipeline (steps 4–6, depends on Phase A)
4. Modify `compute_reward()` in `wellness_env/payoff.py` to accept raw `weights` dict as alternative to `Goal` enum
5. Update `get_coaching_recommendation()` in `backend/inference_service.py` to load custom goal profile and pass weights
6. Update `_generate_llm_recommendation()` to include exercise preferences, target date, days remaining, and periodization phase in the LLM prompt

### Phase C: Frontend (step 7, depends on Phase A)
7. Update `GarminDashboard.tsx` — add free-text goal input + date picker, display interpreted profile (weights, exercise plan, days remaining, active phase badge), keep preset shortcuts

### Phase D: Date-Aware Inference (step 8, depends on B)
8. Add `get_periodization_phase()` and phase-specific weight/load modulation in `backend/inference_service.py`

### Phase E: Build & Deploy (step 9, depends on C + D)
9. Build frontend, copy to static, verify end-to-end

---

## Relevant Files
- `backend/goal_interpreter.py` — **NEW**: parse free-text goal + target date into structured GoalProfile
- `backend/database.py` — Add custom goal columns to `UserProfile`
- `app.py` — New goal API endpoints with date validation
- `wellness_env/payoff.py` — Make `compute_reward()` accept raw weights dict
- `backend/inference_service.py` — Wire custom goals + periodization into recommendation pipeline
- `webapp/src/components/GarminDashboard.tsx` — Free-text goal UI + date picker + phase badge

---

## Verification
1. `curl -X POST /api/user/goal -d '{"goal_text":"Pickleball tournament","target_date":"2026-05-10"}' -H "X-User-ID: 1"` → returns parsed GoalProfile with `days_to_target` and `periodization_phase`
2. `curl /api/recommendations -H "X-User-ID: 1"` → recommendation references pickleball and current phase-specific loading guidance
3. Set a closer target date (e.g., 3 days out) → phase transitions to `taper/event_week` and intensity/volume recommendations reduce
4. Change goal to "half marathon" with a later date → weights shift toward cardio/VO2 max, phase returns to build
5. Clear custom goal → reverts to preset dropdown behavior
6. Frontend: set goal + date → see days remaining + active phase badge → tailored daily recommendation

---

## Decisions
- Custom goals are per-user and persist until changed/cleared
- LLM interpretation is cached in DB — only re-interpreted when the goal text changes (not on every recommendation call)
- Preset goals remain as quick-select shortcuts; they internally use `interpret_goal()` or map to pre-seeded GoalProfiles
- Exercise preferences are advisory to the LLM coach, not hard constraints on the action enum
- Goal date is first-class: each custom goal stores `target_date`, `days_to_target`, and `periodization_phase`
- Backward planning drives recommendations: phase-specific weight multipliers and load constraints are applied before final recommendation text
- If `target_date` is missing, system defaults to non-periodized ongoing progression
- NN/RL path: the NN uses the closest preset goal's weights; date-based phase modulation is applied post-policy during recommendation assembly; full date-conditioned RL retraining is out of scope

## Further Considerations
1. **Exercise type granularity**: The current `ExerciseType` enum has only `none, cardio, strength, flexibility, hiit`. For sport-specific goals (pickleball, marathon), the LLM maps sports to the closest types in coaching text. Adding sports to the enum would require Garmin activity type mapping changes — deferred.
2. **Multi-goal support**: Users can express compound goals (e.g., "reduce stress AND prepare for marathon") as one free-text string. The LLM naturally blends weights. No architecture change needed.
3. **Goal history tracking**: Tracking goal changes over time is deferred — can be added later via the existing recommendation audit trail.
