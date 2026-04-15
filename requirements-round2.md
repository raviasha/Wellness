# RL Wellness App — Requirements

## 1. Vision

A wellness application that uses Reinforcement Learning to guide users toward better **health outcomes** — measured by biomarkers like resting HR, HRV, VO2 max, body fat %, lean mass, sleep efficiency, cortisol, and energy level. The RL agent learns optimal action sequences (sleep, exercise, nutrition recommendations) by observing how actions change each user's biomarkers, then recommends personalized daily actions that maximize outcome improvements for the user's specific health goal.

### 1.1 Terminology

| Term                  | Definition                                                      |
|-----------------------|-----------------------------------------------------------------|
| **Policy**            | The learned decision function that maps states to recommended actions. This is what the RL algorithm produces after training. |
| **Simulated user / Persona** | A model of user behavior used in training. Each persona is defined by its health goal, compliance rate, and hidden physiological response model. |
| **User agent instance** | A specific policy deployed to a specific user in production, together with that user's state and history. |
| **Payoff function**   | Deterministic scoring code that evaluates biomarker deltas and returns a numerical reward. |
| **Episode**           | One complete training run through the simulator (90 simulated days). |
| **Step**              | One day within an episode.                                       |
| **Biomarker**         | A measurable health outcome variable (e.g., resting_hr, hrv, vo2_max). The system tracks 8 biomarkers. |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RL ENVIRONMENT                              │
│                                                                     │
│  State Space ──→ Agent ──→ Actions ──→ Simulator ──→ Next State     │
│  (biomarkers,     ↑       (sleep,       (hidden         │          │
│   deltas,         │        exercise,     response        │          │
│   trends,         │        nutrition)    model)          │          │
│   goal)           │                        │             │          │
│                   └──── Reward (from Payoff Function) ◄──┘          │
│                         (outcome-based: biomarker deltas × weights) │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         USER-FACING APP                             │
│                                                                     │
│  User Input (natural language) → LLM Feature Extractor              │
│       → Structured Actions → Simulator → Biomarker Updates          │
│                                                                     │
│  RL Agent Policy → Recommendations → User                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

---

## 3. Payoff Function

### 3.1 Structure

The payoff function is **outcome-based**: it computes reward from measured biomarker deltas, not from action quality. The scoring formula executes as deterministic code at runtime.

- **Execution time (training & production):** The scoring code runs deterministically — same deltas always produce the same reward. This is critical for RL convergence.

Given biomarker deltas and a user's goal:

$$R = 50 + \sum_{i=1}^{8} \text{normalize}(\Delta_i) \times w_{i,\text{goal}} \times 100 \quad \text{clamped to } [0, 100]$$

Where:
- $\Delta_i$ is the change in biomarker $i$ for this step (computed by the simulator)
- For "lower is better" biomarkers (resting HR, body fat %, cortisol), the delta is negated so positive = improving
- Each delta is normalized by a DELTA_SCALE defining what "excellent daily improvement" looks like
- $w_{i,\text{goal}}$ are goal-specific weights (all sum to 1.0 per goal)
- Baseline 50 = no change; >50 = biomarkers improved; <50 = biomarkers declined

### 3.2 Biomarker Outcomes

The system measures 8 health outcomes:

| Biomarker | Direction | Good Delta Scale | What It Captures |
|-----------|-----------|-----------------|------------------|
| `resting_hr` | Lower = better | −1 bpm/day | Cardiovascular fitness |
| `hrv` | Higher = better | +5 ms/day | Autonomic health, recovery |
| `vo2_max` | Higher = better | +0.3 ml/kg/min/day | Aerobic capacity |
| `body_fat_pct` | Lower = better | −0.05%/day | Body composition |
| `lean_mass_kg` | Higher = better | +0.1 kg/day | Muscle mass |
| `sleep_efficiency` | Higher = better | +2%/day | Sleep quality |
| `cortisol_proxy` | Lower = better | −5 pts/day | Stress level |
| `energy_level` | Higher = better | +10 pts/day | Subjective energy |

### 3.3 Interaction Effects (embedded in simulator)

Cross-action interactions are captured in the **simulator's biomarker transition model**, not as separate bonus terms. The payoff function only sees the resulting biomarker deltas:
- Sleep debt reduces exercise efficacy → smaller VO2/lean mass gains → lower reward
- Overtraining spikes cortisol → cortisol_proxy delta goes positive → penalty in reward
- Post-exercise protein boosts recovery → larger lean mass and VO2 gains → higher reward
- Poor sleep raises resting HR and cortisol → negative deltas → lower reward

### 3.4 Goal-Dependent Weighting

Each goal defines which biomarker improvements matter most. This is the primary personalization mechanism — the same biomarker changes produce different rewards for different goals:

| Goal | Primary Biomarkers (weight ≥ 0.20) | Secondary Biomarkers |
|------|-------------------------------------|---------------------|
| Weight Loss | body_fat_pct (0.35), energy (0.20) | All others at 0.05–0.10 |
| Muscle Gain | lean_mass_kg (0.35), energy (0.20) | vo2_max (0.10), body_fat (0.10) |
| Overall Wellness | All equally weighted (0.125 each) | — |
| Longevity | vo2_max (0.25), hrv (0.20), resting_hr (0.15) | Others at 0.05–0.10 |
| Athletic Performance | vo2_max (0.30), lean_mass (0.20), energy (0.15) | Others at 0.05–0.10 |
| Stress Management | cortisol (0.25), hrv (0.20), sleep_efficiency (0.20) | Others at 0.05–0.10 |

### 3.5 Weight Progression

| Phase | Weight Strategy |
|-------|----------------|
| V1 | Goal-specific weights derived from physiological prioritization (current) |
| V2 | Research-derived weights from relative effect sizes on outcomes |
| V3 | User-goal-dependent fine-tuning from aggregated real data |
| V4 | Learned weights from individual user feedback and outcome tracking |

---

## 4. RL Environment

### 4.1 State Space

The state vector is composed of two layers:

**Layer 1 — Current snapshot (available from day 1):**
- Current biomarker values (resting_hr, hrv, vo2_max, body_fat_pct, lean_mass_kg, sleep_efficiency, cortisol_proxy, energy_level) — 8 values
- Biomarker deltas since last step — 8 values
- User's health goal (categorical)

**Layer 2 — Outcome trends (require 7+ days of history):**
- 7-day slopes for each biomarker (resting_hr_trend, hrv_trend, vo2_max_trend, body_fat_trend, lean_mass_trend, sleep_efficiency_trend) — 6 values
- 7-day reward trend and reward consistency — 2 values

**V1 state vector:** Layer 1 + Layer 2 (biomarker values, deltas, goal, and trends when available)
**V2+ state vector:** Add user profile features (age, fitness level, injury history) for deeper personalization

### 4.2 Action Space (V1 — Discretized)
- **Sleep:** {<6hrs, 6-7hrs, 7-8hrs, 8-9hrs, >9hrs}
- **Exercise:** {none, light_cardio, moderate_cardio, HIIT, strength, yoga}
- **Nutrition:** {high_protein, balanced, high_carb, processed, skipped}

Total: 5 × 6 × 5 = 150 action combinations per time step

### 4.3 Time Step
- One day = one time step
- Episode length: **90 days** (three months of simulated behavior — long enough for the agent to observe the consequences of unsustainable patterns and for 30-day sustainability penalties to apply meaningfully within the episode, not just at the boundary)

### 4.4 Simulator

The simulator is the **most critical and hardest engineering component**. It models how today's actions affect tomorrow's state. Without a realistic simulator, the agent learns a policy that doesn't transfer to real users.

#### 4.4.1 Transition Dynamics

The simulator models how actions change biomarkers through a **hidden ResponseModel** per persona. The agent never sees these parameters — it only observes the resulting biomarker deltas.

**Per-biomarker response to actions:**

| Biomarker | Primary Action Drivers | Key Dynamics |
|-----------|----------------------|--------------|
| Resting HR | Sleep quality, exercise | Sleep ≥7h → improves (persona-specific sensitivity); exercise → improves; overtraining → raises |
| HRV | Sleep, yoga, stress level | Good sleep boosts (persona-specific sensitivity); yoga +2ms; overtraining tanks; high cortisol depresses |
| VO2 Max | Cardio exercise intensity | HIIT > moderate > light (persona-specific gain rate); no exercise → slow decay; sleep debt reduces gains |
| Body Fat % | Exercise intensity, nutrition | Exercise burns fat (persona-specific rate); protein/balanced diet aids loss; poor sleep impairs loss |
| Lean Mass | Strength training, protein | Strength + protein combo (multiplicative); HIIT at 50% strength effect; inactivity + poor diet → loss |
| Sleep Efficiency | Sleep hours, exercise type, cortisol | 7-9h optimal → +1; moderate exercise/yoga help; intense exercise hurts; high cortisol hurts |
| Cortisol Proxy | Sleep, exercise stress, nutrition | Sleep recovery; yoga reduces; intense exercise increases (persona-specific); overtraining spikes; natural regression |
| Energy Level | Sleep, nutrition, fatigue/debt | Good sleep + nutrition boost; intense exercise drains short-term; overtraining/sleep debt drain; high cortisol drains |

**Hidden ResponseModel parameters per persona (~15 params):**

| Parameter Category | Examples | Effect |
|---|---|---|
| Sleep response | `hrv_sleep_sensitivity`, `rhr_sleep_benefit`, `cortisol_sleep_recovery` | How much sleep improves HRV, resting HR, cortisol |
| Exercise response | `vo2_cardio_gain`, `lean_mass_strength_gain`, `body_fat_exercise_loss` | How responsive persona is to exercise |
| Overtraining | `overtraining_threshold`, `overtraining_cortisol_spike` | When overtraining kicks in | 
| Nutrition response | `body_fat_nutrition_sensitivity`, `lean_mass_protein_gain`, `energy_nutrition_sensitivity` | How nutrition affects outcomes |
| Cross-action | `sleep_debt_exercise_penalty`, `protein_recovery_multiplier` | Interaction effects between action categories |

**Cross-action interactions (computed internally):**

| Today's Action Combination | Effect on Biomarkers |
|---|---|
| Sleep debt + exercise | Exercise gains multiplied by debt_factor (reduced) |
| Intense exercise + high protein | Recovery gains multiplied by protein_recovery_multiplier |
| Consecutive intense days ≥ threshold | Cortisol spike, HRV depression, energy drain, RHR increase |
| Poor sleep + any action | Resting HR up, HRV down, cortisol up, fat loss impaired, energy drain |

**Cumulative / multi-day effects:**

| Pattern (over days) | Biomarker Impact |
|---|---|
| Sleep debt (< 7h × 7 days) | Compounds: energy drains, exercise gains reduced proportionally |
| Consistent optimal sleep (7-9h) | Sleep efficiency trends up, cortisol trends down |
| Overtraining (consecutive intense days) | Threshold-dependent: cortisol spikes, energy crashes, VO2/lean mass gains stall |
| No exercise for multiple days | VO2 max slowly decays (−0.02/day) |

#### 4.4.2 Transition Function Structure

```python
def compute_biomarker_changes(action, current_biomarkers, persona, history, rng):
    rm = persona.response_model  # Hidden parameters
    hours = SLEEP_HOURS[action.sleep]
    sleep_debt = _recent_sleep_debt(history)
    consecutive_intense = _consecutive_intense_days(history)
    
    # Sleep debt reduces exercise efficacy
    debt_factor = max(0.2, 1.0 - rm.sleep_debt_exercise_penalty * sleep_debt)
    
    # Protein + intense exercise = recovery boost
    protein_mult = rm.protein_recovery_multiplier if (protein + intense) else 1.0
    
    # Compute each biomarker delta independently using ResponseModel params
    d_rhr = f(hours, exercise, rm.rhr_sleep_benefit, rm.rhr_exercise_benefit, ...)
    d_hrv = f(hours, exercise, rm.hrv_sleep_sensitivity, cortisol_level, ...)
    d_vo2 = f(cardio_type, rm.vo2_cardio_gain, debt_factor, protein_mult, ...)
    d_bf  = f(exercise, nutrition, rm.body_fat_exercise_loss, rm.body_fat_nutrition_sensitivity, ...)
    d_lm  = f(strength, protein, rm.lean_mass_strength_gain, rm.lean_mass_protein_gain, ...)
    d_se  = f(hours, exercise, cortisol_level, ...)
    d_cortisol = f(hours, exercise, rm.cortisol_sleep_recovery, rm.cortisol_exercise_stress, ...)
    d_energy = f(hours, nutrition, exercise, rm.energy_nutrition_sensitivity, sleep_debt, ...)
    
    # Add per-biomarker Gaussian noise
    d_rhr += rng.gauss(0, 0.3)
    d_hrv += rng.gauss(0, 1.5)
    # ... etc
    
    return BiomarkerDeltas(resting_hr=d_rhr, hrv=d_hrv, ...)

def apply_deltas(current_biomarkers, deltas):
    # Clamp each biomarker to valid range
    return Biomarkers(
        resting_hr=clamp(current.resting_hr + d_rhr, 40, 120),
        ...
    )
```

#### 4.4.3 Stochasticity

Real humans are variable. The simulator adds controlled noise:
- **Action noise:** Even if the agent selects "7-8h sleep," the actual duration is sampled from a distribution centered on 7.5h (models realistic variation)
- **Outcome noise:** Same inputs don't always produce the same next-day state (models biological variability)
- **Life events:** Random perturbations (e.g., "bad night's sleep" regardless of bedtime, "social dinner" overriding nutrition plan) — modeled as occasional random state disruptions

#### 4.4.4 Deriving Transition Dynamics from Research

Research literature informs **two types** of simulator parameters:

| Type | Used By | Example |
|-----------------------|------------------|----------------------------------------------------|
| **Response parameters** | Simulator (ResponseModel) | "VO2 max improves ~0.15 ml/kg/min per cardio session for average adults" → `vo2_cardio_gain` |
| **Normalization scales** | Payoff function (DELTA_SCALES) | "A 1 bpm/day resting HR reduction is clinically excellent" → `resting_hr` delta scale |

The extraction process should output both finding types with clear labels.

#### 4.4.5 Simulator Validation

The simulator must be validated before training agents on it:
- **Face validity:** Domain expert reviews transition dynamics — do they make intuitive sense?
- **Research alignment:** Each transition traces back to a cited paper
- **Boundary behavior:** Extreme inputs (0h sleep, 10h exercise) produce reasonable (bad but not nonsensical) outputs
- **Statistical properties:** Simulated multi-day trajectories should resemble real human behavioral data distributions (if available)

### 4.5 RL Algorithm

- **V1:** DQN (Deep Q-Network). With the V1 state vector (biomarker values + discretized action), the 150-action space benefits from a function approximator. A small neural network generalizes better than a Q-table, and it's ready for V2 when history-derived stats are added.
- **V2:** PPO or SAC when the state space expands with continuous history-derived features and potentially continuous action spaces.
- Priority: interpretability of the learned policy (user needs to understand *why* a recommendation is made). Use policy inspection techniques (e.g., feature importance, action-advantage analysis) to generate explanations.

### 4.6 History Store

History is a **first-class component**, not an afterthought. The consistency bonus, trend bonus, and recovery bonus all depend on it.

#### What Is Stored (Per Time Step)

```json
{
  "day": 14,
  "timestamp": "2026-04-03",
  "goal": "athletic_performance",
  "actions": {
    "sleep": "7_to_8h",
    "exercise": "moderate_cardio",
    "nutrition": "balanced"
  },
  "biomarkers": {
    "resting_hr": 60.5,
    "hrv": 58.2,
    "vo2_max": 43.1,
    "body_fat_pct": 17.5,
    "lean_mass_kg": 65.3,
    "sleep_efficiency": 82.0,
    "cortisol_proxy": 30.0,
    "energy_level": 72.0
  },
  "deltas": {
    "resting_hr": -0.3,
    "hrv": +1.5,
    "vo2_max": +0.12,
    "body_fat_pct": -0.02,
    "lean_mass_kg": +0.01,
    "sleep_efficiency": +0.8,
    "cortisol_proxy": -1.5,
    "energy_level": +3.0
  },
  "reward_total": 58.3,
  "recommendation_given": "Add strength training tomorrow",
  "recommendation_followed": null,
  "policy_version": "v1.2",
  "payoff_version": "v2.1"
}
```

#### History Windows Used by the Environment

| Window | Used For |
|--------|-------------------------------------------------------|
| 7-day | Outcome trends (7-day slopes for each biomarker) |
| 7-day | Reward consistency (stddev of reward) |
| 7-day | Sleep debt calculation (accumulated shortfall) |
| History | Consecutive intense exercise day counting |

#### Derived Statistics (Computed from History)

The observation includes these **derived stats from history** (available after day 7):

```
resting_hr_trend     = linear_slope(resting_hr, last 7 days)
hrv_trend            = linear_slope(hrv, last 7 days)
vo2_max_trend        = linear_slope(vo2_max, last 7 days)
body_fat_trend       = linear_slope(body_fat_pct, last 7 days)
lean_mass_trend      = linear_slope(lean_mass_kg, last 7 days)
sleep_efficiency_trend = linear_slope(sleep_efficiency, last 7 days)
reward_trend         = linear_slope(rewards, last 7 days)
reward_consistency   = stddev(rewards, last 7 days)
```

These trends are part of the **observation** the agent sees. The agent can use them to detect whether its strategy is working and adjust accordingly.

#### Training vs. Production History

| Context       | History Source                                       |
|---------------|------------------------------------------------------|
| **Training**  | Simulated — the simulator generates a full episode history as the agent steps through 30 days |
| **Production**| Real — stored in the user database, read by the payoff function and agent at inference time    |

In training, history is ephemeral (lives in the episode buffer). In production, history is persistent (lives in the database). The payoff function and agent interface are identical in both cases — they receive the same state vector with derived stats.

#### History Scoping — Per Agent, Per User

History is stored with **three scoping keys** to keep every agent's trajectory separate:

```
history_record:
    agent_id:          "policy_v1.2"          # which trained agent version
    user_id:           "user_abc123"          # which user
    payoff_version:    "v2.1"                 # which payoff function scored this
    schema_version:    "1.3"                  # which feature set was active
    day:               14
    timestamp:         "2026-04-03"
    actions:           { ... }
    scores:            { ... }
    recommendation:    "..."
    recommendation_followed: true/false/partial
```

**Why all three keys matter:**

| Scope                          | What It Enables                                                   |
|--------------------------------|-------------------------------------------------------------------|
| **Per agent version**          | Compare how agent v1.0 vs. v1.2 performed across all users        |
| **Per user**                   | Each user has their own complete trajectory — consistency and trend bonuses are computed from *their* history, not mixed with other users |
| **Per agent × per user**       | When a user is upgraded from agent v1.0 to v1.2, you can see the before/after. Old history under v1.0 is preserved, new history under v1.2 begins. Both remain queryable. |
| **Per payoff version**         | If the payoff function changes, you can retroactively re-score old history under the new function to compare |

**Transition handling:** When a new agent version is deployed to a user:
- The user's full history is **carried forward** (the new agent can read it for consistency/trend computation)
- New records are tagged with the new agent_id
- The transition point is marked so you can analyze pre/post performance

**Multi-agent comparison:** If you A/B test two agent versions on different user cohorts, the per-agent-version scoping lets you aggregate and compare outcomes cleanly.

---

## 5. User-Facing App

### 5.1 LLM Feature Extractor
- Accepts natural language input from the user (text, voice)
- Extracts structured features matching the payoff function's input schema
- Outputs a confidence score per extracted feature
- Asks follow-up questions when confidence is low
- Optionally connects to food databases (USDA FoodData Central) for nutrition accuracy

### 5.2 User Interaction Flow

```
1. User logs daily activities (natural language)
2. LLM extracts structured features
3. Simulator updates biomarkers based on actual behavior
4. User sees current biomarker values + trends
5. RL agent recommends next actions to improve goal-relevant outcomes
6. Repeat daily
```

### 5.3 User-Facing Outputs
- Current biomarker dashboard — all 8 biomarkers with trends
- Goal-weighted outcome score (0–100) with per-biomarker breakdown
- Biomarker trend over time (weekly, monthly)
- Actionable recommendations ("Go to bed by 10:30pm", "Add a 30-min walk tomorrow")
- Explanation of why each recommendation is made (traceable to expected biomarker impact)

### 5.4 Cold-Start Strategy

On Day 1, the user has no history. History-derived statistics (consistency, trend, etc.) are undefined. The system must handle this gracefully.

#### Onboarding

1. **Brief questionnaire (2–3 minutes):**
   - Typical bedtime and wake time
   - Exercise frequency and type (if any)
   - General diet description (e.g., "mostly home-cooked" vs. "mostly takeout")
   - Primary goal (improve energy, lose weight, better sleep, general wellness)

2. **Seed the initial state:** Questionnaire answers populate the state vector with estimated baselines. These are treated as "Day 0" history entries so the system has a starting point.

3. **Use the pre-trained policy immediately:** The agent trained in simulation already knows good general recommendations. Day 1 recommendations come from this population-level policy — no user-specific data needed.

#### History Window Fill-In

| User Day | Available History | System Behavior                                   |
|----------|-------------------|----------------------------------------------------|
| Day 1    | Onboarding only   | Pre-trained policy; no sequence bonuses applied     |
| Days 2–6 | 1–5 days          | Sequence bonuses computed over available window (partial). Bonus magnitude scaled by `min(days_available / window_size, 1.0)` so partial windows don't produce artificially extreme values. |
| Day 7+   | Full 7-day window  | 7-day bonuses (consistency, recovery) fully active  |
| Day 14+  | Full 14-day window | Trend bonus fully active                            |
| Day 30+  | Full 30-day window | Sustainability penalty fully active                 |

#### Early Recommendations

In the first 7 days, recommendations focus on **establishing a baseline**, not optimizing:
- "Log your meals today" (data collection)
- "Go to bed at your usual time" (measure current habits, not change them)
- Gentle nudges only ("Try drinking an extra glass of water")

This builds user trust and collects data before making demanding recommendations.

### 5.5 User Engagement & Retention

The system requires daily input for months. Without engagement design, users churn before the RL personalization ever takes effect.

#### Missing Day Handling

| Missed Days | System Response                                               |
|-------------|----------------------------------------------------------------|
| 1 day       | Fill with "unknown" — history stats computed with gap handling. No penalty to the user. Gentle reminder: "How was yesterday?"  |
| 2–3 days    | Carry forward last known state. When user returns, ask for a brief retrospective ("Roughly how did you sleep/eat/exercise?")  |
| 4–7 days    | Partial history reset for that window. When user returns, re-onboard with a lighter questionnaire. No guilt messaging.        |
| 7+ days     | Treat as a restart. Clear derived stats, re-seed from onboarding baseline + any prior long-term data. Welcoming re-entry.     |

#### Score Presentation

- **Never show a raw low score without context.** "45/100" is discouraging. Instead: "45/100 — you're building your baseline. Most users start here."
- **Emphasize biomarker deltas over absolutes.** "Your HRV improved by 5ms this week" is more motivating than "HRV: 42ms."
- **Per-biomarker wins.** Even if overall score is low, highlight the best improving biomarker: "Your sleep efficiency improved by 8% this week."
- **Streak tracking.** "You've logged 7 days in a row" — rewards engagement regardless of score.

#### Notification Strategy (V2+)

- Opt-in reminders at user's preferred time
- Contextual nudges ("It's 9:30pm — studies show going to bed in the next hour improves sleep quality")
- Never guilt-trip for missed days or low scores

### 5.6 Progressive Signal Enrichment

Users shouldn't need full wearable data to start. The system adapts its reward computation and recommendations based on whatever biomarker data the user actually provides, improving as more data sources are added.

#### 5.6.1 Data Tiers

| Tier | Data Source | Biomarkers Available | Example User |
|------|------------|---------------------|--------------|
| **Tier 0** | Self-report only | energy_level, cortisol_proxy (subjective) | New user, no devices |
| **Tier 1** | Smart scale | + body_fat_pct, lean_mass_kg | User with a bathroom scale |
| **Tier 2** | Sleep tracker | + sleep_efficiency | User with Oura/Whoop/Apple Watch |
| **Tier 3** | Fitness wearable | + resting_hr, hrv | User with HR-capable wearable |
| **Tier 4** | Full suite | + vo2_max | User with VO2 max estimation (Apple Watch, Garmin) |

#### 5.6.2 Architecture Changes

**Models:** Biomarker fields become `Optional[float]` with `None` meaning "not available." `BiomarkerDeltas` mirrors this — `None` delta for any biomarker the user can't measure.

**Payoff function:** Goal weights are **re-normalized** over available biomarkers only. If a user provides 3 of 8 biomarkers, the weights for those 3 are scaled up proportionally so they still sum to 1.0. The reward formula is unchanged — it just operates over a smaller set.

```python
# Pseudocode
available = {k: v for k, v in deltas.items() if v is not None}
raw_weights = {k: GOAL_WEIGHTS[goal][k] for k in available}
total = sum(raw_weights.values())
normalized_weights = {k: w / total for k, w in raw_weights.items()}
# Then compute reward using normalized_weights instead of GOAL_WEIGHTS
```

**NN input (production):** A mask vector indicates which biomarkers are present. The network learns to make predictions from partial observations, similar to how masked language models handle missing tokens.

```
input = [biomarker_values (8)] + [mask (8)] + [goal (one-hot)] + [trends (8)]
         ↑ 0.0 for missing       ↑ 1=present, 0=missing
```

**Recommendations:** When operating on fewer biomarkers, the system communicates uncertainty: "Based on your sleep and energy data, yoga looks beneficial. Adding a heart rate monitor would let me fine-tune exercise intensity recommendations."

#### 5.6.3 Product Implications

| Aspect | Benefit |
|--------|---------|
| **Low barrier to entry** | Users start with zero hardware — just self-report energy and stress |
| **Natural upsell** | "Your energy improved 15% this week. A sleep tracker would help me optimize your bedtime." |
| **Graceful degradation** | If a user stops wearing their watch for a week, the system falls back to available data instead of breaking |
| **Data quality signal** | More biomarkers → tighter confidence intervals → more specific recommendations |

#### 5.6.4 Versioning

| Phase | Signal Enrichment Strategy |
|-------|---------------------------|
| V1 | All 8 biomarkers required (simulator only — all data is synthetic) |
| V2 | Optional biomarkers with re-normalized weights; Tier 0–1 support |
| V3 | Mask-vector NN input; Tier 0–4 support; uncertainty-aware recommendations |
| V4 | Learned imputation — predict missing biomarkers from available ones using accumulated user data |

---

## 6. Hackathon Demo — Multi-Agent Training Simulation

A live visual simulation showing multiple agents learning in parallel. This demonstrates the core RL concept to an audience: agents start with random behavior and converge toward optimal wellness patterns over time.

### 6.1 Simulation Design

```
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-AGENT SIMULATION                         │
│                                                                 │
│  Agent 1 ("Weight Loss")     ████░░░░░░  Score: 42 → 78  ↑     │
│  Agent 2 ("Stress Mgmt")     ██░░░░░░░░  Score: 35 → 71  ↑     │
│  Agent 3 ("Muscle Gain")     ███░░░░░░░  Score: 38 → 82  ↑     │
│  Agent 4 ("Longevity")       █████░░░░░  Score: 55 → 69  ↑     │
│  Agent 5 ("Random")          █░░░░░░░░░  Score: 30 → 75  ↑     │
│                                                                 │
│  Episode: 247 / 1000        [▓▓▓▓▓▓░░░░░░░░░░░░░░]  25%       │
│                                                                 │
│  ┌──────────────────────┐   ┌──────────────────────┐           │
│  │ Score Over Time      │   │ Action Distribution  │           │
│  │  ^                   │   │                      │           │
│  │  |    ___/‾‾‾‾       │   │  Sleep 7-8h: ██████  │           │
│  │  |   /               │   │  Balanced:   █████   │           │
│  │  |  /                │   │  Mod cardio: ████    │           │
│  │  | /                 │   │  Processed:  █       │           │
│  │  |/                  │   │  No exercise:░       │           │
│  │  +---------->        │   │                      │           │
│  │   Episodes           │   │                      │           │
│  └──────────────────────┘   └──────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Simulated Agent Personas

Each agent starts with a **different behavioral bias** (randomized but themed) to show that RL converges regardless of starting point:

| Agent Persona          | Starting Bias                                              |
|------------------------|------------------------------------------------------------|
| "Weight Loss"          | High body fat, poor nutrition, low exercise                 |
| "Stress Management"    | High cortisol, poor sleep, sedentary                        |
| "Muscle Gain"          | Moderate fitness, needs strength focus, high protein needs   |
| "Longevity"            | Focus on VO2 max, HRV, resting HR improvement               |
| "Random"               | Purely random actions — no bias                             |

Each persona has an **exploration policy** that starts biased toward their persona's habits but gradually shifts as the agent learns which actions yield higher rewards.

### 6.3 User Non-Compliance Model

Real users don't perfectly follow recommendations. The simulation must model this, or convergence is unrealistically fast and the learned policy is brittle.

#### Compliance Probability

When the agent recommends an action, the simulated user follows it with probability $p_{comply}$, otherwise falls back to their persona bias or a random action:

```
recommended_action = agent.policy(state)

if random() < p_comply:
    actual_action = recommended_action          # user follows advice
elif random() < p_persona:
    actual_action = persona_bias()              # user reverts to old habits
else:
    actual_action = random_action()             # user does something unrelated
```

#### Compliance Varies by Dimension and Difficulty

Not all recommendations are equally likely to be followed:

| Factor                        | Effect on Compliance                          |
|-------------------------------|-----------------------------------------------|
| **Magnitude of change**       | "Sleep 15 min earlier" → high comply. "Sleep 3 hrs earlier" → low comply |
| **Action category**           | Nutrition changes are hardest to sustain (social, habitual). Sleep timing second. Exercise most controllable. |
| **Streak length**             | Compliance dips after 5–7 days of sustained change (willpower fatigue) |
| **Weekend vs. weekday**       | Compliance drops on weekends (social pressure, routine disruption) |
| **How far from persona bias** | Recommendations close to current habits → high comply. Drastic changes → low comply |

#### Compliance Profiles (Per Persona)

| Persona              | Overall $p_{comply}$ | Notes                                         |
|----------------------|----------------------|-----------------------------------------------|
| "Weight Loss"        | 0.4                  | Low motivation, resists exercise and nutrition changes |
| "Stress Management"  | 0.5                  | Follows exercise, resists sleep changes       |
| "Muscle Gain"        | 0.55                 | Follows training, resists rest days           |
| "Longevity"          | 0.45                 | Moderate compliance across all categories     |
| "Random"             | 0.5                  | Baseline                                      |
| "Disciplined"        | 0.8                  | High compliance — shows upper bound of outcomes |
| "Resistant"          | 0.25                 | Rarely follows — shows the agent adapting to a difficult user |

**Add two new personas** ("Disciplined" and "Resistant") specifically to show the audience the contrast between high and low compliance and how the agent adapts differently.

#### What This Forces the Agent to Learn

With non-compliance modeled, the agent can't just output the theoretically optimal plan. It must learn:

1. **Gradual recommendations** — small steps toward optimal, not big jumps (because big jumps get ignored)
2. **Prioritization** — if the user only follows 1 of 3 recommendations, which one matters most today?
3. **Recovery strategies** — when the user doesn't comply, what's the best next action given where they actually are (not where they should be)?
4. **User-specific adaptation** — the "Resistant" agent learns to recommend smaller, easier changes. The "Disciplined" agent can recommend more aggressively.
5. **Weekend awareness** — recommendations for Friday/Saturday should account for likely non-compliance

#### Impact on Convergence

| Compliance Level | Convergence Speed     | Ceiling Score |
|------------------|-----------------------|---------------|
| Perfect (1.0)    | Fast (~500 episodes)  | ~95/100       |
| High (0.8)       | Moderate (~800 eps)   | ~85/100       |
| Medium (0.5)     | Slow (~1500 eps)      | ~72/100       |
| Low (0.25)       | Very slow (~3000 eps) | ~58/100       |

This is **exactly what the hackathon demo should show** — agents with highly compliant simulated users converge fast and score high. Agents with resistant users converge slowly but still improve meaningfully. The delta between "no guidance" and "RL guidance with low compliance" is the value the app provides even for unmotivated users.

### 6.4 What the Demo Shows

| Phase                    | What the Audience Sees                                      |
|--------------------------|-------------------------------------------------------------|
| **Episodes 1–50**        | Agents act mostly randomly/biased. Scores are low and erratic. Different agents have different weaknesses visible in the biomarker breakdown. |
| **Episodes 50–200**      | Agents start discovering basic patterns. "Weight Loss" agent starts exercising and eating better. "Stress Management" agent begins sleeping more. Scores trend upward but noisy. |
| **Episodes 200–500**     | Convergence becomes visible. All agents start gravitating toward similar high-reward patterns despite different starting points. Consistency bonuses kick in. |
| **Episodes 500–1000**    | Near-optimal policies. Agents have learned balanced sleep/exercise/nutrition patterns. Scores plateau near the maximum. Action distributions show clear preferences for healthy behaviors. |

### 6.5 Visualization Components

**Real-time dashboard showing:**

1. **Score timeline** — per-agent score curves over episodes (all agents on one chart, color-coded)
2. **Per-biomarker breakdown** — stacked bar per agent showing contribution of each biomarker to the reward
3. **Action heatmap** — what actions each agent is choosing over time (shows behavioral shift)
4. **Convergence indicator** — how similar the agents' policies are becoming
5. **Current episode replay** — animated view of what one agent "did today" (e.g., "Slept 7.5hrs, went for a run, ate balanced meals → Score: 83")
6. **Leaderboard** — ranked agents with current average score

**Controls:**
- Speed slider (1x, 10x, 100x) — speed up training for the audience
- Pause/resume — freeze to explain what's happening
- Select agent — drill into one agent's history and learned policy
- Reset — restart all agents from scratch with new random seeds

### 6.6 Technical Requirements

- Simulation runs **client-side or on a local server** (no cloud dependency during demo)
- Must handle 5–10 agents training in parallel
- Visualization updates in near real-time (at least 10 fps at 1x speed)
- Each agent's full episode history is stored and replayable
- The same payoff function and environment used for real user agents is used here — the demo is not a separate toy, it's the actual system training
- Simulation can be pre-seeded with a known random seed for reproducible demos

### 6.7 Relationship to Production System

```
SIMULATION (hackathon demo)          PRODUCTION (real users)
─────────────────────────            ──────────────────────
Multiple agents                      One agent per user
Simulated user actions               Real user inputs (via LLM extractor)
Same RL environment                  Same RL environment
Same payoff function                 Same payoff function
Same history store                   Same history store
Fast (1000 episodes in minutes)      Slow (1 episode = 1 real day)
Purpose: demonstrate learning        Purpose: guide real behavior
```

The simulation is a **fast-forward preview** of what happens for each real user over weeks/months, compressed into minutes.

---

## 7. Versioning & Phased Rollout

### V0 — Hackathon Demo
- RL environment + simulator + payoff function operational
- Multi-agent training simulation with visualization dashboard
- 7 personas (including Disciplined and Resistant) with non-compliance model
- Demonstrates convergence from random → optimal in real-time
- Pre-trained policy ready for V1 deployment

### V1 — MVP
- Outcome-based biomarker rewards with goal-specific weights
- DQN with biomarker state vector (8 biomarkers + deltas + goal)
- Deterministic payoff function based on biomarker deltas
- Text-based user input with LLM feature extraction
- Basic daily score + recommendations
- Cold-start onboarding questionnaire
- Missing-day handling

### V2 — Refined
- Expanded biomarker set and refined response models
- Research-derived weights
- Layer 2 state vector (history-derived stats: consistency, trends, recovery)
- PPO/SAC for continuous state space
- Interaction effects in payoff function
- Sequence bonuses folded into per-step reward
- Follow-up questions for ambiguous inputs
- Recommendation adherence tracking

### V3 — Personalized
- User-goal-dependent weights
- Deep RL for larger state space
- User profile in state space
- Food database integration

### V4 — Adaptive
- Learned weights from real user outcome data
- Personalized payoff function per user
- Continuous payoff refinement from growing paper corpus

---

## 8. Technical Stack (Suggested)

| Component                | Technology                                          |
|--------------------------|-----------------------------------------------------|
| **RL System**            |                                                     |
| RL Environment           | Python, Gymnasium (OpenAI Gym)                      |
| RL Algorithm             | Stable-Baselines3 (DQN → PPO/SAC)                  |
| **LLMs**                 |                                                     |
| Extraction / Payoff Gen  | OpenAI GPT-4+ or Claude API                         |
| User Feature Extraction  | Smaller/faster model (GPT-4o-mini, Haiku)           |
| **User App**             |                                                     |
| Backend                  | FastAPI or Flask                                    |
| Frontend                 | React Native or Flutter (mobile)                    |
| User Database            | PostgreSQL (user data, scores, history)             |
| History / Event Store    | PostgreSQL or TimescaleDB (time-series optimized)   |
| Experiment Tracking      | MLflow or Weights & Biases (training history)       |
| **Simulation / Demo**    |                                                     |
| Visualization            | Streamlit, Dash, or React (real-time dashboard)     |
| Charting                 | Plotly, D3.js, or Chart.js                          |
| Multi-agent runner       | Python multiprocessing or async                     |

---

## 9. Agent History & Observability

### 9.1 Training-Time History

Every training run is fully reproducible and auditable:

| What                     | Stored Where                                        |
|--------------------------|-----------------------------------------------------|
| Episode trajectories     | (state, action, reward, next_state) tuples per step |
| Policy checkpoints       | Serialized model at regular intervals               |
| Training config           | Hyperparameters, payoff version, schema version    |
| Performance curves       | Average reward, convergence metrics over episodes   |
| Payoff function version  | Which version of scoring rules was used             |
| Feature schema version   | Which features existed during this training run     |

Stored in experiment tracking tool (MLflow / W&B). Enables:
- Comparing agent performance across payoff function versions
- Diagnosing regressions when the payoff function is updated
- Reproducing any historical training run

### 9.2 Production-Time History (Per User)

Every interaction is logged:

| What                     | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| User raw input           | "Had a burger at 10pm"                              |
| Extracted features       | {meal_time: 22:00, food_quality: processed, ...}    |
| Feature confidence       | Which extractions were uncertain                    |
| Payoff score (per biomarker)| resting_hr: −0.3, hrv: +1.5, vo2_max: +0.12, …    |
| Sequence bonuses applied | Consistency: +3, Trend: +5, Recovery: 0             |
| Recommendation given     | "Try a 30-min walk tomorrow morning"                |
| Recommendation followed  | Yes / No / Partial (see §9.2.1)                    |
| Policy version           | Which trained agent generated the recommendation    |
| Payoff function version  | Which scoring rules produced the score              |

#### 9.2.1 Measuring Recommendation Adherence

Determining whether a user followed a recommendation is non-trivial — the user reports what they *did*, not whether they followed advice.

**Approach: Feature-level matching with LLM fallback**

```
Recommendation: "Try a 30-min walk tomorrow morning"
  → Structured: {exercise_type: light_cardio, duration: 30, timing: morning}

User input next day: "I went for a jog after breakfast"
  → Extracted: {exercise_type: moderate_cardio, duration: unknown, timing: morning}

Matching:
  exercise_type: light_cardio ≈ moderate_cardio → PARTIAL (same category)
  duration: 30 vs unknown → UNKNOWN (ask user or default to partial)
  timing: morning = morning → MATCH

Result: PARTIAL
```

**Rules:**
1. Compare recommendation features against extracted user action features
2. If >80% of features match closely → `followed: true`
3. If 30–80% match → `followed: partial`
4. If <30% match or opposite direction → `followed: false`
5. If extraction confidence is low → don't record adherence (data quality > quantity)
6. If ambiguous, the LLM feature extractor can ask: "Yesterday I suggested a morning walk — did you get a chance to try that?"

### 9.3 Lineage Tracing

Full traceability from any recommendation back to research:

```
Recommendation: "Go to bed by 10:30pm"
  ↑ Generated by policy v1.2
  ↑ Trained on payoff function v2.1
  ↑ Sleep timing rule derived from:
    ↑ Finding: "circadian-aligned sleep improves cognitive performance by 15%"
      ↑ Source: Walker 2017, Czeisler 2015 (meta-analyses)
        ↑ Derived from published research literature
```

This lineage is critical for:
- Explaining recommendations to users ("Based on research from...")
- Debugging bad recommendations
- Regulatory compliance if the app makes health claims
- Updating: when a source paper is retracted, trace which rules and recommendations it affected

### 9.4 Feedback Loop

User history feeds back into the system:

```
User history (actions, scores, recommendation adherence)
        ↓
Aggregate across users
        ↓
Validate: Do users who follow recommendations actually improve?
        ↓
If not → flag payoff function rules for review
        ↓
Refine simulator transition dynamics with real user data
```

---

## 10. Key Design Principles

1. **Research-grounded:** Every scoring rule traces back to a cited paper
2. **Incrementally complex:** Start coarse, add features only when they improve outcomes
3. **Modular:** Payoff pipeline, RL environment, and user app are independent systems
4. **Updatable:** New papers → new payoff function → retrained agent, without app changes
5. **Interpretable:** Users can understand why they received a score and recommendation
6. **Safe:** Human review required before any payoff function update is deployed
7. **History-first:** All actions, scores, recommendations, and versions are logged — consistency/trend rewards depend on it, and full lineage tracing requires it
8. **Observable:** Training runs, payoff function changes, and production recommendations are fully auditable
9. **Privacy-respecting:** User health data is treated as sensitive and handled according to applicable regulations

---

## 11. Data Privacy & Security

The system stores sensitive personal health data. This section defines the baseline requirements.

### 11.1 Data Classification

| Data Type                        | Sensitivity  | Retention Policy                          |
|----------------------------------|-------------|-------------------------------------------|
| Raw user input (natural language)| High        | Retained for lineage tracing. User can request deletion. |
| Extracted features               | High        | Retained as long as user account is active |
| Daily scores                     | Medium      | Retained for trend computation             |
| Recommendations given            | Medium      | Retained for adherence tracking + audit    |
| Aggregate/anonymized analytics   | Low         | Retained indefinitely                      |
| Research paper corpus            | Public      | No sensitivity — sourced from published literature |

### 11.2 Requirements

| Requirement                      | Details                                              |
|----------------------------------|------------------------------------------------------|
| **Encryption at rest**           | All user data encrypted in the database (AES-256 or equivalent) |
| **Encryption in transit**        | TLS 1.2+ for all API communication                  |
| **Access control**               | Role-based access. User data only accessible to the user and authorized system components. No admin backdoor to individual user data without audit log. |
| **Data deletion**                | Users can request full deletion of their data (right to be forgotten). Deletion propagates across all stores (user DB, history store, any caches). |
| **Data export**                  | Users can export their complete history in a portable format (JSON/CSV) |
| **Anonymization for analytics**  | Aggregate analysis (e.g., "do users who follow recommendations improve?") uses anonymized, de-identified data only |
| **LLM data handling**            | User inputs sent to external LLM APIs must not be used for model training. Use API providers that contractually guarantee this. Consider on-premise models for V3+. |
| **Consent**                      | Explicit user consent at onboarding for data collection, LLM processing, and score computation. Granular opt-out for specific data types. |
| **Audit logging**                | All access to user data is logged (who, when, what)  |
| **Geographic residency**         | Data stored in the user's region where required by regulation (GDPR, etc.) |

### 11.3 Regulatory Considerations

- **Health data regulations:** HIPAA (US), GDPR (EU), PIPEDA (Canada) may apply depending on jurisdiction and whether the app makes health claims.
- **Not a medical device:** The app provides wellness recommendations, not medical diagnoses or treatment. This distinction should be clear in the UI, terms of service, and any marketing.
- **Disclaimer:** All recommendations are general wellness guidance based on published research, not personalized medical advice.
