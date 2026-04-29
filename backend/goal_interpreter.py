"""Goal Interpreter — translates free-text goals into structured GoalProfiles.

Uses GPT-4o-mini to parse user goals like "pickleball tournament in 5 days"
into outcome weights, sport-specific exercise recommendations, and
periodization phases.  Also contains the sport similarity matrix for
proxy-based compliance scoring.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# GoalProfile — the structured output of interpret_goal()
# ---------------------------------------------------------------------------

@dataclass
class GoalProfile:
    original_text: str                      # "Pickleball tournament in 5 days"
    outcome_weights: dict[str, float]       # 7 biomarker weights summing to ~1.0
    recommended_sport: str                  # "pickleball" — specific activity
    recommended_duration_minutes: int       # 45 — optimal session length
    exercise_preferences: dict[str, float]  # exercise category relevance scores
    focus_summary: str                      # 1-line coaching focus for LLM prompts
    target_date: Optional[str]             # ISO date string or None
    days_to_target: Optional[int]
    periodization_phase: str               # base_build | specific_build | sharpen | taper | event_week
    supporting_exercises: list[str]        # e.g. ["strength (legs/core) 2x/week", "flexibility 1x/week"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoalProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Periodization — date-based backward planning
# ---------------------------------------------------------------------------

def get_periodization_phase(days_to_target: Optional[int]) -> str:
    """Map days-to-target into a training phase."""
    if days_to_target is None:
        return "ongoing"
    if days_to_target > 42:
        return "base_build"
    elif days_to_target > 20:
        return "specific_build"
    elif days_to_target > 6:
        return "sharpen"
    elif days_to_target >= 0:
        return "taper"
    else:
        return "event_week"


PHASE_MODIFIERS: dict[str, dict] = {
    "base_build": {
        "weight_multipliers": {
            "hrv": 1.0, "resting_hr": 1.0, "sleep_score": 1.0,
            "stress_avg": 0.8, "body_battery": 0.8,
            "sleep_stage_quality": 1.0, "vo2_max": 1.2,
        },
        "duration_factor": 1.0,
        "intensity_cap": "vigorous_activity",
        "description": "Build aerobic base, foundational strength, sleep consistency",
    },
    "specific_build": {
        "weight_multipliers": {
            "hrv": 1.0, "resting_hr": 1.0, "sleep_score": 0.9,
            "stress_avg": 0.9, "body_battery": 0.9,
            "sleep_stage_quality": 0.9, "vo2_max": 1.1,
        },
        "duration_factor": 1.1,
        "intensity_cap": "high_intensity",
        "description": "Goal-specific training ramps up, higher specificity",
    },
    "sharpen": {
        "weight_multipliers": {
            "hrv": 1.1, "resting_hr": 0.9, "sleep_score": 1.0,
            "stress_avg": 1.0, "body_battery": 1.0,
            "sleep_stage_quality": 1.0, "vo2_max": 1.0,
        },
        "duration_factor": 0.9,
        "intensity_cap": "vigorous_activity",
        "description": "Controlled load, skill work, higher specificity",
    },
    "taper": {
        "weight_multipliers": {
            "hrv": 1.2, "resting_hr": 0.7, "sleep_score": 1.2,
            "stress_avg": 1.3, "body_battery": 1.3,
            "sleep_stage_quality": 1.2, "vo2_max": 0.5,
        },
        "duration_factor": 0.6,
        "intensity_cap": "moderate_activity",
        "description": "Fatigue reduction, readiness optimization, reduce volume",
    },
    "event_week": {
        "weight_multipliers": {
            "hrv": 1.3, "resting_hr": 0.5, "sleep_score": 1.3,
            "stress_avg": 1.4, "body_battery": 1.4,
            "sleep_stage_quality": 1.3, "vo2_max": 0.3,
        },
        "duration_factor": 0.4,
        "intensity_cap": "light_activity",
        "description": "Minimal load, maximize recovery and readiness",
    },
    "ongoing": {
        "weight_multipliers": {
            "hrv": 1.0, "resting_hr": 1.0, "sleep_score": 1.0,
            "stress_avg": 1.0, "body_battery": 1.0,
            "sleep_stage_quality": 1.0, "vo2_max": 1.0,
        },
        "duration_factor": 1.0,
        "intensity_cap": "high_intensity",
        "description": "Ongoing progression without taper logic",
    },
}


def get_phase_modifiers(phase: str) -> dict:
    """Return weight multipliers and load constraints for a training phase."""
    return PHASE_MODIFIERS.get(phase, PHASE_MODIFIERS["ongoing"])


# ---------------------------------------------------------------------------
# Sport Similarity Matrix — for proxy-based compliance scoring
# ---------------------------------------------------------------------------

# Symmetric pairs — both orderings are checked at lookup time.
_SPORT_SIMILARITY_RAW: dict[tuple[str, str], float] = {
    # Racquet sports cluster
    ("pickleball", "badminton"): 0.85,
    ("pickleball", "tennis"): 0.80,
    ("pickleball", "squash"): 0.75,
    ("pickleball", "table_tennis"): 0.70,
    ("badminton", "tennis"): 0.75,
    ("badminton", "squash"): 0.70,
    ("tennis", "squash"): 0.70,

    # Running / cardio cluster
    ("running", "stair_climbing"): 0.70,
    ("running", "trail_running"): 0.95,
    ("running", "treadmill_running"): 0.95,
    ("running", "hiking"): 0.65,
    ("running", "cycling"): 0.55,
    ("running", "walking"): 0.50,
    ("running", "elliptical"): 0.70,
    ("cycling", "indoor_cycling"): 0.95,
    ("cycling", "road_biking"): 0.95,
    ("cycling", "mountain_biking"): 0.80,
    ("cycling", "elliptical"): 0.60,
    ("swimming", "open_water_swimming"): 0.95,
    ("swimming", "rowing"): 0.55,
    ("walking", "hiking"): 0.80,

    # Strength cluster
    ("strength_training", "weight_training"): 0.95,
    ("strength_training", "crossfit"): 0.70,
    ("strength_training", "gym_and_fitness_equipment"): 0.85,
    ("strength_training", "functional_fitness"): 0.75,

    # Flexibility cluster
    ("yoga", "stretching"): 0.90,
    ("yoga", "pilates"): 0.80,
    ("yoga", "barre"): 0.65,
    ("yoga", "tai_chi"): 0.70,
    ("pilates", "barre"): 0.75,

    # HIIT cluster
    ("hiit", "interval_training"): 0.95,
    ("hiit", "crossfit"): 0.80,
    ("hiit", "circuit_training"): 0.85,

    # Cross-cluster (low similarity)
    ("pickleball", "running"): 0.30,
    ("pickleball", "cycling"): 0.20,
    ("yoga", "running"): 0.15,
    ("yoga", "cycling"): 0.15,
    ("swimming", "running"): 0.45,
    ("strength_training", "running"): 0.30,
    ("strength_training", "yoga"): 0.20,
}

# Build bidirectional lookup
SPORT_SIMILARITY: dict[tuple[str, str], float] = {}
for (a, b), score in _SPORT_SIMILARITY_RAW.items():
    SPORT_SIMILARITY[(a, b)] = score
    SPORT_SIMILARITY[(b, a)] = score

# Category grouping for fallback similarity
_CATEGORY_MAP: dict[str, str] = {
    "running": "cardio", "trail_running": "cardio", "treadmill_running": "cardio",
    "cycling": "cardio", "indoor_cycling": "cardio", "road_biking": "cardio",
    "mountain_biking": "cardio", "swimming": "cardio", "open_water_swimming": "cardio",
    "walking": "cardio", "hiking": "cardio", "elliptical": "cardio",
    "rowing": "cardio", "stair_climbing": "cardio",
    "pickleball": "racquet", "tennis": "racquet", "badminton": "racquet",
    "squash": "racquet", "table_tennis": "racquet",
    "strength_training": "strength", "weight_training": "strength",
    "gym_and_fitness_equipment": "strength", "functional_fitness": "strength",
    "yoga": "flexibility", "pilates": "flexibility", "stretching": "flexibility",
    "barre": "flexibility", "tai_chi": "flexibility", "meditation": "flexibility",
    "hiit": "hiit", "interval_training": "hiit", "crossfit": "hiit",
    "circuit_training": "hiit", "multi_sport": "hiit",
}


def _normalize_sport_name(name: str) -> str:
    """Normalize a sport name for lookup (lowercase, underscored)."""
    if not name:
        return ""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def get_sport_similarity(sport_a: str, sport_b: str) -> float:
    """Get similarity between two sports (0.0–1.0).

    Returns:
        1.0 for exact match
        Matrix value if pair exists
        0.5 if same exercise category
        0.2 if different categories
        0.0 if either sport is empty/none
    """
    a = _normalize_sport_name(sport_a)
    b = _normalize_sport_name(sport_b)

    if not a or not b or a in ("none", "rest") or b in ("none", "rest"):
        return 0.0
    if a == b:
        return 1.0

    # Direct matrix lookup
    if (a, b) in SPORT_SIMILARITY:
        return SPORT_SIMILARITY[(a, b)]

    # Category fallback
    cat_a = _CATEGORY_MAP.get(a)
    cat_b = _CATEGORY_MAP.get(b)
    if cat_a and cat_b:
        if cat_a == cat_b:
            return 0.5
        return 0.2

    # Unknown sport — minimal credit for trying
    return 0.2


def get_sport_compliance(
    recommended_sport: str,
    actual_activity: Optional[str],
    actual_duration_minutes: Optional[int],
    target_duration_minutes: int,
) -> float:
    """Compute activity compliance based on sport similarity × duration adherence.

    Returns:
        0.0–1.0 compliance score.
    """
    if not actual_activity or actual_activity in ("none", "rest", ""):
        return 0.0

    similarity = get_sport_similarity(recommended_sport, actual_activity)

    # Duration compliance: min(1.0, actual / target), but if target is 0 skip
    if target_duration_minutes and target_duration_minutes > 0 and actual_duration_minutes is not None:
        duration_factor = min(1.0, max(0.0, actual_duration_minutes / target_duration_minutes))
    else:
        # No duration data — give benefit of doubt
        duration_factor = 0.7

    return round(similarity * duration_factor, 4)


# ---------------------------------------------------------------------------
# LLM Goal Interpretation
# ---------------------------------------------------------------------------

_OUTCOME_DESCRIPTIONS = """
The 7 biomarker outcomes and their meanings:
1. resting_hr: Resting heart rate (lower is better for fitness)
2. hrv: Heart rate variability RMSSD (higher is better for recovery/fitness)
3. sleep_score: Garmin sleep score 0-100 (higher is better)
4. stress_avg: Average daily stress 0-100 (lower is better)
5. body_battery: Recovery/energy score 0-100 (higher is better)
6. sleep_stage_quality: Deep+REM sleep as % of total (higher is better)
7. vo2_max: VO2max estimate ml/kg/min (higher is better for aerobic fitness)
"""

_EXERCISE_CATEGORIES = """
Exercise type categories with relevance scores (0.0-1.0):
- cardio: running, cycling, swimming, walking, hiking
- strength: weight training, resistance exercises
- flexibility: yoga, stretching, pilates
- hiit: high-intensity interval training, CrossFit
- sport_specific: sports like pickleball, tennis, basketball, etc.
"""


def interpret_goal(
    free_text_goal: str,
    target_date: Optional[date] = None,
) -> GoalProfile:
    """Use LLM to translate a free-text goal into a structured GoalProfile.

    Calls GPT-4o-mini with a structured prompt that maps the goal to:
    - 7 biomarker outcome weights
    - A specific recommended sport/activity
    - Optimal session duration
    - Exercise type preferences
    - A 1-line focus summary
    """
    days_to_target = None
    if target_date:
        days_to_target = (target_date - date.today()).days
        if days_to_target < 0:
            days_to_target = 0  # past date — treat as event_week

    phase = get_periodization_phase(days_to_target)

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

        prompt = f"""You are a sports science expert. A user has set a wellness/fitness goal. Parse it into structured data.

USER GOAL: "{free_text_goal}"
TARGET DATE: {target_date.isoformat() if target_date else "None (ongoing)"}
DAYS TO TARGET: {days_to_target if days_to_target is not None else "N/A (ongoing)"}
CURRENT PHASE: {phase}

{_OUTCOME_DESCRIPTIONS}

{_EXERCISE_CATEGORIES}

You MUST output valid JSON with exactly these fields:
{{
  "outcome_weights": {{
    "resting_hr": <float 0-1>,
    "hrv": <float 0-1>,
    "sleep_score": <float 0-1>,
    "stress_avg": <float 0-1>,
    "body_battery": <float 0-1>,
    "sleep_stage_quality": <float 0-1>,
    "vo2_max": <float 0-1>
  }},
  "recommended_sport": "<specific sport/activity to practice, e.g. pickleball, running, swimming>",
  "recommended_duration_minutes": <int, optimal session length in minutes for the recommended sport given the current phase>,
  "exercise_preferences": {{
    "cardio": <float 0-1>,
    "strength": <float 0-1>,
    "flexibility": <float 0-1>,
    "hiit": <float 0-1>,
    "sport_specific": <float 0-1>
  }},
  "focus_summary": "<1-line coaching focus>",
  "supporting_exercises": ["<e.g. strength (legs/core) 2x/week>", "<flexibility 1x/week>"]
}}

RULES:
- outcome_weights must sum to approximately 1.0
- recommended_sport must be a specific sport/activity name (e.g., "pickleball", "running", "swimming"), not a category
- recommended_duration_minutes should account for the current training phase ({phase}):
  - base_build: full duration
  - specific_build: slightly above full duration
  - sharpen: slightly reduced
  - taper/event_week: significantly reduced (50-60% of normal)
  - ongoing: full duration
- supporting_exercises should list 2-3 complementary exercises with weekly frequency
- exercise_preferences scores should reflect how relevant each category is to the goal"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        parsed = json.loads(response.choices[0].message.content)

        return GoalProfile(
            original_text=free_text_goal,
            outcome_weights=parsed.get("outcome_weights", {}),
            recommended_sport=parsed.get("recommended_sport", "general_fitness"),
            recommended_duration_minutes=int(parsed.get("recommended_duration_minutes", 30)),
            exercise_preferences=parsed.get("exercise_preferences", {}),
            focus_summary=parsed.get("focus_summary", f"Optimize for: {free_text_goal}"),
            target_date=target_date.isoformat() if target_date else None,
            days_to_target=days_to_target,
            periodization_phase=phase,
            supporting_exercises=parsed.get("supporting_exercises", []),
        )

    except Exception as e:
        print(f"[GoalInterpreter] LLM error: {e}")
        # Fallback: return sensible defaults
        return GoalProfile(
            original_text=free_text_goal,
            outcome_weights={
                "resting_hr": 0.14, "hrv": 0.18, "sleep_score": 0.14,
                "stress_avg": 0.14, "body_battery": 0.14,
                "sleep_stage_quality": 0.12, "vo2_max": 0.14,
            },
            recommended_sport="general_fitness",
            recommended_duration_minutes=30,
            exercise_preferences={
                "cardio": 0.3, "strength": 0.25, "flexibility": 0.15,
                "hiit": 0.15, "sport_specific": 0.15,
            },
            focus_summary=f"General wellness: {free_text_goal}",
            target_date=target_date.isoformat() if target_date else None,
            days_to_target=days_to_target,
            periodization_phase=phase,
            supporting_exercises=["cardio 3x/week", "strength 2x/week"],
        )
