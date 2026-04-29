"""Pydantic models for the Outcome-Based Wellness Simulator.

MVP 2: Garmin-only automated data collection.
- Actions: sleep_duration (5) × bedtime_window (5) × activity_level (5) × exercise_type (5) × exercise_duration (5)
  = MultiDiscrete([5,5,5,5,5]) — 5 independent policy heads
- Outcomes: 7 Garmin-measured biomarkers (RHR, HRV, sleep score, stress, body battery,
  sleep stage quality, VO2max)
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SleepDuration(str, Enum):
    VERY_SHORT = "less_than_6h"
    SHORT = "6_to_7h"
    OPTIMAL_LOW = "7_to_8h"
    OPTIMAL_HIGH = "8_to_9h"
    LONG = "more_than_9h"


class ActivityLevel(str, Enum):
    REST_DAY = "rest_day"
    LIGHT_ACTIVITY = "light_activity"
    MODERATE_ACTIVITY = "moderate_activity"
    VIGOROUS_ACTIVITY = "vigorous_activity"
    HIGH_INTENSITY = "high_intensity"


class BedtimeWindow(str, Enum):
    """Target bedtime for circadian rhythm alignment."""
    EARLY = "before_10pm"
    OPTIMAL = "10pm_to_11pm"
    LATE = "11pm_to_midnight"
    VERY_LATE = "midnight_to_1am"
    EXTREMELY_LATE = "after_1am"


class ExerciseType(str, Enum):
    """Primary exercise modality for the day."""
    NONE = "rest"
    CARDIO = "cardio"          # running, cycling, swimming, walking, hiking
    STRENGTH = "strength"      # strength_training, weight-based
    FLEXIBILITY = "flexibility" # yoga, stretching, pilates
    HIIT = "hiit"              # interval_training, CrossFit-like


class ExerciseDuration(str, Enum):
    """Duration bucket for the primary exercise session."""
    NONE = "none"
    SHORT = "15_to_30min"
    MODERATE = "30_to_45min"
    LONG = "45_to_60min"
    EXTENDED = "over_60min"


class Goal(str, Enum):
    STRESS_MANAGEMENT = "stress_management"
    CARDIOVASCULAR_FITNESS = "cardiovascular_fitness"
    SLEEP_OPTIMIZATION = "sleep_optimization"
    RECOVERY_ENERGY = "recovery_energy"
    ACTIVE_LIVING = "active_living"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    sleep: SleepDuration = Field(description="Target sleep duration category")
    activity: ActivityLevel = Field(description="Activity level for today")
    bedtime: BedtimeWindow = Field(
        default=BedtimeWindow.OPTIMAL, description="Target bedtime window"
    )
    exercise_type: ExerciseType = Field(
        default=ExerciseType.NONE, description="Primary exercise modality"
    )
    exercise_duration: ExerciseDuration = Field(
        default=ExerciseDuration.NONE, description="Duration of primary exercise session"
    )


# ---------------------------------------------------------------------------
# Biomarkers — the outcome variables the agent observes (all Garmin-measured)
# ---------------------------------------------------------------------------

class Biomarkers(BaseModel):
    """Measurable health outcomes — all from Garmin wearable."""
    resting_hr: float = Field(ge=40, le=120, description="Resting heart rate (bpm)")
    hrv: float = Field(ge=5, le=150, description="Heart rate variability (ms RMSSD)")
    sleep_score: float = Field(ge=0, le=100, description="Garmin sleep score (0-100)")
    stress_avg: float = Field(ge=0, le=100, description="Garmin stress level (0=low, 100=high)")
    body_battery: float = Field(ge=0, le=100, description="Garmin body battery / recovery (0-100)")
    sleep_stage_quality: float = Field(
        ge=0, le=100, default=35.0,
        description="Deep + REM sleep as % of total sleep window (0-100)"
    )
    vo2_max: float = Field(
        ge=10, le=90, default=40.0,
        description="VO2max estimate (ml/kg/min)"
    )


class BiomarkerDeltas(BaseModel):
    """Change in each biomarker since last step. This IS the outcome signal."""
    resting_hr: float = Field(description="Δ resting HR (negative = improving)")
    hrv: float = Field(description="Δ HRV (positive = improving)")
    sleep_score: float = Field(description="Δ sleep score (positive = improving)")
    stress_avg: float = Field(description="Δ stress level (negative = improving)")
    body_battery: float = Field(description="Δ body battery (positive = improving)")
    sleep_stage_quality: float = Field(
        default=0.0, description="Δ sleep stage quality — deep+REM % (positive = improving)"
    )
    vo2_max: float = Field(
        default=0.0, description="Δ VO2max (positive = improving)"
    )


# ---------------------------------------------------------------------------
# Observation — what the agent sees
# ---------------------------------------------------------------------------

class OutcomeTrends(BaseModel):
    """7-day trends for outcome variables."""
    resting_hr_trend: float = Field(description="7-day slope of resting HR")
    hrv_trend: float = Field(description="7-day slope of HRV")
    sleep_score_trend: float = Field(description="7-day slope of sleep score")
    stress_avg_trend: float = Field(description="7-day slope of stress level")
    body_battery_trend: float = Field(description="7-day slope of body battery")
    sleep_stage_quality_trend: float = Field(
        default=0.0, description="7-day slope of sleep stage quality"
    )
    vo2_max_trend: float = Field(default=0.0, description="7-day slope of VO2max")
    reward_trend: float = Field(description="7-day slope of total reward")
    reward_consistency: float = Field(description="Stddev of reward over last 7 days")


class Observation(BaseModel):
    day: int = Field(ge=0, description="Current day in the episode")
    total_days: int = Field(description="Total days in this episode")
    goal: Goal = Field(description="User's primary health goal")
    biomarkers: Biomarkers = Field(description="Current biomarker values")
    deltas: BiomarkerDeltas = Field(description="Change since last step")
    trends: Optional[OutcomeTrends] = Field(
        default=None, description="7-day outcome trends (available after day 7)"
    )
    persona_name: str = Field(description="Name of the simulated user persona")
    compliance_rate: float = Field(ge=0, le=1, description="How often this persona follows recs")


# ---------------------------------------------------------------------------
# Reward — based on outcome deltas weighted by goal
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    resting_hr_reward: float = Field(description="Reward from resting HR change")
    hrv_reward: float = Field(description="Reward from HRV change")
    sleep_score_reward: float = Field(description="Reward from sleep score change")
    stress_avg_reward: float = Field(description="Reward from stress level change")
    body_battery_reward: float = Field(description="Reward from body battery change")
    sleep_stage_quality_reward: float = Field(
        default=0.0, description="Reward from sleep stage quality change"
    )
    vo2_max_reward: float = Field(default=0.0, description="Reward from VO2max change")
    total: float = Field(description="Goal-weighted total reward (0-100)")


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: RewardBreakdown
    done: bool = Field(description="Whether the episode has ended")
    info: dict = Field(default_factory=dict, description="Additional metadata")


# ---------------------------------------------------------------------------
# Full environment state (for grading — not seen by agent in practice)
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    day: int
    total_days: int
    goal: Goal
    persona_name: str
    compliance_rate: float
    biomarkers: Biomarkers
    history: list
    cumulative_reward: float
    action_space_description: str = Field(
        default="MultiDiscrete([5,5,5,5,5]) — sleep_duration × bedtime × activity_level × exercise_type × exercise_duration"
    )


# ---------------------------------------------------------------------------
# Sleep duration midpoints in hours
# ---------------------------------------------------------------------------

SLEEP_HOURS: dict[SleepDuration, float] = {
    SleepDuration.VERY_SHORT: 5.0,
    SleepDuration.SHORT: 6.5,
    SleepDuration.OPTIMAL_LOW: 7.5,
    SleepDuration.OPTIMAL_HIGH: 8.5,
    SleepDuration.LONG: 9.5,
}


# ---------------------------------------------------------------------------
# Activity level intensity estimates (for simulator)
# ---------------------------------------------------------------------------

ACTIVITY_INTENSITY: dict[ActivityLevel, dict] = {
    ActivityLevel.REST_DAY: {"active_minutes": 5, "active_calories": 50, "is_intense": False},
    ActivityLevel.LIGHT_ACTIVITY: {"active_minutes": 20, "active_calories": 150, "is_intense": False},
    ActivityLevel.MODERATE_ACTIVITY: {"active_minutes": 45, "active_calories": 300, "is_intense": False},
    ActivityLevel.VIGOROUS_ACTIVITY: {"active_minutes": 60, "active_calories": 450, "is_intense": True},
    ActivityLevel.HIGH_INTENSITY: {"active_minutes": 75, "active_calories": 550, "is_intense": True},
}

# ---------------------------------------------------------------------------
# Bedtime window midpoint decimal hours (e.g. 22.5 = 10:30pm)
# ---------------------------------------------------------------------------

BEDTIME_HOUR: dict[BedtimeWindow, float] = {
    BedtimeWindow.EARLY: 21.5,
    BedtimeWindow.OPTIMAL: 22.5,
    BedtimeWindow.LATE: 23.5,
    BedtimeWindow.VERY_LATE: 0.5,
    BedtimeWindow.EXTREMELY_LATE: 1.5,
}

# ---------------------------------------------------------------------------
# Exercise duration midpoints in minutes
# ---------------------------------------------------------------------------

EXERCISE_DURATION_MINUTES: dict[ExerciseDuration, float] = {
    ExerciseDuration.NONE: 0,
    ExerciseDuration.SHORT: 22,
    ExerciseDuration.MODERATE: 37,
    ExerciseDuration.LONG: 52,
    ExerciseDuration.EXTENDED: 75,
}

# ---------------------------------------------------------------------------
# Exercise type physiological properties used by the simulator
# ---------------------------------------------------------------------------

EXERCISE_TYPE_PROPERTIES: dict[ExerciseType, dict] = {
    ExerciseType.NONE: {
        "cardio_benefit": 0.0, "strength_benefit": 0.0,
        "flexibility_benefit": 0.0, "cortisol_factor": 0.0, "recovery_cost": 0.0,
    },
    ExerciseType.CARDIO: {
        "cardio_benefit": 1.0, "strength_benefit": 0.2,
        "flexibility_benefit": 0.1, "cortisol_factor": 0.6, "recovery_cost": 0.5,
    },
    ExerciseType.STRENGTH: {
        "cardio_benefit": 0.2, "strength_benefit": 1.0,
        "flexibility_benefit": 0.1, "cortisol_factor": 0.8, "recovery_cost": 0.7,
    },
    ExerciseType.FLEXIBILITY: {
        "cardio_benefit": 0.05, "strength_benefit": 0.05,
        "flexibility_benefit": 1.0, "cortisol_factor": -0.3, "recovery_cost": 0.1,
    },
    ExerciseType.HIIT: {
        "cardio_benefit": 1.2, "strength_benefit": 0.6,
        "flexibility_benefit": 0.0, "cortisol_factor": 1.2, "recovery_cost": 1.0,
    },
}

# ---------------------------------------------------------------------------
# Garmin activity typeKey → ExerciseType mapping
# ---------------------------------------------------------------------------

GARMIN_ACTIVITY_TYPE_MAP: dict[str, ExerciseType] = {
    # Cardio
    "running": ExerciseType.CARDIO,
    "cycling": ExerciseType.CARDIO,
    "swimming": ExerciseType.CARDIO,
    "walking": ExerciseType.CARDIO,
    "hiking": ExerciseType.CARDIO,
    "open_water_swimming": ExerciseType.CARDIO,
    "road_biking": ExerciseType.CARDIO,
    "mountain_biking": ExerciseType.CARDIO,
    "trail_running": ExerciseType.CARDIO,
    "treadmill_running": ExerciseType.CARDIO,
    "indoor_cycling": ExerciseType.CARDIO,
    "elliptical": ExerciseType.CARDIO,
    "rowing": ExerciseType.CARDIO,
    "stair_climbing": ExerciseType.CARDIO,
    "cardio": ExerciseType.CARDIO,
    # Strength
    "strength_training": ExerciseType.STRENGTH,
    "weight_training": ExerciseType.STRENGTH,
    "gym_and_fitness_equipment": ExerciseType.STRENGTH,
    "indoor_cardio": ExerciseType.STRENGTH,
    # Flexibility
    "yoga": ExerciseType.FLEXIBILITY,
    "pilates": ExerciseType.FLEXIBILITY,
    "stretching": ExerciseType.FLEXIBILITY,
    "barre": ExerciseType.FLEXIBILITY,
    "meditation": ExerciseType.FLEXIBILITY,
    "tai_chi": ExerciseType.FLEXIBILITY,
    # HIIT
    "interval_training": ExerciseType.HIIT,
    "hiit": ExerciseType.HIIT,
    "crossfit": ExerciseType.HIIT,
    "multi_sport": ExerciseType.HIIT,
    "functional_fitness": ExerciseType.HIIT,
    "circuit_training": ExerciseType.HIIT,
    # Fallback
    "other": ExerciseType.CARDIO,
    "none": ExerciseType.NONE,
}
