"""Pydantic models for the Outcome-Based Wellness Simulator."""

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


class ExerciseType(str, Enum):
    NONE = "none"
    LIGHT_CARDIO = "light_cardio"
    MODERATE_CARDIO = "moderate_cardio"
    HIIT = "hiit"
    STRENGTH = "strength"
    YOGA = "yoga"


class NutritionType(str, Enum):
    HIGH_PROTEIN = "high_protein"
    BALANCED = "balanced"
    HIGH_CARB = "high_carb"
    PROCESSED = "processed"
    SKIPPED = "skipped"


class Goal(str, Enum):
    WEIGHT_LOSS = "weight_loss"
    OVERALL_WELLNESS = "overall_wellness"
    STRESS_MANAGEMENT = "stress_management"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    sleep: SleepDuration = Field(description="Target sleep duration category")
    exercise: ExerciseType = Field(description="Exercise type for today")
    nutrition: NutritionType = Field(description="Nutrition category for today")


# ---------------------------------------------------------------------------
# Biomarkers — the outcome variables the agent observes
# ---------------------------------------------------------------------------

class Biomarkers(BaseModel):
    """Measurable health outcomes."""
    resting_hr: float = Field(ge=40, le=120, description="Resting heart rate (bpm)")
    hrv: float = Field(ge=5, le=150, description="Heart rate variability (ms RMSSD)")
    vo2_max: float = Field(ge=15, le=70, description="Estimated VO2 max (ml/kg/min)")
    body_fat_pct: float = Field(ge=3, le=50, description="Body fat percentage")
    lean_mass_kg: float = Field(ge=30, le=100, description="Lean body mass (kg)")
    sleep_efficiency: float = Field(ge=0, le=100, description="Sleep efficiency %")
    cortisol_proxy: float = Field(ge=0, le=100, description="Stress proxy (0=low, 100=high)")
    energy_level: float = Field(ge=0, le=100, description="Subjective energy (0-100)")


class BiomarkerDeltas(BaseModel):
    """Change in each biomarker since last step. This IS the outcome signal."""
    resting_hr: float = Field(description="Δ resting HR (negative = improving)")
    hrv: float = Field(description="Δ HRV (positive = improving)")
    vo2_max: float = Field(description="Δ VO2 max (positive = improving)")
    body_fat_pct: float = Field(description="Δ body fat % (negative = improving for weight loss)")
    lean_mass_kg: float = Field(description="Δ lean mass (positive = improving for muscle gain)")
    sleep_efficiency: float = Field(description="Δ sleep efficiency (positive = improving)")
    cortisol_proxy: float = Field(description="Δ cortisol proxy (negative = improving)")
    energy_level: float = Field(description="Δ energy (positive = improving)")


# ---------------------------------------------------------------------------
# Observation — what the agent sees
# ---------------------------------------------------------------------------

class OutcomeTrends(BaseModel):
    """7-day trends for outcome variables."""
    resting_hr_trend: float = Field(description="7-day slope of resting HR")
    hrv_trend: float = Field(description="7-day slope of HRV")
    vo2_max_trend: float = Field(description="7-day slope of VO2 max")
    body_fat_trend: float = Field(description="7-day slope of body fat %")
    lean_mass_trend: float = Field(description="7-day slope of lean mass")
    sleep_efficiency_trend: float = Field(description="7-day slope of sleep efficiency")
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
    vo2_max_reward: float = Field(description="Reward from VO2 max change")
    body_fat_reward: float = Field(description="Reward from body fat change")
    lean_mass_reward: float = Field(description="Reward from lean mass change")
    sleep_efficiency_reward: float = Field(description="Reward from sleep efficiency change")
    cortisol_reward: float = Field(description="Reward from cortisol proxy change")
    energy_reward: float = Field(description="Reward from energy change")
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
        default="sleep × exercise × nutrition = 5 × 6 × 5 = 150 combos"
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
