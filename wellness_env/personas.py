"""Persona definitions with hidden physiological response models.

MVP 1: Garmin-only, 2D action space (sleep × activity), 5 biomarkers.
Each persona has a unique response model — the mapping from actions to
biomarker changes is different per person.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .models import (
    Action,
    ActivityLevel,
    Biomarkers,
    Goal,
    SleepDuration,
)


@dataclass
class ResponseModel:
    """Hidden parameters controlling how a persona responds to actions.

    These are NOT visible to the agent.  They define this simulated
    person's unique physiology.
    """

    # ----- Sleep response -----
    hrv_sleep_sensitivity: float = 3.0
    rhr_sleep_benefit: float = -0.3
    cortisol_sleep_recovery: float = -3.0

    # ----- Activity response -----
    rhr_exercise_benefit: float = -0.1
    cortisol_exercise_stress: float = 5.0
    overtraining_threshold: int = 3

    # ----- Cross-action sensitivities -----
    sleep_debt_exercise_penalty: float = 0.1
    overtraining_cortisol_spike: float = 15.0

    # ----- Body battery / energy sensitivity -----
    energy_sensitivity: float = 8.0

    # ----- Circadian sensitivity (new) -----
    # Multiplier on how much bedtime alignment affects outcomes (1.0 = average)
    circadian_sensitivity: float = 1.0


@dataclass
class PersonaConfig:
    """Full persona definition: identity + defaults + hidden response model."""

    name: str
    compliance_rate: float
    goal: Goal
    sleep_default: SleepDuration
    activity_default: ActivityLevel
    starting_biomarkers: Biomarkers
    response_model: ResponseModel
    random_defaults: bool = False


# ---------------------------------------------------------------------------
# Persona library
# ---------------------------------------------------------------------------

PERSONAS: dict[str, PersonaConfig] = {
    "digital_twin": PersonaConfig(
        name="digital_twin",
        compliance_rate=0.8,
        goal=Goal.ACTIVE_LIVING,
        sleep_default=SleepDuration.SHORT,
        activity_default=ActivityLevel.REST_DAY,
        starting_biomarkers=Biomarkers(
            resting_hr=65.0, hrv=50.0, sleep_score=75.0,
            stress_avg=40.0, body_battery=60.0,
            sleep_stage_quality=35.0, vo2_max=40.0,
        ),
        response_model=ResponseModel(),
    ),
    "cardiovascular_fitness": PersonaConfig(
        name="cardiovascular_fitness",
        compliance_rate=0.9,
        goal=Goal.CARDIOVASCULAR_FITNESS,
        sleep_default=SleepDuration.OPTIMAL_HIGH,
        activity_default=ActivityLevel.HIGH_INTENSITY,
        starting_biomarkers=Biomarkers(
            resting_hr=60.0, hrv=60.0, sleep_score=80.0,
            stress_avg=30.0, body_battery=75.0,
            sleep_stage_quality=42.0, vo2_max=50.0,
        ),
        response_model=ResponseModel(rhr_exercise_benefit=-0.15),
    ),
    "stress_management": PersonaConfig(
        name="stress_management",
        compliance_rate=0.7,
        goal=Goal.STRESS_MANAGEMENT,
        sleep_default=SleepDuration.SHORT,
        activity_default=ActivityLevel.LIGHT_ACTIVITY,
        starting_biomarkers=Biomarkers(
            resting_hr=75.0, hrv=30.0, sleep_score=60.0,
            stress_avg=70.0, body_battery=40.0,
            sleep_stage_quality=28.0, vo2_max=35.0,
        ),
        response_model=ResponseModel(
            hrv_sleep_sensitivity=5.0, cortisol_sleep_recovery=-6.0, rhr_sleep_benefit=-0.5,
            circadian_sensitivity=1.3,  # more sensitive to circadian disruption
        ),
    ),
    "sedentary": PersonaConfig(
        name="sedentary",
        compliance_rate=0.4,
        goal=Goal.ACTIVE_LIVING,
        sleep_default=SleepDuration.OPTIMAL_LOW,
        activity_default=ActivityLevel.REST_DAY,
        starting_biomarkers=Biomarkers(
            resting_hr=72.0, hrv=40.0, sleep_score=70.0,
            stress_avg=50.0, body_battery=50.0,
            sleep_stage_quality=32.0, vo2_max=30.0,
        ),
        response_model=ResponseModel(),
    ),
    "poor_sleeper": PersonaConfig(
        name="poor_sleeper",
        compliance_rate=0.6,
        goal=Goal.SLEEP_OPTIMIZATION,
        sleep_default=SleepDuration.VERY_SHORT,
        activity_default=ActivityLevel.MODERATE_ACTIVITY,
        starting_biomarkers=Biomarkers(
            resting_hr=70.0, hrv=35.0, sleep_score=55.0,
            stress_avg=55.0, body_battery=45.0,
            sleep_stage_quality=24.0, vo2_max=37.0,
        ),
        response_model=ResponseModel(
            hrv_sleep_sensitivity=4.0, cortisol_sleep_recovery=-5.0,
            circadian_sensitivity=1.5,  # very sensitive to late bedtimes
        ),
    ),
}


# ---------------------------------------------------------------------------
# Compliance model
# ---------------------------------------------------------------------------

def apply_compliance(
    recommended: Action,
    persona: PersonaConfig,
    rng: random.Random,
) -> tuple[Action, bool]:
    """Apply persona compliance model.  Returns (actual_action, complied)."""
    if rng.random() < persona.compliance_rate:
        return recommended, True

    # Non-compliant: 60% revert to defaults, 40% random
    if rng.random() < 0.6:
        if persona.random_defaults:
            actual = Action(
                sleep=rng.choice(list(SleepDuration)),
                activity=rng.choice(list(ActivityLevel)),
            )
        else:
            actual = Action(
                sleep=persona.sleep_default,
                activity=persona.activity_default,
            )
    else:
        actual = Action(
            sleep=rng.choice(list(SleepDuration)),
            activity=rng.choice(list(ActivityLevel)),
        )
    return actual, False
