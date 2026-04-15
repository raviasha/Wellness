"""Persona definitions with hidden physiological response models.

Each persona has a unique response model — the mapping from actions to
biomarker changes is different per person.  The agent never sees these
parameters; it only sees the resulting biomarker deltas.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .models import (
    Action,
    Biomarkers,
    ExerciseType,
    Goal,
    NutritionType,
    SleepDuration,
)


@dataclass
class ResponseModel:
    """Hidden parameters controlling how a persona responds to actions.

    These are NOT visible to the agent.  They define this simulated
    person's unique physiology.
    """

    # ----- Sleep response -----
    # How much HRV improves per hour of sleep above 7h (ms/hour)
    hrv_sleep_sensitivity: float = 3.0
    # How much resting HR drops per optimal-sleep night (bpm)
    rhr_sleep_benefit: float = -0.3
    # Sleep efficiency baseline (%) — some people sleep more efficiently
    sleep_efficiency_base: float = 85.0
    # How much cortisol drops per optimal-sleep night
    cortisol_sleep_recovery: float = -3.0

    # ----- Exercise response -----
    # VO2 max improvement per cardio session (ml/kg/min)
    vo2_cardio_gain: float = 0.15
    # Lean mass gain per strength session (kg)
    lean_mass_strength_gain: float = 0.05
    # Body fat loss per intense exercise session (%)
    body_fat_exercise_loss: float = -0.03
    # Resting HR improvement per exercise session (bpm)
    rhr_exercise_benefit: float = -0.1
    # Cortisol rise from intense exercise (0-100 scale)
    cortisol_exercise_stress: float = 5.0
    # Overtraining threshold — consecutive intense days before harm
    overtraining_threshold: int = 3

    # ----- Nutrition response -----
    # Body fat change per day from nutrition quality (-0.05 to +0.05)
    body_fat_nutrition_sensitivity: float = 0.02
    # Lean mass response to protein (kg per high-protein day)
    lean_mass_protein_gain: float = 0.02
    # Energy response to nutrition quality (0-100 scale per day)
    energy_nutrition_sensitivity: float = 8.0
    # Cortisol response to poor nutrition
    cortisol_nutrition_stress: float = 3.0

    # ----- Cross-action sensitivities -----
    # How much sleep debt hurts exercise gains (multiplier reduction per debt hour)
    sleep_debt_exercise_penalty: float = 0.1
    # How much protein intake boosts post-exercise recovery (multiplier)
    protein_recovery_multiplier: float = 1.3
    # Overtraining cortisol spike
    overtraining_cortisol_spike: float = 15.0


@dataclass
class PersonaConfig:
    """Full persona definition: identity + defaults + hidden response model."""

    name: str
    compliance_rate: float
    goal: Goal
    sleep_default: SleepDuration
    exercise_default: ExerciseType
    nutrition_default: NutritionType
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
        goal=Goal.OVERALL_WELLNESS,
        sleep_default=SleepDuration.SHORT,
        exercise_default=ExerciseType.NONE,
        nutrition_default=NutritionType.BALANCED,
        starting_biomarkers=Biomarkers(
            resting_hr=65.0, hrv=50.0, vo2_max=35.0, body_fat_pct=20.0,
            lean_mass_kg=60.0, sleep_efficiency=80.0, cortisol_proxy=40.0, energy_level=60.0,
        ),
        response_model=ResponseModel(),
    ),
    "athletic_performance": PersonaConfig(
        name="athletic_performance",
        compliance_rate=0.9,
        goal=Goal.OVERALL_WELLNESS,
        sleep_default=SleepDuration.OPTIMAL_HIGH,
        exercise_default=ExerciseType.HIIT,
        nutrition_default=NutritionType.HIGH_PROTEIN,
        starting_biomarkers=Biomarkers(
            resting_hr=60.0, hrv=60.0, vo2_max=45.0, body_fat_pct=15.0,
            lean_mass_kg=65.0, sleep_efficiency=85.0, cortisol_proxy=30.0, energy_level=80.0,
        ),
        response_model=ResponseModel(
            vo2_cardio_gain=0.2, lean_mass_strength_gain=0.08, body_fat_exercise_loss=-0.05
        ),
    ),
    "stress_management": PersonaConfig(
        name="stress_management",
        compliance_rate=0.7,
        goal=Goal.STRESS_MANAGEMENT,
        sleep_default=SleepDuration.SHORT,
        exercise_default=ExerciseType.YOGA,
        nutrition_default=NutritionType.PROCESSED,
        starting_biomarkers=Biomarkers(
            resting_hr=75.0, hrv=30.0, vo2_max=32.0, body_fat_pct=25.0,
            lean_mass_kg=55.0, sleep_efficiency=70.0, cortisol_proxy=70.0, energy_level=40.0,
        ),
        response_model=ResponseModel(
            hrv_sleep_sensitivity=5.0, cortisol_sleep_recovery=-6.0, rhr_sleep_benefit=-0.5
        ),
    ),
    "weight_loss": PersonaConfig(
        name="weight_loss",
        compliance_rate=0.5,
        goal=Goal.WEIGHT_LOSS,
        sleep_default=SleepDuration.OPTIMAL_LOW,
        exercise_default=ExerciseType.NONE,
        nutrition_default=NutritionType.HIGH_CARB,
        starting_biomarkers=Biomarkers(
            resting_hr=72.0, hrv=40.0, vo2_max=30.0, body_fat_pct=30.0,
            lean_mass_kg=50.0, sleep_efficiency=75.0, cortisol_proxy=50.0, energy_level=50.0,
        ),
        response_model=ResponseModel(
            body_fat_nutrition_sensitivity=0.04, body_fat_exercise_loss=-0.06
        ),
    ),
}


# ---------------------------------------------------------------------------
# Compliance model (same as original)
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
                exercise=rng.choice(list(ExerciseType)),
                nutrition=rng.choice(list(NutritionType)),
            )
        else:
            actual = Action(
                sleep=persona.sleep_default,
                exercise=persona.exercise_default,
                nutrition=persona.nutrition_default,
            )
    else:
        actual = Action(
            sleep=rng.choice(list(SleepDuration)),
            exercise=rng.choice(list(ExerciseType)),
            nutrition=rng.choice(list(NutritionType)),
        )
    return actual, False
