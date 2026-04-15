"""Physiological simulator — computes how actions change biomarkers.

This is the hidden response model.  Each persona responds differently to
the same action because they have different ResponseModel parameters.
The agent never sees the ResponseModel; it only sees the resulting
biomarker changes.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Any

from .models import (
    Action,
    Biomarkers,
    BiomarkerDeltas,
    ExerciseType,
    NutritionType,
    SleepDuration,
    SLEEP_HOURS,
)
from .personas import PersonaConfig, ResponseModel


INTENSE_EXERCISES = {ExerciseType.HIIT, ExerciseType.STRENGTH}
CARDIO_EXERCISES = {ExerciseType.LIGHT_CARDIO, ExerciseType.MODERATE_CARDIO, ExerciseType.HIIT}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _consecutive_intense_days(history: list[dict[str, Any]]) -> int:
    """Count consecutive intense exercise days looking backward."""
    count = 0
    for h in reversed(history):
        ex = h.get("actual_action", {}).get("exercise", "none")
        if ex in {e.value for e in INTENSE_EXERCISES}:
            count += 1
        else:
            break
    return count


def _recent_sleep_debt(history: list[dict[str, Any]], window: int = 7) -> float:
    """Accumulated sleep debt over the last `window` days."""
    debt = 0.0
    for h in history[-window:]:
        sleep_val = h.get("actual_action", {}).get("sleep", "7_to_8h")
        hours = SLEEP_HOURS.get(SleepDuration(sleep_val), 7.5)
        if hours < 7.0:
            debt += 7.0 - hours
    return debt


def compute_biomarker_changes(
    action: Action,
    current: Biomarkers,
    persona: PersonaConfig,
    history: list[dict[str, Any]],
    rng: stdlib_random.Random,
) -> BiomarkerDeltas:
    """Compute the change in each biomarker for one day.

    This is the core of the simulator — the hidden function the RL agent
    must learn through experience.
    """
    rm = persona.response_model
    hours = SLEEP_HOURS[action.sleep]
    sleep_debt = _recent_sleep_debt(history)
    consecutive_intense = _consecutive_intense_days(history)
    is_intense = action.exercise in INTENSE_EXERCISES
    is_cardio = action.exercise in CARDIO_EXERCISES

    # Track whether we're in overtraining
    is_overtraining = (
        is_intense and consecutive_intense >= rm.overtraining_threshold
    )

    # --- Sleep debt penalty factor (reduces exercise efficacy) ---
    debt_factor = max(0.2, 1.0 - rm.sleep_debt_exercise_penalty * sleep_debt)

    # --- Protein multiplier (boosts exercise recovery) ---
    protein_mult = (
        rm.protein_recovery_multiplier
        if action.nutrition == NutritionType.HIGH_PROTEIN and is_intense
        else 1.0
    )

    # =====================================================================
    # Resting Heart Rate (lower is better, down to ~45 bpm)
    # =====================================================================
    d_rhr = 0.0
    # Sleep benefit
    if hours >= 7.0:
        d_rhr += rm.rhr_sleep_benefit * ((hours - 7.0) / 2.0 + 1.0)
    elif hours < 6.0:
        d_rhr += 0.3  # poor sleep raises RHR
    # Exercise benefit
    if action.exercise != ExerciseType.NONE and not is_overtraining:
        d_rhr += rm.rhr_exercise_benefit * debt_factor
    if is_overtraining:
        d_rhr += 0.5  # overtraining raises RHR
    # Noise (Garmin variance baseline)
    d_rhr += rng.gauss(0, 1.5)

    # =====================================================================
    # Heart Rate Variability (higher is better, typical 20-100ms)
    # =====================================================================
    d_hrv = 0.0
    # Sleep benefit
    if hours >= 7.0:
        d_hrv += rm.hrv_sleep_sensitivity * (hours - 7.0)
    elif hours < 6.0:
        d_hrv -= rm.hrv_sleep_sensitivity * 1.5  # poor sleep tanks HRV
    # Overtraining tanks HRV
    if is_overtraining:
        d_hrv -= 5.0
    # Yoga specifically boosts HRV
    if action.exercise == ExerciseType.YOGA:
        d_hrv += 2.0 * debt_factor
    # Stress (cortisol proxy) depresses HRV
    if current.cortisol_proxy > 60:
        d_hrv -= 1.0
    # Noise (Garmin variance baseline)
    d_hrv += rng.gauss(0, 5.0)

    # =====================================================================
    # VO2 Max (higher is better, changes slowly)
    # =====================================================================
    d_vo2 = 0.0
    if is_cardio and not is_overtraining:
        intensity_mult = {
            ExerciseType.LIGHT_CARDIO: 0.5,
            ExerciseType.MODERATE_CARDIO: 1.0,
            ExerciseType.HIIT: 1.5,
        }.get(action.exercise, 0.0)
        d_vo2 += rm.vo2_cardio_gain * intensity_mult * debt_factor * protein_mult
    # Detraining: no exercise → slow decay
    if action.exercise == ExerciseType.NONE:
        d_vo2 -= 0.02
    # Noise (Garmin variance baseline)
    d_vo2 += rng.gauss(0, 0.1)

    # =====================================================================
    # Body Fat % (lower is better for weight loss, changes slowly)
    # =====================================================================
    d_bf = 0.0
    # Exercise burns fat
    if action.exercise != ExerciseType.NONE and not is_overtraining:
        intensity_mult = {
            ExerciseType.LIGHT_CARDIO: 0.4,
            ExerciseType.YOGA: 0.2,
            ExerciseType.MODERATE_CARDIO: 0.8,
            ExerciseType.STRENGTH: 0.7,
            ExerciseType.HIIT: 1.0,
        }.get(action.exercise, 0.0)
        d_bf += rm.body_fat_exercise_loss * intensity_mult * debt_factor
    # Nutrition effect
    nutrition_fat_map = {
        NutritionType.SKIPPED: -0.01,  # slight loss from not eating, but unhealthy
        NutritionType.PROCESSED: 0.0,  # no fat loss from junk
        NutritionType.HIGH_CARB: 0.0,  # positive = gaining if not burning
        NutritionType.BALANCED: -0.01,
        NutritionType.HIGH_PROTEIN: -0.015,
    }
    d_bf += rm.body_fat_nutrition_sensitivity * nutrition_fat_map.get(action.nutrition, 0.0) / 0.02
    # Poor sleep impairs fat loss
    if hours < 6.0:
        d_bf += 0.01  # poor sleep promotes fat retention
    # Noise (Garmin variance baseline)
    d_bf += rng.gauss(0, 0.2)

    # =====================================================================
    # Lean Mass (higher is better for muscle gain, changes slowly)
    # =====================================================================
    d_lm = 0.0
    if action.exercise == ExerciseType.STRENGTH and not is_overtraining:
        d_lm += rm.lean_mass_strength_gain * debt_factor * protein_mult
    elif action.exercise == ExerciseType.HIIT and not is_overtraining:
        d_lm += rm.lean_mass_strength_gain * 0.5 * debt_factor * protein_mult
    # Protein nutrition bonus
    if action.nutrition == NutritionType.HIGH_PROTEIN:
        d_lm += rm.lean_mass_protein_gain
    # Muscle loss from inactivity + poor nutrition
    if action.exercise == ExerciseType.NONE and action.nutrition in {
        NutritionType.SKIPPED, NutritionType.PROCESSED
    }:
        d_lm -= 0.01
    # Noise (Garmin variance baseline)
    d_lm += rng.gauss(0, 0.1)

    # =====================================================================
    # Sleep Efficiency (higher is better)
    # =====================================================================
    d_se = 0.0
    if hours >= 7.0 and hours <= 9.0:
        d_se += 1.0  # good sleep begets better sleep
    elif hours < 6.0:
        d_se -= 2.0
    elif hours > 9.0:
        d_se -= 0.5  # oversleep reduces efficiency
    # Exercise improves sleep (but not intense evening exercise)
    if action.exercise in {ExerciseType.MODERATE_CARDIO, ExerciseType.YOGA}:
        d_se += 0.8
    elif is_intense:
        d_se -= 0.5  # intense exercise can hurt sleep quality
    # High cortisol hurts sleep
    if current.cortisol_proxy > 60:
        d_se -= 1.0
    # Noise (Garmin variance baseline)
    d_se += rng.gauss(0, 3.0)

    # =====================================================================
    # Cortisol Proxy (lower is better, 0-100 scale)
    # =====================================================================
    d_cortisol = 0.0
    # Sleep recovery
    if hours >= 7.0:
        d_cortisol += rm.cortisol_sleep_recovery * (hours - 7.0) / 2.0
    elif hours < 6.0:
        d_cortisol += 5.0  # poor sleep raises cortisol
    # Exercise stress (acute)
    if is_intense:
        d_cortisol += rm.cortisol_exercise_stress
    elif action.exercise == ExerciseType.YOGA:
        d_cortisol -= 3.0  # yoga explicitly reduces stress
    elif action.exercise in {ExerciseType.LIGHT_CARDIO, ExerciseType.MODERATE_CARDIO}:
        d_cortisol -= 1.0  # moderate exercise mild stress relief
    # Overtraining cortisol spike
    if is_overtraining:
        d_cortisol += rm.overtraining_cortisol_spike
    # Nutrition stress
    if action.nutrition in {NutritionType.SKIPPED, NutritionType.PROCESSED}:
        d_cortisol += rm.cortisol_nutrition_stress
    elif action.nutrition in {NutritionType.BALANCED, NutritionType.HIGH_PROTEIN}:
        d_cortisol -= 1.0
    # Natural recovery tendency (cortisol wants to normalize)
    if current.cortisol_proxy > 50:
        d_cortisol -= 1.0  # slow natural recovery
    # Noise (Garmin variance baseline)
    d_cortisol += rng.gauss(0, 5.0)

    # =====================================================================
    # Energy Level (higher is better, subjective)
    # =====================================================================
    d_energy = 0.0
    # Sleep contribution
    if hours >= 7.0 and hours <= 9.0:
        d_energy += rm.energy_nutrition_sensitivity * 0.6
    elif hours < 6.0:
        d_energy -= rm.energy_nutrition_sensitivity
    elif hours > 9.0:
        d_energy -= rm.energy_nutrition_sensitivity * 0.3  # grogginess
    # Nutrition contribution
    nutrition_energy_map = {
        NutritionType.SKIPPED: -1.5,
        NutritionType.PROCESSED: -1.0,
        NutritionType.HIGH_CARB: 0.5,
        NutritionType.BALANCED: 1.0,
        NutritionType.HIGH_PROTEIN: 1.0,
    }
    d_energy += rm.energy_nutrition_sensitivity * nutrition_energy_map.get(action.nutrition, 0.0)
    # Exercise: acute energy cost, but post-exercise energy boost
    if action.exercise == ExerciseType.NONE:
        pass  # no change from exercise
    elif is_intense:
        d_energy -= 3.0  # short-term fatigue
    else:
        d_energy += 2.0  # light/moderate exercise boosts energy
    # Overtraining drains energy
    if is_overtraining:
        d_energy -= 8.0
    # Sleep debt drains energy
    d_energy -= sleep_debt * 1.5
    # High cortisol drains energy
    if current.cortisol_proxy > 60:
        d_energy -= 2.0
    # Noise (Garmin variance baseline)
    d_energy += rng.gauss(0, 5.0)

    return BiomarkerDeltas(
        resting_hr=round(d_rhr, 3),
        hrv=round(d_hrv, 3),
        vo2_max=round(d_vo2, 4),
        body_fat_pct=round(d_bf, 4),
        lean_mass_kg=round(d_lm, 4),
        sleep_efficiency=round(d_se, 3),
        cortisol_proxy=round(d_cortisol, 3),
        energy_level=round(d_energy, 3),
    )


def apply_deltas(current: Biomarkers, deltas: BiomarkerDeltas) -> Biomarkers:
    """Apply biomarker deltas to current state, clamping to valid ranges."""
    return Biomarkers(
        resting_hr=round(_clamp(current.resting_hr + deltas.resting_hr, 40.0, 120.0), 2),
        hrv=round(_clamp(current.hrv + deltas.hrv, 5.0, 150.0), 2),
        vo2_max=round(_clamp(current.vo2_max + deltas.vo2_max, 15.0, 70.0), 2),
        body_fat_pct=round(_clamp(current.body_fat_pct + deltas.body_fat_pct, 3.0, 50.0), 2),
        lean_mass_kg=round(_clamp(current.lean_mass_kg + deltas.lean_mass_kg, 30.0, 100.0), 2),
        sleep_efficiency=round(_clamp(current.sleep_efficiency + deltas.sleep_efficiency, 0.0, 100.0), 2),
        cortisol_proxy=round(_clamp(current.cortisol_proxy + deltas.cortisol_proxy, 0.0, 100.0), 2),
        energy_level=round(_clamp(current.energy_level + deltas.energy_level, 0.0, 100.0), 2),
    )


def apply_life_event(
    action: Action, rng: stdlib_random.Random
) -> Action:
    """5% chance of a random life disruption per day."""
    if rng.random() > 0.05:
        return action

    event = rng.choice(["bad_sleep", "missed_exercise", "social_dinner"])
    if event == "bad_sleep":
        return Action(sleep=SleepDuration.VERY_SHORT, exercise=action.exercise, nutrition=action.nutrition)
    elif event == "missed_exercise":
        return Action(sleep=action.sleep, exercise=ExerciseType.NONE, nutrition=action.nutrition)
    else:  # social dinner
        return Action(sleep=action.sleep, exercise=action.exercise, nutrition=NutritionType.PROCESSED)
