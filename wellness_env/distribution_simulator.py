"""Distribution-based biomarker change computation.

Replaces the rule-based compute_biomarker_changes() in simulator.py with
samples drawn from the user's fitted Gaussian copula for the five Garmin-
observable biomarkers.  The three unobservable biomarkers (VO2 max, lean
mass, energy level) fall back to literature-based heuristics because no
Garmin ground truth exists to inform the distribution.

Hybrid design:
  - Distribution handles:  RHR, HRV, SleepEfficiency, BodyFat, CortisolProxy
  - Heuristics handle:     VO2Max, LeanMass, EnergyLevel
  - Multi-day modifiers:   sleep debt, overtraining — applied post-sampling
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .models import (
    Action,
    Biomarkers,
    BiomarkerDeltas,
    ExerciseType,
    NutritionType,
    SLEEP_HOURS,
)
from .personas import PersonaConfig
from .simulator import (
    CARDIO_EXERCISES,
    INTENSE_EXERCISES,
    _clamp,
    _consecutive_intense_days,
    _recent_sleep_debt,
)
from backend.distribution_calibration import (
    JointDistribution,
    encode_action_to_features,
    sample_conditional,
)

# ---------------------------------------------------------------------------
# Physiologically plausible daily change limits (used for final clamping)
# ---------------------------------------------------------------------------
_DELTA_BOUNDS: dict[str, tuple[float, float]] = {
    "resting_hr":       (-5.0,  5.0),
    "hrv":             (-20.0, 20.0),
    "vo2_max":          (-0.5,  0.5),
    "body_fat_pct":     (-0.5,  0.5),
    "lean_mass_kg":     (-0.2,  0.2),
    "sleep_efficiency":(-15.0, 15.0),
    "cortisol_proxy":  (-15.0, 15.0),
    "energy_level":    (-20.0, 20.0),
}

# Scale factors converting Garmin outcomes to simulator biomarker deltas
_SLEEP_SCORE_TO_EFF = 1.0       # Both are 0-100 scales; Garmin score ≈ efficiency
_STRESS_TO_CORTISOL = 1.0       # Both are 0-100 scales, higher = worse
_WEIGHT_TO_BF_FRACTION = 0.8    # ~80% of daily weight change assumed to be fat


# ---------------------------------------------------------------------------
# Heuristics for unobservable biomarkers
# ---------------------------------------------------------------------------

def _heuristic_vo2_delta(
    action: Action,
    persona: PersonaConfig,
    debt_factor: float,
    protein_mult: float,
    is_overtraining: bool,
    rng: random.Random,
) -> float:
    """VO2 max change — literature-based; no Garmin ground truth available."""
    rm = persona.response_model
    d = 0.0
    if action.exercise in CARDIO_EXERCISES and not is_overtraining:
        intensity_mult = {
            ExerciseType.LIGHT_CARDIO:    0.5,
            ExerciseType.MODERATE_CARDIO: 1.0,
            ExerciseType.HIIT:            1.5,
        }.get(action.exercise, 0.0)
        d += rm.vo2_cardio_gain * intensity_mult * debt_factor * protein_mult
    if action.exercise == ExerciseType.NONE:
        d -= 0.02   # detraining decay
    d += rng.gauss(0, 0.1)
    return d


def _heuristic_lean_mass_delta(
    action: Action,
    persona: PersonaConfig,
    debt_factor: float,
    protein_mult: float,
    is_overtraining: bool,
    rng: random.Random,
) -> float:
    """Lean mass change — literature-based; no Garmin ground truth available."""
    rm = persona.response_model
    d = 0.0
    if action.exercise == ExerciseType.STRENGTH and not is_overtraining:
        d += rm.lean_mass_strength_gain * debt_factor * protein_mult
    elif action.exercise == ExerciseType.HIIT and not is_overtraining:
        d += rm.lean_mass_strength_gain * 0.5 * debt_factor * protein_mult
    if action.nutrition == NutritionType.HIGH_PROTEIN:
        d += rm.lean_mass_protein_gain
    if action.exercise == ExerciseType.NONE and action.nutrition in {
        NutritionType.SKIPPED, NutritionType.PROCESSED
    }:
        d -= 0.01   # muscle loss from inactivity + poor nutrition
    d += rng.gauss(0, 0.1)
    return d


def _heuristic_energy_delta(
    action: Action,
    persona: PersonaConfig,
    sleep_debt: float,
    is_intense: bool,
    is_overtraining: bool,
    current: Biomarkers,
) -> float:
    """Energy level change — subjective; no Garmin ground truth available."""
    rm = persona.response_model
    hours = SLEEP_HOURS[action.sleep]
    d = 0.0

    if 7.0 <= hours <= 9.0:
        d += rm.energy_nutrition_sensitivity * 0.6
    elif hours < 6.0:
        d -= rm.energy_nutrition_sensitivity
    elif hours > 9.0:
        d -= rm.energy_nutrition_sensitivity * 0.3   # post-oversleep grogginess

    nutrition_energy_map = {
        NutritionType.SKIPPED:      -1.5,
        NutritionType.PROCESSED:    -1.0,
        NutritionType.HIGH_CARB:     0.5,
        NutritionType.BALANCED:      1.0,
        NutritionType.HIGH_PROTEIN:  1.0,
    }
    d += rm.energy_nutrition_sensitivity * nutrition_energy_map.get(action.nutrition, 0.0)

    if action.exercise != ExerciseType.NONE:
        d += -3.0 if is_intense else 2.0

    if is_overtraining:
        d -= 8.0
    d -= sleep_debt * 1.5
    if current.cortisol_proxy > 60:
        d -= 2.0

    return d


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_biomarker_changes_from_distribution(
    action: Action,
    current: Biomarkers,
    persona: PersonaConfig,
    distribution: JointDistribution,
    history: list[dict[str, Any]],
    rng: random.Random,
) -> BiomarkerDeltas:
    """Compute one-day biomarker changes using the fitted joint distribution.

    Steps
    -----
    1. Encode today's action as a calibration feature vector X.
    2. Sample Y ~ P(Y | X) from the Gaussian copula.
    3. Map sampled Y columns to observed biomarker deltas.
    4. Apply multi-day modifiers (sleep debt, overtraining) post-sampling.
    5. Compute heuristic deltas for unobserved biomarkers.
    6. Clamp all deltas to physiologically plausible daily ranges.
    """
    rm = persona.response_model
    is_intense = action.exercise in INTENSE_EXERCISES
    sleep_debt = _recent_sleep_debt(history)
    consecutive_intense = _consecutive_intense_days(history)
    is_overtraining = is_intense and consecutive_intense >= rm.overtraining_threshold
    debt_factor = max(0.2, 1.0 - rm.sleep_debt_exercise_penalty * sleep_debt)
    protein_mult = (
        rm.protein_recovery_multiplier
        if action.nutrition == NutritionType.HIGH_PROTEIN and is_intense
        else 1.0
    )

    # --- Step 1: encode action ---
    x_vec = encode_action_to_features(action)

    # --- Step 2: sample from P(Y | X) ---
    # Bridge stdlib rng → numpy Generator with a fresh seed each call
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    y_sampled = sample_conditional(distribution, x_vec, np_rng)
    # y_sampled layout: [delta_sleep_score, delta_hrv, delta_rhr, delta_weight, delta_stress]

    # --- Step 3: map to biomarker deltas ---
    d_sleep_eff = float(y_sampled[0]) * _SLEEP_SCORE_TO_EFF
    d_hrv       = float(y_sampled[1])
    d_rhr       = float(y_sampled[2])
    delta_weight = float(y_sampled[3])
    d_cortisol  = float(y_sampled[4]) * _STRESS_TO_CORTISOL

    # Weight → body fat conversion (accounts for current body composition)
    total_mass = current.lean_mass_kg / max(0.01, 1.0 - current.body_fat_pct / 100.0)
    d_bf = delta_weight * _WEIGHT_TO_BF_FRACTION / max(total_mass, 30.0) * 100.0

    # --- Step 4: multi-day modifiers (applied on top of distribution samples) ---
    if is_overtraining:
        d_hrv      -= 5.0
        d_rhr      += 0.5
        d_cortisol += float(rm.overtraining_cortisol_spike)

    # Sleep debt reduces fat-loss benefit of exercise (scale only if losing fat)
    if delta_weight < 0:
        d_bf *= debt_factor

    # --- Step 5: heuristic deltas for unobservable biomarkers ---
    d_vo2    = _heuristic_vo2_delta(action, persona, debt_factor, protein_mult, is_overtraining, rng)
    d_lm     = _heuristic_lean_mass_delta(action, persona, debt_factor, protein_mult, is_overtraining, rng)
    d_energy = _heuristic_energy_delta(action, persona, sleep_debt, is_intense, is_overtraining, current)

    # --- Step 6: clamp to physiological daily change limits ---
    def _cd(val: float, key: str) -> float:
        lo, hi = _DELTA_BOUNDS[key]
        return _clamp(val, lo, hi)

    return BiomarkerDeltas(
        resting_hr=       _cd(d_rhr,       "resting_hr"),
        hrv=              _cd(d_hrv,       "hrv"),
        vo2_max=          _cd(d_vo2,       "vo2_max"),
        body_fat_pct=     _cd(d_bf,        "body_fat_pct"),
        lean_mass_kg=     _cd(d_lm,        "lean_mass_kg"),
        sleep_efficiency= _cd(d_sleep_eff, "sleep_efficiency"),
        cortisol_proxy=   _cd(d_cortisol,  "cortisol_proxy"),
        energy_level=     _cd(d_energy,    "energy_level"),
    )
