"""Distribution-based biomarker change computation.

Replaces the rule-based compute_biomarker_changes() in simulator.py with
samples drawn from the user's fitted Gaussian copula for the seven
Garmin-observable biomarkers.

MVP 2: All 7 biomarkers are handled directly (RHR, HRV, sleep_score,
stress_avg, body_battery, sleep_stage_quality, vo2_max).

Multi-day modifiers (sleep debt, overtraining) are applied post-sampling.
Backward compat: if distribution has n_y=5, the two new biomarkers fall
back to zero-delta defaults.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .models import (
    Action,
    Biomarkers,
    BiomarkerDeltas,
    SLEEP_HOURS,
    ACTIVITY_INTENSITY,
)
from .personas import PersonaConfig
from .simulator import (
    INTENSE_ACTIVITIES,
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
    "resting_hr":          (-5.0,  5.0),
    "hrv":                 (-20.0, 20.0),
    "sleep_score":         (-15.0, 15.0),
    "stress_avg":          (-15.0, 15.0),
    "body_battery":        (-20.0, 20.0),
    "sleep_stage_quality": (-10.0, 10.0),
    "vo2_max":             (-0.3,   0.3),
}


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
    3. Map sampled Y columns to biomarker deltas.
    4. Apply multi-day modifiers (sleep debt, overtraining) post-sampling.
    5. Clamp all deltas to physiologically plausible daily ranges.
    """
    rm = persona.response_model
    is_intense = ACTIVITY_INTENSITY[action.activity]["is_intense"]
    sleep_debt = _recent_sleep_debt(history)
    consecutive_intense = _consecutive_intense_days(history)
    is_overtraining = is_intense and consecutive_intense >= rm.overtraining_threshold
    debt_factor = max(0.2, 1.0 - rm.sleep_debt_exercise_penalty * sleep_debt)

    # --- Step 1: encode action (pass dist for compat detection) ---
    x_vec = encode_action_to_features(action, distribution)

    # --- Step 2: sample from P(Y | X) ---
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    y_sampled = sample_conditional(distribution, x_vec, np_rng)
    # y_sampled layout (v2): [sleep_score, hrv, rhr, stress, battery, stage_quality, vo2_max]
    # y_sampled layout (v1): [sleep_score, hrv, rhr, stress, battery]

    d_sleep_score = float(y_sampled[0])
    d_hrv = float(y_sampled[1])
    d_rhr = float(y_sampled[2])
    d_stress = float(y_sampled[3])
    d_battery = float(y_sampled[4])
    # v2 extras (default to 0 if legacy distribution)
    d_stage_quality = float(y_sampled[5]) if len(y_sampled) > 5 else 0.0
    d_vo2 = float(y_sampled[6]) if len(y_sampled) > 6 else 0.0

    # --- Step 3: Apply multi-day modifiers ---
    if is_overtraining:
        d_rhr += 2.0
        d_hrv -= 5.0
        d_stress += 5.0
        d_battery -= 8.0

    # Sleep debt penalty
    d_hrv -= sleep_debt * 1.5
    d_rhr += sleep_debt * 0.5
    d_sleep_score -= sleep_debt * 2.0
    d_battery -= sleep_debt * 2.0

    # Debt factor scales beneficial exercise deltas
    if d_hrv > 0:
        d_hrv *= debt_factor
    if d_rhr < 0:
        d_rhr *= debt_factor

    # --- Step 4: Clamp to physiologically plausible ranges ---
    d_rhr = _clamp(d_rhr, *_DELTA_BOUNDS["resting_hr"])
    d_hrv = _clamp(d_hrv, *_DELTA_BOUNDS["hrv"])
    d_sleep_score = _clamp(d_sleep_score, *_DELTA_BOUNDS["sleep_score"])
    d_stress = _clamp(d_stress, *_DELTA_BOUNDS["stress_avg"])
    d_battery = _clamp(d_battery, *_DELTA_BOUNDS["body_battery"])
    d_stage_quality = _clamp(d_stage_quality, *_DELTA_BOUNDS["sleep_stage_quality"])
    d_vo2 = _clamp(d_vo2, *_DELTA_BOUNDS["vo2_max"])

    return BiomarkerDeltas(
        resting_hr=round(d_rhr, 4),
        hrv=round(d_hrv, 4),
        sleep_score=round(d_sleep_score, 4),
        stress_avg=round(d_stress, 4),
        body_battery=round(d_battery, 4),
        sleep_stage_quality=round(d_stage_quality, 4),
        vo2_max=round(d_vo2, 5),
    )
