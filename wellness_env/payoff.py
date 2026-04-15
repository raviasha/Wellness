"""Outcome-based reward computation.

Reward = sum of goal-weighted biomarker deltas, normalized to [0, 100].
Each goal defines which outcomes matter most.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .models import BiomarkerDeltas, Biomarkers, Goal, RewardBreakdown


# ---------------------------------------------------------------------------
# Goal-specific weights for each biomarker delta.
#
# Convention:
#   - Negative deltas are "good" for resting_hr, body_fat_pct, cortisol_proxy
#     → we negate them before weighting so positive = better for all.
#   - All weights for a goal sum to ~1.0.
# ---------------------------------------------------------------------------

GOAL_WEIGHTS: dict[Goal, dict[str, float]] = {
    Goal.WEIGHT_LOSS: {
        "resting_hr": 0.05,
        "hrv": 0.05,
        "vo2_max": 0.05,
        "body_fat_pct": 0.35,
        "lean_mass_kg": 0.10,
        "sleep_efficiency": 0.10,
        "cortisol_proxy": 0.10,
        "energy_level": 0.20,
    },

    Goal.OVERALL_WELLNESS: {
        "resting_hr": 0.125,
        "hrv": 0.125,
        "vo2_max": 0.125,
        "body_fat_pct": 0.125,
        "lean_mass_kg": 0.125,
        "sleep_efficiency": 0.125,
        "cortisol_proxy": 0.125,
        "energy_level": 0.125,
    },

    Goal.STRESS_MANAGEMENT: {
        "resting_hr": 0.10,
        "hrv": 0.20,
        "vo2_max": 0.05,
        "body_fat_pct": 0.05,
        "lean_mass_kg": 0.05,
        "sleep_efficiency": 0.20,
        "cortisol_proxy": 0.25,
        "energy_level": 0.10,
    },
}

# ---------------------------------------------------------------------------
# Normalization scales: how much delta is "very good" for each biomarker.
# These are used to convert raw deltas into a 0-100 reward contribution.
# ---------------------------------------------------------------------------

DELTA_SCALES: dict[str, float] = {
    "resting_hr": 1.0,       # −1 bpm/day is excellent
    "hrv": 5.0,              # +5 ms/day is excellent
    "vo2_max": 0.3,          # +0.3 ml/kg/min per day is excellent
    "body_fat_pct": 0.05,    # −0.05% per day is excellent
    "lean_mass_kg": 0.1,     # +0.1 kg per day is excellent
    "sleep_efficiency": 2.0,  # +2% per day is excellent
    "cortisol_proxy": 5.0,   # −5 points per day is excellent
    "energy_level": 10.0,    # +10 per day is excellent
}

# Biomarkers where LOWER is better (we negate so positive = good)
LOWER_IS_BETTER = {"resting_hr", "body_fat_pct", "cortisol_proxy"}

# ---------------------------------------------------------------------------
# State quality ranges for normalizing absolute biomarker values.
# ---------------------------------------------------------------------------

STATE_RANGES: dict[str, tuple[float, float]] = {
    "resting_hr": (40, 120),
    "hrv": (5, 150),
    "vo2_max": (15, 70),
    "body_fat_pct": (3, 50),
    "lean_mass_kg": (30, 100),
    "sleep_efficiency": (0, 100),
    "cortisol_proxy": (0, 100),
    "energy_level": (0, 100),
}

# Blend ratio: how much of the reward comes from deltas vs state quality
_DELTA_WEIGHT = 0.7
_STATE_WEIGHT = 0.3


def _compute_state_quality(biomarkers: Biomarkers, weights: dict[str, float]) -> float:
    """Compute state quality score from absolute biomarker values.

    Returns a value in [0, 1] representing how good the current state is.
    Uses the same goal weights as the delta component.
    """
    bio_dict = biomarkers.model_dump()
    quality = 0.0
    for marker, value in bio_dict.items():
        lo, hi = STATE_RANGES[marker]
        normalized = (value - lo) / (hi - lo) if hi > lo else 0.5
        normalized = max(0.0, min(1.0, normalized))
        if marker in LOWER_IS_BETTER:
            normalized = 1.0 - normalized
        quality += normalized * weights[marker]
    return quality


def compute_reward(
    deltas: BiomarkerDeltas,
    goal: Goal,
    current_biomarkers: Optional[Biomarkers] = None,
) -> RewardBreakdown:
    """Compute blended reward from biomarker deltas and absolute state quality.

    Reward = delta_score * 0.7 + state_score * 0.3

    - delta_score (0-100): rewards improving biomarkers (same as original formula).
    - state_score (0-100): rewards maintaining good absolute biomarker values.

    When current_biomarkers is None (e.g. in tests), state quality defaults to
    0.5 so the baseline remains 50.
    """
    weights = GOAL_WEIGHTS[goal]
    delta_dict = deltas.model_dump()

    per_marker: dict[str, float] = {}
    weighted_sum = 0.0

    for marker, raw_delta in delta_dict.items():
        w = weights[marker]
        scale = DELTA_SCALES[marker]

        # Normalize: flip sign for "lower is better" markers
        if marker in LOWER_IS_BETTER:
            normalized = -raw_delta / scale
        else:
            normalized = raw_delta / scale

        # Clamp to [-2, +2] to prevent outlier noise from dominating
        normalized = max(-2.0, min(2.0, normalized))

        # Scale to a per-marker reward contribution.
        marker_reward = normalized * w * 100.0
        per_marker[marker] = round(marker_reward, 3)
        weighted_sum += marker_reward

    # Delta score: baseline 50 + improvements
    delta_score = max(0.0, min(100.0, 50.0 + weighted_sum))

    # State quality score: how good are current absolute values?
    if current_biomarkers is not None:
        state_quality = _compute_state_quality(current_biomarkers, weights)
    else:
        state_quality = 0.5  # Neutral default preserves baseline = 50
    state_score = state_quality * 100.0

    # Blend delta and state components
    total = delta_score * _DELTA_WEIGHT + state_score * _STATE_WEIGHT
    total = max(0.0, min(100.0, total))

    return RewardBreakdown(
        resting_hr_reward=per_marker["resting_hr"],
        hrv_reward=per_marker["hrv"],
        vo2_max_reward=per_marker["vo2_max"],
        body_fat_reward=per_marker["body_fat_pct"],
        lean_mass_reward=per_marker["lean_mass_kg"],
        sleep_efficiency_reward=per_marker["sleep_efficiency"],
        cortisol_reward=per_marker["cortisol_proxy"],
        energy_reward=per_marker["energy_level"],
        total=round(total, 2),
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _stddev(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _linear_slope(values: list[float]) -> float:
    """Slope of a simple linear regression over the index."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator
