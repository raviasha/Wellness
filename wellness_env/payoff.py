"""Outcome-based reward computation.

Reward = sum of goal-weighted biomarker deltas, normalized to [0, 100].
Each goal defines which outcomes matter most.

MVP 1: 5 Garmin-measured biomarkers × 5 goals.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .models import BiomarkerDeltas, Biomarkers, Goal, RewardBreakdown


# ---------------------------------------------------------------------------
# Goal-specific weights for each biomarker delta.
#
# Convention:
#   - Negative deltas are "good" for resting_hr, stress_avg
#     → we negate them before weighting so positive = better for all.
#   - All weights for a goal sum to ~1.0.
# ---------------------------------------------------------------------------

GOAL_WEIGHTS: dict[Goal, dict[str, float]] = {
    Goal.STRESS_MANAGEMENT: {
        "resting_hr": 0.08,
        "hrv": 0.22,
        "sleep_score": 0.15,
        "stress_avg": 0.28,
        "body_battery": 0.12,
        "sleep_stage_quality": 0.10,
        "vo2_max": 0.05,
    },

    Goal.CARDIOVASCULAR_FITNESS: {
        "resting_hr": 0.26,
        "hrv": 0.26,
        "sleep_score": 0.10,
        "stress_avg": 0.08,
        "body_battery": 0.10,
        "sleep_stage_quality": 0.05,
        "vo2_max": 0.15,
    },

    Goal.SLEEP_OPTIMIZATION: {
        "resting_hr": 0.08,
        "hrv": 0.15,
        "sleep_score": 0.28,
        "stress_avg": 0.15,
        "body_battery": 0.12,
        "sleep_stage_quality": 0.17,
        "vo2_max": 0.05,
    },

    Goal.RECOVERY_ENERGY: {
        "resting_hr": 0.08,
        "hrv": 0.18,
        "sleep_score": 0.12,
        "stress_avg": 0.22,
        "body_battery": 0.25,
        "sleep_stage_quality": 0.12,
        "vo2_max": 0.03,
    },

    Goal.ACTIVE_LIVING: {
        "resting_hr": 0.16,
        "hrv": 0.16,
        "sleep_score": 0.16,
        "stress_avg": 0.16,
        "body_battery": 0.16,
        "sleep_stage_quality": 0.10,
        "vo2_max": 0.10,
    },
}

# ---------------------------------------------------------------------------
# Normalization scales: how much delta is "very good" for each biomarker.
# These are used to convert raw deltas into a 0-100 reward contribution.
# ---------------------------------------------------------------------------

DELTA_SCALES: dict[str, float] = {
    "resting_hr": 1.0,             # −1 bpm/day is excellent
    "hrv": 5.0,                    # +5 ms/day is excellent
    "sleep_score": 3.0,            # +3 points/day is excellent
    "stress_avg": 5.0,             # −5 points/day is excellent
    "body_battery": 5.0,           # +5 points/day is excellent
    "sleep_stage_quality": 3.0,    # +3 % deep+REM/day is excellent
    "vo2_max": 0.3,                # +0.3 ml/kg/min/day is excellent
}

# Biomarkers where LOWER is better (we negate so positive = good)
LOWER_IS_BETTER = {"resting_hr", "stress_avg"}

# ---------------------------------------------------------------------------
# State quality ranges for normalizing absolute biomarker values.
# ---------------------------------------------------------------------------

STATE_RANGES: dict[str, tuple[float, float]] = {
    "resting_hr": (40, 120),
    "hrv": (5, 150),
    "sleep_score": (0, 100),
    "stress_avg": (0, 100),
    "body_battery": (0, 100),
    "sleep_stage_quality": (0, 100),
    "vo2_max": (10, 90),
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
    goal: Optional[Goal] = None,
    current_biomarkers: Optional[Biomarkers] = None,
    weights: Optional[dict[str, float]] = None,
) -> RewardBreakdown:
    """Compute blended reward from biomarker deltas and absolute state quality.

    Reward = delta_score * 0.7 + state_score * 0.3

    - delta_score (0-100): rewards improving biomarkers (same as original formula).
    - state_score (0-100): rewards maintaining good absolute biomarker values.

    When current_biomarkers is None (e.g. in tests), state quality defaults to
    0.5 so the baseline remains 50.

    Parameters
    ----------
    weights : dict, optional
        When provided, use these raw biomarker weights directly instead of
        looking up GOAL_WEIGHTS[goal]. This enables custom goal-derived weights.
    goal : Goal, optional
        Used to look up weights from GOAL_WEIGHTS when ``weights`` is not given.
        At least one of ``goal`` or ``weights`` must be provided.
    """
    if weights is None:
        if goal is None:
            goal = Goal.ACTIVE_LIVING  # safe fallback
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
        sleep_score_reward=per_marker["sleep_score"],
        stress_avg_reward=per_marker["stress_avg"],
        body_battery_reward=per_marker["body_battery"],
        sleep_stage_quality_reward=per_marker.get("sleep_stage_quality", 0.0),
        vo2_max_reward=per_marker.get("vo2_max", 0.0),
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
