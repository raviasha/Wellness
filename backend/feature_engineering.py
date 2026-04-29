"""Lagged feature engineering for per-outcome ML models.

Extracts (X, Y) training pairs from wearable_syncs with:
  - 7 controllable input features (same as Gaussian copula)
  - 7 autoregressive previous-day outcome values  (strongest predictors)
  - 2 compliance flags from recommendations table
  - Extended rolling aggregates (unlocked at ml_lag2_min_days)

Output naming follows the same conventions as distribution_calibration.py so
features can be compared or transferred between the two systems.

Action@T → Outcomes@T+1 alignment:
  X: features extracted from sync_date = T
  Y: delta computed from sync_{T+1} − sync_{T}  (same as copula calibration)
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from backend.distribution_calibration import X_FEATURE_NAMES, Y_OUTCOME_NAMES
from wellness_env.models import ExerciseType, GARMIN_ACTIVITY_TYPE_MAP

# Feature name groups  (order matters — ML model stores coefficients by index)
_BASE_INPUT_NAMES: list[str] = list(X_FEATURE_NAMES)  # 7 controllable inputs

_AUTOREGRESSIVE_NAMES: list[str] = [
    "prev_delta_sleep_score",
    "prev_delta_hrv_ms",
    "prev_delta_rhr_bpm",
    "prev_delta_stress",
    "prev_delta_body_battery",
    "prev_delta_sleep_stage_quality",
    "prev_delta_vo2_max",
]  # 7 autoregressive (yesterday's outcome deltas)

_COMPLIANCE_NAMES: list[str] = [
    "compliance_sleep",
    "compliance_activity",
]  # 2 compliance flags

# Rolling aggregate suffixes (added at ml_lag2_min_days)
_ROLLING_SUFFIXES: list[str] = ["_roll7_mean", "_roll7_std"]

OUTCOME_NAMES: list[str] = list(Y_OUTCOME_NAMES)  # 7 outcomes, same as copula


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_sleep_hours(sync: dict) -> float:
    hours = sync.get("sleep_duration_hours")
    if hours:
        return float(hours)
    try:
        rp = sync.get("raw_payload") or {}
        raw = json.loads(rp) if isinstance(rp, str) else rp
        secs = raw.get("sleep", {}).get("durationInSeconds", 0)
        return secs / 3600.0 if secs else 7.0
    except Exception:
        return 7.0


def _encode_input_row(sync: dict) -> list[float]:
    """Encode a single day's sync row into the 6-feature controllable-input vector."""
    sleep_hours = _extract_sleep_hours(sync)

    sleep_start_hour = sync.get("sleep_start_hour")
    if sleep_start_hour is None:
        sleep_start_hour = 23.0
    angle = 2 * math.pi * sleep_start_hour / 24.0
    bedtime_cos = math.cos(angle)
    bedtime_sin = math.sin(angle)

    raw_ex_type = (sync.get("exercise_type") or "none").lower()
    mapped_ex = GARMIN_ACTIVITY_TYPE_MAP.get(raw_ex_type, ExerciseType.NONE)
    ex_types = list(ExerciseType)
    ex_type_idx = float(ex_types.index(mapped_ex)) if mapped_ex in ex_types else 0.0

    active_cals = float(sync.get("active_calories") or 0)
    intensity_mins = float(sync.get("active_minutes") or 0)

    return [
        sleep_hours,
        bedtime_cos,
        bedtime_sin,
        ex_type_idx,
        active_cals / 100.0,
        intensity_mins / 60.0,
    ]


def _outcome_delta(t0: dict, t1: dict) -> list[float]:
    """Compute the 7 outcome deltas: sync_{T+1} − sync_T."""
    def _g(s: dict, key: str, default: float = 0.0) -> float:
        v = s.get(key)
        return float(v) if v is not None else default

    return [
        _g(t1, "sleep_score",         70.0) - _g(t0, "sleep_score",         70.0),
        _g(t1, "hrv_rmssd",            45.0) - _g(t0, "hrv_rmssd",            45.0),
        _g(t1, "resting_hr",           70.0) - _g(t0, "resting_hr",           70.0),
        _g(t1, "stress_avg",           50.0) - _g(t0, "stress_avg",           50.0),
        _g(t1, "recovery_score",       50.0) - _g(t0, "recovery_score",       50.0),
        (_g(t1, "sleep_stage_quality", 35.0) - _g(t0, "sleep_stage_quality",  35.0)),
        (_g(t1, "vo2_max",              0.0) - _g(t0, "vo2_max",               0.0)),
    ]


def _get_compliance_row(rec_map: dict, date_str: str) -> list[float]:
    """Return [compliance_sleep, compliance_activity] for a date, defaulting to 0.5."""
    rec = rec_map.get(date_str, {})
    return [
        float(rec.get("compliance_sleep") or 0.5),
        float(rec.get("compliance_activity") or 0.5),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_Xy_matrix(
    user_id: int,
    include_rolling: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build the full (X, Y) training matrix for a user.

    Parameters
    ----------
    user_id : int
    include_rolling : bool
        When True, adds 7-day rolling mean and std for each base input
        (14 additional features).  Set True when paired_days ≥ ml_lag2_min_days.

    Returns
    -------
    X : (n, n_features) float array
    Y : (n, 7) float array  — outcome deltas
    feature_names : list of column names for X
    outcome_names : list of column names for Y (always OUTCOME_NAMES)
    """
    from backend.database import SessionLocal, WearableSync, Recommendation

    today_str = date.today().isoformat()
    db = SessionLocal()
    try:
        # Load all syncs for this user ordered by date
        sync_rows = (
            db.query(WearableSync)
            .filter(WearableSync.user_id == user_id)
            .order_by(WearableSync.sync_date.asc())
            .all()
        )
        # Build date → dict map
        sync_map: dict[str, dict] = {}
        for s in sync_rows:
            d = {c.name: getattr(s, c.name) for c in s.__table__.columns}
            sync_map[s.sync_date] = d

        # Load recommendations for compliance
        rec_rows = (
            db.query(Recommendation)
            .filter(Recommendation.user_id == user_id)
            .all()
        )
        rec_map: dict[str, dict] = {
            r.rec_date: {
                "compliance_sleep":    r.compliance_sleep,
                "compliance_activity": r.compliance_activity,
            }
            for r in rec_rows
        }
    finally:
        db.close()

    sorted_dates = sorted(sync_map.keys())

    # Build sequential pairs: (T, T+1) where T+1 ≠ today and the dates are consecutive
    X_rows: list[list[float]] = []
    Y_rows: list[list[float]] = []
    prev_delta: list[float] = [0.0] * 7  # autoregressive seed

    for i in range(len(sorted_dates) - 1):
        t0_str = sorted_dates[i]
        t1_str = sorted_dates[i + 1]

        if t1_str == today_str:
            break  # T+1 still accumulating

        # Only use consecutive days to avoid garbage-in from large gaps
        try:
            d0 = datetime.strptime(t0_str, "%Y-%m-%d").date()
            d1 = datetime.strptime(t1_str, "%Y-%m-%d").date()
            if (d1 - d0).days != 1:
                prev_delta = [0.0] * 7  # reset autoregressive on gap
                continue
        except Exception:
            continue

        sync_t  = sync_map[t0_str]
        sync_t1 = sync_map[t1_str]

        x_base = _encode_input_row(sync_t)           # 7 controllable inputs
        x_ar   = list(prev_delta)                     # 7 autoregressive deltas
        x_comp = _get_compliance_row(rec_map, t0_str) # 2 compliance flags

        X_rows.append(x_base + x_ar + x_comp)

        delta = _outcome_delta(sync_t, sync_t1)       # 7 outcomes
        Y_rows.append(delta)

        prev_delta = delta  # carry forward for next step

    if not X_rows:
        # Return empty arrays with correct shapes
        base_feat = _BASE_INPUT_NAMES + _AUTOREGRESSIVE_NAMES + _COMPLIANCE_NAMES
        return np.zeros((0, len(base_feat))), np.zeros((0, 7)), base_feat, OUTCOME_NAMES

    X = np.array(X_rows, dtype=float)
    Y = np.array(Y_rows, dtype=float)

    base_feat_names = _BASE_INPUT_NAMES + _AUTOREGRESSIVE_NAMES + _COMPLIANCE_NAMES

    if include_rolling and len(X_rows) >= 14:
        X, base_feat_names = _add_rolling_features(X, X_rows, base_feat_names)

    return X, Y, base_feat_names, OUTCOME_NAMES


def _add_rolling_features(
    X: np.ndarray,
    X_rows: list[list[float]],
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Add 7-day rolling mean and std for the 7 base inputs (first 7 columns)."""
    n, n_feat = X.shape
    n_base = len(_BASE_INPUT_NAMES)  # 7

    roll_cols: list[np.ndarray] = []
    new_names: list[str] = []

    for j in range(n_base):
        col = X[:, j]
        roll_mean = np.array([
            col[max(0, i - 6): i + 1].mean() for i in range(n)
        ])
        roll_std = np.array([
            col[max(0, i - 6): i + 1].std() for i in range(n)
        ])
        roll_cols.extend([roll_mean, roll_std])
        new_names.extend([
            f"{_BASE_INPUT_NAMES[j]}_roll7_mean",
            f"{_BASE_INPUT_NAMES[j]}_roll7_std",
        ])

    X_extended = np.hstack([X, np.column_stack(roll_cols)])
    return X_extended, feature_names + new_names


def get_feature_names(include_rolling: bool = False) -> list[str]:
    """Return ordered feature names matching build_Xy_matrix output."""
    base = _BASE_INPUT_NAMES + _AUTOREGRESSIVE_NAMES + _COMPLIANCE_NAMES
    if include_rolling:
        for bname in _BASE_INPUT_NAMES:
            base = base + [f"{bname}_roll7_mean", f"{bname}_roll7_std"]
    return base
