"""Joint distribution calibration using a Gaussian copula with Ledoit-Wolf shrinkage.

MVP 2: Garmin-only inputs and outputs with circadian + exercise-type enrichment.

Architecture: Gaussian copula == multivariate Normal on standardised data.
At simulation time, P(Y | X) is computed in closed form and sampled.

X features (6, controllable inputs only):
  sleep_duration_hours, bedtime_hour_cos, bedtime_hour_sin,
  exercise_type_idx, active_calories_100s, intensity_minutes_h

Y outcomes (7):
  delta_sleep_score, delta_hrv_ms, delta_rhr_bpm, delta_stress, delta_body_battery,
  delta_sleep_stage_quality, delta_vo2_max

Backward compat: loaded distributions with n_x=5 / n_y=5 use the legacy encoding.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

# ---------------------------------------------------------------------------
# Column layout (v2 — 7 inputs, 7 outputs)
# ---------------------------------------------------------------------------

X_FEATURE_NAMES: list[str] = [
    "sleep_duration_hours",
    "bedtime_hour_cos",
    "bedtime_hour_sin",
    "exercise_type_idx",
    "active_calories_100s",
    "intensity_minutes_h",
]
Y_OUTCOME_NAMES: list[str] = [
    "delta_sleep_score",
    "delta_hrv_ms",
    "delta_rhr_bpm",
    "delta_stress",
    "delta_body_battery",
    "delta_sleep_stage_quality",
    "delta_vo2_max",
]

# Legacy v1 column lists (for backward compat when loading old distributions)
X_FEATURE_NAMES_V1: list[str] = [
    "sleep_duration_hours", "sleep_score_pct", "active_minutes_h",
    "active_calories_100s", "steps_1000s",
]
Y_OUTCOME_NAMES_V1: list[str] = [
    "delta_sleep_score", "delta_hrv_ms", "delta_rhr_bpm",
    "delta_stress", "delta_body_battery",
]

N_X = len(X_FEATURE_NAMES)   # 7
N_Y = len(Y_OUTCOME_NAMES)   # 7
N_JOINT = N_X + N_Y          # 14


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class JointDistribution:
    """Gaussian copula representation of P(X, Y).

    All list fields are JSON-serialisable so the distribution can be saved
    and loaded without numpy.
    """
    n_x: int                    # number of input features (5)
    n_y: int                    # number of output dimensions (5)
    means: list[float]          # length n_x + n_y
    stds: list[float]           # length n_x + n_y  (> 0)
    corr_matrix: list[list[float]]  # (n_x + n_y) × (n_x + n_y)
    n_samples: int
    feature_names: list[str]
    outcome_names: list[str]
    condition_number: float = 1.0
    ledoit_wolf_shrinkage: float = 0.0


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def fit_joint_distribution(X: np.ndarray, Y: np.ndarray) -> JointDistribution:
    """Fit a Gaussian copula to the joint distribution P(X, Y).

    Parameters
    ----------
    X : (n, N_X) array of input features without intercept.
    Y : (n, N_Y) array of output deltas.

    Returns
    -------
    JointDistribution
    """
    n = X.shape[0]
    n_x = X.shape[1]
    n_y = Y.shape[1]

    Z = np.hstack([X, Y])          # (n, n_x + n_y)

    means = Z.mean(axis=0)
    stds = Z.std(axis=0, ddof=1)
    stds = np.where(stds < 1e-8, 1e-8, stds)   # guard against constant columns

    Z_std = (Z - means) / stds     # standardise to zero-mean unit-variance

    lw = LedoitWolf()
    lw.fit(Z_std)

    # Covariance → correlation
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    d = np.where(d < 1e-8, 1e-8, d)
    corr = (cov / d[:, None]) / d[None, :]
    corr = np.clip(corr, -1.0, 1.0)

    # Select feature/outcome names based on dimensionality
    feat_names = X_FEATURE_NAMES if n_x == N_X else X_FEATURE_NAMES_V1
    out_names = Y_OUTCOME_NAMES if n_y == N_Y else Y_OUTCOME_NAMES_V1

    return JointDistribution(
        n_x=n_x,
        n_y=n_y,
        means=means.tolist(),
        stds=stds.tolist(),
        corr_matrix=corr.tolist(),
        n_samples=n,
        feature_names=feat_names,
        outcome_names=out_names,
        condition_number=float(np.linalg.cond(corr)),
        ledoit_wolf_shrinkage=float(lw.shrinkage_),
    )


# ---------------------------------------------------------------------------
# Conditional sampling
# ---------------------------------------------------------------------------

def sample_conditional(
    dist: JointDistribution,
    x_observed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Y ~ P(Y | X = x_observed) from the Gaussian copula.

    Uses the standard Gaussian conditional formula in standardised space.

    Parameters
    ----------
    dist        : fitted JointDistribution
    x_observed  : (N_X,) array in original (unstandardised) scale
    rng         : numpy Generator for reproducibility

    Returns
    -------
    y_sampled : (N_Y,) array in original scale
    """
    n_x = dist.n_x
    n_y = dist.n_y
    means = np.array(dist.means)
    stds = np.array(dist.stds)
    R = np.array(dist.corr_matrix)

    # Standardise observed X
    z_x = (x_observed - means[:n_x]) / stds[:n_x]

    # Partition correlation matrix: R = [[R_xx, R_xy], [R_yx, R_yy]]
    R_xx = R[:n_x, :n_x]
    R_xy = R[:n_x, n_x:]
    R_yx = R[n_x:, :n_x]
    R_yy = R[n_x:, n_x:]

    # Solve R_xx @ alpha = z_x  →  alpha = R_xx^{-1} z_x  (numerically stable)
    alpha = np.linalg.solve(R_xx, z_x)

    # Conditional mean and covariance in standardised space
    mu_zy = R_yx @ alpha
    Sigma_zy = R_yy - R_yx @ np.linalg.solve(R_xx, R_xy)

    # Ensure positive semi-definiteness (numerical jitter if needed)
    min_eig = float(np.linalg.eigvalsh(Sigma_zy).min())
    if min_eig < 1e-8:
        Sigma_zy += (1e-8 - min_eig) * np.eye(n_y)

    # Sample from conditional Gaussian
    z_y = rng.multivariate_normal(mu_zy, Sigma_zy)

    # Back-transform to original scale
    return z_y * stds[n_x:] + means[n_x:]


# ---------------------------------------------------------------------------
# Feature encoding: Action → calibration feature vector
# ---------------------------------------------------------------------------

def encode_action_to_features(action: Any, dist: "JointDistribution | None" = None) -> np.ndarray:
    """Map a simulator Action to a calibration-compatible feature vector.

    If `dist` has n_x == 5 (legacy), returns the old 5-feature encoding for
    backward compatibility.  Otherwise returns the full 7-feature v2 vector.

    v2 columns (6):
      sleep_duration_hours, bedtime_hour_cos, bedtime_hour_sin,
      exercise_type_idx, active_calories_100s, intensity_minutes_h

    v1 columns (5, legacy):
      sleep_duration_hours, sleep_score_pct, active_minutes_h,
      active_calories_100s, steps_1000s
    """
    from wellness_env.models import (
        SLEEP_HOURS, ACTIVITY_INTENSITY, BEDTIME_HOUR,
        ExerciseType, BedtimeWindow,
    )

    sleep_hours = SLEEP_HOURS.get(action.sleep, 7.5)
    activity_info = ACTIVITY_INTENSITY.get(action.activity, {"active_minutes": 30, "active_calories": 200})
    active_mins = activity_info["active_minutes"]
    active_cals = activity_info["active_calories"]

    # Backward compat: return legacy 5-feature vector
    if dist is not None and getattr(dist, "n_x", N_X) == 5:
        sleep_score_pct = min(100.0, max(0.0, 50.0 + (sleep_hours - 5.0) * 10.0))
        return np.array([
            sleep_hours,
            sleep_score_pct / 100.0,
            active_mins / 60.0,
            active_cals / 100.0,
            active_mins * 100 / 1000.0,
        ], dtype=float)

    # v2: 6-feature vector with bedtime circular encoding + exercise type
    bedtime = getattr(action, "bedtime", BedtimeWindow.OPTIMAL)
    bedtime_hour = BEDTIME_HOUR.get(bedtime, 22.5)
    # Circular encoding (handles midnight wrap-around)
    angle = 2 * math.pi * bedtime_hour / 24.0
    bedtime_cos = math.cos(angle)
    bedtime_sin = math.sin(angle)

    exercise_type = getattr(action, "exercise_type", ExerciseType.NONE)

    exercise_type_list = list(ExerciseType)
    exercise_type_idx = exercise_type_list.index(exercise_type) if exercise_type in exercise_type_list else 0

    return np.array([
        sleep_hours,
        bedtime_cos,
        bedtime_sin,
        float(exercise_type_idx),
        active_cals / 100.0,
        active_mins / 60.0,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_distribution(dist: JointDistribution, path: str) -> None:
    """Serialise JointDistribution to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "n_x": dist.n_x,
        "n_y": dist.n_y,
        "means": dist.means,
        "stds": dist.stds,
        "corr_matrix": dist.corr_matrix,
        "n_samples": dist.n_samples,
        "feature_names": dist.feature_names,
        "outcome_names": dist.outcome_names,
        "condition_number": dist.condition_number,
        "ledoit_wolf_shrinkage": dist.ledoit_wolf_shrinkage,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_distribution(path: str) -> JointDistribution:
    """Deserialise JointDistribution from JSON."""
    with open(path, "r") as fh:
        data = json.load(fh)
    return JointDistribution(**data)


# ---------------------------------------------------------------------------
# Full calibration pipeline (replaces calibrate_user_persona)
# ---------------------------------------------------------------------------

def calibrate_user_distribution(user_id: int) -> dict[str, Any]:
    """Fit a Gaussian copula from the user's Garmin history.

    Same data extraction as calibrate_user_persona() in calibration.py.
    Saves the fitted distribution to models/user_{id}/distribution.json.

    Returns a status dict with quality metrics.
    """
    from backend.database import get_recent_history

    history = get_recent_history(user_id, limit=30)

    if len(history["syncs"]) < 15:
        return {
            "error": (
                f"Insufficient data. Need at least 15 days of health history "
                f"(currently have {len(history['syncs'])})."
            )
        }

    sync_map = {s["sync_date"]: s for s in history["syncs"]}
    log_map: dict[str, list] = {}
    for log in history["logs"]:
        date = log["log_date"]
        log_map.setdefault(date, []).append(log)

    sorted_dates = sorted(sync_map.keys())
    today_str = datetime.date.today().isoformat()

    X_rows: list[list[float]] = []
    Y_rows: list[list[float]] = []

    for i in range(len(sorted_dates) - 1):
        date_t = sorted_dates[i]
        date_t1 = sorted_dates[i + 1]

        # Never train on outcomes still being accumulated today
        if date_t1 == today_str:
            continue

        sync_t = sync_map[date_t]
        sync_t1 = sync_map[date_t1]

        # ---- Outcomes (Day T → T+1 deltas) — use per-field fallbacks so no pair is dropped ----
        delta_sleep_score = (
            (sync_t1.get("sleep_score") or 70) - (sync_t.get("sleep_score") or 70)
        )
        delta_hrv = (
            (sync_t1.get("hrv_rmssd") or 0) - (sync_t.get("hrv_rmssd") or 0)
        )
        delta_rhr = (
            (sync_t1.get("resting_hr") or 0) - (sync_t.get("resting_hr") or 0)
        )
        delta_stress = (
            (sync_t1.get("stress_avg") or 50) - (sync_t.get("stress_avg") or 50)
        )
        delta_body_battery = (
            (sync_t1.get("recovery_score") or 50) - (sync_t.get("recovery_score") or 50)
        )
        delta_sleep_stage_quality = (
            (sync_t1.get("sleep_stage_quality") or 35.0)
            - (sync_t.get("sleep_stage_quality") or 35.0)
        )
        delta_vo2_max = (
            (sync_t1.get("vo2_max") or 0.0) - (sync_t.get("vo2_max") or 0.0)
        )

        # ---- Input features (controllable only, v2) ----
        active_mins = sync_t.get("active_minutes", 0)
        steps = sync_t.get("steps", 0) or 0

        # Sleep hours
        sleep_hours = sync_t.get("sleep_duration_hours") or 0
        if not sleep_hours:
            raw: dict = {}
            try:
                rp = sync_t.get("raw_payload")
                raw = json.loads(rp) if isinstance(rp, str) else (rp or {})
            except Exception:
                pass
            sleep_seconds = raw.get("sleep", {}).get("durationInSeconds", 28800)
            sleep_hours = sleep_seconds / 3600.0

        # Bedtime circular encoding
        sleep_start_hour = sync_t.get("sleep_start_hour")
        if sleep_start_hour is None:
            sleep_start_hour = 23.0  # fallback: assume midnight-ish
        angle = 2 * math.pi * sleep_start_hour / 24.0
        bedtime_cos = math.cos(angle)
        bedtime_sin = math.sin(angle)

        # Exercise type index (0 = rest, 1 = cardio, 2 = strength, 3 = flexibility, 4 = hiit)
        from wellness_env.models import ExerciseType, GARMIN_ACTIVITY_TYPE_MAP
        raw_ex_type = sync_t.get("exercise_type") or "none"
        mapped_ex_type = GARMIN_ACTIVITY_TYPE_MAP.get(raw_ex_type.lower(), ExerciseType.NONE)
        ex_type_list = list(ExerciseType)
        exercise_type_idx = float(ex_type_list.index(mapped_ex_type)) if mapped_ex_type in ex_type_list else 0.0

        exercise_duration_minutes = sync_t.get("exercise_duration_minutes") or 0
        exercise_duration_h = exercise_duration_minutes / 60.0

        X_rows.append([
            sleep_hours,
            bedtime_cos,
            bedtime_sin,
            active_mins / 60.0,
            exercise_type_idx,
            exercise_duration_h,
            steps / 1000.0,
        ])
        Y_rows.append([
            delta_sleep_score, delta_hrv, delta_rhr, delta_stress, delta_body_battery,
            delta_sleep_stage_quality, delta_vo2_max,
        ])

    if len(X_rows) < 7:
        return {
            "error": (
                f"Not enough contiguous days with behavioural logs. "
                f"Need 7 paired samples (currently have {len(X_rows)})."
            )
        }

    X_mat = np.array(X_rows)
    Y_mat = np.array(Y_rows)

    try:
        dist = fit_joint_distribution(X_mat, Y_mat)
    except Exception as exc:
        return {"error": f"Distribution fitting error: {exc}"}

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "models", f"user_{user_id}"
    )
    output_path = os.path.join(output_dir, "distribution.json")
    save_distribution(dist, output_path)

    warnings: list[str] = []
    if dist.n_samples < 20:
        warnings.append(
            f"Small sample size (n={dist.n_samples}): correlations are heavily "
            "regularised toward independence. More data recommended."
        )
    if dist.condition_number > 1000:
        warnings.append(
            f"High condition number ({dist.condition_number:.0f}): distribution "
            "may be ill-conditioned."
        )
    if dist.ledoit_wolf_shrinkage > 0.8:
        warnings.append(
            f"High Ledoit-Wolf shrinkage ({dist.ledoit_wolf_shrinkage:.2f}): "
            "more data will improve correlation estimates."
        )

    return {
        "status": "success",
        "method": "gaussian_copula_ledoit_wolf",
        "samples": dist.n_samples,
        "features": dist.feature_names,
        "outcomes": dist.outcome_names,
        "condition_number": round(dist.condition_number, 2),
        "ledoit_wolf_shrinkage": round(dist.ledoit_wolf_shrinkage, 4),
        "distribution_path": output_path,
        "warnings": warnings,
    }
