"""Joint distribution calibration using a Gaussian copula with Ledoit-Wolf shrinkage.

Replaces the OLS-based calibrate_user_persona() with a richer statistical model that:
  - Preserves cross-biomarker correlations lost by OLS compression
  - Captures real noise structure instead of hardcoded Gaussian sigmas
  - Regularises the covariance estimate for small samples (n=15-30, d=12)
    using Ledoit-Wolf shrinkage

Architecture: Gaussian copula == multivariate Normal on standardised data.
At simulation time, P(Y | X) is computed in closed form and sampled.

Unobserved biomarkers (VO2, LeanMass, Energy) have no Garmin ground truth
and are handled by heuristic rules in wellness_env/distribution_simulator.py.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

X_FEATURE_NAMES: list[str] = [
    "sleep_hours",
    "protein_100g",
    "carbs_100g",
    "fat_100g",
    "quality_0_1",
    "intensity_mins_h",
    "active_cals_100s",
]
Y_OUTCOME_NAMES: list[str] = [
    "delta_sleep_score",
    "delta_hrv_ms",
    "delta_rhr_bpm",
    "delta_weight_kg",
    "delta_stress",
]

N_X = len(X_FEATURE_NAMES)   # 7
N_Y = len(Y_OUTCOME_NAMES)   # 5
N_JOINT = N_X + N_Y          # 12


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class JointDistribution:
    """Gaussian copula representation of P(X, Y).

    All list fields are JSON-serialisable so the distribution can be saved
    and loaded without numpy.
    """
    n_x: int                    # number of input features (7)
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
    if X.shape[1] != N_X:
        raise ValueError(f"Expected {N_X} input features, got {X.shape[1]}")
    if Y.shape[1] != N_Y:
        raise ValueError(f"Expected {N_Y} output dimensions, got {Y.shape[1]}")

    Z = np.hstack([X, Y])          # (n, N_JOINT)

    means = Z.mean(axis=0)
    stds = Z.std(axis=0, ddof=1)
    stds = np.where(stds < 1e-8, 1e-8, stds)   # guard against constant columns

    Z_std = (Z - means) / stds     # standardise to zero-mean unit-variance

    lw = LedoitWolf()
    lw.fit(Z_std)

    # Covariance → correlation (standardised data should already give corr ≈ cov,
    # but numerical precision can cause slight deviations)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    d = np.where(d < 1e-8, 1e-8, d)
    corr = (cov / d[:, None]) / d[None, :]
    corr = np.clip(corr, -1.0, 1.0)

    return JointDistribution(
        n_x=N_X,
        n_y=N_Y,
        means=means.tolist(),
        stds=stds.tolist(),
        corr_matrix=corr.tolist(),
        n_samples=n,
        feature_names=list(X_FEATURE_NAMES),
        outcome_names=list(Y_OUTCOME_NAMES),
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

def encode_action_to_features(action: Any) -> np.ndarray:
    """Map a simulator Action to a calibration-compatible feature vector.

    Returns an (N_X,) array with columns:
      [sleep_hours, protein_100g, carbs_100g, fat_100g,
       quality_0_1, intensity_mins_h, active_cals_100s]

    Values match the encoding used when building calibration data from
    Garmin syncs, so the distribution can be queried at simulation time.
    """
    from wellness_env.models import (
        ExerciseType, NutritionType, SLEEP_HOURS,
    )

    # ---- Sleep ----
    sleep_hours = SLEEP_HOURS.get(action.sleep, 7.5)

    # ---- Exercise ----
    EXERCISE_INTENSITY: dict = {
        ExerciseType.NONE:            (0.0,   0.0),
        ExerciseType.LIGHT_CARDIO:    (30.0, 200.0),
        ExerciseType.MODERATE_CARDIO: (45.0, 350.0),
        ExerciseType.HIIT:            (30.0, 400.0),
        ExerciseType.STRENGTH:        (45.0, 300.0),
        ExerciseType.YOGA:            (45.0, 150.0),
    }
    intensity_mins, active_cals = EXERCISE_INTENSITY.get(action.exercise, (0.0, 0.0))

    # ---- Nutrition ----
    # Representative macro amounts (grams) per nutrition category
    NUTRITION_MACROS: dict = {
        NutritionType.HIGH_PROTEIN: (50.0, 100.0, 30.0),
        NutritionType.BALANCED:     (30.0, 150.0, 40.0),
        NutritionType.HIGH_CARB:    (15.0, 200.0, 20.0),
        NutritionType.PROCESSED:    (10.0, 150.0, 50.0),
        NutritionType.SKIPPED:      ( 0.0,   0.0,  0.0),
    }
    protein_g, carbs_g, fat_g = NUTRITION_MACROS.get(action.nutrition, (30.0, 150.0, 40.0))

    NUTRITION_QUALITY: dict = {
        NutritionType.HIGH_PROTEIN: 0.8,
        NutritionType.BALANCED:     1.0,
        NutritionType.HIGH_CARB:    0.4,
        NutritionType.PROCESSED:    0.1,
        NutritionType.SKIPPED:      0.0,
    }
    quality = NUTRITION_QUALITY.get(action.nutrition, 1.0)

    return np.array([
        sleep_hours,
        protein_g / 100.0,
        carbs_g / 100.0,
        fat_g / 100.0,
        quality,
        intensity_mins / 60.0,
        active_cals / 100.0,
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
        logs_t = log_map.get(date_t, [])

        # ---- Outcomes (Day T → T+1 deltas) ----
        delta_sleep_score = (
            sync_t1.get("sleep_score", 70) - sync_t.get("sleep_score", 70)
        )
        delta_hrv = sync_t1["hrv_rmssd"] - sync_t["hrv_rmssd"]
        delta_rhr = sync_t1["resting_hr"] - sync_t["resting_hr"]
        delta_stress = (
            sync_t1.get("stress_avg", 50) - sync_t.get("stress_avg", 50)
        )

        # ---- Nutrition features from food logs ----
        protein_g = carbs_g = fat_g = 0.0
        q_sum = q_count = 0.0
        for log in logs_t:
            if log["log_type"] == "food" and log.get("raw_input"):
                try:
                    parsed = json.loads(log["raw_input"])
                    if isinstance(parsed, dict) and "parsed" in parsed:
                        p = parsed["parsed"]
                        protein_g += float(p.get("protein_g", 0))
                        carbs_g += float(p.get("carbs_g", 0))
                        fat_g += float(p.get("fat_g", 0))
                        qmap = {
                            "high_protein": 0.8, "balanced": 1.0,
                            "high_carb": 0.4, "processed": 0.1, "skipped": 0.0,
                        }
                        q_sum += qmap.get(p.get("nutrition_type", "balanced"), 1.0)
                        q_count += 1
                except Exception:
                    pass
        quality_score = (q_sum / q_count) if q_count > 0 else 1.0

        # ---- Exercise features from Garmin ----
        intensity_mins = sync_t.get("active_minutes", 0)
        active_cals = sync_t.get("active_calories", 0)

        # ---- Sleep hours from raw payload ----
        raw: dict = {}
        try:
            rp = sync_t.get("raw_payload")
            raw = json.loads(rp) if isinstance(rp, str) else (rp or {})
        except Exception:
            pass
        sleep_seconds = raw.get("sleep", {}).get("durationInSeconds", 28800)
        sleep_hours = sleep_seconds / 3600.0

        # ---- Weight delta ----
        w_t = next(
            (log["value"] for log in logs_t if log["log_type"] == "weight"), None
        )
        w_t1 = next(
            (log["value"] for log in log_map.get(date_t1, []) if log["log_type"] == "weight"),
            None,
        )
        delta_weight = float(w_t1 - w_t) if (w_t is not None and w_t1 is not None) else 0.0

        X_rows.append([
            sleep_hours,
            protein_g / 100.0,
            carbs_g / 100.0,
            fat_g / 100.0,
            quality_score,
            intensity_mins / 60.0,
            active_cals / 100.0,
        ])
        Y_rows.append([delta_sleep_score, delta_hrv, delta_rhr, delta_weight, delta_stress])

    if len(X_rows) < 14:
        return {
            "error": (
                f"Not enough contiguous days with behavioural logs. "
                f"Need 14 paired samples (currently have {len(X_rows)})."
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
