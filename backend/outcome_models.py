"""Per-outcome Ridge regression models for the ML model tier.

Trains one Ridge regression per biomarker outcome using leave-one-out
cross-validation for an honest R² estimate.  Saves/loads as JSON so no
PyTorch dependency is needed at this tier.

The ML model simultaneously serves as:
  1. Evals display  — R², feature importances, human explanations
  2. Simulator mean — conditional mean prediction for the ml_model simulator mode
  3. Inference path — expected delta computation alongside the copula noise
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from backend.feature_engineering import (
    build_Xy_matrix,
    OUTCOME_NAMES,
)
from backend.maturity_config import get_user_thresholds

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")

# Human-readable outcome labels
OUTCOME_LABELS: dict[str, str] = {
    "delta_sleep_score":       "Sleep Score Change",
    "delta_hrv_ms":            "HRV Change",
    "delta_rhr_bpm":           "Resting HR Change",
    "delta_stress":            "Stress Level Change",
    "delta_body_battery":      "Body Battery Change",
    "delta_sleep_stage_quality": "Sleep Quality Change",
    "delta_vo2_max":           "VO2 Max Change",
}

# Suggestions for low-R² outcomes (investor-facing)
_LOW_R2_SUGGESTIONS: dict[str, str] = {
    "delta_sleep_score":       "Adding consistent bedtime logging or sleep environment data could improve this.",
    "delta_hrv_ms":            "Tracking nutrition or hydration may explain more of your HRV variation.",
    "delta_rhr_bpm":           "Temperature, caffeine, or alcohol data could improve resting HR prediction.",
    "delta_stress":            "Mental workload or nutrition tracking could explain more stress variation.",
    "delta_body_battery":      "Hydration and nutrition data are likely missing contributors.",
    "delta_sleep_stage_quality": "Consistent bedtime and environmental factors (temperature, noise) matter here.",
    "delta_vo2_max":           "VO2 max changes slowly — more weeks of consistent cardio data will improve this.",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OutcomeModel:
    outcome_name: str
    feature_names: list[str]
    coefficients: list[float]          # unstandardised (original scale)
    intercept: float
    feature_importances: dict[str, float]  # standardised |coeff| normalised to sum=1
    r_squared: float                   # leave-one-out cross-validated
    r_squared_train: float             # in-sample (for comparison)
    n_samples: int
    mse: float                         # on held-out LOO predictions
    fitted_at: str                     # ISO timestamp

    # Scaler params for inference
    scaler_mean: list[float]
    scaler_std: list[float]


@dataclass
class OutcomeModelSuite:
    models: dict[str, OutcomeModel]    # keyed by outcome name
    data_days: int
    include_rolling: bool
    fitted_at: str

    def to_dict(self) -> dict:
        return {
            "models": {k: asdict(v) for k, v in self.models.items()},
            "data_days": self.data_days,
            "include_rolling": self.include_rolling,
            "fitted_at": self.fitted_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutcomeModelSuite":
        models = {
            k: OutcomeModel(**v) for k, v in d["models"].items()
        }
        return cls(
            models=models,
            data_days=d["data_days"],
            include_rolling=d.get("include_rolling", False),
            fitted_at=d["fitted_at"],
        )


# ---------------------------------------------------------------------------
# LOO cross-validated R²
# ---------------------------------------------------------------------------

def _loo_r2_mse(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[float, float]:
    """Leave-one-out cross-validated R² and MSE for Ridge regression."""
    n = len(y)
    if n < 4:
        return float("nan"), float("nan")

    loo_residuals = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_val, y_val = X[i:i+1], y[i]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_s, y_tr)
        loo_residuals[i] = y_val - ridge.predict(X_val_s)[0]

    ss_res = float(np.sum(loo_residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    mse = float(np.mean(loo_residuals ** 2))
    return max(-1.0, min(1.0, r2)), mse


def _in_sample_r2(X_s: np.ndarray, y: np.ndarray, ridge: Ridge) -> float:
    y_pred = ridge.predict(X_s)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_outcome_models(user_id: int) -> OutcomeModelSuite | None:
    """Fit one Ridge regression per outcome for a user.

    Returns None if insufficient data (below ml_model_min_days threshold).
    Saves the suite to models/user_{id}/outcome_models.json and appends to
    the trajectory log.
    """
    thresholds = get_user_thresholds(user_id)
    min_days = thresholds["ml_model_min_days"]
    lag2_min = thresholds["ml_lag2_min_days"]

    # Decide whether to include rolling features
    from backend.maturity_config import count_paired_days
    paired_days = count_paired_days(user_id)

    if paired_days < min_days:
        print(f"[OutcomeModels] user {user_id}: only {paired_days} paired days, need {min_days}")
        return None

    include_rolling = paired_days >= lag2_min

    X, Y, feature_names, outcome_names = build_Xy_matrix(
        user_id, include_rolling=include_rolling
    )

    if X.shape[0] < 4:
        print(f"[OutcomeModels] user {user_id}: too few rows ({X.shape[0]}) after pair extraction")
        return None

    models: dict[str, OutcomeModel] = {}
    alpha = 1.0

    for j, outcome in enumerate(outcome_names):
        y = Y[:, j]

        # Fit final model on all data (for coefficients + scaler)
        sc = StandardScaler()
        X_s = sc.fit_transform(X)
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_s, y)

        # LOO-CV metrics
        r2_loo, mse = _loo_r2_mse(X, y, alpha=alpha)
        r2_train = _in_sample_r2(X_s, y, ridge)

        # Feature importances: standardised absolute coefficients, normalised to sum=1
        std_coeffs = ridge.coef_ * np.array(sc.scale_)  # back to original-scale magnitude proxy
        abs_coeffs = np.abs(ridge.coef_)  # standardised space importance
        total = abs_coeffs.sum()
        if total < 1e-10:
            importances = {fn: 0.0 for fn in feature_names}
        else:
            importances = {
                feature_names[k]: float(abs_coeffs[k] / total)
                for k in range(len(feature_names))
            }

        models[outcome] = OutcomeModel(
            outcome_name=outcome,
            feature_names=feature_names,
            coefficients=ridge.coef_.tolist(),
            intercept=float(ridge.intercept_),
            feature_importances=importances,
            r_squared=round(float(r2_loo), 4),
            r_squared_train=round(float(r2_train), 4),
            n_samples=int(X.shape[0]),
            mse=round(float(mse), 4) if math.isfinite(mse) else 0.0,
            fitted_at=datetime.utcnow().isoformat(),
            scaler_mean=sc.mean_.tolist(),
            scaler_std=sc.scale_.tolist(),
        )

    suite = OutcomeModelSuite(
        models=models,
        data_days=paired_days,
        include_rolling=include_rolling,
        fitted_at=datetime.utcnow().isoformat(),
    )

    save_outcome_models(user_id, suite)
    _append_trajectory(user_id, suite)
    print(f"[OutcomeModels] user {user_id}: trained {len(models)} models on {X.shape[0]} rows")
    return suite


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_from_suite(suite: OutcomeModelSuite, x_features: np.ndarray) -> dict[str, float]:
    """Predict one day's outcome deltas from the ML suite.

    Parameters
    ----------
    suite : OutcomeModelSuite
    x_features : (n_features,) array in original (unscaled) space

    Returns
    -------
    dict mapping outcome_name → predicted delta (float)
    """
    result: dict[str, float] = {}
    for outcome, model in suite.models.items():
        sc_mean = np.array(model.scaler_mean)
        sc_std  = np.array(model.scaler_std)
        sc_std  = np.where(sc_std < 1e-8, 1e-8, sc_std)
        x_s = (x_features - sc_mean) / sc_std
        pred = float(np.dot(x_s, model.coefficients) + model.intercept)
        result[outcome] = pred
    return result


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _model_dir(user_id: int) -> str:
    path = os.path.join(MODELS_ROOT, f"user_{user_id}")
    os.makedirs(path, exist_ok=True)
    return path


def save_outcome_models(user_id: int, suite: OutcomeModelSuite) -> None:
    path = os.path.join(_model_dir(user_id), "outcome_models.json")
    with open(path, "w") as fh:
        json.dump(suite.to_dict(), fh, indent=2)


def load_outcome_models(user_id: int) -> OutcomeModelSuite | None:
    path = os.path.join(MODELS_ROOT, f"user_{user_id}", "outcome_models.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return OutcomeModelSuite.from_dict(json.load(fh))
    except Exception as e:
        print(f"[OutcomeModels] load error for user {user_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# R² trajectory log (one JSON line per training run)
# ---------------------------------------------------------------------------

def _append_trajectory(user_id: int, suite: OutcomeModelSuite) -> None:
    """Append a snapshot of this training run to the trajectory JSONL log."""
    path = os.path.join(_model_dir(user_id), "outcome_model_history.jsonl")
    entry = {
        "fitted_at": suite.fitted_at,
        "data_days": suite.data_days,
        "r_squared": {
            outcome: model.r_squared
            for outcome, model in suite.models.items()
        },
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_trajectory(user_id: int) -> list[dict]:
    """Load all training run snapshots for the trajectory chart."""
    path = os.path.join(MODELS_ROOT, f"user_{user_id}", "outcome_model_history.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    return entries


# ---------------------------------------------------------------------------
# Eval-facing helpers
# ---------------------------------------------------------------------------

def _top_features(model: OutcomeModel, n: int = 3) -> list[dict[str, Any]]:
    """Return top-N features sorted by importance."""
    sorted_items = sorted(
        model.feature_importances.items(), key=lambda x: x[1], reverse=True
    )[:n]
    return [{"name": k, "importance": round(v, 4)} for k, v in sorted_items]


def _human_explanation(model: OutcomeModel) -> str:
    """Generate a one-sentence plain-English explanation of the model."""
    r2_pct = int(model.r_squared * 100)
    top = _top_features(model, n=2)
    label = OUTCOME_LABELS.get(model.outcome_name, model.outcome_name)

    # Map feature machine names → readable
    _readable = {
        "sleep_duration_hours":           "sleep duration",
        "bedtime_hour_cos":               "bedtime timing",
        "bedtime_hour_sin":               "bedtime timing",
        "active_minutes_h":               "active time",
        "exercise_type_idx":              "exercise type",
        "exercise_duration_h":            "exercise duration",
        "steps_1000s":                    "daily steps",
        "prev_delta_sleep_score":         "yesterday's sleep score change",
        "prev_delta_hrv_ms":              "yesterday's HRV change",
        "prev_delta_rhr_bpm":             "yesterday's resting HR change",
        "prev_delta_stress":              "yesterday's stress change",
        "prev_delta_body_battery":        "yesterday's body battery change",
        "prev_delta_sleep_stage_quality": "yesterday's sleep quality change",
        "prev_delta_vo2_max":             "yesterday's VO2 max change",
        "compliance_sleep":               "sleep compliance",
        "compliance_activity":            "activity compliance",
    }
    feature_words = []
    for f in top:
        name = f["name"]
        # strip rolling suffix for readable name
        base = name.replace("_roll7_mean", "").replace("_roll7_std", "")
        word = _readable.get(base, base.replace("_", " "))
        feature_words.append(word)

    if not feature_words:
        return f"{r2_pct}% of your {label} is captured by the model."

    drivers = " and ".join(dict.fromkeys(feature_words))  # deduplicate (cos/sin)
    return f"{r2_pct}% of your {label} is explained by {drivers}."


def get_eval_payload(user_id: int) -> dict[str, Any]:
    """Build the full evals payload for the /api/evals/models endpoint."""
    suite = load_outcome_models(user_id)
    if suite is None:
        return {"available": False, "reason": "ML models not yet trained."}

    outcome_cards = []
    for outcome in OUTCOME_NAMES:
        model = suite.models.get(outcome)
        if model is None:
            continue
        r2 = model.r_squared
        card: dict[str, Any] = {
            "outcome_name": outcome,
            "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
            "r_squared": r2,
            "r_squared_train": model.r_squared_train,
            "n_samples": model.n_samples,
            "top_features": _top_features(model, n=3),
            "explanation": _human_explanation(model),
            "low_r2": r2 < 0.3,
            "suggestion": _LOW_R2_SUGGESTIONS.get(outcome, "") if r2 < 0.3 else "",
        }
        outcome_cards.append(card)

    avg_r2 = float(np.mean([m.r_squared for m in suite.models.values()]))

    return {
        "available": True,
        "fitted_at": suite.fitted_at,
        "data_days": suite.data_days,
        "avg_r_squared": round(avg_r2, 4),
        "include_rolling": suite.include_rolling,
        "outcome_cards": outcome_cards,
        "trajectory": load_trajectory(user_id),
    }
