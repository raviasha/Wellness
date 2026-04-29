"""Gap analysis for the calibrated distribution model.

Computes residuals, R², and detects systematic patterns that indicate
where the persona model diverges from real Garmin data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.distribution_calibration import (
    JointDistribution,
    Y_OUTCOME_NAMES,
    sample_conditional,
)


@dataclass
class GapReport:
    """Summary of model gaps for a single user."""
    r_squared: dict[str, float] = field(default_factory=dict)
    mean_residuals: dict[str, float] = field(default_factory=dict)
    residual_patterns: list[str] = field(default_factory=list)
    overall_fit: str = "unknown"  # "good", "fair", "poor"


def compute_residuals(
    dist: JointDistribution,
    X: np.ndarray,
    Y_actual: np.ndarray,
    n_samples: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Compute residuals = actual - mean(predicted) for each observation.

    Args:
        dist: fitted joint distribution
        X: (n, n_x) input features
        Y_actual: (n, n_y) actual outcomes
        n_samples: Monte Carlo samples per observation for conditional mean
        seed: random seed

    Returns:
        (n, n_y) residual matrix
    """
    n = X.shape[0]
    n_y = Y_actual.shape[1]
    predicted = np.zeros((n, n_y))

    for i in range(n):
        samples = np.array([
            sample_conditional(dist, X[i], np.random.default_rng(seed + i * n_samples + s))
            for s in range(n_samples)
        ])
        predicted[i] = samples.mean(axis=0)

    return Y_actual - predicted


def compute_r_squared(Y_actual: np.ndarray, residuals: np.ndarray) -> dict[str, float]:
    """Compute R² per outcome variable.

    Returns:
        dict mapping outcome name to R² value.
    """
    result = {}
    for j, name in enumerate(Y_OUTCOME_NAMES):
        ss_res = np.sum(residuals[:, j] ** 2)
        ss_tot = np.sum((Y_actual[:, j] - Y_actual[:, j].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        result[name] = round(float(r2), 4)
    return result


def detect_residual_patterns(residuals: np.ndarray) -> list[str]:
    """Detect systematic patterns in residuals.

    Checks for:
      - Persistent bias (mean residual significantly != 0)
      - Increasing variance (heteroscedasticity proxy)
      - Autocorrelation (sequential residuals correlated)

    Returns:
        List of human-readable pattern descriptions.
    """
    patterns = []
    n = residuals.shape[0]

    for j, name in enumerate(Y_OUTCOME_NAMES):
        r = residuals[:, j]
        mean_r = float(np.mean(r))
        std_r = float(np.std(r)) if n > 1 else 0.0

        # Bias detection: mean residual > 0.5 std of outcomes
        if abs(mean_r) > 0.3 * (std_r + 1e-6):
            direction = "over-predicting" if mean_r < 0 else "under-predicting"
            patterns.append(f"{name}: systematic bias ({direction}, mean residual={mean_r:.3f})")

        # Trend in residuals (are errors growing over time?)
        if n >= 7:
            first_half = np.mean(np.abs(r[: n // 2]))
            second_half = np.mean(np.abs(r[n // 2 :]))
            if second_half > first_half * 1.5 and second_half > 0.5:
                patterns.append(f"{name}: errors increasing over time (model may be drifting)")

        # Autocorrelation (lag-1)
        if n >= 5:
            r_centered = r - r.mean()
            autocorr = np.sum(r_centered[:-1] * r_centered[1:]) / (np.sum(r_centered**2) + 1e-12)
            if autocorr > 0.5:
                patterns.append(f"{name}: residuals are autocorrelated (lag-1 r={autocorr:.2f}), model missing a temporal pattern")

    return patterns


def generate_gap_report(
    dist: JointDistribution,
    X: np.ndarray,
    Y_actual: np.ndarray,
    n_samples: int = 50,
    seed: int = 0,
) -> GapReport:
    """Full gap analysis pipeline.

    Args:
        dist: fitted joint distribution
        X: (n, n_x) encoded action features
        Y_actual: (n, n_y) actual biomarker deltas

    Returns:
        GapReport with R², mean residuals, detected patterns, and overall fit.
    """
    residuals = compute_residuals(dist, X, Y_actual, n_samples=n_samples, seed=seed)
    r2 = compute_r_squared(Y_actual, residuals)
    mean_res = {
        name: round(float(np.mean(residuals[:, j])), 4)
        for j, name in enumerate(Y_OUTCOME_NAMES)
    }
    patterns = detect_residual_patterns(residuals)

    # Overall fit classification
    avg_r2 = np.mean(list(r2.values()))
    if avg_r2 >= 0.5 and len(patterns) == 0:
        overall = "good"
    elif avg_r2 >= 0.2 or len(patterns) <= 2:
        overall = "fair"
    else:
        overall = "poor"

    return GapReport(
        r_squared=r2,
        mean_residuals=mean_res,
        residual_patterns=patterns,
        overall_fit=overall,
    )
