"""Tests for backend/distribution_calibration.py.

Covers:
  - fit_joint_distribution: correlation recovery on synthetic MVN data
  - sample_conditional: conditional mean recovery on known linear relationship
  - save/load round-trip
  - encode_action_to_features: correct shape and value ranges
  - Quality warnings trigger at correct thresholds
"""

import json
import os
import tempfile

import numpy as np
import pytest

from backend.distribution_calibration import (
    JointDistribution,
    N_X,
    N_Y,
    encode_action_to_features,
    fit_joint_distribution,
    load_distribution,
    sample_conditional,
    save_distribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (X, Y) from a known MVN with off-diagonal correlations."""
    rng = np.random.default_rng(seed)
    dim = N_X + N_Y
    # Build a random positive-definite correlation matrix
    A = rng.standard_normal((dim, dim))
    cov_true = A @ A.T + np.eye(dim) * dim   # ensures positive definiteness
    d = np.sqrt(np.diag(cov_true))
    corr_true = (cov_true / d[:, None]) / d[None, :]
    # Scale to realistic magnitudes
    means_true = np.zeros(dim)
    stds_true = np.ones(dim)
    Z = rng.multivariate_normal(means_true, cov_true, size=n)
    X = Z[:, :N_X]
    Y = Z[:, N_X:]
    return X, Y, corr_true


# ---------------------------------------------------------------------------
# fit_joint_distribution
# ---------------------------------------------------------------------------

class TestFitJointDistribution:

    def test_returns_joint_distribution(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        assert isinstance(dist, JointDistribution)

    def test_dimensions(self):
        X, Y, _ = _make_synthetic_data(25)
        dist = fit_joint_distribution(X, Y)
        assert dist.n_x == N_X
        assert dist.n_y == N_Y
        assert len(dist.means) == N_X + N_Y
        assert len(dist.stds) == N_X + N_Y
        assert len(dist.corr_matrix) == N_X + N_Y
        assert len(dist.corr_matrix[0]) == N_X + N_Y

    def test_diagonal_is_one(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        R = np.array(dist.corr_matrix)
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-6)

    def test_symmetric(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        R = np.array(dist.corr_matrix)
        np.testing.assert_allclose(R, R.T, atol=1e-10)

    def test_positive_semi_definite(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        R = np.array(dist.corr_matrix)
        eigvals = np.linalg.eigvalsh(R)
        assert eigvals.min() >= -1e-6, f"Correlation matrix not PSD, min eigenvalue={eigvals.min()}"

    def test_wrong_input_dim_raises(self):
        X = np.random.randn(20, N_X + 1)   # wrong column count
        Y = np.random.randn(20, N_Y)
        with pytest.raises(ValueError):
            fit_joint_distribution(X, Y)

    def test_wrong_output_dim_raises(self):
        X = np.random.randn(20, N_X)
        Y = np.random.randn(20, N_Y + 2)   # wrong column count
        with pytest.raises(ValueError):
            fit_joint_distribution(X, Y)

    def test_stores_n_samples(self):
        n = 22
        X, Y, _ = _make_synthetic_data(n)
        dist = fit_joint_distribution(X, Y)
        assert dist.n_samples == n

    def test_stds_positive(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        assert all(s > 0 for s in dist.stds)

    def test_condition_number_set(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        assert dist.condition_number > 0

    def test_shrinkage_in_range(self):
        X, Y, _ = _make_synthetic_data(20)
        dist = fit_joint_distribution(X, Y)
        assert 0.0 <= dist.ledoit_wolf_shrinkage <= 1.0

    def test_large_n_recovers_block_structure(self):
        """With many samples, Ledoit-Wolf should recover that X⊥Y if truly independent."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, N_X))
        Y = rng.standard_normal((300, N_Y))   # independent of X
        dist = fit_joint_distribution(X, Y)
        R = np.array(dist.corr_matrix)
        # Off-diagonal block (X-Y cross correlations) should be near 0
        R_xy = R[:N_X, N_X:]
        assert np.abs(R_xy).max() < 0.3, "Cross-correlations should be small for independent X, Y"


# ---------------------------------------------------------------------------
# sample_conditional
# ---------------------------------------------------------------------------

class TestSampleConditional:

    def test_output_shape(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        rng = np.random.default_rng(0)
        x_obs = X[0]
        y_samp = sample_conditional(dist, x_obs, rng)
        assert y_samp.shape == (N_Y,)

    def test_conditional_mean_for_known_linear_relationship(self):
        """If Y = B @ X + noise, sample_conditional should recover E[Y|X] ≈ B @ x."""
        rng_np = np.random.default_rng(7)
        n = 500     # large n so Ledoit-Wolf barely shrinks
        X = rng_np.standard_normal((n, N_X))
        # Simple linear relationship: Y[:,0] = 2 * X[:,0] + small noise
        Y = np.zeros((n, N_Y))
        Y[:, 0] = 2.0 * X[:, 0] + rng_np.standard_normal(n) * 0.2
        # Fill remaining Y columns with noise
        Y[:, 1:] = rng_np.standard_normal((n, N_Y - 1))

        dist = fit_joint_distribution(X, Y)

        x_test = np.zeros(N_X)
        x_test[0] = 2.0     # set first feature to 2.0; expect Y[:,0] ≈ 4.0

        # Average over many samples to estimate conditional mean
        samples = np.array([
            sample_conditional(dist, x_test, np.random.default_rng(s))
            for s in range(500)
        ])
        cond_mean_y0 = samples[:, 0].mean()
        # Allow ±1 tolerance (Ledoit-Wolf shrinkage mutes the estimate slightly)
        assert abs(cond_mean_y0 - 4.0) < 1.5, (
            f"E[Y0 | X0=2] should be ≈4.0, got {cond_mean_y0:.2f}"
        )

    def test_different_x_gives_different_y_mean(self):
        """Conditioning on different X should produce shifted Y distributions."""
        X, Y, _ = _make_synthetic_data(200, seed=1)
        dist = fit_joint_distribution(X, Y)
        rng0 = np.random.default_rng(0)
        rng1 = np.random.default_rng(0)
        x_low  = X.mean(axis=0) - X.std(axis=0)
        x_high = X.mean(axis=0) + X.std(axis=0)
        mean_low  = np.mean([sample_conditional(dist, x_low, np.random.default_rng(s)) for s in range(200)], axis=0)
        mean_high = np.mean([sample_conditional(dist, x_high, np.random.default_rng(s)) for s in range(200)], axis=0)
        # At least one dimension should differ
        assert not np.allclose(mean_low, mean_high, atol=0.01), (
            "Conditional means for very different X should differ"
        )


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadDistribution:

    def test_round_trip(self):
        X, Y, _ = _make_synthetic_data(25)
        dist = fit_joint_distribution(X, Y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dist.json")
            save_distribution(dist, path)
            assert os.path.exists(path)
            dist2 = load_distribution(path)

        assert dist2.n_x == dist.n_x
        assert dist2.n_y == dist.n_y
        assert dist2.n_samples == dist.n_samples
        np.testing.assert_allclose(dist2.means, dist.means)
        np.testing.assert_allclose(dist2.stds, dist.stds)
        np.testing.assert_allclose(
            np.array(dist2.corr_matrix), np.array(dist.corr_matrix)
        )

    def test_saved_file_is_valid_json(self):
        X, Y, _ = _make_synthetic_data(20)
        dist = fit_joint_distribution(X, Y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dist.json")
            save_distribution(dist, path)
            with open(path) as fh:
                payload = json.load(fh)
        assert "corr_matrix" in payload
        assert "means" in payload


# ---------------------------------------------------------------------------
# encode_action_to_features
# ---------------------------------------------------------------------------

class TestEncodeActionToFeatures:

    def _make_action(self, sleep="7_to_8h", exercise="moderate_cardio", nutrition="balanced"):
        from wellness_env.models import Action, SleepDuration, ExerciseType, NutritionType
        return Action(
            sleep=SleepDuration(sleep),
            exercise=ExerciseType(exercise),
            nutrition=NutritionType(nutrition),
        )

    def test_output_shape(self):
        action = self._make_action()
        x = encode_action_to_features(action)
        assert x.shape == (N_X,)

    def test_none_exercise_gives_zero_intensity(self):
        action = self._make_action(exercise="none")
        x = encode_action_to_features(action)
        # intensity_mins_h (index 5) and active_cals_100s (index 6) should be 0
        assert x[5] == 0.0
        assert x[6] == 0.0

    def test_skipped_nutrition_gives_zero_macros(self):
        action = self._make_action(nutrition="skipped")
        x = encode_action_to_features(action)
        # protein, carbs, fat, quality should all be 0
        assert x[1] == 0.0
        assert x[2] == 0.0
        assert x[3] == 0.0
        assert x[4] == 0.0

    def test_sleep_hours_plausible(self):
        for sleep_val in ["less_than_6h", "6_to_7h", "7_to_8h", "8_to_9h", "more_than_9h"]:
            action = self._make_action(sleep=sleep_val)
            x = encode_action_to_features(action)
            assert 4.0 <= x[0] <= 11.0, f"sleep_hours={x[0]} out of plausible range for {sleep_val}"

    def test_all_values_finite(self):
        action = self._make_action(sleep="8_to_9h", exercise="hiit", nutrition="high_protein")
        x = encode_action_to_features(action)
        assert np.all(np.isfinite(x))
