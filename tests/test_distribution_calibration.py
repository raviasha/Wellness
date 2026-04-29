"""Tests for backend/distribution_calibration.py."""

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


def _make_synthetic_data(n, seed=0):
    rng = np.random.default_rng(seed)
    dim = N_X + N_Y
    A = rng.standard_normal((dim, dim))
    cov_true = A @ A.T + np.eye(dim) * dim
    means_true = np.zeros(dim)
    Z = rng.multivariate_normal(means_true, cov_true, size=n)
    X = Z[:, :N_X]
    Y = Z[:, N_X:]
    corr_true = None
    return X, Y, corr_true


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
        assert eigvals.min() >= -1e-6

    def test_wrong_input_dim_raises(self):
        # fit_joint_distribution now auto-detects dimensions—any dim is valid.
        X = np.random.randn(20, N_X + 1)
        Y = np.random.randn(20, N_Y)
        dist = fit_joint_distribution(X, Y)
        assert dist.n_x == N_X + 1
        assert dist.n_y == N_Y

    def test_wrong_output_dim_raises(self):
        # fit_joint_distribution now auto-detects dimensions—any dim is valid.
        X = np.random.randn(20, N_X)
        Y = np.random.randn(20, N_Y + 2)
        dist = fit_joint_distribution(X, Y)
        assert dist.n_x == N_X
        assert dist.n_y == N_Y + 2

    def test_stores_n_samples(self):
        n = 22
        X, Y, _ = _make_synthetic_data(n)
        dist = fit_joint_distribution(X, Y)
        assert dist.n_samples == n

    def test_stds_positive(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        assert all(s > 0 for s in dist.stds)

    def test_shrinkage_in_range(self):
        X, Y, _ = _make_synthetic_data(20)
        dist = fit_joint_distribution(X, Y)
        assert 0.0 <= dist.ledoit_wolf_shrinkage <= 1.0


class TestSampleConditional:

    def test_output_shape(self):
        X, Y, _ = _make_synthetic_data(30)
        dist = fit_joint_distribution(X, Y)
        rng = np.random.default_rng(0)
        y_samp = sample_conditional(dist, X[0], rng)
        assert y_samp.shape == (N_Y,)

    def test_conditional_mean_for_known_linear_relationship(self):
        rng_np = np.random.default_rng(7)
        n = 500
        X = rng_np.standard_normal((n, N_X))
        Y = np.zeros((n, N_Y))
        Y[:, 0] = 2.0 * X[:, 0] + rng_np.standard_normal(n) * 0.2
        Y[:, 1:] = rng_np.standard_normal((n, N_Y - 1))

        dist = fit_joint_distribution(X, Y)
        x_test = np.zeros(N_X)
        x_test[0] = 2.0

        samples = np.array([
            sample_conditional(dist, x_test, np.random.default_rng(s))
            for s in range(500)
        ])
        cond_mean_y0 = samples[:, 0].mean()
        assert abs(cond_mean_y0 - 4.0) < 1.5

    def test_different_x_gives_different_y_mean(self):
        X, Y, _ = _make_synthetic_data(200, seed=1)
        dist = fit_joint_distribution(X, Y)
        x_low = X.mean(axis=0) - X.std(axis=0)
        x_high = X.mean(axis=0) + X.std(axis=0)
        mean_low = np.mean([sample_conditional(dist, x_low, np.random.default_rng(s)) for s in range(200)], axis=0)
        mean_high = np.mean([sample_conditional(dist, x_high, np.random.default_rng(s)) for s in range(200)], axis=0)
        assert not np.allclose(mean_low, mean_high, atol=0.01)


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
        np.testing.assert_allclose(dist2.means, dist.means)

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


class TestEncodeActionToFeatures:

    def _make_action(self, sleep="7_to_8h", activity="moderate_activity"):
        from wellness_env.models import Action, SleepDuration, ActivityLevel
        return Action(
            sleep=SleepDuration(sleep),
            activity=ActivityLevel(activity),
        )

    def test_output_shape(self):
        action = self._make_action()
        x = encode_action_to_features(action)
        assert x.shape == (N_X,)

    def test_rest_day_gives_low_activity_features(self):
        action = self._make_action(activity="rest_day")
        x = encode_action_to_features(action)
        # active_minutes and active_calories should be low/zero
        assert x[2] <= 0.1  # active_minutes_h
        assert x[3] <= 0.1  # active_calories_100s

    def test_sleep_hours_plausible(self):
        for sleep_val in ["less_than_6h", "6_to_7h", "7_to_8h", "8_to_9h", "more_than_9h"]:
            action = self._make_action(sleep=sleep_val)
            x = encode_action_to_features(action)
            assert 4.0 <= x[0] <= 11.0

    def test_all_values_finite(self):
        action = self._make_action(sleep="8_to_9h", activity="high_intensity")
        x = encode_action_to_features(action)
        assert np.all(np.isfinite(x))
