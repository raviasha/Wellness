"""Tests for wellness_env/distribution_simulator.py and WellnessEnv distribution mode."""

import copy
import random

import numpy as np
import pytest

from backend.distribution_calibration import (
    fit_joint_distribution,
    N_X,
    N_Y,
)
from wellness_env.distribution_simulator import (
    _DELTA_BOUNDS,
    compute_biomarker_changes_from_distribution,
)
from wellness_env.models import (
    Action,
    Biomarkers,
    BiomarkerDeltas,
    ActivityLevel,
    SleepDuration,
)
from wellness_env.personas import PERSONAS


@pytest.fixture
def default_biomarkers():
    return Biomarkers(resting_hr=65, hrv=45, sleep_score=75, stress_avg=45, body_battery=55)

@pytest.fixture
def default_action():
    return Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.MODERATE_ACTIVITY)

@pytest.fixture
def synthetic_distribution():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, N_X))
    Y = rng.standard_normal((30, N_Y))
    return fit_joint_distribution(X, Y)

@pytest.fixture
def digital_twin_persona():
    return copy.deepcopy(PERSONAS["digital_twin"])


class TestComputeBiomarkerChangesFromDistribution:

    def test_returns_biomarker_deltas(self, default_action, default_biomarkers, digital_twin_persona, synthetic_distribution):
        rng = random.Random(0)
        deltas = compute_biomarker_changes_from_distribution(
            default_action, default_biomarkers, digital_twin_persona,
            synthetic_distribution, [], rng,
        )
        assert isinstance(deltas, BiomarkerDeltas)

    def test_all_deltas_within_physiological_bounds(self, default_action, default_biomarkers, digital_twin_persona, synthetic_distribution):
        rng = random.Random(1)
        for _ in range(50):
            deltas = compute_biomarker_changes_from_distribution(
                default_action, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], rng,
            )
            d = deltas.model_dump()
            for key, (lo, hi) in _DELTA_BOUNDS.items():
                v = d[key]
                assert lo <= v <= hi

    def test_different_actions_produce_different_distribution_means(self, default_biomarkers, digital_twin_persona, synthetic_distribution):
        action_intense = Action(sleep=SleepDuration.VERY_SHORT, activity=ActivityLevel.HIGH_INTENSITY)
        action_rest = Action(sleep=SleepDuration.OPTIMAL_HIGH, activity=ActivityLevel.REST_DAY)
        n_samples = 100

        intense_deltas = [
            compute_biomarker_changes_from_distribution(
                action_intense, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], random.Random(42),
            ).model_dump()
            for _ in range(n_samples)
        ]
        rest_deltas = [
            compute_biomarker_changes_from_distribution(
                action_rest, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], random.Random(42),
            ).model_dump()
            for _ in range(n_samples)
        ]
        keys = list(_DELTA_BOUNDS.keys())
        intense_means = {k: np.mean([d[k] for d in intense_deltas]) for k in keys}
        rest_means = {k: np.mean([d[k] for d in rest_deltas]) for k in keys}
        any_differ = any(abs(intense_means[k] - rest_means[k]) > 1e-4 for k in keys)
        assert any_differ

    def test_all_delta_fields_finite(self, default_action, default_biomarkers, digital_twin_persona, synthetic_distribution):
        rng = random.Random(99)
        for _ in range(20):
            deltas = compute_biomarker_changes_from_distribution(
                default_action, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], rng,
            )
            for key, val in deltas.model_dump().items():
                assert np.isfinite(val)


class TestWellnessEnvDistributionMode:

    def test_full_episode_runs_without_error(self, synthetic_distribution):
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=0, simulator_mode="distribution", distribution=synthetic_distribution)
        obs = env.reset("personal_coaching")
        for _ in range(obs.total_days):
            action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.MODERATE_ACTIVITY)
            result = env.step(action)
            if result.done:
                break

    def test_biomarkers_stay_within_model_bounds(self, synthetic_distribution):
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=1, simulator_mode="distribution", distribution=synthetic_distribution)
        obs = env.reset("personal_coaching")
        action = Action(sleep=SleepDuration.SHORT, activity=ActivityLevel.HIGH_INTENSITY)
        for _ in range(obs.total_days):
            result = env.step(action)
            b = result.observation.biomarkers
            assert 40 <= b.resting_hr <= 120
            assert 5 <= b.hrv <= 150
            assert 0 <= b.sleep_score <= 100
            assert 0 <= b.stress_avg <= 100
            assert 0 <= b.body_battery <= 100
            if result.done:
                break

    def test_set_distribution_switches_mode(self, synthetic_distribution):
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=0)
        assert env._simulator_mode == "rules"
        env.set_distribution(synthetic_distribution)
        assert env._simulator_mode == "distribution"
        assert env._distribution is synthetic_distribution

    def test_rules_mode_unchanged(self):
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=42)
        obs = env.reset("cardiovascular_fitness")
        for _ in range(obs.total_days):
            action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.MODERATE_ACTIVITY)
            result = env.step(action)
            if result.done:
                break
        assert result.done

    def test_reward_is_finite_in_distribution_mode(self, synthetic_distribution):
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=2, simulator_mode="distribution", distribution=synthetic_distribution)
        obs = env.reset("personal_coaching")
        action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.LIGHT_ACTIVITY)
        for _ in range(5):
            result = env.step(action)
            assert np.isfinite(result.reward.total)
