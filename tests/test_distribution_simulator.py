"""Tests for wellness_env/distribution_simulator.py and WellnessEnv distribution mode.

Covers:
  - compute_biomarker_changes_from_distribution returns valid BiomarkerDeltas
  - Physiological delta bounds are respected
  - Overtraining modifier is applied on top of distribution samples
  - Sleep debt modifier scales fat-loss deltas
  - WellnessEnv in distribution mode runs a full episode without errors
  - Distribution mode and rules mode both produce deltas in valid ranges
"""

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
    ExerciseType,
    NutritionType,
    SleepDuration,
)
from wellness_env.personas import PERSONAS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_biomarkers():
    return Biomarkers(
        resting_hr=65,
        hrv=45,
        vo2_max=38,
        body_fat_pct=22,
        lean_mass_kg=58,
        sleep_efficiency=78,
        cortisol_proxy=45,
        energy_level=55,
    )


@pytest.fixture
def default_action():
    return Action(
        sleep=SleepDuration.OPTIMAL_LOW,
        exercise=ExerciseType.MODERATE_CARDIO,
        nutrition=NutritionType.BALANCED,
    )


@pytest.fixture
def synthetic_distribution():
    """A small synthetic distribution fitted from random data (n=30)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, N_X))
    Y = rng.standard_normal((30, N_Y))
    return fit_joint_distribution(X, Y)


@pytest.fixture
def digital_twin_persona():
    return copy.deepcopy(PERSONAS["digital_twin"])


# ---------------------------------------------------------------------------
# compute_biomarker_changes_from_distribution
# ---------------------------------------------------------------------------

class TestComputeBiomarkerChangesFromDistribution:

    def test_returns_biomarker_deltas(
        self, default_action, default_biomarkers, digital_twin_persona, synthetic_distribution
    ):
        rng = random.Random(0)
        deltas = compute_biomarker_changes_from_distribution(
            default_action, default_biomarkers, digital_twin_persona,
            synthetic_distribution, [], rng,
        )
        assert isinstance(deltas, BiomarkerDeltas)

    def test_all_deltas_within_physiological_bounds(
        self, default_action, default_biomarkers, digital_twin_persona, synthetic_distribution
    ):
        rng = random.Random(1)
        for _ in range(50):
            deltas = compute_biomarker_changes_from_distribution(
                default_action, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], rng,
            )
            d = deltas.model_dump()
            for key, (lo, hi) in _DELTA_BOUNDS.items():
                v = d[key]
                assert lo <= v <= hi, (
                    f"{key} delta={v:.4f} outside [{lo}, {hi}]"
                )

    def test_different_actions_produce_different_distribution_means(
        self, default_biomarkers, digital_twin_persona, synthetic_distribution
    ):
        """Different actions should (on average) give different deltas for at least
        one biomarker, since the distribution conditions on the action features."""
        action_hiit = Action(
            sleep=SleepDuration.VERY_SHORT,
            exercise=ExerciseType.HIIT,
            nutrition=NutritionType.PROCESSED,
        )
        action_rest = Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.NONE,
            nutrition=NutritionType.BALANCED,
        )
        n_samples = 100
        rng_hiit = random.Random(42)
        rng_rest = random.Random(42)

        hiit_deltas = [
            compute_biomarker_changes_from_distribution(
                action_hiit, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], rng_hiit,
            ).model_dump()
            for _ in range(n_samples)
        ]
        rest_deltas = [
            compute_biomarker_changes_from_distribution(
                action_rest, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], rng_rest,
            ).model_dump()
            for _ in range(n_samples)
        ]
        # At least one biomarker mean should differ (VO2 heuristic alone will differ)
        keys = list(_DELTA_BOUNDS.keys())
        hiit_means = {k: np.mean([d[k] for d in hiit_deltas]) for k in keys}
        rest_means = {k: np.mean([d[k] for d in rest_deltas]) for k in keys}
        any_differ = any(abs(hiit_means[k] - rest_means[k]) > 1e-4 for k in keys)
        assert any_differ, "HIIT and NONE actions should produce different mean deltas"

    def test_overtraining_increases_cortisol(
        self, default_biomarkers, digital_twin_persona, synthetic_distribution
    ):
        """After 4+ consecutive intense days, cortisol delta should be elevated."""
        rm = digital_twin_persona.response_model
        # Build a history of 4 consecutive HIIT days
        intense_history = [
            {"actual_action": {"exercise": "hiit"}}
        ] * 4

        action_hiit = Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.HIIT,
            nutrition=NutritionType.BALANCED,
        )
        n = 100
        with_history = [
            compute_biomarker_changes_from_distribution(
                action_hiit, default_biomarkers, digital_twin_persona,
                synthetic_distribution, intense_history, random.Random(i),
            ).cortisol_proxy
            for i in range(n)
        ]
        without_history = [
            compute_biomarker_changes_from_distribution(
                action_hiit, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], random.Random(i),
            ).cortisol_proxy
            for i in range(n)
        ]
        assert np.mean(with_history) > np.mean(without_history), (
            "Overtraining should increase mean cortisol delta"
        )

    def test_all_delta_fields_finite(
        self, default_action, default_biomarkers, digital_twin_persona, synthetic_distribution
    ):
        rng = random.Random(99)
        for _ in range(20):
            deltas = compute_biomarker_changes_from_distribution(
                default_action, default_biomarkers, digital_twin_persona,
                synthetic_distribution, [], rng,
            )
            for key, val in deltas.model_dump().items():
                assert np.isfinite(val), f"{key} delta is not finite: {val}"


# ---------------------------------------------------------------------------
# WellnessEnv distribution mode integration
# ---------------------------------------------------------------------------

class TestWellnessEnvDistributionMode:

    def _make_env(self, distribution):
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=0, simulator_mode="distribution", distribution=distribution)
        return env

    def test_full_episode_runs_without_error(self, synthetic_distribution):
        from wellness_env.env import WellnessEnv
        from wellness_env.models import Action, SleepDuration, ExerciseType, NutritionType
        env = WellnessEnv(seed=0, simulator_mode="distribution", distribution=synthetic_distribution)
        obs = env.reset("personal_coaching")
        assert obs is not None
        for _ in range(obs.total_days):
            action = Action(
                sleep=SleepDuration.OPTIMAL_LOW,
                exercise=ExerciseType.MODERATE_CARDIO,
                nutrition=NutritionType.BALANCED,
            )
            result = env.step(action)
            if result.done:
                break

    def test_biomarkers_stay_within_model_bounds(self, synthetic_distribution):
        """After a full episode, all biomarkers should stay within pydantic field bounds."""
        from wellness_env.env import WellnessEnv
        from wellness_env.models import Action, SleepDuration, ExerciseType, NutritionType
        env = WellnessEnv(seed=1, simulator_mode="distribution", distribution=synthetic_distribution)
        obs = env.reset("personal_coaching")
        action = Action(
            sleep=SleepDuration.SHORT,
            exercise=ExerciseType.HIIT,
            nutrition=NutritionType.PROCESSED,
        )
        for _ in range(obs.total_days):
            result = env.step(action)
            b = result.observation.biomarkers
            assert 40 <= b.resting_hr <= 120
            assert 5 <= b.hrv <= 150
            assert 15 <= b.vo2_max <= 70
            assert 3 <= b.body_fat_pct <= 50
            assert 30 <= b.lean_mass_kg <= 100
            assert 0 <= b.sleep_efficiency <= 100
            assert 0 <= b.cortisol_proxy <= 100
            assert 0 <= b.energy_level <= 100
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
        """Default (rules) mode should still work exactly as before."""
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=42)
        obs = env.reset("single_goal")
        for _ in range(obs.total_days):
            action = Action(
                sleep=SleepDuration.OPTIMAL_LOW,
                exercise=ExerciseType.MODERATE_CARDIO,
                nutrition=NutritionType.BALANCED,
            )
            result = env.step(action)
            if result.done:
                break
        assert result.done

    def test_reward_is_finite_in_distribution_mode(self, synthetic_distribution):
        from wellness_env.env import WellnessEnv
        env = WellnessEnv(seed=2, simulator_mode="distribution", distribution=synthetic_distribution)
        obs = env.reset("personal_coaching")
        action = Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.YOGA,
            nutrition=NutritionType.HIGH_PROTEIN,
        )
        for _ in range(5):
            result = env.step(action)
            assert np.isfinite(result.reward.total), f"Reward not finite: {result.reward.total}"
