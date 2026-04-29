"""Tests for the outcome-based payoff (reward) function."""
from __future__ import annotations

import random
import pytest

from wellness_env.models import BiomarkerDeltas, Goal
from wellness_env.payoff import DELTA_SCALES, GOAL_WEIGHTS, LOWER_IS_BETTER, _linear_slope, _stddev, compute_reward

ZERO_DELTAS = BiomarkerDeltas(resting_hr=0, hrv=0, sleep_score=0, stress_avg=0, body_battery=0)


class TestGoalWeights:
    def test_all_goals_present(self):
        for goal in Goal:
            assert goal in GOAL_WEIGHTS

    def test_all_biomarkers_present(self):
        markers = {"resting_hr", "hrv", "sleep_score", "stress_avg", "body_battery",
                   "sleep_stage_quality", "vo2_max"}
        for goal in Goal:
            assert set(GOAL_WEIGHTS[goal].keys()) == markers

    def test_weights_sum_to_one(self):
        for goal in Goal:
            total = sum(GOAL_WEIGHTS[goal].values())
            assert abs(total - 1.0) < 0.01, f"{goal}: weights sum to {total}"

    def test_all_weights_positive(self):
        for goal in Goal:
            for marker, w in GOAL_WEIGHTS[goal].items():
                assert w >= 0, f"{goal}/{marker} has negative weight"


class TestDeltaScales:
    def test_all_biomarkers_have_scale(self):
        for marker in ["resting_hr", "hrv", "sleep_score", "stress_avg", "body_battery"]:
            assert marker in DELTA_SCALES

    def test_all_scales_positive(self):
        for marker, scale in DELTA_SCALES.items():
            assert scale > 0


class TestComputeReward:
    def test_zero_deltas_gives_baseline_50(self):
        reward = compute_reward(ZERO_DELTAS, Goal.STRESS_MANAGEMENT)
        assert reward.total == 50.0

    def test_zero_deltas_all_goals(self):
        for goal in Goal:
            reward = compute_reward(ZERO_DELTAS, goal)
            assert reward.total == 50.0

    def test_positive_improvement_above_50(self):
        deltas = BiomarkerDeltas(resting_hr=-1.0, hrv=5.0, sleep_score=2.0, stress_avg=-5.0, body_battery=10.0)
        for goal in Goal:
            reward = compute_reward(deltas, goal)
            assert reward.total > 50.0

    def test_negative_changes_below_50(self):
        deltas = BiomarkerDeltas(resting_hr=1.0, hrv=-5.0, sleep_score=-2.0, stress_avg=5.0, body_battery=-10.0)
        for goal in Goal:
            reward = compute_reward(deltas, goal)
            assert reward.total < 50.0

    def test_reward_clamped_to_0_100(self):
        extreme = BiomarkerDeltas(resting_hr=-20, hrv=100, sleep_score=50, stress_avg=-50, body_battery=100)
        for goal in Goal:
            reward = compute_reward(extreme, goal)
            assert 0.0 <= reward.total <= 100.0

    def test_reward_breakdown_fields(self):
        deltas = BiomarkerDeltas(resting_hr=-1.0, hrv=3.0, sleep_score=1.0, stress_avg=-2.0, body_battery=5.0)
        reward = compute_reward(deltas, Goal.CARDIOVASCULAR_FITNESS)
        assert hasattr(reward, 'resting_hr_reward')
        assert hasattr(reward, 'hrv_reward')
        assert hasattr(reward, 'sleep_score_reward')
        assert hasattr(reward, 'stress_avg_reward')
        assert hasattr(reward, 'body_battery_reward')

    def test_lower_is_better_markers(self):
        assert "resting_hr" in LOWER_IS_BETTER
        assert "stress_avg" in LOWER_IS_BETTER
        assert "hrv" not in LOWER_IS_BETTER


class TestUtilities:
    def test_stddev_empty(self):
        assert _stddev([]) == 0.0

    def test_stddev_single(self):
        assert _stddev([5.0]) == 0.0

    def test_stddev_constant(self):
        assert _stddev([5, 5, 5]) == pytest.approx(0.0, abs=0.01)

    def test_stddev_known(self):
        assert _stddev([0, 10]) == pytest.approx(5.0, abs=0.01)

    def test_linear_slope_empty(self):
        assert _linear_slope([]) == 0.0

    def test_linear_slope_single(self):
        assert _linear_slope([5.0]) == 0.0

    def test_linear_slope_ascending(self):
        assert _linear_slope([1, 2, 3, 4, 5]) == pytest.approx(1.0, abs=0.01)

    def test_linear_slope_descending(self):
        assert _linear_slope([5, 3, 1]) == pytest.approx(-2.0, abs=0.01)

    def test_linear_slope_flat(self):
        assert _linear_slope([5, 5, 5]) == pytest.approx(0.0, abs=0.01)
