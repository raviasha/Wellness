"""Tests for the outcome-based payoff (reward) function."""
from __future__ import annotations

import pytest

from wellness_env.models import BiomarkerDeltas, Goal
from wellness_env.payoff import (
    DELTA_SCALES,
    GOAL_WEIGHTS,
    LOWER_IS_BETTER,
    _linear_slope,
    _stddev,
    compute_reward,
)


# ============================================================================
# Helpers
# ============================================================================

ZERO_DELTAS = BiomarkerDeltas(
    resting_hr=0, hrv=0, vo2_max=0, body_fat_pct=0,
    lean_mass_kg=0, sleep_efficiency=0, cortisol_proxy=0, energy_level=0,
)


# ============================================================================
# Goal weight structure
# ============================================================================

class TestGoalWeights:
    """Verify the GOAL_WEIGHTS table is well-formed."""

    def test_all_goals_present(self):
        for goal in Goal:
            assert goal in GOAL_WEIGHTS

    def test_all_biomarkers_present(self):
        expected = {
            "resting_hr", "hrv", "vo2_max", "body_fat_pct",
            "lean_mass_kg", "sleep_efficiency", "cortisol_proxy", "energy_level",
        }
        for goal, weights in GOAL_WEIGHTS.items():
            assert set(weights.keys()) == expected, f"{goal} missing keys"

    def test_weights_sum_to_one(self):
        for goal, weights in GOAL_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"{goal} weights sum to {total}"

    def test_all_weights_positive(self):
        for goal, weights in GOAL_WEIGHTS.items():
            for marker, w in weights.items():
                assert w > 0, f"{goal}.{marker} weight is {w}"


# ============================================================================
# Delta scales
# ============================================================================

class TestDeltaScales:
    def test_all_biomarkers_have_scale(self):
        expected = {
            "resting_hr", "hrv", "vo2_max", "body_fat_pct",
            "lean_mass_kg", "sleep_efficiency", "cortisol_proxy", "energy_level",
        }
        assert set(DELTA_SCALES.keys()) == expected

    def test_all_scales_positive(self):
        for marker, scale in DELTA_SCALES.items():
            assert scale > 0, f"{marker} scale is {scale}"


# ============================================================================
# Reward computation
# ============================================================================

class TestComputeReward:
    def test_zero_deltas_gives_baseline_50(self):
        reward = compute_reward(ZERO_DELTAS, Goal.OVERALL_WELLNESS)
        assert reward.total == 50.0

    def test_zero_deltas_all_goals(self):
        for goal in Goal:
            reward = compute_reward(ZERO_DELTAS, goal)
            assert reward.total == 50.0, f"{goal} baseline != 50"

    def test_positive_improvement_above_50(self):
        """All biomarkers improving should yield reward > 50."""
        good_deltas = BiomarkerDeltas(
            resting_hr=-1.0, hrv=5.0, vo2_max=0.3, body_fat_pct=-0.05,
            lean_mass_kg=0.1, sleep_efficiency=2.0, cortisol_proxy=-5.0, energy_level=10.0,
        )
        for goal in Goal:
            reward = compute_reward(good_deltas, goal)
            assert reward.total > 50.0, f"{goal} reward {reward.total} not > 50"

    def test_negative_changes_below_50(self):
        """All biomarkers declining should yield reward < 50."""
        bad_deltas = BiomarkerDeltas(
            resting_hr=1.0, hrv=-5.0, vo2_max=-0.3, body_fat_pct=0.05,
            lean_mass_kg=-0.1, sleep_efficiency=-2.0, cortisol_proxy=5.0, energy_level=-10.0,
        )
        for goal in Goal:
            reward = compute_reward(bad_deltas, goal)
            assert reward.total < 50.0, f"{goal} reward {reward.total} not < 50"

    def test_reward_clamped_to_0_100(self):
        """Extreme deltas should still produce rewards in [0, 100]."""
        extreme_good = BiomarkerDeltas(
            resting_hr=-20.0, hrv=100.0, vo2_max=5.0, body_fat_pct=-5.0,
            lean_mass_kg=5.0, sleep_efficiency=50.0, cortisol_proxy=-50.0, energy_level=100.0,
        )
        extreme_bad = BiomarkerDeltas(
            resting_hr=20.0, hrv=-100.0, vo2_max=-5.0, body_fat_pct=5.0,
            lean_mass_kg=-5.0, sleep_efficiency=-50.0, cortisol_proxy=50.0, energy_level=-100.0,
        )
        for goal in Goal:
            r_good = compute_reward(extreme_good, goal)
            r_bad = compute_reward(extreme_bad, goal)
            assert 0.0 <= r_good.total <= 100.0
            assert 0.0 <= r_bad.total <= 100.0

    def test_reward_breakdown_fields(self):
        reward = compute_reward(ZERO_DELTAS, Goal.WEIGHT_LOSS)
        assert hasattr(reward, "resting_hr_reward")
        assert hasattr(reward, "hrv_reward")
        assert hasattr(reward, "vo2_max_reward")
        assert hasattr(reward, "body_fat_reward")
        assert hasattr(reward, "lean_mass_reward")
        assert hasattr(reward, "sleep_efficiency_reward")
        assert hasattr(reward, "cortisol_reward")
        assert hasattr(reward, "energy_reward")
        assert hasattr(reward, "total")

    def test_lower_is_better_markers(self):
        """Negative delta for lower-is-better markers should contribute positive reward."""
        assert LOWER_IS_BETTER == {"resting_hr", "body_fat_pct", "cortisol_proxy"}

    def test_weight_loss_heavily_weights_body_fat(self):
        """Body fat change should dominate reward for weight loss goal."""
        bf_only = BiomarkerDeltas(
            resting_hr=0, hrv=0, vo2_max=0, body_fat_pct=-0.05,
            lean_mass_kg=0, sleep_efficiency=0, cortisol_proxy=0, energy_level=0,
        )
        reward = compute_reward(bf_only, Goal.WEIGHT_LOSS)
        assert reward.total > 50.0
        # body_fat should have the largest contribution
        assert abs(reward.body_fat_reward) > abs(reward.hrv_reward)


# ============================================================================
# Utility functions
# ============================================================================

class TestUtilities:
    def test_stddev_empty(self):
        assert _stddev([]) == 0.0

    def test_stddev_single(self):
        assert _stddev([5.0]) == 0.0

    def test_stddev_constant(self):
        assert _stddev([3.0, 3.0, 3.0]) == 0.0

    def test_stddev_known(self):
        # stddev of [1, 2, 3] = sqrt(2/3) ≈ 0.8165
        result = _stddev([1.0, 2.0, 3.0])
        assert abs(result - 0.8165) < 0.001

    def test_linear_slope_empty(self):
        assert _linear_slope([]) == 0.0

    def test_linear_slope_single(self):
        assert _linear_slope([5.0]) == 0.0

    def test_linear_slope_ascending(self):
        slope = _linear_slope([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(slope - 1.0) < 0.001

    def test_linear_slope_descending(self):
        slope = _linear_slope([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(slope - (-1.0)) < 0.001

    def test_linear_slope_flat(self):
        slope = _linear_slope([3.0, 3.0, 3.0])
        assert abs(slope) < 0.001
