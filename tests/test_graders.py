"""Tests for task graders — determinism, range, and scoring logic."""
from __future__ import annotations

import random

import pytest

from wellness_env import WellnessEnv, Action
from wellness_env.models import ExerciseType, NutritionType, SleepDuration
from wellness_env.graders import (
    grade_multi_outcome,
    grade_resistant_adaptation,
    grade_single_goal,
)


# ============================================================================
# Helpers
# ============================================================================

SEED = 42

GOOD_ACTION = Action(
    sleep=SleepDuration.OPTIMAL_LOW,
    exercise=ExerciseType.MODERATE_CARDIO,
    nutrition=NutritionType.BALANCED,
)

BAD_ACTION = Action(
    sleep=SleepDuration.VERY_SHORT,
    exercise=ExerciseType.NONE,
    nutrition=NutritionType.PROCESSED,
)


def _run_episode(env: WellnessEnv, task_name: str, action: Action):
    """Run a full episode with a fixed action. Returns history."""
    obs = env.reset(task_name)
    for _ in range(obs.total_days):
        env.step(action)
    return env._history


# ============================================================================
# Empty history
# ============================================================================

class TestEmptyHistory:
    def test_single_goal_empty(self):
        assert grade_single_goal([]) == 0.0

    def test_multi_outcome_empty(self):
        assert grade_multi_outcome([]) == 0.0

    def test_resistant_adaptation_empty(self):
        assert grade_resistant_adaptation([]) == 0.0


# ============================================================================
# Score range [0.0, 1.0]
# ============================================================================

class TestScoreRange:
    def test_single_goal_range(self):
        env = WellnessEnv(seed=SEED)
        history = _run_episode(env, "single_goal", GOOD_ACTION)
        score = grade_single_goal(history)
        assert 0.0 <= score <= 1.0

    def test_multi_outcome_range(self):
        env = WellnessEnv(seed=SEED)
        history = _run_episode(env, "multi_outcome", GOOD_ACTION)
        score = grade_multi_outcome(history)
        assert 0.0 <= score <= 1.0

    def test_resistant_adaptation_range(self):
        env = WellnessEnv(seed=SEED)
        history = _run_episode(env, "resistant_adaptation", GOOD_ACTION)
        score = grade_resistant_adaptation(history)
        assert 0.0 <= score <= 1.0

    def test_bad_action_range(self):
        """Even with bad actions, scores should stay in [0, 1]."""
        env = WellnessEnv(seed=SEED)
        for task in ["single_goal", "multi_outcome", "resistant_adaptation"]:
            history = _run_episode(env, task, BAD_ACTION)
            score = env.grade()
            assert 0.0 <= score <= 1.0, f"{task} bad action score {score} out of range"


# ============================================================================
# Determinism — same seed → same score
# ============================================================================

class TestDeterminism:
    def test_single_goal_deterministic(self):
        scores = []
        for _ in range(3):
            env = WellnessEnv(seed=SEED)
            history = _run_episode(env, "single_goal", GOOD_ACTION)
            scores.append(grade_single_goal(history))
        assert all(s == scores[0] for s in scores), f"Non-deterministic: {scores}"

    def test_multi_outcome_deterministic(self):
        scores = []
        for _ in range(3):
            env = WellnessEnv(seed=SEED)
            history = _run_episode(env, "multi_outcome", GOOD_ACTION)
            scores.append(grade_multi_outcome(history))
        assert all(s == scores[0] for s in scores), f"Non-deterministic: {scores}"

    def test_resistant_adaptation_deterministic(self):
        scores = []
        for _ in range(3):
            env = WellnessEnv(seed=SEED)
            history = _run_episode(env, "resistant_adaptation", GOOD_ACTION)
            scores.append(grade_resistant_adaptation(history))
        assert all(s == scores[0] for s in scores), f"Non-deterministic: {scores}"


# ============================================================================
# Good actions score higher than bad actions
# ============================================================================

class TestGoodBeatsBad:
    def test_single_goal(self):
        env_good = WellnessEnv(seed=SEED)
        env_bad = WellnessEnv(seed=SEED)
        h_good = _run_episode(env_good, "single_goal", GOOD_ACTION)
        h_bad = _run_episode(env_bad, "single_goal", BAD_ACTION)
        assert grade_single_goal(h_good) > grade_single_goal(h_bad)

    def test_multi_outcome(self):
        env_good = WellnessEnv(seed=SEED)
        env_bad = WellnessEnv(seed=SEED)
        h_good = _run_episode(env_good, "multi_outcome", GOOD_ACTION)
        h_bad = _run_episode(env_bad, "multi_outcome", BAD_ACTION)
        assert grade_multi_outcome(h_good) > grade_multi_outcome(h_bad)

    def test_resistant_adaptation(self):
        env_good = WellnessEnv(seed=SEED)
        env_bad = WellnessEnv(seed=SEED)
        h_good = _run_episode(env_good, "resistant_adaptation", GOOD_ACTION)
        h_bad = _run_episode(env_bad, "resistant_adaptation", BAD_ACTION)
        assert grade_resistant_adaptation(h_good) > grade_resistant_adaptation(h_bad)


# ============================================================================
# Grade via env.grade() matches direct grader call
# ============================================================================

class TestEnvGradeConsistency:
    def test_grade_matches_grader(self):
        for task in ["single_goal", "multi_outcome", "resistant_adaptation"]:
            env = WellnessEnv(seed=SEED)
            _run_episode(env, task, GOOD_ACTION)
            env_score = env.grade()
            # Also call the grader directly
            graders = {
                "single_goal": grade_single_goal,
                "multi_outcome": grade_multi_outcome,
                "resistant_adaptation": grade_resistant_adaptation,
            }
            direct_score = graders[task](env._history)
            assert env_score == direct_score, (
                f"{task}: env.grade()={env_score} != grader={direct_score}"
            )
