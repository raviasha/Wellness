"""Tests for task graders — determinism, range, and scoring logic."""
from __future__ import annotations

import random
import pytest

from wellness_env import WellnessEnv, Action
from wellness_env.models import ActivityLevel, SleepDuration
from wellness_env.graders import (
    grade_cardiovascular_fitness,
    grade_stress_recovery,
    grade_sedentary_activation,
    grade_sleep_optimization,
)

SEED = 42

GOOD_ACTION = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.MODERATE_ACTIVITY)
BAD_ACTION = Action(sleep=SleepDuration.VERY_SHORT, activity=ActivityLevel.REST_DAY)

def _run_episode(env, task_name, action):
    obs = env.reset(task_name)
    for _ in range(obs.total_days):
        env.step(action)
    return env._history


class TestEmptyHistory:
    def test_cardiovascular_empty(self): assert grade_cardiovascular_fitness([]) == 0.0
    def test_stress_recovery_empty(self): assert grade_stress_recovery([]) == 0.0
    def test_sedentary_activation_empty(self): assert grade_sedentary_activation([]) == 0.0
    def test_sleep_optimization_empty(self): assert grade_sleep_optimization([]) == 0.0


class TestScoreRange:
    @pytest.mark.parametrize("task,grader", [
        ("cardiovascular_fitness", grade_cardiovascular_fitness),
        ("stress_recovery", grade_stress_recovery),
        ("sedentary_activation", grade_sedentary_activation),
        ("sleep_optimization", grade_sleep_optimization),
    ])
    def test_good_action_range(self, task, grader):
        env = WellnessEnv(seed=SEED)
        history = _run_episode(env, task, GOOD_ACTION)
        score = grader(history)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("task,grader", [
        ("cardiovascular_fitness", grade_cardiovascular_fitness),
        ("stress_recovery", grade_stress_recovery),
        ("sedentary_activation", grade_sedentary_activation),
        ("sleep_optimization", grade_sleep_optimization),
    ])
    def test_bad_action_range(self, task, grader):
        env = WellnessEnv(seed=SEED)
        history = _run_episode(env, task, BAD_ACTION)
        score = grader(history)
        assert 0.0 <= score <= 1.0


class TestDeterminism:
    @pytest.mark.parametrize("task,grader", [
        ("cardiovascular_fitness", grade_cardiovascular_fitness),
        ("stress_recovery", grade_stress_recovery),
        ("sedentary_activation", grade_sedentary_activation),
        ("sleep_optimization", grade_sleep_optimization),
    ])
    def test_deterministic(self, task, grader):
        env1 = WellnessEnv(seed=SEED)
        env2 = WellnessEnv(seed=SEED)
        h1 = _run_episode(env1, task, GOOD_ACTION)
        h2 = _run_episode(env2, task, GOOD_ACTION)
        assert grader(h1) == grader(h2)


class TestGoodBeatsBad:
    @pytest.mark.parametrize("task,grader", [
        ("cardiovascular_fitness", grade_cardiovascular_fitness),
        ("stress_recovery", grade_stress_recovery),
        ("sleep_optimization", grade_sleep_optimization),
    ])
    def test_good_beats_bad(self, task, grader):
        env_good = WellnessEnv(seed=SEED)
        env_bad = WellnessEnv(seed=SEED)
        h_good = _run_episode(env_good, task, GOOD_ACTION)
        h_bad = _run_episode(env_bad, task, BAD_ACTION)
        assert grader(h_good) >= grader(h_bad)


class TestEnvGradeConsistency:
    def test_grade_matches_grader(self):
        env = WellnessEnv(seed=SEED)
        _run_episode(env, "cardiovascular_fitness", GOOD_ACTION)
        env_score = env.grade()
        grader_score = grade_cardiovascular_fitness(env._history)
        assert env_score == pytest.approx(grader_score, abs=1e-6)
