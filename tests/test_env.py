"""Tests for the Outcome-Based Wellness Simulator (Garmin-only MVP)."""
from __future__ import annotations

import math
import random

import pytest

from wellness_env import WellnessEnv, Action, Observation
from wellness_env.models import (
    Biomarkers,
    BiomarkerDeltas,
    EnvState,
    ActivityLevel,
    Goal,
    OutcomeTrends,
    RewardBreakdown,
    SleepDuration,
    StepResult,
    SLEEP_HOURS,
)
from wellness_env.payoff import (
    DELTA_SCALES,
    GOAL_WEIGHTS,
    LOWER_IS_BETTER,
    _linear_slope,
    _stddev,
    compute_reward,
)
from wellness_env.personas import PERSONAS, PersonaConfig, ResponseModel, apply_compliance
from wellness_env.simulator import (
    apply_deltas,
    apply_life_event,
    compute_biomarker_changes,
)
from wellness_env.graders import (
    grade_cardiovascular_fitness,
    grade_stress_recovery,
    grade_sedentary_activation,
    grade_sleep_optimization,
)


SEED = 42

GOOD_ACTION = Action(
    sleep=SleepDuration.OPTIMAL_LOW,
    activity=ActivityLevel.MODERATE_ACTIVITY,
)

BAD_ACTION = Action(
    sleep=SleepDuration.VERY_SHORT,
    activity=ActivityLevel.REST_DAY,
)


def _run_episode(env, task_name, action, steps=None):
    obs = env.reset(task_name)
    total_days = obs.total_days
    if steps is not None:
        total_days = min(steps, total_days)
    rewards = []
    observations = [obs]
    for _ in range(total_days):
        result = env.step(action)
        rewards.append(result.reward.total)
        observations.append(result.observation)
    return rewards, observations


class TestModels:
    def test_action_valid(self):
        a = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.LIGHT_ACTIVITY)
        assert a.sleep == SleepDuration.OPTIMAL_LOW

    def test_action_from_string(self):
        a = Action(sleep="7_to_8h", activity="moderate_activity")
        assert a.sleep == SleepDuration.OPTIMAL_LOW
        assert a.activity == ActivityLevel.MODERATE_ACTIVITY

    def test_biomarkers_valid_range(self):
        b = Biomarkers(resting_hr=60, hrv=50, sleep_score=80, stress_avg=40, body_battery=60)
        assert b.resting_hr == 60

    def test_sleep_hours_mapping(self):
        assert SLEEP_HOURS[SleepDuration.OPTIMAL_LOW] == 7.5
        assert len(SLEEP_HOURS) == 5

    def test_goal_enum(self):
        assert Goal.STRESS_MANAGEMENT.value == "stress_management"
        assert len(Goal) == 5

    def test_reward_breakdown_fields(self):
        rb = RewardBreakdown(
            resting_hr_reward=1.0, hrv_reward=2.0, sleep_score_reward=3.0,
            stress_avg_reward=4.0, body_battery_reward=5.0, total=50.0,
        )
        assert rb.total == 50.0
        assert rb.hrv_reward == 2.0


class TestPersonas:
    def test_all_personas_exist(self):
        assert "cardiovascular_fitness" in PERSONAS
        assert "stress_management" in PERSONAS
        assert "sedentary" in PERSONAS

    def test_persona_goals(self):
        assert PERSONAS["cardiovascular_fitness"].goal == Goal.CARDIOVASCULAR_FITNESS
        assert PERSONAS["stress_management"].goal == Goal.STRESS_MANAGEMENT

    def test_compliance_rates_valid(self):
        for name, p in PERSONAS.items():
            assert 0 < p.compliance_rate <= 1, f"{name} compliance invalid"

    def test_starting_biomarkers_valid(self):
        for name, p in PERSONAS.items():
            b = p.starting_biomarkers
            assert 40 <= b.resting_hr <= 120
            assert 5 <= b.hrv <= 150
            assert 0 <= b.sleep_score <= 100

    def test_apply_compliance_deterministic_when_complied(self):
        persona = PersonaConfig(
            name="test", compliance_rate=1.0, goal=Goal.STRESS_MANAGEMENT,
            sleep_default=SleepDuration.VERY_SHORT, activity_default=ActivityLevel.REST_DAY,
            starting_biomarkers=PERSONAS["cardiovascular_fitness"].starting_biomarkers,
            response_model=ResponseModel(),
        )
        rng = random.Random(SEED)
        actual, complied = apply_compliance(GOOD_ACTION, persona, rng)
        assert complied is True
        assert actual == GOOD_ACTION

    def test_apply_compliance_never_when_zero(self):
        persona = PersonaConfig(
            name="test", compliance_rate=0.0, goal=Goal.STRESS_MANAGEMENT,
            sleep_default=SleepDuration.VERY_SHORT, activity_default=ActivityLevel.REST_DAY,
            starting_biomarkers=PERSONAS["cardiovascular_fitness"].starting_biomarkers,
            response_model=ResponseModel(),
        )
        rng = random.Random(SEED)
        actual, complied = apply_compliance(GOOD_ACTION, persona, rng)
        assert complied is False

    def test_response_model_defaults(self):
        rm = ResponseModel()
        assert rm.overtraining_threshold == 3


class TestSimulator:
    def test_compute_returns_deltas(self):
        persona = PERSONAS["cardiovascular_fitness"]
        rng = random.Random(SEED)
        deltas = compute_biomarker_changes(GOOD_ACTION, persona.starting_biomarkers, persona, [], rng)
        assert isinstance(deltas, BiomarkerDeltas)

    def test_good_sleep_improves_hrv(self):
        persona = PERSONAS["stress_management"]
        rng = random.Random(SEED)
        action = Action(sleep=SleepDuration.OPTIMAL_HIGH, activity=ActivityLevel.LIGHT_ACTIVITY)
        bio = persona.starting_biomarkers.model_copy()
        hrv_changes = []
        for _ in range(20):
            deltas = compute_biomarker_changes(action, bio, persona, [], rng)
            hrv_changes.append(deltas.hrv)
            bio = apply_deltas(bio, deltas)
        assert sum(hrv_changes) / len(hrv_changes) > 0

    def test_overtraining_hurts_hrv(self):
        persona = PERSONAS["cardiovascular_fitness"]
        rng = random.Random(99)
        action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.HIGH_INTENSITY)
        history = [{"actual_action": {"activity": "high_intensity", "sleep": "7_to_8h"}} for _ in range(5)]
        deltas = compute_biomarker_changes(action, persona.starting_biomarkers, persona, history, rng)
        assert deltas.hrv < 0

    def test_apply_deltas_clamps(self):
        bio = Biomarkers(resting_hr=45, hrv=10, sleep_score=5, stress_avg=5, body_battery=5)
        extreme_deltas = BiomarkerDeltas(resting_hr=-50, hrv=-100, sleep_score=-50, stress_avg=-50, body_battery=-50)
        result = apply_deltas(bio, extreme_deltas)
        assert result.resting_hr >= 40
        assert result.hrv >= 5
        assert result.sleep_score >= 0
        assert result.body_battery >= 0

    def test_apply_deltas_clamps_upper(self):
        bio = Biomarkers(resting_hr=115, hrv=145, sleep_score=98, stress_avg=98, body_battery=98)
        extreme_deltas = BiomarkerDeltas(resting_hr=50, hrv=50, sleep_score=50, stress_avg=50, body_battery=50)
        result = apply_deltas(bio, extreme_deltas)
        assert result.resting_hr <= 120
        assert result.hrv <= 150
        assert result.sleep_score <= 100
        assert result.body_battery <= 100

    def test_life_event_rare(self):
        rng = random.Random(SEED)
        changed = 0
        trials = 1000
        for _ in range(trials):
            result = apply_life_event(GOOD_ACTION, rng)
            if result != GOOD_ACTION:
                changed += 1
        assert 20 < changed < 100, f"Expected ~50 life events, got {changed}"

    def test_sleep_debt_reduces_exercise_gains(self):
        persona = PERSONAS["cardiovascular_fitness"]
        rng1 = random.Random(SEED)
        rng2 = random.Random(SEED)
        action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.HIGH_INTENSITY)
        deltas_good = compute_biomarker_changes(action, persona.starting_biomarkers, persona, [], rng1)
        history_debt = [{"actual_action": {"sleep": "less_than_6h", "activity": "rest_day"}} for _ in range(7)]
        deltas_bad = compute_biomarker_changes(action, persona.starting_biomarkers, persona, history_debt, rng2)
        assert isinstance(deltas_bad, BiomarkerDeltas)


class TestPayoff:
    def test_all_goals_have_weights(self):
        for goal in Goal:
            assert goal in GOAL_WEIGHTS
            weights = GOAL_WEIGHTS[goal]
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{goal}: weights sum to {total}"

    def test_all_markers_have_scales(self):
        for marker in DELTA_SCALES:
            assert DELTA_SCALES[marker] > 0

    def test_no_change_gives_baseline(self):
        zero_deltas = BiomarkerDeltas(resting_hr=0, hrv=0, sleep_score=0, stress_avg=0, body_battery=0)
        reward = compute_reward(zero_deltas, Goal.STRESS_MANAGEMENT)
        assert reward.total == 50.0

    def test_reward_range(self):
        rng = random.Random(SEED)
        for _ in range(100):
            deltas = BiomarkerDeltas(
                resting_hr=rng.gauss(0, 2), hrv=rng.gauss(0, 5),
                sleep_score=rng.gauss(0, 3), stress_avg=rng.gauss(0, 5),
                body_battery=rng.gauss(0, 10),
            )
            for goal in Goal:
                reward = compute_reward(deltas, goal)
                assert 0.0 <= reward.total <= 100.0

    def test_positive_deltas_above_baseline(self):
        deltas = BiomarkerDeltas(resting_hr=-1.0, hrv=5.0, sleep_score=2.0, stress_avg=-5.0, body_battery=10.0)
        for goal in Goal:
            reward = compute_reward(deltas, goal)
            assert reward.total > 50.0, f"Goal {goal}: got {reward.total}"

    def test_negative_deltas_below_baseline(self):
        deltas = BiomarkerDeltas(resting_hr=1.0, hrv=-5.0, sleep_score=-2.0, stress_avg=5.0, body_battery=-10.0)
        for goal in Goal:
            reward = compute_reward(deltas, goal)
            assert reward.total < 50.0, f"Goal {goal}: got {reward.total}"

    def test_linear_slope(self):
        assert _linear_slope([1, 2, 3, 4, 5]) == pytest.approx(1.0, abs=0.01)
        assert _linear_slope([5, 5, 5]) == pytest.approx(0.0, abs=0.01)
        assert _linear_slope([5, 3, 1]) == pytest.approx(-2.0, abs=0.01)

    def test_stddev(self):
        assert _stddev([5, 5, 5]) == pytest.approx(0.0, abs=0.01)
        assert _stddev([0, 10]) == pytest.approx(5.0, abs=0.01)

    def test_lower_is_better_set(self):
        assert "resting_hr" in LOWER_IS_BETTER
        assert "stress_avg" in LOWER_IS_BETTER
        assert "hrv" not in LOWER_IS_BETTER


class TestEnv:
    def test_reset_returns_observation(self):
        env = WellnessEnv(seed=SEED)
        obs = env.reset("cardiovascular_fitness")
        assert isinstance(obs, Observation)
        assert obs.day == 0
        assert obs.total_days == 14
        assert obs.goal == Goal.CARDIOVASCULAR_FITNESS

    def test_reset_all_tasks(self):
        env = WellnessEnv(seed=SEED)
        for task in ["cardiovascular_fitness", "stress_recovery", "sedentary_activation", "sleep_optimization"]:
            obs = env.reset(task)
            assert isinstance(obs, Observation)
            assert obs.total_days > 0

    def test_reset_invalid_task(self):
        env = WellnessEnv(seed=SEED)
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset("nonexistent_task")

    def test_step_returns_step_result(self):
        env = WellnessEnv(seed=SEED)
        env.reset("cardiovascular_fitness")
        result = env.step(GOOD_ACTION)
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, Observation)
        assert isinstance(result.reward, RewardBreakdown)
        assert result.done is False

    def test_step_advances_day(self):
        env = WellnessEnv(seed=SEED)
        env.reset("cardiovascular_fitness")
        r1 = env.step(GOOD_ACTION)
        assert r1.observation.day == 1
        r2 = env.step(GOOD_ACTION)
        assert r2.observation.day == 2

    def test_episode_completes(self):
        env = WellnessEnv(seed=SEED)
        obs = env.reset("cardiovascular_fitness")
        for i in range(obs.total_days):
            result = env.step(GOOD_ACTION)
        assert result.done is True

    def test_step_after_done_raises(self):
        env = WellnessEnv(seed=SEED)
        env.reset("cardiovascular_fitness")
        for _ in range(14):
            env.step(GOOD_ACTION)
        with pytest.raises(RuntimeError, match="done"):
            env.step(GOOD_ACTION)

    def test_state_returns_env_state(self):
        env = WellnessEnv(seed=SEED)
        env.reset("cardiovascular_fitness")
        env.step(GOOD_ACTION)
        st = env.state()
        assert isinstance(st, EnvState)
        assert st.day == 1
        assert st.total_days == 14
        assert st.goal == Goal.CARDIOVASCULAR_FITNESS
        assert len(st.history) == 1

    def test_grade_returns_valid_score(self):
        env = WellnessEnv(seed=SEED)
        _run_episode(env, "cardiovascular_fitness", GOOD_ACTION)
        score = env.grade()
        assert 0.0 <= score <= 1.0

    def test_grade_before_reset_raises(self):
        env = WellnessEnv(seed=SEED)
        with pytest.raises(RuntimeError, match="No task"):
            env.grade()

    def test_reward_total_in_range(self):
        env = WellnessEnv(seed=SEED)
        env.reset("cardiovascular_fitness")
        for _ in range(14):
            result = env.step(GOOD_ACTION)
            assert 0.0 <= result.reward.total <= 100.0

    def test_trends_appear_after_day_7(self):
        env = WellnessEnv(seed=SEED)
        env.reset("cardiovascular_fitness")
        for i in range(7):
            result = env.step(GOOD_ACTION)
            if i < 6:
                assert result.observation.trends is None, f"Trends present on day {i+1}"
            else:
                assert result.observation.trends is not None, "Trends missing after day 7"

    def test_deterministic_with_seed(self):
        env1 = WellnessEnv(seed=SEED)
        env2 = WellnessEnv(seed=SEED)
        rewards1, _ = _run_episode(env1, "cardiovascular_fitness", GOOD_ACTION)
        rewards2, _ = _run_episode(env2, "cardiovascular_fitness", GOOD_ACTION)
        assert rewards1 == rewards2
