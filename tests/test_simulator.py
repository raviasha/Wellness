"""Tests for the physiological simulator (biomarker transition dynamics)."""
from __future__ import annotations

import random
import pytest

from wellness_env.models import (
    Action, Biomarkers, BiomarkerDeltas, ActivityLevel, SleepDuration, SLEEP_HOURS,
    ExerciseType, ExerciseDuration,
)
from wellness_env.personas import PERSONAS
from wellness_env.simulator import apply_deltas, apply_life_event, compute_biomarker_changes

SEED = 42

BASELINE_BIOMARKERS = Biomarkers(
    resting_hr=70, hrv=40, sleep_score=70, stress_avg=50, body_battery=50,
)

GOOD_ACTION = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.MODERATE_ACTIVITY)
BAD_ACTION = Action(sleep=SleepDuration.VERY_SHORT, activity=ActivityLevel.REST_DAY)


class TestComputeBiomarkerChanges:
    def test_returns_biomarker_deltas(self):
        persona = PERSONAS["cardiovascular_fitness"]
        rng = random.Random(SEED)
        deltas = compute_biomarker_changes(GOOD_ACTION, persona.starting_biomarkers, persona, [], rng)
        assert isinstance(deltas, BiomarkerDeltas)

    def test_all_five_fields_present(self):
        persona = PERSONAS["cardiovascular_fitness"]
        rng = random.Random(SEED)
        deltas = compute_biomarker_changes(GOOD_ACTION, persona.starting_biomarkers, persona, [], rng)
        d = deltas.model_dump()
        for field in ["resting_hr", "hrv", "sleep_score", "stress_avg", "body_battery"]:
            assert field in d

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

    def test_different_personas_produce_different_deltas(self):
        # Use an action that includes exercise to activate persona-specific rhr_exercise_benefit
        action_with_exercise = Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            activity=ActivityLevel.VIGOROUS_ACTIVITY,
            exercise_type=ExerciseType.CARDIO,
            exercise_duration=ExerciseDuration.MODERATE,
        )
        rng1 = random.Random(SEED)
        rng2 = random.Random(SEED)
        d1 = compute_biomarker_changes(action_with_exercise, BASELINE_BIOMARKERS, PERSONAS["cardiovascular_fitness"], [], rng1)
        d2 = compute_biomarker_changes(action_with_exercise, BASELINE_BIOMARKERS, PERSONAS["sedentary"], [], rng2)
        assert d1 != d2

    def test_overtraining_effect(self):
        persona = PERSONAS["cardiovascular_fitness"]
        rng = random.Random(99)
        action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.HIGH_INTENSITY)
        history = [{"actual_action": {"activity": "high_intensity", "sleep": "7_to_8h"}} for _ in range(5)]
        deltas = compute_biomarker_changes(action, persona.starting_biomarkers, persona, history, rng)
        assert deltas.hrv < 0


class TestApplyDeltas:
    def test_applies_deltas_correctly(self):
        bio = BASELINE_BIOMARKERS.model_copy()
        deltas = BiomarkerDeltas(resting_hr=-1, hrv=2, sleep_score=1, stress_avg=-3, body_battery=5)
        result = apply_deltas(bio, deltas)
        assert result.resting_hr == 69
        assert result.hrv == 42

    def test_clamps_to_valid_ranges(self):
        bio = Biomarkers(resting_hr=42, hrv=7, sleep_score=2, stress_avg=2, body_battery=2)
        extreme = BiomarkerDeltas(resting_hr=-50, hrv=-50, sleep_score=-50, stress_avg=-50, body_battery=-50)
        result = apply_deltas(bio, extreme)
        assert result.resting_hr >= 40
        assert result.hrv >= 5

    def test_negative_clamp(self):
        bio = Biomarkers(resting_hr=115, hrv=148, sleep_score=99, stress_avg=99, body_battery=99)
        extreme = BiomarkerDeltas(resting_hr=50, hrv=50, sleep_score=50, stress_avg=50, body_battery=50)
        result = apply_deltas(bio, extreme)
        assert result.resting_hr <= 120
        assert result.hrv <= 150


class TestApplyLifeEvent:
    def test_usually_no_change(self):
        rng = random.Random(SEED)
        same = sum(1 for _ in range(100) if apply_life_event(GOOD_ACTION, rng) == GOOD_ACTION)
        assert same > 80

    def test_life_event_changes_action(self):
        rng = random.Random(SEED)
        changed = sum(1 for _ in range(1000) if apply_life_event(GOOD_ACTION, rng) != GOOD_ACTION)
        assert changed > 20


class TestSleepHours:
    def test_all_durations_mapped(self):
        for sd in SleepDuration:
            assert sd in SLEEP_HOURS

    def test_hours_are_ordered(self):
        vals = [SLEEP_HOURS[sd] for sd in SleepDuration]
        assert vals == sorted(vals)
