"""Additional tests to boost code coverage for simulator, personas, and utilities."""
import random
import pytest
from wellness_env.models import (
    Action, ActivityLevel, SleepDuration, ExerciseType, ExerciseDuration,
    BedtimeWindow, Biomarkers
)
from wellness_env.simulator import (
    compute_biomarker_changes, apply_life_event, _duration_factor, 
    _consecutive_hiit_days, _consecutive_intense_days, _circadian_alignment
)
from wellness_env.personas import PersonaConfig, ResponseModel, apply_compliance, Goal

def test_consecutive_hiit_days():
    history = [
        {"actual_action": {"exercise_type": "hiit"}},
        {"actual_action": {"exercise_type": "hiit"}},
        {"actual_action": {"exercise_type": "cardio"}},
        {"actual_action": {"exercise_type": "hiit"}},
    ]
    assert _consecutive_hiit_days(history) == 1
    assert _consecutive_hiit_days(history[:2]) == 2
    assert _consecutive_hiit_days([]) == 0

def test_consecutive_intense_days():
    history = [
        {"actual_action": {"activity": "high_intensity"}},
        {"actual_action": {"activity": "vigorous_activity"}},
        {"actual_action": {"activity": "moderate_activity"}},
    ]
    assert _consecutive_intense_days(history) == 0 # moderate is not intense
    
    history2 = [
        {"actual_action": {"activity": "high_intensity"}},
        {"actual_action": {"activity": "vigorous_activity"}},
    ]
    assert _consecutive_intense_days(history2) == 2

def test_duration_factor():
    assert _duration_factor(ExerciseDuration.NONE, ExerciseType.CARDIO) == 0.0
    assert _duration_factor(ExerciseDuration.MODERATE, ExerciseType.HIIT) > 1.0
    assert _duration_factor(ExerciseDuration.MODERATE, ExerciseType.FLEXIBILITY) < 2.0
    assert _duration_factor(ExerciseDuration.MODERATE, ExerciseType.CARDIO) > 0.5

def test_circadian_alignment():
    assert _circadian_alignment(BedtimeWindow.OPTIMAL) == 1.0
    assert _circadian_alignment(BedtimeWindow.EXTREMELY_LATE) == 0.1
    assert _circadian_alignment(None) == 0.6

def test_apply_compliance_random_defaults():
    persona = PersonaConfig(
        name="test", compliance_rate=0.0, goal=Goal.STRESS_MANAGEMENT,
        sleep_default=SleepDuration.OPTIMAL_LOW, activity_default=ActivityLevel.REST_DAY,
        starting_biomarkers=Biomarkers(resting_hr=70, hrv=50, sleep_score=70, stress_avg=50, body_battery=50),
        response_model=ResponseModel(),
        random_defaults=True
    )
    rng = random.Random(42)
    # With compliance_rate=0 and random_defaults=True, it should pick random actions
    action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.MODERATE_ACTIVITY)
    for _ in range(10):
        actual, complied = apply_compliance(action, persona, rng)
        assert complied is False

def test_simulator_edge_cases():
    persona = PersonaConfig(
        name="test", compliance_rate=1.0, goal=Goal.STRESS_MANAGEMENT,
        sleep_default=SleepDuration.OPTIMAL_LOW, activity_default=ActivityLevel.REST_DAY,
        starting_biomarkers=Biomarkers(resting_hr=70, hrv=50, sleep_score=70, stress_avg=70, body_battery=50),
        response_model=ResponseModel()
    )
    rng = random.Random(42)
    
    # Test flexibility hrv boost
    action = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.LIGHT_ACTIVITY, 
                    exercise_type=ExerciseType.FLEXIBILITY, exercise_duration=ExerciseDuration.MODERATE)
    deltas = compute_biomarker_changes(action, persona.starting_biomarkers, persona, [], rng)
    assert deltas.hrv > 0
    
    # Test overtraining with HIIT
    action_hiit = Action(sleep=SleepDuration.OPTIMAL_LOW, activity=ActivityLevel.HIGH_INTENSITY,
                         exercise_type=ExerciseType.HIIT, exercise_duration=ExerciseDuration.MODERATE)
    history_hiit = [{"actual_action": {"exercise_type": "hiit", "activity": "high_intensity"}} for _ in range(3)]
    deltas_hiit = compute_biomarker_changes(action_hiit, persona.starting_biomarkers, persona, history_hiit, rng)
    assert deltas_hiit.hrv < 0
    
    # Test long sleep penalty
    action_long = Action(sleep=SleepDuration.LONG, activity=ActivityLevel.REST_DAY)
    deltas_long = compute_biomarker_changes(action_long, persona.starting_biomarkers, persona, [], rng)
    assert deltas_long.sleep_score < 1.0 # should be small negative or zero
