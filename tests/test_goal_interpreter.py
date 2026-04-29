import pytest
from unittest.mock import patch, MagicMock
from datetime import date, timedelta
from backend.goal_interpreter import (
    get_periodization_phase,
    get_phase_modifiers,
    get_sport_similarity,
    get_sport_compliance,
    interpret_goal,
    GoalProfile
)

def test_get_periodization_phase():
    assert get_periodization_phase(None) == "ongoing"
    assert get_periodization_phase(50) == "base_build"
    assert get_periodization_phase(30) == "specific_build"
    assert get_periodization_phase(10) == "sharpen"
    assert get_periodization_phase(2) == "taper"
    assert get_periodization_phase(-5) == "event_week"

def test_get_phase_modifiers():
    assert "weight_multipliers" in get_phase_modifiers("taper")
    assert get_phase_modifiers("taper")["duration_factor"] == 0.6
    assert get_phase_modifiers("unknown_phase")["duration_factor"] == 1.0

def test_get_sport_similarity():
    # Exact match
    assert get_sport_similarity("running", "Running") == 1.0
    # Matrix match
    assert get_sport_similarity("pickleball", "tennis") == 0.80
    assert get_sport_similarity("tennis", "pickleball") == 0.80
    # Category match
    assert get_sport_similarity("indoor_cycling", "mountain_biking") == 0.50
    # Unknown/different category fallback
    assert get_sport_similarity("yoga", "running") == 0.15 # In matrix
    assert get_sport_similarity("yoga", "weight_training") == 0.20 # Not in matrix, diff cat
    # None/Rest
    assert get_sport_similarity("rest", "running") == 0.0
    assert get_sport_similarity(None, "running") == 0.0

def test_get_sport_compliance():
    # Perfect match and duration
    assert get_sport_compliance("running", "running", 60, 60) == 1.0
    # Match but half duration
    assert get_sport_compliance("running", "running", 30, 60) == 0.5
    # Similar sport, full duration
    assert get_sport_compliance("running", "trail_running", 60, 60) == 0.95
    # Zero target duration
    assert get_sport_compliance("running", "running", 30, 0) == 0.7 # No duration target -> 0.7 fallback
    # No actual activity
    assert get_sport_compliance("running", "rest", 0, 60) == 0.0

@patch("backend.goal_interpreter.OpenAI")
def test_interpret_goal_success(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='''{
            "outcome_weights": {"hrv": 0.5, "resting_hr": 0.5},
            "recommended_sport": "swimming",
            "recommended_duration_minutes": 45,
            "exercise_preferences": {"cardio": 0.8},
            "focus_summary": "Swim fast",
            "supporting_exercises": ["core"]
        }'''))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    target = date.today() + timedelta(days=10)
    profile = interpret_goal("Swim a mile", target_date=target)

    assert isinstance(profile, GoalProfile)
    assert profile.recommended_sport == "swimming"
    assert profile.periodization_phase == "sharpen"
    assert profile.days_to_target == 10
    assert profile.outcome_weights["hrv"] == 0.5

@patch("backend.goal_interpreter.OpenAI")
def test_interpret_goal_fallback(mock_openai):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("LLM Error")
    mock_openai.return_value = mock_client

    profile = interpret_goal("Get fit")
    assert profile.recommended_sport == "general_fitness"
    assert profile.periodization_phase == "ongoing"
    assert profile.days_to_target is None
