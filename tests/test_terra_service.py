import pytest
import os
import hmac
import hashlib
from backend.terra_service import (
    _safe_get,
    _extract_from_daily,
    _extract_from_body,
    _extract_from_sleep,
    normalize_terra_payload,
    normalize_terra_webhook,
    verify_webhook_signature
)

pytestmark = pytest.mark.skip(reason="Terra service is for future use")

def test_safe_get():
    obj = {"a": {"b": {"c": 1}}}
    assert _safe_get(obj, "a", "b", "c") == 1
    assert _safe_get(obj, "a", "b", "d") is None
    assert _safe_get(obj, "a", "b", "d", default=0) == 0

def test_extract_from_daily():
    payload = {
        "distance_data": {"steps": 5000, "distance_metres": 4000, "floors_climbed": 10},
        "calories_data": {"net_activity_calories": 500, "total_calories_expenditure": 2000},
        "active_durations_data": {"activity_seconds": 3600},
        "stress_data": {"avg_stress_level": 40},
        "heart_rate_data": {"summary": {"avg_hr_bpm": 65, "max_hr_bpm": 150, "resting_hr_bpm": 55}, "hrv": {"avg_hrv_rmssd": 60}},
        "oxygen_saturation_data": {"avg_saturation_percentage": 98},
        "breathing_data": {"avg_breaths_per_min": 15},
        "readiness_data": {"readiness": 80},
        "temperature_data": {"delta": 0.5}
    }
    extracted = _extract_from_daily(payload)
    assert extracted["steps"] == 5000
    assert extracted["active_calories"] == 500
    assert extracted["calories_total"] == 2000
    assert extracted["active_minutes"] == 60
    assert extracted["stress_avg"] == 40
    assert extracted["resting_hr"] == 55
    assert extracted["hrv_rmssd"] == 60
    assert extracted["recovery_score"] == 80

def test_extract_from_body():
    payload = {
        "heart_rate_data": {"summary": {"resting_hr_bpm": 50}, "hrv": {"avg_hrv_rmssd": 65}},
        "oxygen_data": {"avg_saturation_percentage": 99},
        "vo2max_ml_per_min_per_kg": 45,
        "stress_data": {"avg_stress_level": 35},
        "readiness_data": {"readiness": 85}
    }
    extracted = _extract_from_body(payload)
    assert extracted["resting_hr"] == 50
    assert extracted["vo2_max"] == 45

def test_extract_from_sleep():
    payload = {
        "sleep_durations_data": {"total_duration_seconds": 28800, "asleep": {"duration_deep_sleep_state_seconds": 7200, "duration_REM_sleep_state_seconds": 7200, "duration_light_sleep_state_seconds": 14400}},
        "sleep_efficiency": 90,
        "heart_rate_data": {"hrv": {"avg_hrv_rmssd": 55}}
    }
    extracted = _extract_from_sleep(payload)
    assert extracted["sleep_duration_hours"] == 8.0
    assert extracted["sleep_score"] == 90
    assert extracted["sleep_deep_pct"] == 25.0
    assert extracted["sleep_rem_pct"] == 25.0
    assert extracted["sleep_light_pct"] == 50.0

def test_normalize_terra_payload():
    assert normalize_terra_payload({"distance_data": {"steps": 1000}}, "daily")["steps"] == 1000
    assert normalize_terra_payload({"vo2max_ml_per_min_per_kg": 40}, "body")["vo2_max"] == 40
    assert normalize_terra_payload({"sleep_efficiency": 85}, "sleep")["sleep_score"] == 85
    assert normalize_terra_payload({"calories_data": {"net_activity_calories": 300}}, "activity")["active_calories"] == 300
    assert normalize_terra_payload({}, "unknown") == {}

def test_normalize_terra_webhook():
    payload = {
        "type": "daily",
        "user": {"user_id": "terra-123", "provider": "OURA"},
        "data": [
            {
                "metadata": {"start_time": "2023-10-01T00:00:00Z"},
                "distance_data": {"steps": 5000}
            }
        ]
    }
    result = normalize_terra_webhook(payload)
    assert len(result) == 1
    uid, sync_date, flat = result[0]
    assert uid == "terra-123"
    assert sync_date == "2023-10-01"
    assert flat["source"] == "oura"
    assert flat["steps"] == 5000

def test_verify_webhook_signature():
    secret = "my_super_secret"
    payload = b"test payload"
    signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(payload, signature, secret=secret) is True
    assert verify_webhook_signature(payload, "invalid", secret=secret) is False
