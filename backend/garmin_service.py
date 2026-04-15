import os
import datetime
import logging
import json
from garminconnect import Garmin, GarminConnectAuthenticationError

logger = logging.getLogger(__name__)

def fetch_garmin_data(email=None, password=None, target_date: datetime.date = None, simulate=False):
    """
    Fetches HRV, Resting HR, and Body Battery data from Garmin Connect.
    No automatic mocks; strictly real data unless simulate=True.
    """
    if target_date is None:
        target_date = datetime.date.today()
        
    date_str = target_date.isoformat()
    
    # Use passed credentials or fallback to ENV for testing
    email = email or os.environ.get("GARMIN_EMAIL")
    password = password or os.environ.get("GARMIN_PASSWORD")
    
    if not email or not password:
        if simulate: return get_mock_payload(date_str)
        logger.error("Missing Garmin credentials.")
        return {"error": "Missing credentials. Please check Settings.", "source": "garmin"}
        
    try:
        client = Garmin(email, password)
        client.login()
    except GarminConnectAuthenticationError as e:
        if "429" in str(e):
            return {"error": "IP rate limited by Garmin (429)", "source": "garmin"}
        if simulate: return get_mock_payload(date_str)
        logger.error("Garmin authentication failed.")
        return {"error": "Authentication failed", "source": "garmin"}
    except Exception as e:
        if "429" in str(e):
            return {"error": "IP rate limited by Garmin (429)", "source": "garmin"}
        if simulate: return get_mock_payload(date_str)
        logger.error(f"Failed to connect to Garmin: {e}")
        return {"error": str(e), "source": "garmin"}
        
    # Fetch real data (Indented properly inside fetch_garmin_data)
        # Hybrid Sync: Combine multiple Garmin endpoints for maximum fidelity
        try:
            summary = client.get_user_summary(date_str)
            stats = client.get_stats(date_str)
            
            # 1. Active Calories (Robust Fallback)
            # Try activeCalories, activeKilocalories, or calculate difference
            active_cals = summary.get("activeCalories") or stats.get("activeCalories") or summary.get("activeKilocalories")
            if (active_cals is None or active_cals == 0) and summary.get("totalCalories") and summary.get("bmrCalories"):
                 active_cals = max(0, summary["totalCalories"] - summary["bmrCalories"])
            active_cals = active_cals or 0
            
            # 2. Intensity Minutes
            intensity_mins = summary.get("activeSeconds", 0) // 60
            if intensity_mins == 0:
                intensity = client.get_intensity_minutes(date_str)
                intensity_mins = intensity.get("total", 0)
                
            # 3. Steps (Robust Fallback)
            total_steps = stats.get("totalSteps") or summary.get("totalSteps") or stats.get("steps") or 0
            
        except Exception as e:
            logger.warning(f"Partial fetch error for {date_str}: {e}")
            summary = summary if 'summary' in locals() else {}
            stats = stats if 'stats' in locals() else {}
            active_cals = 0
            intensity_mins = 0
            total_steps = 0

        # Fetch other biometrics
        try: body_battery = client.get_body_battery(date_str)
        except Exception: body_battery = None
        
        try: hrv = client.get_hrv_data(date_str)
        except Exception: hrv = None
        
        try: rhr = client.get_rhr_day(date_str)
        except Exception: rhr = None
        
        try: sleep_data = client.get_sleep_data(date_str)
        except Exception: sleep_data = None
            
        try: stress_data = client.get_stress_data(date_str)
        except Exception: stress_data = None
            
        try: spo2_data = client.get_spo2_data(date_str)
        except Exception: spo2_data = None
            
        try: resp_data = client.get_respiration_data(date_str)
        except Exception: resp_data = None
        
        return {
            "source": "garmin",
            "date": date_str,
            "body_battery": body_battery,
            "hrv": hrv,
            "rhr": rhr,
            "intensity_minutes": intensity_mins,
            "active_calories": active_cals,
            "steps": {"totalSteps": total_steps},
            "sleep": sleep_data,
            "stress": stress_data,
            "spo2": spo2_data,
            "respiration": resp_data
        }
    except Exception as e:
        logger.error(f"Error fetching Garmin data: {e}")
        return {"error": str(e), "source": "garmin"}

def get_mock_payload(date_str):
    return {
        "source": "mock",
        "date": date_str,
        "body_battery": {"charged": 55, "drained": 42, "latestValue": 72},
        "hrv": {"lastNightAvg": 45, "baseline": {"currentValue": 43}},
        "rhr": {"restingHeartRate": 58},
        "intensity_minutes": {"total": 45, "moderate": 20, "vigorous": 25},
        "active_calories": 450,
        "sleep": {
            "durationInSeconds": 28800, 
            "dailySleepDTO": {"sleepScores": {"personal": {"overallScore": 85}}}
        },
        "stress": {"averageStressLevel": 32},
        "steps": {"totalSteps": 12500},
        "spo2": {"latestSpO2": 98},
        "respiration": {"latestRespiration": 14}
    }
