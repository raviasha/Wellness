import os
import datetime
import logging
import json
from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectTooManyRequestsError

logger = logging.getLogger(__name__)

# Session cache: avoids re-login (and 429s) by reusing OAuth tokens
_GARMIN_SESSIONS: dict[str, Garmin] = {}
_TOKEN_DIR = os.path.join(os.path.dirname(__file__), "..", ".garmin_tokens")

def _get_tokenstore(email: str) -> str:
    """Per-user token directory for garth session persistence."""
    safe = email.replace("@", "_at_").replace(".", "_")
    path = os.path.join(_TOKEN_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path

def _get_client(email: str, password: str) -> Garmin:
    """Return a cached Garmin client, restoring from persisted tokens when possible.

    Auth priority (least rate-limit risk first):
    1. In-memory session cache — zero API calls
    2. Persisted garth token files — no password login needed
    3. Fresh login — only when tokens are missing or truly expired
       Uses garminconnect 0.3.2 multi-strategy auth (widget+cffi bypasses the
       per-clientId rate-limit bucket entirely).
    """
    if email in _GARMIN_SESSIONS:
        return _GARMIN_SESSIONS[email]

    tokenstore = _get_tokenstore(email)
    client = Garmin(email=email, password=password, is_cn=False)

    # Strategy 1: restore persisted tokens (no login request)
    token_file = os.path.join(tokenstore, "oauth2_token.json")
    if os.path.exists(token_file):
        try:
            client.login(tokenstore)
            _GARMIN_SESSIONS[email] = client
            logger.info(f"[Garmin] Restored token session for {email[:6]}…")
            return client
        except GarminConnectTooManyRequestsError:
            raise  # bubble up — don't attempt fresh login on top of a 429
        except Exception as e:
            logger.warning(f"[Garmin] Token restore failed ({e}), will re-authenticate")
            # Remove stale tokens so we don't keep retrying them
            try:
                import shutil
                shutil.rmtree(tokenstore, ignore_errors=True)
                os.makedirs(tokenstore, exist_ok=True)
            except Exception:
                pass

    # Strategy 2: fresh login via garminconnect 0.3.2 multi-strategy cascade
    # (widget+cffi → portal+cffi → portal+requests → mobile+cffi → mobile+requests)
    try:
        client.login()
    except GarminConnectTooManyRequestsError:
        raise
    except Exception as e:
        raise

    # Persist tokens immediately so next restart skips this login
    try:
        client.garth.dump(tokenstore)
        logger.info(f"[Garmin] Persisted new tokens for {email[:6]}…")
    except Exception as e:
        logger.warning(f"[Garmin] Could not persist tokens: {e}")

    _GARMIN_SESSIONS[email] = client
    return client

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
        client = _get_client(email, password)
    except GarminConnectAuthenticationError as e:
        if "429" in str(e):
            return {"error": "IP rate limited by Garmin (429)", "source": "garmin"}
        if simulate: return get_mock_payload(date_str)
        logger.error("Garmin authentication failed.")
        return {"error": "Authentication failed", "source": "garmin"}
    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            return {"error": "IP rate limited by Garmin (429)", "source": "garmin"}
        # Token might be stale — evict cache and retry once
        if email in _GARMIN_SESSIONS:
            del _GARMIN_SESSIONS[email]
            try:
                client = _get_client(email, password)
            except Exception as e2:
                if simulate: return get_mock_payload(date_str)
                logger.error(f"Garmin retry also failed: {e2}")
                return {"error": str(e2), "source": "garmin"}
        else:
            if simulate: return get_mock_payload(date_str)
            logger.error(f"Failed to connect to Garmin: {e}")
            return {"error": err, "source": "garmin"}

    # Helper: detect 429 in any exception during data fetch
    def _is_rate_limit(exc):
        msg = str(exc)
        return "429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower()

    # Fetch real data after successful login
    try:
        # Hybrid Sync: Combine multiple Garmin endpoints for maximum fidelity
        summary = client.get_user_summary(date_str)
        stats = client.get_stats(date_str)

        # 1. Active Calories (Robust Fallback)
        active_cals = summary.get("activeCalories") or stats.get("activeCalories") or summary.get("activeKilocalories")
        if (active_cals is None or active_cals == 0) and summary.get("totalCalories") and summary.get("bmrCalories"):
            active_cals = max(0, summary["totalCalories"] - summary["bmrCalories"])
        active_cals = active_cals or 0

        # 2. Intensity Minutes (Garmin HR-zone-based, NOT generic activeSeconds)
        moderate = summary.get("moderateIntensityMinutes", 0) or 0
        vigorous = summary.get("vigorousIntensityMinutes", 0) or 0
        intensity_mins = moderate + vigorous
        if intensity_mins == 0:
            try:
                intensity = client.get_intensity_minutes(date_str)
                if isinstance(intensity, dict):
                    intensity_mins = (
                        (intensity.get("moderateIntensityMinutes", 0) or 0) +
                        (intensity.get("vigorousIntensityMinutes", 0) or 0)
                    ) or intensity.get("total", 0) or 0
            except Exception:
                pass
        logger.info(f"[Garmin] {date_str} intensity: moderate={moderate} vigorous={vigorous} total={intensity_mins}")

        # 3. Steps (Robust Fallback)
        total_steps = stats.get("totalSteps") or summary.get("totalSteps") or stats.get("steps") or 0

        logger.info(f"[Garmin] {date_str} summary keys: {sorted(summary.keys()) if summary else 'None'}")
        logger.info(f"[Garmin] {date_str} active_cals={active_cals} steps={total_steps}")

    except Exception as e:
        if _is_rate_limit(e):
            # Evict cached session so next attempt re-authenticates
            if email in _GARMIN_SESSIONS:
                del _GARMIN_SESSIONS[email]
            return {"error": "IP rate limited by Garmin (429)", "source": "garmin"}
        logger.warning(f"Partial fetch error for {date_str}: {e}")
        summary = {}
        stats = {}
        active_cals = 0
        intensity_mins = 0
        total_steps = 0

    # Fetch other biometrics (each isolated so one failure doesn't block the rest)
    rate_limited = False

    def _safe_fetch(fn, label):
        nonlocal rate_limited
        try:
            return fn()
        except Exception as e:
            if _is_rate_limit(e):
                rate_limited = True
                logger.warning(f"[Garmin] 429 rate limit during {label} fetch for {date_str}")
            return None

    body_battery = _safe_fetch(lambda: client.get_body_battery(date_str), "body_battery")
    hrv = _safe_fetch(lambda: client.get_hrv_data(date_str), "hrv")
    rhr = _safe_fetch(lambda: client.get_rhr_day(date_str), "rhr")
    sleep_data = _safe_fetch(lambda: client.get_sleep_data(date_str), "sleep")
    stress_data = _safe_fetch(lambda: client.get_stress_data(date_str), "stress")
    spo2_data = _safe_fetch(lambda: client.get_spo2_data(date_str), "spo2")
    resp_data = _safe_fetch(lambda: client.get_respiration_data(date_str), "respiration")

    # If we got rate-limited on any biometric AND have no useful data, report as error
    if rate_limited and all(v is None for v in [body_battery, hrv, rhr, sleep_data, stress_data]):
        if email in _GARMIN_SESSIONS:
            del _GARMIN_SESSIONS[email]
        return {"error": "IP rate limited by Garmin (429)", "source": "garmin"}

    # Extract sleep duration in hours directly so we don't rely on frontend parsing
    sleep_duration_hours = None
    if sleep_data and isinstance(sleep_data, dict):
        dur_sec = None
        dto = sleep_data.get("dailySleepDTO") or {}
        dur_sec = sleep_data.get("durationInSeconds") or dto.get("sleepDurationInSeconds") or dto.get("sleepTimeSeconds")
        if dur_sec and isinstance(dur_sec, (int, float)) and dur_sec > 0:
            sleep_duration_hours = round(dur_sec / 3600, 1)

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
        "sleep_duration_hours": sleep_duration_hours,
        "stress": stress_data,
        "spo2": spo2_data,
        "respiration": resp_data
    }

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
