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
    2. Persisted garmin_tokens.json — token restore, no password login needed
    3. Fresh login — only when tokens are missing or truly expired
       garminconnect 0.3.2 login(tokenstore) handles both cases and auto-saves
       tokens after a fresh login so the next call restores without re-authentication.
    """
    if email in _GARMIN_SESSIONS:
        return _GARMIN_SESSIONS[email]

    tokenstore = _get_tokenstore(email)
    client = Garmin(email=email, password=password, is_cn=False)

    token_file = os.path.join(tokenstore, "garmin_tokens.json")
    token_exists = os.path.exists(token_file)

    try:
        # Pass tokenstore so login() tries token restore first, then falls back
        # to fresh credential login — and auto-saves tokens on fresh login too.
        client.login(tokenstore if token_exists else None)
        _GARMIN_SESSIONS[email] = client
        if token_exists:
            logger.info(f"[Garmin] Restored token session for {email[:6]}…")
        else:
            # Persist new tokens for next call
            try:
                client.client.dump(tokenstore)
                logger.info(f"[Garmin] Persisted new tokens for {email[:6]}…")
            except Exception as dump_err:
                logger.warning(f"[Garmin] Could not persist tokens: {dump_err}")
        return client
    except GarminConnectTooManyRequestsError:
        raise
    except Exception as e:
        if token_exists:
            logger.warning(f"[Garmin] Token restore failed ({e}), clearing and retrying fresh login")
            try:
                import shutil
                shutil.rmtree(tokenstore, ignore_errors=True)
                os.makedirs(tokenstore, exist_ok=True)
            except Exception:
                pass
            # Retry with fresh credentials only
            client2 = Garmin(email=email, password=password, is_cn=False)
            client2.login()
            try:
                client2.client.dump(tokenstore)
                logger.info(f"[Garmin] Persisted new tokens for {email[:6]}…")
            except Exception as dump_err:
                logger.warning(f"[Garmin] Could not persist tokens: {dump_err}")
            _GARMIN_SESSIONS[email] = client2
            return client2
        raise

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

    # -------------------------------------------------------------------------
    # Fetch activities and VO2max (new endpoints)
    # -------------------------------------------------------------------------
    activities_data = _safe_fetch(lambda: client.get_activities_fordate(date_str), "activities")
    max_metrics_data = _safe_fetch(lambda: client.get_max_metrics(date_str), "max_metrics")
    training_status_data = _safe_fetch(lambda: client.get_training_status(date_str), "training_status")

    # -------------------------------------------------------------------------
    # Extract sleep fields
    # -------------------------------------------------------------------------
    sleep_duration_hours = None
    sleep_start_local = None
    sleep_end_local = None
    sleep_start_hour = None
    sleep_deep_pct = None
    sleep_rem_pct = None
    sleep_light_pct = None
    sleep_awake_pct = None
    sleep_stage_quality = None
    sleep_score_val = None

    if sleep_data and isinstance(sleep_data, dict):
        dto = sleep_data.get("dailySleepDTO") or {}

        # Duration
        dur_sec = (
            sleep_data.get("durationInSeconds")
            or dto.get("sleepDurationInSeconds")
            or dto.get("sleepTimeSeconds")
        )
        if dur_sec and isinstance(dur_sec, (int, float)) and dur_sec > 0:
            sleep_duration_hours = round(dur_sec / 3600, 1)

        # Bedtime timestamp
        start_ts = dto.get("sleepStartTimestampLocal")
        end_ts = dto.get("sleepEndTimestampLocal")
        if start_ts:
            try:
                import datetime as _dt
                # sleepStartTimestampLocal = GMT_epoch_ms + IST_offset_ms
                # utcfromtimestamp(Local/1000) numerically equals IST wall-clock time
                dt_start = _dt.datetime.utcfromtimestamp(int(start_ts) / 1000)
                sleep_start_local = dt_start.isoformat()
                sleep_start_hour = dt_start.hour + dt_start.minute / 60.0
            except Exception:
                sleep_start_local = str(start_ts)
        if end_ts:
            try:
                import datetime as _dt
                dt_end = _dt.datetime.utcfromtimestamp(int(end_ts) / 1000)
                sleep_end_local = dt_end.isoformat()
            except Exception:
                sleep_end_local = str(end_ts)

        # Sleep stage seconds
        deep_sec = dto.get("deepSleepSeconds") or 0
        rem_sec = dto.get("remSleepSeconds") or 0
        light_sec = dto.get("lightSleepSeconds") or 0
        awake_sec = dto.get("awakeSleepSeconds") or 0
        total_sec = deep_sec + rem_sec + light_sec + awake_sec
        if total_sec > 0:
            sleep_deep_pct = round(deep_sec / total_sec * 100, 1)
            sleep_rem_pct = round(rem_sec / total_sec * 100, 1)
            sleep_light_pct = round(light_sec / total_sec * 100, 1)
            sleep_awake_pct = round(awake_sec / total_sec * 100, 1)
            sleep_stage_quality = round((deep_sec + rem_sec) / total_sec * 100, 1)

        # Sleep score
        scores = dto.get("sleepScores") or sleep_data.get("sleepScores") or {}
        sleep_score_val = (
            (scores.get("overall") or {}).get("value")
            or scores.get("overall")
            or (scores.get("personal") or {}).get("overallScore")
            or scores.get("overallScore")
        )
        if isinstance(sleep_score_val, dict):
            sleep_score_val = sleep_score_val.get("value")

    # -------------------------------------------------------------------------
    # Extract activity fields
    # -------------------------------------------------------------------------
    primary_exercise_type = "none"
    exercise_duration_minutes = 0
    exercise_calories = 0

    if activities_data and isinstance(activities_data, list) and len(activities_data) > 0:
        # Pick the longest activity as the "primary" one for the day
        primary = max(
            activities_data,
            key=lambda a: a.get("duration", 0) or 0,
            default=None,
        )
        if primary:
            type_key = (primary.get("activityType") or {}).get("typeKey") or "other"
            exercise_duration_minutes = round((primary.get("duration") or 0) / 60)
            exercise_calories = primary.get("calories") or 0
            primary_exercise_type = type_key

    # -------------------------------------------------------------------------
    # Extract VO2max  (max_metrics endpoint + summary/stats/training_status)
    # -------------------------------------------------------------------------
    def _extract_vo2(obj):
        def _is_plausible_vo2(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool) and 20 <= float(v) <= 90

        if obj is None or isinstance(obj, bool):
            return None
        if isinstance(obj, (int, float)):
            return float(obj) if _is_plausible_vo2(obj) else None
        if isinstance(obj, list):
            for item in obj:
                v = _extract_vo2(item)
                if v is not None:
                    return v
            return None
        if not isinstance(obj, dict):
            return None

        candidates = [
            obj.get("vo2Max"),
            obj.get("vo2max"),
            obj.get("vo2MaxValue"),
            obj.get("latestEnhancedAverageVO2Max"),
            obj.get("latestVo2Max"),
            (obj.get("generic") or {}).get("vo2MaxPreciseValue"),
            (obj.get("mostRecentVO2Max") or {}).get("value"),
            (obj.get("mostRecentVO2Max") or {}).get("vo2MaxValue"),
            (obj.get("mostRecentVO2Max") or {}).get("vo2Max"),
            (obj.get("mostRecentVO2MaxRunning") or {}).get("vo2MaxValue"),
            (obj.get("mostRecentVO2MaxCycling") or {}).get("vo2MaxValue"),
        ]
        for c in candidates:
            if _is_plausible_vo2(c):
                return float(c)

        for k, v in obj.items():
            if "vo2" in str(k).lower():
                parsed = _extract_vo2(v)
                if parsed is not None:
                    return parsed

        return None

    vo2_max = None
    if max_metrics_data:
        vo2_max = _extract_vo2(max_metrics_data)

    # Fallback: summary and stats often carry vo2MaxValue / vo2Max
    if vo2_max is None and summary:
        vo2_max = _extract_vo2(summary)
    if vo2_max is None and stats:
        vo2_max = _extract_vo2(stats)

    # Fallback: training_status (what Garmin Connect website shows)
    if vo2_max is None and training_status_data:
        vo2_max = _extract_vo2(training_status_data)

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
        "sleep_start_local": sleep_start_local,
        "sleep_end_local": sleep_end_local,
        "sleep_start_hour": sleep_start_hour,
        "sleep_deep_pct": sleep_deep_pct,
        "sleep_rem_pct": sleep_rem_pct,
        "sleep_light_pct": sleep_light_pct,
        "sleep_awake_pct": sleep_awake_pct,
        "sleep_stage_quality": sleep_stage_quality,
        "sleep_score": sleep_score_val,
        "stress": stress_data,
        "spo2": spo2_data,
        "respiration": resp_data,
        "activities": activities_data,
        "primary_exercise_type": primary_exercise_type,
        "exercise_duration_minutes": exercise_duration_minutes,
        "exercise_calories": exercise_calories,
        "vo2_max": vo2_max,
        "summary": summary,   # stored for backfill / debugging
        "max_metrics": max_metrics_data,
        "training_status": training_status_data,
    }

def get_mock_payload(date_str):
    return {
        "source": "mock_garmin",
        "date": date_str,
        "body_battery": {"charged": 55, "drained": 42, "latestValue": 72},
        "hrv": {"lastNightAvg": 45, "baseline": {"currentValue": 43}},
        "hrv_rmssd": 45,
        "rhr": {"restingHeartRate": 58},
        "resting_hr": 58,
        "intensity_minutes": {"total": 45, "moderate": 20, "vigorous": 25},
        "active_calories": 450,
        "sleep": {
            "durationInSeconds": 27000,
            "dailySleepDTO": {
                "sleepStartTimestampLocal": 1745366400000,  # ~10:00pm mock epoch ms
                "sleepEndTimestampLocal":   1745393400000,  # ~5:30am mock
                "deepSleepSeconds": 5400,
                "remSleepSeconds": 5400,
                "lightSleepSeconds": 12600,
                "awakeSleepSeconds": 3600,
                "sleepScores": {"overall": {"value": 82}},
            },
        },
        "sleep_duration_hours": 7.5,
        "sleep_start_local": "2026-04-20T22:00:00",
        "sleep_end_local": "2026-04-21T05:30:00",
        "sleep_start_hour": 22.0,
        "sleep_deep_pct": 20.0,
        "sleep_rem_pct": 20.0,
        "sleep_light_pct": 46.7,
        "sleep_awake_pct": 13.3,
        "sleep_stage_quality": 40.0,
        "sleep_score": 82,
        "stress": {"averageStressLevel": 32},
        "steps": {"totalSteps": 12500},
        "spo2": {"latestSpO2": 98},
        "respiration": {"latestRespiration": 14},
        "activities": [
            {
                "activityType": {"typeKey": "running"},
                "duration": 2400.0,
                "calories": 320,
                "startTimeLocal": f"{date_str}T07:00:00",
            }
        ],
        "primary_exercise_type": "running",
        "exercise_duration_minutes": 40,
        "exercise_calories": 320,
        "vo2_max": 48.2,
    }
