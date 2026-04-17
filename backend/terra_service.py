import os
import hmac
import hashlib
import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Terra client is initialized lazily to avoid import errors when env vars aren't set
_terra_client = None

def _get_terra_client():
    global _terra_client
    if _terra_client is None:
        from terra import Terra
        dev_id = os.environ.get("TERRA_DEV_ID")
        api_key = os.environ.get("TERRA_API_KEY")
        if not dev_id or not api_key:
            raise RuntimeError("TERRA_DEV_ID and TERRA_API_KEY must be set in environment")
        _terra_client = Terra(dev_id=dev_id, api_key=api_key)
    return _terra_client


def _safe_get(obj, *keys, default=None):
    """Safely traverse nested dicts/objects without raising."""
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            cur = getattr(cur, k, None)
    return cur if cur is not None else default


def _extract_from_daily(daily_data: dict) -> dict:
    """Extract fields from Terra's daily summary payload."""
    out = {}

    # Steps
    out["steps"] = _safe_get(daily_data, "distance_data", "steps")

    # Active calories
    out["active_calories"] = _safe_get(daily_data, "calories_data", "net_activity_calories")
    if not out["active_calories"]:
        out["active_calories"] = _safe_get(daily_data, "calories_data", "active_calories")

    # Total calories
    out["calories_total"] = _safe_get(daily_data, "calories_data", "total_calories_expenditure")

    # Active minutes
    active_secs = _safe_get(daily_data, "active_durations_data", "activity_seconds")
    if active_secs is not None:
        out["active_minutes"] = int(active_secs / 60)
    else:
        out["active_minutes"] = None

    # Distance
    out["distance_meters"] = _safe_get(daily_data, "distance_data", "distance_metres")

    # Floors
    out["floors_climbed"] = _safe_get(daily_data, "distance_data", "floors_climbed")

    # Stress
    out["stress_avg"] = _safe_get(daily_data, "stress_data", "avg_stress_level")

    # Average and max HR from daily
    out["avg_hr"] = _safe_get(daily_data, "heart_rate_data", "summary", "avg_hr_bpm")
    out["hr_max"] = _safe_get(daily_data, "heart_rate_data", "summary", "max_hr_bpm")

    # Resting HR from daily (primary source)
    out["resting_hr"] = _safe_get(daily_data, "heart_rate_data", "summary", "resting_hr_bpm")

    # HRV from daily
    out["hrv_rmssd"] = _safe_get(daily_data, "heart_rate_data", "hrv", "avg_hrv_rmssd")

    # SpO2
    out["spo2"] = _safe_get(daily_data, "oxygen_saturation_data", "avg_saturation_percentage")

    # Respiration rate
    out["respiration_rate"] = _safe_get(daily_data, "breathing_data", "avg_breaths_per_min")

    # Recovery / readiness score (device-agnostic, some provide it as daily readiness)
    out["recovery_score"] = _safe_get(daily_data, "readiness_data", "readiness")
    if not out["recovery_score"]:
        out["recovery_score"] = _safe_get(daily_data, "scores", "recovery")

    # Skin temperature deviation
    out["skin_temp_delta"] = _safe_get(daily_data, "temperature_data", "delta")

    return out


def _extract_from_body(body_data: dict) -> dict:
    """Extract fields from Terra's body measurement payload."""
    out = {}

    out["resting_hr"] = _safe_get(body_data, "heart_rate_data", "summary", "resting_hr_bpm")
    out["hrv_rmssd"] = _safe_get(body_data, "heart_rate_data", "hrv", "avg_hrv_rmssd")
    out["spo2"] = _safe_get(body_data, "oxygen_data", "avg_saturation_percentage")
    out["respiration_rate"] = _safe_get(body_data, "respiration_data", "avg_breaths_per_min")
    out["vo2_max"] = _safe_get(body_data, "vo2max_ml_per_min_per_kg")
    out["skin_temp_delta"] = _safe_get(body_data, "temperature_data", "delta")
    out["stress_avg"] = _safe_get(body_data, "stress_data", "avg_stress_level")
    out["recovery_score"] = _safe_get(body_data, "readiness_data", "readiness")

    return out


def _extract_from_sleep(sleep_data: dict) -> dict:
    """Extract fields from Terra's sleep payload."""
    out = {}

    durations = sleep_data.get("sleep_durations_data") or {}

    # Total sleep duration
    total_secs = durations.get("asleep", {}).get("duration_light") if isinstance(durations.get("asleep"), dict) else None
    # Use top-level sleep duration if available
    total_min = durations.get("awake", {})  # reset
    total_secs = _safe_get(sleep_data, "sleep_durations_data", "total_duration_seconds")
    if total_secs is not None:
        out["sleep_duration_hours"] = round(total_secs / 3600, 2)
    else:
        out["sleep_duration_hours"] = None

    # Sleep efficiency
    out["sleep_score"] = _safe_get(sleep_data, "sleep_efficiency")
    if out["sleep_score"]:
        # Terra returns 0-100 efficiency
        out["sleep_score"] = int(out["sleep_score"])

    # Sleep stages — compute percentages
    total = total_secs or 1  # avoid division by zero
    deep_secs = _safe_get(sleep_data, "sleep_durations_data", "asleep", "duration_deep_sleep_state_seconds")
    rem_secs = _safe_get(sleep_data, "sleep_durations_data", "asleep", "duration_REM_sleep_state_seconds")
    light_secs = _safe_get(sleep_data, "sleep_durations_data", "asleep", "duration_light_sleep_state_seconds")

    out["sleep_deep_pct"] = round(deep_secs / total * 100, 1) if deep_secs else None
    out["sleep_rem_pct"] = round(rem_secs / total * 100, 1) if rem_secs else None
    out["sleep_light_pct"] = round(light_secs / total * 100, 1) if light_secs else None

    # HRV during sleep
    if not out.get("hrv_rmssd"):
        out["hrv_rmssd"] = _safe_get(sleep_data, "heart_rate_data", "hrv", "avg_hrv_rmssd")

    # Respiration during sleep
    out["respiration_rate"] = _safe_get(sleep_data, "respiration_data", "avg_breaths_per_min")

    return out


def _merge_dicts(*dicts) -> dict:
    """Merge multiple extraction dicts — first non-None value wins per key."""
    result = {}
    for d in dicts:
        for k, v in d.items():
            if result.get(k) is None and v is not None:
                result[k] = v
    return result


def normalize_terra_payload(payload: dict, data_type: str) -> dict:
    """
    Convert a single Terra webhook payload item into a flat dict
    matching WearableSync column names.

    data_type: 'daily', 'body', 'sleep', 'activity'
    """
    if data_type == "daily":
        return _extract_from_daily(payload)
    elif data_type == "body":
        return _extract_from_body(payload)
    elif data_type == "sleep":
        return _extract_from_sleep(payload)
    elif data_type == "activity":
        # Activity is workout-level; extract aggregate metrics
        return {
            "active_calories": _safe_get(payload, "calories_data", "net_activity_calories"),
            "active_minutes": int(_safe_get(payload, "active_durations_data", "activity_seconds", default=0) / 60) or None,
            "distance_meters": _safe_get(payload, "distance_data", "distance_metres"),
            "avg_hr": _safe_get(payload, "heart_rate_data", "summary", "avg_hr_bpm"),
            "hr_max": _safe_get(payload, "heart_rate_data", "summary", "max_hr_bpm"),
            "strain_score": _safe_get(payload, "strain_data", "strain_level"),
        }
    return {}


def normalize_terra_webhook(webhook_body: dict) -> list:
    """
    Parse a full Terra webhook body and return a list of
    (terra_user_id, sync_date, source, flat_data_dict) tuples.

    Terra sends one event type per webhook call, with one or more data items.
    """
    event_type = webhook_body.get("type", "")
    user_obj = webhook_body.get("user", {})
    terra_user_id = user_obj.get("user_id") if isinstance(user_obj, dict) else None
    source = (user_obj.get("provider", "unknown").lower().replace(" ", "_")
               if isinstance(user_obj, dict) else "unknown")

    results = []

    # Map Terra event types to our handler
    type_map = {
        "daily": "daily",
        "body": "body",
        "sleep": "sleep",
        "activity": "activity",
    }

    data_type = None
    for key in type_map:
        if key in event_type.lower():
            data_type = type_map[key]
            break

    if not data_type:
        logger.info(f"[Terra Webhook] Unhandled event type: {event_type}")
        return []

    data_items = webhook_body.get("data", [])
    if not isinstance(data_items, list):
        data_items = [data_items]

    for item in data_items:
        if not isinstance(item, dict):
            continue
        # Extract date from metadata
        meta = item.get("metadata", {})
        start_time = meta.get("start_time") or meta.get("end_time")
        if start_time:
            try:
                sync_date = start_time[:10]  # YYYY-MM-DD
            except Exception:
                sync_date = datetime.date.today().isoformat()
        else:
            sync_date = datetime.date.today().isoformat()

        flat = normalize_terra_payload(item, data_type)
        flat["source"] = source
        results.append((terra_user_id, sync_date, flat))

    return results


def fetch_terra_data(terra_user_id: str, target_date: datetime.date) -> dict:
    """
    Pull a full day's data for a Terra user via the REST API (polling mode).
    Returns a flat dict matching WearableSync column names, or {"error": "..."} on failure.
    """
    try:
        client = _get_terra_client()
        date_str = target_date.isoformat()
        # Terra accepts ISO8601 date strings directly
        start = date_str
        # End date is the next day to capture the full 24h window
        end = (target_date + datetime.timedelta(days=1)).isoformat()

        daily_raw = None
        body_raw = None
        sleep_raw = None

        try:
            resp = client.daily.fetch(user_id=terra_user_id, start_date=start, end_date=end, to_webhook=False)
            data_list = getattr(resp, "data", None) or []
            if data_list:
                item = data_list[0]
                daily_raw = item if isinstance(item, dict) else (item.__dict__ if hasattr(item, "__dict__") else {})
        except Exception as e:
            logger.warning(f"[Terra] daily.fetch failed for {terra_user_id}: {e}")

        try:
            resp = client.body.fetch(user_id=terra_user_id, start_date=start, end_date=end, to_webhook=False)
            data_list = getattr(resp, "data", None) or []
            if data_list:
                item = data_list[0]
                body_raw = item if isinstance(item, dict) else (item.__dict__ if hasattr(item, "__dict__") else {})
        except Exception as e:
            logger.warning(f"[Terra] body.fetch failed for {terra_user_id}: {e}")

        try:
            resp = client.sleep.fetch(user_id=terra_user_id, start_date=start, end_date=end, to_webhook=False)
            data_list = getattr(resp, "data", None) or []
            if data_list:
                item = data_list[0]
                sleep_raw = item if isinstance(item, dict) else (item.__dict__ if hasattr(item, "__dict__") else {})
        except Exception as e:
            logger.warning(f"[Terra] sleep.fetch failed for {terra_user_id}: {e}")

        # Merge all three sources — first non-None wins per key
        merged = _merge_dicts(
            _extract_from_daily(daily_raw or {}),
            _extract_from_body(body_raw or {}),
            _extract_from_sleep(sleep_raw or {}),
        )

        merged["source"] = "terra"
        merged["date"] = date_str
        merged["raw_payload"] = {
            "daily": daily_raw,
            "body": body_raw,
            "sleep": sleep_raw,
        }

        return merged

    except Exception as e:
        logger.error(f"[Terra] fetch_terra_data failed: {e}")
        return {"error": str(e), "source": "terra"}


def generate_widget_session(reference_id: str, redirect_url: str = None) -> str:
    """
    Generate a Terra widget session URL for user OAuth.
    reference_id should be your internal user_id (as string).
    Returns the widget URL the frontend should redirect the user to.
    """
    client = _get_terra_client()
    kwargs = {"reference_id": reference_id}
    if redirect_url:
        kwargs["auth_success_redirect_url"] = redirect_url
        kwargs["auth_failure_redirect_url"] = redirect_url + "?status=failed"

    resp = client.authentication.generatewidgetsession(**kwargs)
    # The SDK returns an object; extract the URL
    if hasattr(resp, "url"):
        return resp.url
    if isinstance(resp, dict):
        return resp.get("url") or resp.get("widget_url", "")
    return str(resp)


def deauthenticate_user(terra_user_id: str) -> bool:
    """Revoke Terra's access for a user and delete their records on Terra's side."""
    try:
        client = _get_terra_client()
        client.authentication.deauthenticateuser(user_id=terra_user_id)
        return True
    except Exception as e:
        logger.error(f"[Terra] deauthenticate failed for {terra_user_id}: {e}")
        return False


def verify_webhook_signature(payload_bytes: bytes, signature: str, secret: str = None) -> bool:
    """
    Verify Terra's HMAC-SHA256 webhook signature.
    Terra sets the header: terra-signature: <hex_digest>
    """
    secret = secret or os.environ.get("TERRA_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("[Terra] TERRA_WEBHOOK_SECRET not set — skipping signature verification")
        return True  # fail open in dev; set secret in prod
    expected = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()  # type: ignore[attr-defined]
    return hmac.compare_digest(expected, signature.lower())
