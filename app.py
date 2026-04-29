"""FastAPI server exposing WellnessEnv over HTTP for HF Space deployment."""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load variables from .env right at the start (override=True so .env wins over stale shell vars)
load_dotenv(override=True)

import threading
import time
from typing import Any
from datetime import date, datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from wellness_env import WellnessEnv
from wellness_env.models import Action
from backend.garmin_service import fetch_garmin_data
from backend.terra_service import (
    fetch_terra_data, normalize_terra_webhook,
    generate_widget_session, deauthenticate_user, verify_webhook_signature
)
from backend.database import (
    init_db, save_garmin_sync, save_wearable_sync, add_manual_log,
    get_user_profile, update_user_profile, get_recent_history,
    get_users, update_garmin_creds, get_garmin_creds,
    update_terra_creds, get_wearable_creds,
    approve_simulator, save_recommendation, create_user,
    set_user_device, set_custom_goal, get_custom_goal, clear_custom_goal
)
from wellness_env.payoff import GOAL_WEIGHTS, DELTA_SCALES, _DELTA_WEIGHT, _STATE_WEIGHT
from backend.llm_nutrition import parse_nutrition_text
from backend.distribution_calibration import calibrate_user_distribution as calibrate_user_persona
from backend.inference_service import get_coaching_recommendation
from backend.eval_service import evaluate_past_recommendations, evaluate_user_performance
from backend.upload_service import parse_apple_health_xml, parse_csv_upload, parse_json_upload

def _safe_extract(data: dict, key: str, subkey: str) -> Any:
    """Helper to defensively extract values from Garmin's changing API schemas."""
    val = data.get(key)
    if not val: return None
    
    if key == "hrv" and isinstance(val, dict):
        return val.get("hrvSummary", {}).get(subkey, val.get(subkey))
        
    if key == "rhr" and isinstance(val, dict):
        try:
            return val["allMetrics"]["metricsMap"]["WELLNESS_RESTING_HEART_RATE"][0]["value"]
        except Exception:
            return val.get("restingHeartRate", val.get(subkey))

    if key == "sleep" and isinstance(val, dict):
        try:
            return val.get("dailySleepDTO", {}).get("sleepScores", {}).get("overall", {}).get("value")
        except Exception:
            return None
            
    if key == "stress" and isinstance(val, dict):
        return val.get("avgStressLevel")

    if key == "spo2" and isinstance(val, dict):
        return val.get("latestSpO2") or val.get("averageSpO2")

    if key == "respiration" and isinstance(val, dict):
        return val.get("latestRespiration") or val.get("avgSleepRespirationValue") or val.get("avgWakingRespirationValue")

    if isinstance(val, list):
        if not val: return None
        item = val[-1]
        if isinstance(item, dict):
            return item.get(subkey, item.get("charged", item.get("restingHeartRate", item.get("lastNightAvg"))))
        return None
        
    if isinstance(val, (int, float)):
        return val
        
    if isinstance(val, dict):
        return val.get(subkey)
    return None


def _extract_vo2_from_obj(obj: Any) -> float | None:
    """Best-effort VO2 extraction across Garmin schema variants."""
    def _is_plausible_vo2(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and 20 <= float(v) <= 90

    if obj is None:
        return None
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return float(obj) if _is_plausible_vo2(obj) else None
    if isinstance(obj, list):
        for item in obj:
            v = _extract_vo2_from_obj(item)
            if v is not None:
                return v
        return None
    if not isinstance(obj, dict):
        return None

    # Priority keys seen in Garmin payloads
    direct_candidates = [
        obj.get("vo2MaxValue"),
        obj.get("vo2Max"),
        obj.get("latestEnhancedAverageVO2Max"),
        obj.get("latestVo2Max"),
        (obj.get("generic") or {}).get("vo2MaxPreciseValue"),
        (obj.get("mostRecentVO2Max") or {}).get("value"),
        (obj.get("mostRecentVO2Max") or {}).get("vo2MaxValue"),
        (obj.get("mostRecentVO2Max") or {}).get("vo2Max"),
        (obj.get("mostRecentVO2MaxRunning") or {}).get("vo2MaxValue"),
        (obj.get("mostRecentVO2MaxCycling") or {}).get("vo2MaxValue"),
    ]
    for c in direct_candidates:
        if _is_plausible_vo2(c):
            return float(c)

    # Generic fallback: recurse through VO2-related keys first
    for k, v in obj.items():
        if "vo2" in str(k).lower():
            parsed = _extract_vo2_from_obj(v)
            if parsed is not None:
                return parsed

    return None

app = FastAPI(title="Wellness-Outcome OpenEnv", version="1.0.0")

# --- Sync Backoff Tracker (per-user exponential backoff) ---
import random as _random

_sync_backoff: dict[int, dict] = {}  # user_id → {failures, next_retry_after}
_BACKOFF_BASE = 300       # 5 minutes base
_BACKOFF_MAX  = 7200      # 2 hours max wait
_BACKOFF_JITTER = 60      # ±60s random jitter


def _record_sync_failure(user_id: int, is_rate_limit: bool = False):
    """Record a sync failure and compute next retry time with exponential backoff + jitter."""
    state = _sync_backoff.get(user_id, {"failures": 0})
    state["failures"] = state.get("failures", 0) + 1
    state["is_rate_limit"] = is_rate_limit
    state["last_failure"] = datetime.utcnow().isoformat()
    # Exponential backoff: base * 2^(failures-1), capped at max
    delay = min(_BACKOFF_BASE * (2 ** (state["failures"] - 1)), _BACKOFF_MAX)
    jitter = _random.uniform(-_BACKOFF_JITTER, _BACKOFF_JITTER)
    state["next_retry_after"] = (datetime.utcnow() + timedelta(seconds=delay + jitter)).isoformat()
    _sync_backoff[user_id] = state
    print(f"[Sync Backoff] User {user_id}: failure #{state['failures']}, next retry after {delay + jitter:.0f}s (rate_limit={is_rate_limit})")


def _record_sync_success(user_id: int):
    """Clear backoff state on success."""
    if user_id in _sync_backoff:
        del _sync_backoff[user_id]


def _should_skip_user(user_id: int) -> bool:
    """Check if user is in backoff period."""
    state = _sync_backoff.get(user_id)
    if not state:
        return False
    retry_after = state.get("next_retry_after")
    if not retry_after:
        return False
    return datetime.utcnow() < datetime.fromisoformat(retry_after)


def get_sync_backoff_status(user_id: int):
    """Get backoff status for a user (exposed via API)."""
    return _sync_backoff.get(user_id)


# --- Auto Garmin Sync Scheduler ---
def _do_wearable_sync(user_id: int, dt: date) -> dict:
    """Sync one day for a user, routing to Terra or legacy Garmin based on their wearable_source."""
    auth_type, cred_a, cred_b = get_wearable_creds(user_id)

    if auth_type == "terra" and cred_a:  # cred_a = terra_user_id, cred_b = wearable_source
        try:
            data = fetch_terra_data(terra_user_id=cred_a, target_date=dt)
        except Exception as e:
            _record_sync_failure(user_id)
            return {"status": "error", "message": f"Terra fetch failed: {e}"}
        if data and "error" not in data:
            raw = data.pop("raw_payload", data)
            save_wearable_sync(
                user_id=user_id,
                sync_date=dt.isoformat(),
                source=cred_b or "terra",
                raw_payload=raw,
                **{k: v for k, v in data.items() if k not in ("date", "source")}
            )
            _record_sync_success(user_id)
            return {"status": "success", "source": cred_b}
        error_msg = (data.get("error") if data else None) or "Terra returned no data"
        _record_sync_failure(user_id)
        return {"status": "error", "message": error_msg}

    elif auth_type == "garmin" and cred_a:  # cred_a = email, cred_b = password
        try:
            data = fetch_garmin_data(cred_a, cred_b, dt)
        except Exception as e:
            err = str(e)
            is_rl = "429" in err or "rate limit" in err.lower()
            _record_sync_failure(user_id, is_rate_limit=is_rl)
            if is_rl:
                return {
                    "status": "rate_limited",
                    "message": "Garmin rate limited — will retry with backoff.",
                }
            return {"status": "error", "message": f"Garmin fetch failed: {err}"}

        if data and "error" not in data:
            save_garmin_sync(  # legacy wrapper → maps to save_wearable_sync internally
                user_id=user_id,
                sync_date=dt.isoformat(),
                hrv_avg=_safe_extract(data, "hrv", "lastNightAvg"),
                resting_hr=_safe_extract(data, "rhr", "restingHeartRate"),
                body_battery=_safe_extract(data, "body_battery", "latestValue"),
                intensity_minutes=data.get("intensity_minutes", 0) or 0,
                active_calories=data.get("active_calories", 0),
                training_load=data.get("training_load", 0.0),
                sleep_score=data.get("sleep_score"),
                sleep_duration_hours=data.get("sleep_duration_hours"),
                stress_avg=_safe_extract(data, "stress", "avg"),
                steps_total=(data.get("steps") or {}).get("totalSteps"),
                spo2_avg=_safe_extract(data, "spo2", "latestSpO2"),
                respiration_avg=_safe_extract(data, "respiration", "latestRespiration"),
                # Circadian + exercise enrichment
                sleep_start_local=data.get("sleep_start_local"),
                sleep_end_local=data.get("sleep_end_local"),
                sleep_start_hour=data.get("sleep_start_hour"),
                sleep_deep_pct=data.get("sleep_deep_pct"),
                sleep_rem_pct=data.get("sleep_rem_pct"),
                sleep_light_pct=data.get("sleep_light_pct"),
                sleep_awake_pct=data.get("sleep_awake_pct"),
                sleep_stage_quality=data.get("sleep_stage_quality"),
                exercise_type=data.get("primary_exercise_type"),
                exercise_duration_minutes=data.get("exercise_duration_minutes"),
                vo2_max=data.get("vo2_max"),
                raw_payload=data
            )
            _record_sync_success(user_id)
            return {"status": "success", "source": "garmin"}

        # Error path — extract the actual message
        error_msg = (data.get("error") if data else None) or "Garmin returned no data"
        is_rate_limit = "429" in error_msg or "rate limit" in error_msg.lower()
        _record_sync_failure(user_id, is_rate_limit=is_rate_limit)
        if is_rate_limit:
            backoff = _sync_backoff.get(user_id, {})
            return {
                "status": "rate_limited",
                "message": f"Garmin rate limited — will retry automatically. (attempt #{backoff.get('failures', 1)})",
                "next_retry": backoff.get("next_retry_after"),
            }
        return {"status": "error", "message": error_msg}

    return {"status": "skipped", "message": "No wearable credentials configured"}


def _run_sync_cycle():
    """Run one sync cycle for all users with exponential backoff awareness."""
    try:
        users = get_users()
        for i, u in enumerate(users):
            uid = u["id"]
            # Skip users in backoff period
            if _should_skip_user(uid):
                state = _sync_backoff.get(uid, {})
                print(f"[Auto-Sync] Skipping user {uid} ({u['username']}) — in backoff until {state.get('next_retry_after', '?')}")
                continue
            try:
                user_rate_limited = False
                for dt in [date.today(), date.today() - timedelta(days=1)]:
                    if user_rate_limited:
                        # Don't keep hitting Garmin if already rate-limited this cycle
                        print(f"[Auto-Sync] Skipping {dt.isoformat()} for user {uid} — rate limited earlier in this cycle")
                        continue
                    result = _do_wearable_sync(uid, dt)
                    if result["status"] == "success":
                        print(f"[Auto-Sync] Synced {dt.isoformat()} for user {uid} ({u['username']}) via {result['source']}")
                    elif result["status"] == "rate_limited":
                        print(f"[Auto-Sync] Rate limited for user {uid} on {dt.isoformat()}: {result.get('message')}")
                        user_rate_limited = True
                    elif result["status"] == "error":
                        print(f"[Auto-Sync] Error for user {uid} on {dt.isoformat()}: {result.get('message')}")
                    time.sleep(3)  # 3s between date fetches to reduce rate-limit risk
                if not user_rate_limited:
                    evaluate_past_recommendations(uid)
                    evaluate_user_performance(uid)
                    # Auto-trigger ML model training when user crosses the min_days gate
                    try:
                        from backend.maturity_config import count_paired_days, get_user_thresholds
                        from backend.outcome_models import load_outcome_models, train_outcome_models
                        paired = count_paired_days(uid)
                        thr = get_user_thresholds(uid)
                        if paired >= thr["ml_model_min_days"]:
                            existing = load_outcome_models(uid)
                            # Retrain once a week (when data_days has grown by 7+)
                            if existing is None or (paired - existing.data_days) >= 7:
                                print(f"[Auto-Sync] Training ML models for user {uid} ({paired} paired days)")
                                train_outcome_models(uid)
                    except Exception as train_e:
                        print(f"[Auto-Sync] ML training error for user {uid}: {train_e}")
            except Exception as e:
                print(f"[Auto-Sync] Error for user {uid}: {e}")
            if i < len(users) - 1:
                # Jittered delay between users (5-15s)
                delay = 5 + _random.uniform(0, 10)
                time.sleep(delay)
    except Exception as e:
        print(f"[Auto-Sync] Scheduler error: {e}")
    try:
        from backend.persist import persist_to_repo
        persist_to_repo()
    except Exception as e:
        print(f"[Auto-Sync] Persistence error: {e}")


def _auto_sync_all_users():
    """Background thread: sync at 9am and 9pm daily, with exponential backoff."""
    import datetime as _dt
    SYNC_HOURS = {10}  # 10am daily (all overnight data settled by then)
    last_synced_hour: set = set()
    # Add small random offset at startup so multiple instances don't sync simultaneously
    startup_jitter = _random.uniform(0, 120)
    print(f"[Auto-Sync] Scheduler sleeping {startup_jitter:.0f}s startup jitter")
    time.sleep(startup_jitter)
    while True:
        now = _dt.datetime.now()
        hour = now.hour
        day_hour = (now.date(), hour)
        if hour in SYNC_HOURS and day_hour not in last_synced_hour:
            print(f"[Auto-Sync] Triggering scheduled sync at {now.strftime('%H:%M')}")
            _run_sync_cycle()
            last_synced_hour.add(day_hour)
            # Prune old entries to prevent unbounded growth
            cutoff = now.date() - timedelta(days=2)
            last_synced_hour = {dh for dh in last_synced_hour if dh[0] >= cutoff}
        time.sleep(60)  # Check every minute


# Initialize database and start auto-sync on startup
@app.on_event("startup")
def startup_event():
    init_db()
    sync_thread = threading.Thread(target=_auto_sync_all_users, daemon=True)
    sync_thread.start()
    print("[Startup] Auto-sync scheduler started (9am and 9pm daily)")

# One shared env instance per container — the validator expects session state
_env = WellnessEnv(seed=int(os.environ.get("SEED", "42")))


@app.post("/reset")
def reset(body: dict[str, Any]) -> JSONResponse:
    """Start a new episode. Body: {"task_name": "single_goal"}"""
    task_name = body.get("task_name", "single_goal")
    try:
        obs = _env.reset(task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(obs.model_dump())


@app.post("/step")
def step(body: dict[str, Any]) -> JSONResponse:
    """Apply one action. Body: Action fields as JSON."""
    try:
        action = Action(**body)
        result = _env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(result.model_dump())


@app.get("/state")
def state() -> JSONResponse:
    """Return current environment state."""
    try:
        s = _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(s.model_dump())


@app.get("/grade")
def grade() -> JSONResponse:
    """Return grader score for current episode."""
    try:
        score = _env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"score": score})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

# --- Multi-Tenant User Routes ---

@app.get("/api/history")
def history(limit: int = 30, x_user_id: int = Header(...)):
    return get_recent_history(x_user_id, limit=limit)

@app.get("/api/debug/raw-sync")
def debug_raw_sync(target_date: str = None, x_user_id: int = Header(...)):
    """Diagnostic endpoint: returns raw_payload + extracted column values side by side for a date."""
    from backend.database import SessionLocal, WearableSync
    import json as _json
    db = SessionLocal()
    try:
        q = db.query(WearableSync).filter(WearableSync.user_id == x_user_id)
        if target_date:
            q = q.filter(WearableSync.sync_date == target_date)
        q = q.order_by(WearableSync.sync_date.desc()).limit(7)
        rows = q.all()
        result = []
        for s in rows:
            raw = s.raw_payload
            if isinstance(raw, str):
                try:
                    raw = _json.loads(raw)
                except Exception:
                    pass
            extracted = {
                "hrv_rmssd": s.hrv_rmssd,
                "resting_hr": s.resting_hr,
                "recovery_score": s.recovery_score,
                "active_minutes": s.active_minutes,
                "active_calories": s.active_calories,
                "sleep_score": s.sleep_score,
                "sleep_duration_hours": s.sleep_duration_hours,
                "stress_avg": s.stress_avg,
                "steps": s.steps,
                "spo2": s.spo2,
                "respiration_rate": s.respiration_rate,
                "strain_score": s.strain_score,
            }
            # Pull key fields from raw for comparison
            raw_highlights = {}
            if isinstance(raw, dict):
                raw_highlights["intensity_minutes_raw"] = raw.get("intensity_minutes")
                raw_highlights["active_calories_raw"] = raw.get("active_calories")
                raw_highlights["sleep_duration_hours_raw"] = raw.get("sleep_duration_hours")
                stress = raw.get("stress")
                if isinstance(stress, dict):
                    raw_highlights["stress_avgStressLevel"] = stress.get("avgStressLevel")
                sleep = raw.get("sleep")
                if isinstance(sleep, dict):
                    dto = sleep.get("dailySleepDTO") or {}
                    dur = sleep.get("durationInSeconds") or dto.get("sleepDurationInSeconds") or dto.get("sleepTimeSeconds")
                    raw_highlights["sleep_durationInSeconds"] = dur
                    if dur and isinstance(dur, (int, float)):
                        raw_highlights["sleep_duration_hours_raw"] = round(dur / 3600, 1)
                    scores = dto.get("sleepScores") or {}
                    overall = scores.get("overall") or scores.get("personal") or {}
                    raw_highlights["sleep_score_raw"] = overall.get("value") or overall.get("overallScore")
                hrv_raw = raw.get("hrv")
                if isinstance(hrv_raw, dict):
                    raw_highlights["hrv_lastNightAvg"] = hrv_raw.get("hrvSummary", {}).get("lastNightAvg", hrv_raw.get("lastNightAvg"))
                rhr_raw = raw.get("rhr")
                if isinstance(rhr_raw, dict):
                    am = rhr_raw.get("allMetrics") or {}
                    mm = am.get("metricsMap") or {}
                    rhr_list = mm.get("WELLNESS_RESTING_HEART_RATE") or []
                    raw_highlights["resting_hr_raw"] = rhr_list[0].get("value") if rhr_list else rhr_raw.get("restingHeartRate")
                steps_raw = raw.get("steps")
                if isinstance(steps_raw, dict):
                    raw_highlights["steps_raw"] = steps_raw.get("totalSteps")
                spo2 = raw.get("spo2")
                if isinstance(spo2, dict):
                    raw_highlights["spo2_raw"] = spo2.get("latestSpO2") or spo2.get("averageSpO2")
                resp = raw.get("respiration")
                if isinstance(resp, dict):
                    raw_highlights["respiration_raw"] = resp.get("latestRespiration") or resp.get("avgSleepRespirationValue") or resp.get("avgWakingRespirationValue")
                bb = raw.get("body_battery")
                if isinstance(bb, list) and bb:
                    item = bb[-1] if isinstance(bb[-1], dict) else {}
                    raw_highlights["body_battery_raw"] = item.get("charged")
            result.append({
                "sync_date": s.sync_date,
                "source": s.source,
                "created_at": str(s.created_at) if s.created_at else None,
                "extracted": extracted,
                "raw_highlights": raw_highlights,
            })
        return result
    finally:
        db.close()

@app.get("/api/profile")
def profile(x_user_id: int = Header(...)):
    return get_user_profile(x_user_id)

@app.get("/api/users")
def list_users():
    return get_users()

@app.get("/api/sync/status")
def get_sync_status(x_user_id: int = Header(None)):
    """Returns last sync timestamp, last sync date, and backoff status for every user (or one user)."""
    users = get_users()
    for u in users:
        backoff = get_sync_backoff_status(u["id"])
        if backoff:
            u["sync_backoff"] = {
                "failures": backoff.get("failures", 0),
                "is_rate_limit": backoff.get("is_rate_limit", False),
                "next_retry_after": backoff.get("next_retry_after"),
                "last_failure": backoff.get("last_failure"),
            }
    if x_user_id:
        return [u for u in users if u["id"] == x_user_id]
    return users

@app.post("/api/users/creds")
def set_creds(body: dict, x_user_id: int = Header(...)):
    email = body.get("email")
    password = body.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Missing email or password")
    update_garmin_creds(x_user_id, email, password)
    return {"status": "success"}

@app.post("/api/users/create")
def create_new_user(body: dict):
    """Create a new user with optional Garmin credentials and device selection."""
    username = body.get("username")
    if not username:
        raise HTTPException(status_code=400, detail="Missing username")
    device = body.get("device", "garmin")
    result = create_user(
        username=username,
        name=body.get("name", username),
        garmin_email=body.get("email"),
        garmin_password=body.get("password"),
        wearable_source=device
    )
    return JSONResponse(result)

@app.post("/api/users/device")
def set_device(body: dict, x_user_id: int = Header(...)):
    """Set the user's wearable device type."""
    device = body.get("device")
    if device not in ("garmin", "apple_watch", "oneplus", "other"):
        raise HTTPException(status_code=400, detail="Invalid device. Must be: garmin, apple_watch, oneplus, other")
    set_user_device(x_user_id, device)
    return {"status": "success", "device": device}

@app.post("/api/wearable/upload")
async def upload_wearable_data(
    file: UploadFile = File(...),
    source: str = Form("other"),
    x_user_id: int = Header(...)
):
    """Upload health data file (Apple Health .zip/.xml, CSV, or JSON)."""
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    filename = (file.filename or "").lower()
    rows, errors = [], []

    if filename.endswith(".zip") or filename.endswith(".xml"):
        rows, errors = parse_apple_health_xml(content)
        source = "apple_health"
    elif filename.endswith(".csv"):
        rows, errors = parse_csv_upload(content)
    elif filename.endswith(".json"):
        rows, errors = parse_json_upload(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .zip, .xml, .csv, or .json")

    saved = 0
    for row in rows:
        sync_date = row.pop("sync_date", None)
        if not sync_date:
            errors.append(f"Row missing sync_date, skipped")
            continue
        try:
            save_wearable_sync(
                user_id=x_user_id,
                sync_date=sync_date,
                source=source,
                raw_payload={"upload": filename, "source": source},
                **row
            )
            saved += 1
        except Exception as e:
            errors.append(f"{sync_date}: {str(e)}")

    return JSONResponse({
        "status": "success",
        "rows_processed": saved,
        "rows_skipped": len(rows) - saved,
        "total_in_file": len(rows),
        "errors": errors[:20]  # cap error list
    })

@app.get("/api/wearable/upload-template")
def download_upload_template():
    """Download a CSV template for manual health data upload."""
    template_path = os.path.join(os.path.dirname(__file__), "static", "upload_template.csv")
    return FileResponse(template_path, filename="health_data_template.csv", media_type="text/csv")


@app.post("/api/health/apple-push")
async def apple_health_push(body: dict[str, Any], x_user_id: int = Header(...)):
    """Receive a single day's Apple Health metrics pushed from an iOS Shortcut.

    Expected JSON body:
      { "date": "2025-01-15",   // optional — defaults to today
        "hrv": 45.2,            // HRV RMSSD ms
        "resting_hr": 58,       // bpm
        "sleep_score": 82,      // 0–100
        "sleep_hours": 7.5,     // hours
        "steps": 8200,
        "active_calories": 420,
        "stress": 30 }          // avg stress 0–100

    Authenticate using the x-user-id header (numeric user ID from /api/users).
    """
    sync_date = body.get("date") or date.today().isoformat()
    save_wearable_sync(
        user_id=x_user_id,
        sync_date=sync_date,
        source="apple_health",
        hrv_rmssd=body.get("hrv"),
        resting_hr=body.get("resting_hr"),
        sleep_score=body.get("sleep_score"),
        sleep_duration_hours=body.get("sleep_hours"),
        steps=body.get("steps"),
        active_calories=body.get("active_calories"),
        stress_avg=body.get("stress"),
        raw_payload={"source": "apple_health_shortcut", **body},
    )
    try:
        evaluate_past_recommendations(x_user_id)
    except Exception:
        pass
    return {"status": "ok", "sync_date": sync_date}


@app.get("/api/wearable/sync")
@app.get("/api/garmin/backfill")
def backfill_garmin(days: int = 30, x_user_id: int = Header(...)) -> JSONResponse:
    """Fetch last N days of Garmin data for a user in one shot. Skips days already stored."""
    import time as _time
    auth_type, cred_a, cred_b = get_wearable_creds(x_user_id)
    if auth_type != "garmin" or not cred_a:
        raise HTTPException(status_code=400, detail="No Garmin credentials configured for this user")
    results = []
    skipped_existing = 0
    for i in range(days):
        dt = date.today() - timedelta(days=i)
        result = _do_wearable_sync(x_user_id, dt)
        if result["status"] == "success":
            results.append({"date": dt.isoformat(), "status": "success"})
        elif result["status"] == "rate_limited":
            results.append({"date": dt.isoformat(), "status": "rate_limited"})
            break  # Stop immediately once rate-limited; don't hammer the API
        elif result["status"] == "skipped":
            skipped_existing += 1
        else:
            results.append({"date": dt.isoformat(), "status": "error", "message": result.get("message")})
        _time.sleep(1.5)  # Brief pause between requests to reduce rate-limit risk
    try:
        evaluate_past_recommendations(x_user_id)
        evaluate_user_performance(x_user_id)
    except Exception:
        pass
    fetched = sum(1 for r in results if r["status"] == "success")
    return JSONResponse({"fetched": fetched, "skipped_existing": skipped_existing, "results": results})


@app.get("/api/garmin/sync")  # legacy alias
def sync_wearable(target_date: str = None, simulate: bool = False, x_user_id: int = Header(...)) -> JSONResponse:
    """Trigger wearable sync for this user (Terra or legacy Garmin). Optional simulate mode for Garmin users."""
    try:
        dates_to_sync = [date.fromisoformat(target_date)] if target_date else [date.today(), date.today() - timedelta(days=1)]

        results = []
        rate_limited_this_call = False
        for dt in dates_to_sync:
            if rate_limited_this_call:
                results.append({"date": dt.isoformat(), "status": "skipped", "message": "Skipped — rate limited on previous date"})
                continue
            if simulate:
                # Simulate mode only works for legacy Garmin path
                email, password = get_garmin_creds(x_user_id)
                data = fetch_garmin_data(email, password, dt, simulate=True)
                if data and "error" not in data:
                    save_garmin_sync(
                        user_id=x_user_id,
                        sync_date=dt.isoformat(),
                        hrv_avg=_safe_extract(data, "hrv", "lastNightAvg"),
                        resting_hr=_safe_extract(data, "rhr", "restingHeartRate"),
                        body_battery=_safe_extract(data, "body_battery", "latestValue"),
                        intensity_minutes=data.get("intensity_minutes", 0) or 0,
                        active_calories=data.get("active_calories", 0),
                        training_load=data.get("training_load", 0.0),
                        sleep_score=_safe_extract(data, "sleep", "score"),
                        sleep_duration_hours=data.get("sleep_duration_hours"),
                        stress_avg=_safe_extract(data, "stress", "avg"),
                        steps_total=(data.get("steps") or {}).get("totalSteps"),
                        spo2_avg=_safe_extract(data, "spo2", "latestSpO2"),
                        respiration_avg=_safe_extract(data, "respiration", "latestRespiration"),
                        raw_payload=data
                    )
                    results.append({"date": dt.isoformat(), "status": "success", "source": "simulate"})
                else:
                    results.append({"date": dt.isoformat(), "status": "error", "message": (data or {}).get("error", "Simulate returned no data")})
            else:
                result = _do_wearable_sync(x_user_id, dt)
                results.append({"date": dt.isoformat(), **result})
                if result.get("status") == "rate_limited":
                    rate_limited_this_call = True

        # Post-Sync Ecosystem Tasks (skip if fully rate limited)
        if not rate_limited_this_call:
            try:
                from backend.eval_service import evaluate_past_recommendations
                evaluate_past_recommendations(x_user_id)
            except Exception as system_err:
                print(f"[Wearable Sync] Non-fatal post-sync ecosystem error: {system_err}")

        # Include backoff info in response
        backoff = get_sync_backoff_status(x_user_id)
        resp = {"status": "complete", "results": results}
        if backoff:
            resp["backoff"] = {
                "failures": backoff.get("failures", 0),
                "is_rate_limit": backoff.get("is_rate_limit", False),
                "next_retry_after": backoff.get("next_retry_after"),
            }
        return JSONResponse(resp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Terra OAuth & Webhook Endpoints ---

@app.post("/api/terra/connect")
def terra_connect(x_user_id: int = Header(...)) -> JSONResponse:
    """Returns a Terra widget URL the client should redirect the user to."""
    try:
        widget_url = generate_widget_session(
            reference_id=str(x_user_id),
            redirect_url=os.environ.get("TERRA_REDIRECT_URL", "")
        )
        return JSONResponse({"widget_url": widget_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/terra/disconnect")
def terra_disconnect(x_user_id: int = Header(...)) -> JSONResponse:
    """Deauthorize Terra for this user and clear their terra_user_id."""
    from backend.database import SessionLocal, User
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == x_user_id).first()
        if not user or not user.terra_user_id:
            raise HTTPException(status_code=404, detail="No Terra connection found for this user.")
        tid = user.terra_user_id
        deauthenticate_user(tid)
        user.terra_user_id = None
        user.wearable_source = "garmin"  # fallback
        db.commit()
        return JSONResponse({"status": "success", "message": "Terra disconnected."})
    finally:
        db.close()


@app.post("/api/terra/callback")
async def terra_callback(request) -> JSONResponse:
    """Terra POSTs user info after OAuth completes. Stores terra_user_id for this user."""
    from fastapi import Request
    body = await request.json()
    # Terra sends: {"user": {"user_id": "...", "provider": "APPLE", ...}, "reference_id": "<our user_id>"}
    user_obj = body.get("user", {})
    terra_user_id = user_obj.get("user_id")
    reference_id = body.get("reference_id") or user_obj.get("reference_id")
    provider = (user_obj.get("provider") or "unknown").lower().replace(" ", "_")

    if not terra_user_id or not reference_id:
        raise HTTPException(status_code=400, detail="Missing user_id or reference_id in Terra callback")

    try:
        internal_user_id = int(reference_id)
        update_terra_creds(internal_user_id, terra_user_id, provider)
        return JSONResponse({"status": "success", "terra_user_id": terra_user_id, "provider": provider})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/terra/webhook")
async def terra_webhook(request) -> JSONResponse:
    """Receives real-time push data from Terra. Verifies HMAC, normalizes, saves to WearableSync."""
    from fastapi import Request
    payload_bytes = await request.body()
    signature = request.headers.get("terra-signature", "")

    if not verify_webhook_signature(payload_bytes, signature):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    try:
        import json as _json
        body = _json.loads(payload_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Map terra_user_id → internal user_id
    from backend.database import SessionLocal, User
    db = SessionLocal()
    try:
        user_obj = body.get("user", {})
        terra_user_id = user_obj.get("user_id") if isinstance(user_obj, dict) else None
        if not terra_user_id:
            return JSONResponse({"status": "ignored", "reason": "no user_id in payload"})

        user = db.query(User).filter(User.terra_user_id == terra_user_id).first()
        if not user:
            # Could be a new user connecting — log and ignore
            print(f"[Terra Webhook] Unknown terra_user_id: {terra_user_id}")
            return JSONResponse({"status": "ignored", "reason": "unknown terra_user_id"})

        normalized_items = normalize_terra_webhook(body)
        saved_count = 0
        for (tid, sync_date, flat_data) in normalized_items:
            source = flat_data.pop("source", user.wearable_source or "terra")
            save_wearable_sync(
                user_id=user.id,
                sync_date=sync_date,
                source=source,
                raw_payload=body,
                **{k: v for k, v in flat_data.items()}
            )
            saved_count += 1

        # Trigger post-sync eval
        try:
            from backend.eval_service import evaluate_past_recommendations
            evaluate_past_recommendations(user.id)
        except Exception as e:
            print(f"[Terra Webhook] eval error: {e}")

        return JSONResponse({"status": "ok", "saved": saved_count})
    finally:
        db.close()


@app.post("/api/nutrition/parse")
def parse_nutrition(body: dict[str, Any]) -> JSONResponse:
    """Parse natural language food logs into structured nutrition info."""
    text = body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text field")
    
    result = parse_nutrition_text(text)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return JSONResponse(result)


@app.post("/api/logs/manual")
def post_manual_log(body: dict, x_user_id: int = Header(...)):
    """Receives natural language or structured health logs."""
    try:
        log_type = body.get("log_type", body.get("type", "food"))
        raw_input = body.get("raw_input", body.get("input", ""))
        value = body.get("value", 0.0)
        log_date = body.get("log_date", date.today().isoformat())
        log_time = body.get("log_time", None)
        
        add_manual_log(x_user_id, log_date, log_type, value, raw_input, log_time)
        return JSONResponse({"status": "success", "message": "Log saved"})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/profile")
def save_profile(body: dict[str, Any], x_user_id: int = Header(...)) -> JSONResponse:
    """Update user profile."""
    try:
        update_user_profile(x_user_id, body)
        return JSONResponse({"status": "success", "message": "Profile updated"})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --- Admin / Metrics ---

@app.get("/api/admin/config")
def admin_config():
    """Return system constants for UI transparency (reward weights, scaling)."""
    return JSONResponse({
        "goal_weights": {g.value: w for g, w in GOAL_WEIGHTS.items()},
        "delta_scales": DELTA_SCALES,
        "delta_weight_ratio": _DELTA_WEIGHT,
        "state_weight_ratio": _STATE_WEIGHT
    })

@app.get("/api/persona/draft")
def get_draft_persona(x_user_id: int = Header(...)):
    """Fetch the latest draft calibration result."""
    draft_path = os.path.join("models", f"user_{x_user_id}", "draft_persona.json")
    if not os.path.exists(draft_path):
        raise HTTPException(status_code=404, detail="No draft persona found. Run Deep Calibration first.")
    with open(draft_path, "r") as f:
        return JSONResponse(json.load(f))

@app.post("/api/persona/approve")
def post_approve_persona(body: dict, x_user_id: int = Header(...)):
    """Finalize the draft persona, save to calibrated_persona.json, and mark as approved."""
    try:
        # 1. Save to final location
        output_dir = os.path.join("models", f"user_{x_user_id}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "calibrated_persona.json")
        with open(output_path, "w") as f:
            json.dump(body, f, indent=2)
            
        # 2. Update DB
        approve_simulator(x_user_id, True)
        
        # 3. Persistence
        from backend.persist import persist_model
        persist_model(x_user_id)
        
        return JSONResponse({"status": "success", "message": "Simulator approved and finalized."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Calibration (Regression only, saves to DRAFT) ---

@app.post("/api/calibrate")
def calibrate(x_user_id: int = Header(...)):
    """Run regression calibration. Returns R², features, weights."""
    try:
        result = calibrate_user_persona(x_user_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        # Push to HF repo in the background
        def _bg_persist():
            try:
                from backend.persist import persist_model
                persist_model(x_user_id)
            except Exception as e:
                print(f"Calibration persistence error: {e}")
        
        threading.Thread(target=_bg_persist, daemon=True).start()
        
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- NN Training (separate, background) ---
_training_status: dict[int, dict] = {}

@app.post("/api/train")
def train_nn(x_user_id: int = Header(...)):
    """Trigger NN training in background thread (separate from calibration)."""
    if _training_status.get(x_user_id, {}).get("status") == "running":
        return JSONResponse({"status": "running", "message": "Training already in progress."})
    
    # NEW: Check approval
    profile = get_user_profile(x_user_id)
    if not profile or not profile.get("simulator_approved"):
        raise HTTPException(status_code=400, detail="Simulator must be reviewed and approved before training NN.")
    
    persona_path = os.path.join("models", f"user_{x_user_id}", "calibrated_persona.json")
    if not os.path.exists(persona_path):
        raise HTTPException(status_code=400, detail="Run Deep Calibration first to generate persona.")
    
    _training_status[x_user_id] = {"status": "running", "message": "Training neural network..."}
    
    def _run():
        try:
            from rl_training.train import train as run_training
            from backend.persist import persist_model
            model_path = run_training(
                user_id=x_user_id,
                persona_path=persona_path,
                total_steps=10000
            )
            # Auto-persist to HF repo
            persist_model(x_user_id)
            _training_status[x_user_id] = {
                "status": "complete",
                "message": "Neural network training complete! Model persisted to repo.",
                "model_path": model_path
            }
        except Exception as e:
            _training_status[x_user_id] = {"status": "error", "message": str(e)}
    
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    
    return JSONResponse({"status": "running", "message": "NN training started."})


@app.get("/api/train/status")
def train_status(x_user_id: int = Header(...)):
    """Check NN training progress."""
    status = _training_status.get(x_user_id, {"status": "idle", "message": "No training run yet."})
    return JSONResponse(status)


# ---------------------------------------------------------------------------
# Maturity tier endpoints
# ---------------------------------------------------------------------------

@app.get("/api/maturity/status")
def maturity_status(x_user_id: int = Header(...)):
    """Return the user's current model maturity status (tier, data days, gates)."""
    from backend.maturity_config import get_maturity_status
    status = get_maturity_status(x_user_id)
    return JSONResponse(status.to_dict())


@app.post("/api/maturity/transition")
def maturity_advance(x_user_id: int = Header(...)):
    """Advance the user's active tier by one step (if they qualify)."""
    from backend.maturity_config import get_maturity_status, set_active_tier, TIER_ORDER
    status = get_maturity_status(x_user_id)
    if not status.can_advance:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot advance from '{status.active_tier}': {status.recommendation_text}",
        )
    new_tier = status.next_tier
    set_active_tier(x_user_id, new_tier)
    return JSONResponse({"ok": True, "active_tier": new_tier})


@app.post("/api/maturity/revert")
def maturity_revert(x_user_id: int = Header(...)):
    """Revert the user's active tier by one step."""
    from backend.maturity_config import get_maturity_status, set_active_tier
    status = get_maturity_status(x_user_id)
    if not status.can_revert:
        raise HTTPException(
            status_code=400,
            detail="Already at the base tier (rules); cannot revert further.",
        )
    set_active_tier(x_user_id, status.prev_tier)
    return JSONResponse({"ok": True, "active_tier": status.prev_tier})


@app.post("/api/maturity/jump")
async def maturity_jump(request: Request, x_user_id: int = Header(...)):
    """Jump directly to any previous tier (allows reverting more than one step)."""
    from backend.maturity_config import get_active_tier, set_active_tier, TIER_ORDER
    body = await request.json()
    target_tier = body.get("tier")
    if target_tier not in TIER_ORDER:
        raise HTTPException(status_code=400, detail=f"Unknown tier: {target_tier}. Must be one of {TIER_ORDER}")
    current_tier = get_active_tier(x_user_id)
    current_idx = TIER_ORDER.index(current_tier)
    target_idx = TIER_ORDER.index(target_tier)
    if target_idx >= current_idx:
        raise HTTPException(
            status_code=400,
            detail="Jump only allowed to a previous tier. Use /api/maturity/transition to advance.",
        )
    set_active_tier(x_user_id, target_tier)
    return JSONResponse({"ok": True, "active_tier": target_tier})


@app.put("/api/maturity/thresholds")
async def maturity_set_thresholds(request: Request, x_user_id: int = Header(...)):
    """Override per-user maturity threshold values (body: JSON dict of threshold key→value)."""
    from backend.maturity_config import set_user_thresholds, DEFAULT_THRESHOLDS
    body = await request.json()
    invalid = {k for k in body if k not in DEFAULT_THRESHOLDS}
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown threshold keys: {sorted(invalid)}. Valid keys: {sorted(DEFAULT_THRESHOLDS.keys())}",
        )
    set_user_thresholds(x_user_id, body)
    return JSONResponse({"ok": True, "updated": body})


@app.post("/api/maturity/train")
def maturity_train(x_user_id: int = Header(...)):
    """Manually trigger ML model training for the requesting user."""
    from backend.outcome_models import train_outcome_models
    suite = train_outcome_models(x_user_id)
    if suite is None:
        from backend.maturity_config import count_paired_days, get_user_thresholds
        days = count_paired_days(x_user_id)
        thr = get_user_thresholds(x_user_id)
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data: {days} paired days, need {thr['ml_model_min_days']}.",
        )
    return JSONResponse({
        "ok": True,
        "data_days": suite.data_days,
        "n_models": len(suite.models),
        "fitted_at": suite.fitted_at,
    })


# ---------------------------------------------------------------------------
# ML evals endpoints
# ---------------------------------------------------------------------------

@app.get("/api/evals/models")
def evals_models(x_user_id: int = Header(...)):
    """Return per-outcome R², feature importances, and maturity status."""
    from backend.eval_models_service import get_outcome_model_evals
    return JSONResponse(get_outcome_model_evals(x_user_id))


@app.get("/api/evals/trajectory")
def evals_trajectory(x_user_id: int = Header(...)):
    """Return the R² training history for trend charts."""
    from backend.eval_models_service import get_model_trajectory
    return JSONResponse(get_model_trajectory(x_user_id))


@app.get("/api/evals/inference-comparison")
def evals_inference_comparison(x_user_id: int = Header(...)):
    """Compare primary vs alt inference path accuracy over the last 7 days."""
    from backend.eval_models_service import get_inference_comparison
    return JSONResponse(get_inference_comparison(x_user_id))


@app.post("/api/persist")
def persist_data():
    """Manually trigger persistence of all models + DB to HF repo."""
    from backend.persist import persist_to_repo
    result = persist_to_repo()
    return JSONResponse(result)


# --- Custom Goal Endpoints ---

@app.post("/api/user/goal")
def post_user_goal(body: dict[str, Any], x_user_id: int = Header(...)) -> JSONResponse:
    """Set a custom free-text goal with optional target date. LLM interprets it into a GoalProfile."""
    from backend.goal_interpreter import interpret_goal
    from datetime import date as _date

    goal_text = body.get("goal_text", "").strip()
    if not goal_text:
        raise HTTPException(status_code=400, detail="Missing goal_text")

    target_date_str = body.get("target_date")
    target_date = None
    if target_date_str:
        try:
            target_date = _date.fromisoformat(target_date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid target_date format. Use YYYY-MM-DD.")

    # Interpret goal via LLM
    try:
        goal_profile = interpret_goal(goal_text, target_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Goal interpretation failed: {e}")

    # Cache in DB
    import json as _json
    profile_json = _json.dumps(goal_profile.to_dict())
    set_custom_goal(x_user_id, goal_text, target_date_str, profile_json)

    return JSONResponse({
        "status": "success",
        "message": f"Goal set: {goal_text}",
        "goal_profile": goal_profile.to_dict(),
    })


@app.get("/api/user/goal")
def get_user_goal(x_user_id: int = Header(...)) -> JSONResponse:
    """Get the current custom goal for a user, or null if using preset."""
    custom = get_custom_goal(x_user_id)
    if not custom:
        profile = get_user_profile(x_user_id)
        return JSONResponse({
            "has_custom_goal": False,
            "preset_goal": profile.get("goal", "stress_management") if profile else "stress_management",
        })
    return JSONResponse({
        "has_custom_goal": True,
        **custom,
    })


@app.delete("/api/user/goal")
def delete_user_goal(x_user_id: int = Header(...)) -> JSONResponse:
    """Clear custom goal, reverting to preset dropdown."""
    clear_custom_goal(x_user_id)
    return JSONResponse({"status": "success", "message": "Custom goal cleared."})


@app.get("/api/recommendations")
def recommendations(x_user_id: int = Header(...), goal: str = "stress_management", mode: str = "auto"):
    res = get_coaching_recommendation(x_user_id, goal=goal, force_mode=mode)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    
    # Extract sport-specific info from goal_profile if present
    gp = res.get("goal_profile")
    rec_sport = gp.get("recommended_sport") if gp else None
    rec_duration = gp.get("recommended_duration_minutes") if gp else None
    
    # Save for future Evaluation
    try:
        import json as _json
        from datetime import date
        today = date.today().isoformat()
        deltas = res.get("expected_deltas", {})
        alt_deltas = res.get("expected_deltas_alt")
        save_recommendation(
            user_id=x_user_id,
            rec_date=today,
            sleep_rec=res["recommendation"]["sleep"],
            activity_rec=res["recommendation"]["activity"],
            expected_hrv=deltas.get("hrv", 0.0),
            expected_rhr=deltas.get("resting_hr", 0.0),
            expected_sleep=deltas.get("sleep_score", 0.0),
            expected_stress=deltas.get("stress_avg", 0.0),
            expected_battery=deltas.get("body_battery", 0.0),
            expected_sleep_stage=deltas.get("sleep_stage_quality", 0.0),
            expected_vo2=deltas.get("vo2_max", 0.0),
            long_term_impact=res.get("long_term_impact"),
            inference_path=res.get("inference_path"),
            expected_deltas_alt=_json.dumps(alt_deltas) if alt_deltas else None,
            recommended_sport=rec_sport,
            recommended_duration=rec_duration,
        )
    except Exception as e:
        print(f"Error saving recommendation for eval: {e}")
        
    return res


# --- User Trust & Evals ---

@app.get("/api/persona/evals")
def get_user_evals(x_user_id: int = Header(...)):
    """Fetch historical evaluation deltas and fidelity scores."""
    from backend.database import get_recommendations
    recs = get_recommendations(x_user_id, limit=30)
    # Filter for recs that HAVE been evaluated (fidelity_score is not null)
    evaluated = [r for r in recs if r.get("fidelity_score") is not None]
    # Ensure datetime fields are serializable
    for r in evaluated:
        for k, v in r.items():
            if isinstance(v, datetime):
                r[k] = v.isoformat()
    return JSONResponse(evaluated)

@app.post("/api/persona/evals/recalculate")
def recalculate_evals(x_user_id: int = Header(...)):
    """Force re-run compliance + fidelity on all existing recommendation records."""
    from backend.eval_service import force_recalculate_evals
    n = force_recalculate_evals(x_user_id)
    return JSONResponse({"ok": True, "records_updated": n})


@app.get("/api/persona/evals/backtest")
def backtest_evals(tier: str, x_user_id: int = Header(...)):
    """
    Read-only backtest: compute what fidelity score a given tier would achieve
    on all historical recommendations without modifying the DB.

    Query param: tier = rules | copula | ml_model | nn
    """
    from backend.eval_service import backtest_tier_fidelity
    result = backtest_tier_fidelity(x_user_id, tier)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(result)


@app.get("/api/dashboard/metrics")
def get_dashboard_metrics(x_user_id: int = Header(...)):
    """Return dashboard aggregates like 7-Day Compliance Avg."""
    from backend.database import get_recommendations
    recs = get_recommendations(x_user_id, limit=7)
    
    scored_recs = [r.get("compliance_score", 0.0) for r in recs if r.get("compliance_score") is not None]
    avg_compliance = sum(scored_recs) / len(scored_recs) if scored_recs else 0.0
    
    return JSONResponse({
        "avg_compliance_7d": round(avg_compliance * 100, 1)
    })

@app.post("/api/admin/backfill-raw")
def backfill_raw_payload():
    """Re-extract sleep stages, bedtime, exercise, vo2_max from raw_payload for all rows that have nulls."""
    from backend.database import SessionLocal, WearableSync
    import json as _json, datetime as _dt
    db = SessionLocal()
    updated = 0
    try:
        rows = db.query(WearableSync).all()
        for s in rows:
            if not s.raw_payload:
                continue
            try:
                raw = _json.loads(s.raw_payload) if isinstance(s.raw_payload, str) else s.raw_payload
            except Exception:
                continue

            changed = False

            # ── body_battery → recovery_score ──
            if s.recovery_score is None:
                bb = raw.get("body_battery")
                if isinstance(bb, list) and bb:
                    val = bb[-1].get("charged") if isinstance(bb[-1], dict) else None
                    if val is not None:
                        s.recovery_score = val; changed = True
                elif isinstance(bb, dict):
                    val = bb.get("latestValue") or bb.get("charged")
                    if val is not None:
                        s.recovery_score = val; changed = True

            # ── sleep fields ──
            sleep = raw.get("sleep") or {}
            dto = sleep.get("dailySleepDTO") or {}

            if s.sleep_duration_hours is None:
                dur = (sleep.get("durationInSeconds") or dto.get("sleepDurationInSeconds") or dto.get("sleepTimeSeconds"))
                if dur:
                    s.sleep_duration_hours = round(dur / 3600, 1); changed = True

            # Always recalculate bedtime (fix IST double-offset bug)
            ts = dto.get("sleepStartTimestampLocal")
            if ts:
                try:
                    # sleepStartTimestampLocal = GMT_epoch_ms + IST_offset_ms
                    # utcfromtimestamp(Local/1000) numerically equals IST wall-clock time
                    dt_start = _dt.datetime.utcfromtimestamp(int(ts) / 1000)
                    s.sleep_start_local = dt_start.isoformat()
                    s.sleep_start_hour = dt_start.hour + dt_start.minute / 60.0
                    changed = True
                except Exception:
                    pass

            if s.sleep_stage_quality is None:
                deep = dto.get("deepSleepSeconds") or 0
                rem = dto.get("remSleepSeconds") or 0
                light = dto.get("lightSleepSeconds") or 0
                awake = dto.get("awakeSleepSeconds") or 0
                total = deep + rem + light + awake
                if total > 0:
                    s.sleep_deep_pct = round(deep / total * 100, 1)
                    s.sleep_rem_pct = round(rem / total * 100, 1)
                    s.sleep_light_pct = round(light / total * 100, 1)
                    s.sleep_awake_pct = round(awake / total * 100, 1)
                    s.sleep_stage_quality = round((deep + rem) / total * 100, 1)
                    changed = True

            if s.sleep_score is None:
                scores = dto.get("sleepScores") or sleep.get("sleepScores") or {}
                sv = (scores.get("overall") or {}).get("value") or scores.get("overallScore")
                if sv is not None:
                    s.sleep_score = sv; changed = True

            # ── stress ──
            if s.stress_avg is None:
                stress = raw.get("stress") or {}
                sv = stress.get("avgStressLevel") or stress.get("averageStressLevel")
                if sv is not None:
                    s.stress_avg = sv; changed = True

            # ── exercise ──
            if s.exercise_type is None or s.exercise_duration_minutes is None:
                acts = raw.get("activities") or []
                if acts and isinstance(acts, list):
                    primary = max(acts, key=lambda a: a.get("duration") or 0, default=None)
                    if primary:
                        if s.exercise_type is None:
                            s.exercise_type = (primary.get("activityType") or {}).get("typeKey") or "other"
                        if s.exercise_duration_minutes is None:
                            s.exercise_duration_minutes = round((primary.get("duration") or 0) / 60)
                        changed = True
                elif s.exercise_type is None:
                    # Mark as "no activity" so it doesn't stay null
                    s.exercise_type = "none"
                    s.exercise_duration_minutes = 0
                    changed = True

            # ── vo2_max ──
            if s.vo2_max is None:
                mm = raw.get("max_metrics")
                if isinstance(mm, list) and mm:
                    v = mm[0].get("vo2Max") or mm[0].get("vo2max") or (mm[0].get("generic") or {}).get("vo2MaxPreciseValue")
                    if v is not None:
                        s.vo2_max = v; changed = True
                elif isinstance(mm, dict):
                    v = mm.get("vo2Max") or mm.get("vo2max") or (mm.get("generic") or {}).get("vo2MaxPreciseValue")
                    if v is not None:
                        s.vo2_max = v; changed = True
            if s.vo2_max is None:
                sm = raw.get("summary") or {}
                v = sm.get("vo2Max") or sm.get("vo2MaxValue") or sm.get("latestVo2Max")
                if v is not None:
                    s.vo2_max = v; changed = True
            if s.vo2_max is None:
                ts_d = raw.get("training_status") or {}
                if isinstance(ts_d, dict):
                    v = _extract_vo2_from_obj(ts_d)
                    if v is not None:
                        s.vo2_max = v; changed = True

            # ── hrv fallback ──
            if s.hrv_rmssd is None:
                hrv = raw.get("hrv") or {}
                v = (hrv.get("hrvSummary") or {}).get("lastNightAvg") or hrv.get("lastNightAvg")
                if v is not None:
                    s.hrv_rmssd = v; changed = True

            # ── steps fallback ──
            if s.steps is None:
                v = (raw.get("steps") or {}).get("totalSteps")
                if v is not None:
                    s.steps = v; changed = True

            if changed:
                updated += 1

        # ── Forward-fill vo2_max per user (carry last known value forward) ──
        from backend.database import User as _User
        user_ids = [u.id for u in db.query(_User).all()]
        ffill_count = 0
        for uid in user_ids:
            user_rows = (
                db.query(WearableSync)
                .filter(WearableSync.user_id == uid)
                .order_by(WearableSync.sync_date.asc())
                .all()
            )
            last_vo2 = None
            for row in user_rows:
                if row.vo2_max is not None:
                    last_vo2 = row.vo2_max
                elif last_vo2 is not None:
                    row.vo2_max = last_vo2
                    ffill_count += 1

        db.commit()
        return {"updated_rows": updated, "total_rows": len(rows), "vo2_ffilled": ffill_count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/api/admin/backfill-activities")
def backfill_activities():
    """Re-fetch activities and training_status from Garmin API for all historical dates
    where exercise_type is null or 'none', and vo2_max is null."""
    from backend.database import SessionLocal, WearableSync
    from backend.garmin_service import _get_client
    import json as _json, datetime as _dt

    db = SessionLocal()
    updated = 0
    errors = []
    debug_info = []
    try:
        rows = db.query(WearableSync).filter(
            WearableSync.source == "garmin"
        ).order_by(WearableSync.sync_date.asc()).all()

        from collections import defaultdict
        by_user: dict = defaultdict(list)
        for s in rows:
            by_user[s.user_id].append(s)

        for uid, user_rows in by_user.items():
            try:
                email, password = get_garmin_creds(uid)
            except Exception:
                errors.append(f"user {uid}: no credentials")
                continue
            if not email or not password:
                errors.append(f"user {uid}: empty credentials")
                continue

            try:
                client = _get_client(email, password)
            except Exception as e:
                errors.append(f"user {uid}: login failed – {e}")
                continue

            # Fetch ALL activities in one call (sorted date range)
            dates = sorted([s.sync_date if isinstance(s.sync_date, str) else s.sync_date.isoformat() for s in user_rows])
            start_date, end_date = dates[0], dates[-1]
            acts_by_date: dict = defaultdict(list)
            try:
                all_acts = client.get_activities_by_date(start_date, end_date)
                debug_info.append(f"user {uid}: fetched {len(all_acts)} total activities {start_date}→{end_date}")
                for act in (all_acts or []):
                    # startTimeLocal is "YYYY-MM-DD HH:MM:SS"
                    st = act.get("startTimeLocal") or act.get("startTime") or ""
                    act_date = st[:10] if st else None
                    if act_date:
                        acts_by_date[act_date].append(act)
            except Exception as e:
                errors.append(f"user {uid}: get_activities_by_date failed – {e}")

            # Fetch training_status for today (most recent VO2 max)
            today_ts = None
            try:
                today_ts = client.get_training_status(_dt.date.today().isoformat())
                debug_info.append(f"user {uid}: training_status keys={list(today_ts.keys()) if isinstance(today_ts, dict) else type(today_ts).__name__}")
            except Exception as e:
                errors.append(f"user {uid}: get_training_status failed – {e}")

            # Extract VO2 from training_status
            vo2_from_ts = None
            if today_ts and isinstance(today_ts, dict):
                vo2_from_ts = _extract_vo2_from_obj(today_ts)
                debug_info.append(f"user {uid}: vo2_from_ts={vo2_from_ts}")

            for s in user_rows:
                date_str = s.sync_date if isinstance(s.sync_date, str) else s.sync_date.isoformat()
                changed = False

                # ── Activities for this date ──
                day_acts = acts_by_date.get(date_str, [])
                if day_acts and (s.exercise_type is None or s.exercise_type == "none"):
                    primary = max(day_acts, key=lambda a: a.get("duration", 0) or 0)
                    s.exercise_type = (primary.get("activityType") or {}).get("typeKey") or "other"
                    s.exercise_duration_minutes = round((primary.get("duration") or 0) / 60)
                    changed = True
                    if s.raw_payload:
                        try:
                            raw = _json.loads(s.raw_payload) if isinstance(s.raw_payload, str) else dict(s.raw_payload)
                            raw["activities"] = day_acts
                            s.raw_payload = _json.dumps(raw)
                        except Exception:
                            pass

                # ── VO2 max (use training_status value for all rows) ──
                if s.vo2_max is None and vo2_from_ts is not None:
                    s.vo2_max = vo2_from_ts
                    changed = True

                if changed:
                    updated += 1

        db.commit()
        return {
            "updated_rows": updated,
            "total_rows": sum(len(v) for v in by_user.values()),
            "errors": errors,
            "debug": debug_info,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/admin/data")
def admin_data():
    """Return all users, profiles, syncs, and logs for admin overview."""
    from backend.database import SessionLocal, User, UserProfile, WearableSync, ManualLog
    db = SessionLocal()
    try:
        users = db.query(User).all()
        profiles = db.query(UserProfile).all()
        syncs = db.query(WearableSync).order_by(WearableSync.created_at.desc()).all()
        logs = db.query(ManualLog).order_by(ManualLog.created_at.desc()).all()
        
        def safe_serialize(obj, date_cols=("created_at", "updated_at")):
            result = {}
            for c in obj.__table__.columns:
                val = getattr(obj, c.name)
                if c.name in date_cols and val is not None:
                    val = str(val)
                result[c.name] = val
            return result
        
        return {
            "users": [
                {"id": u.id, "username": u.username, "garmin_email": u.garmin_email or "",
                 "wearable_source": u.wearable_source, "terra_user_id": u.terra_user_id, "created_at": str(u.created_at)}
                for u in users
            ],
            "profiles": [safe_serialize(p) for p in profiles],
            "syncs": [safe_serialize(s) for s in syncs],
            "logs": [safe_serialize(l) for l in logs],
            "counts": {
                "users": len(users),
                "profiles": len(profiles),
                "syncs": len(syncs),
                "logs": len(logs),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin data error: {str(e)}")
    finally:
        db.close()

# --- Static File Serving & SPA Routing ---

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # 1. API routes should never be handled by the SPA router
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail=f"API endpoint '{full_path}' not found")
        
    static_dir = "static"
    
    # 2. If the root is requested, serve index.html directly
    if not full_path or full_path == "/":
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, headers={"Cache-Control": "no-store"})
        return JSONResponse({"detail": "Build Error: static/index.html not found."}, status_code=404)

    # 3. Check for the literal file (CSS, JS, Images, Favicon)
    file_path = os.path.join(static_dir, full_path)
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            return FileResponse(file_path)
    
    # 4. Check for Next.js 'trailingSlash: true' patterns
    # e.g. user visits /dashboard, it might be at static/dashboard/index.html
    html_path = os.path.join(static_dir, full_path.strip("/"), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, headers={"Cache-Control": "no-store"})
        
    # 5. Check for 'trailingSlash: false' patterns 
    # e.g. /dashboard.html
    direct_html_path = os.path.join(static_dir, full_path.strip("/") + ".html")
    if os.path.exists(direct_html_path):
        return FileResponse(direct_html_path, headers={"Cache-Control": "no-store"})
        
    # 6. Final SPA Fallback: Everything else serves the main index.html
    # This allows client-side routing to take over
    index_fallback = os.path.join(static_dir, "index.html")
    if os.path.exists(index_fallback):
        return FileResponse(index_fallback, headers={"Cache-Control": "no-store"})
    
    return JSONResponse({
        "detail": "Frontend missing",
        "debug_info": {
            "requested": full_path,
            "static_folder_exists": os.path.exists(static_dir),
            "files_present": os.listdir(static_dir) if os.path.exists(static_dir) else []
        }
    }, status_code=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
