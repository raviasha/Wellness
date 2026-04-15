"""FastAPI server exposing WellnessEnv over HTTP for HF Space deployment."""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load variables from .env right at the start
load_dotenv()

import threading
import time
from typing import Any
from datetime import date, datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from wellness_env import WellnessEnv
from wellness_env.models import Action
from backend.garmin_service import fetch_garmin_data
from backend.database import (
    init_db, save_garmin_sync, add_manual_log,
    get_user_profile, update_user_profile, get_recent_history,
    get_users, update_garmin_creds, get_garmin_creds, approve_simulator,
    save_recommendation, create_user
)
from wellness_env.payoff import GOAL_WEIGHTS, DELTA_SCALES, _DELTA_WEIGHT, _STATE_WEIGHT
from backend.llm_nutrition import parse_nutrition_text
from backend.calibration import calibrate_user_persona
from backend.inference_service import get_coaching_recommendation
from backend.evals import evaluate_user_performance

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

app = FastAPI(title="Wellness-Outcome OpenEnv", version="1.0.0")

# --- Auto Garmin Sync Scheduler ---
def _auto_sync_all_users():
    """Background thread: sync Garmin data for all users with credentials."""
    while True:
        try:
            users = get_users()
            today = date.today()
            for u in users:
                try:
                    email, password = get_garmin_creds(u["id"])
                    if email and password:
                        # Fetch today AND yesterday to ensure no behavioral gaps (Intensity/Cals)
                        for dt in [date.today(), date.today() - timedelta(days=1)]:
                            data = fetch_garmin_data(email, password, dt)
                            if data and "error" not in data:
                                save_garmin_sync(
                                    user_id=u["id"],
                                    sync_date=dt.isoformat(),
                                    hrv_avg=_safe_extract(data, "hrv", "lastNightAvg"),
                                    resting_hr=_safe_extract(data, "rhr", "restingHeartRate"),
                                    body_battery=_safe_extract(data, "body_battery", "latestValue"),
                                    intensity_minutes=_safe_extract(data, "intensity_minutes", "total") or 0,
                                    active_calories=data.get("active_calories", 0),
                                    training_load=data.get("training_load", 0.0),
                                    sleep_score=_safe_extract(data, "sleep", "score"),
                                    stress_avg=_safe_extract(data, "stress", "avg"),
                                    steps_total=(data.get("steps") or {}).get("totalSteps"),
                                    spo2_avg=(data.get("spo2") or {}).get("latestSpO2"),
                                    respiration_avg=(data.get("respiration") or {}).get("latestRespiration"),
                                    raw_payload=data
                                )
                                print(f"[Auto-Sync] Synced {dt.isoformat()} for user {u['id']} ({u['username']})")
                        
                        # Trigger evaluation loop
                        evaluate_user_performance(u['id'])
                except Exception as e:
                    print(f"[Auto-Sync] Error for user {u['id']}: {e}")
        except Exception as e:
            print(f"[Auto-Sync] Scheduler error: {e}")
        
        # Periodically backup the database back to the repo
        try:
            from backend.persist import persist_to_repo
            persist_to_repo()
        except Exception as e:
            print(f"[Auto-Sync] Persistence error: {e}")
            
        time.sleep(12 * 3600)  # Every 12 hours (Captures Morning/Evening)


# Initialize database and start auto-sync on startup
@app.on_event("startup")
def startup_event():
    init_db()
    sync_thread = threading.Thread(target=_auto_sync_all_users, daemon=True)
    sync_thread.start()
    print("[Startup] Auto-sync scheduler started (every 12 hours)")

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

@app.get("/api/profile")
def profile(x_user_id: int = Header(...)):
    return get_user_profile(x_user_id)

@app.get("/api/users")
def list_users():
    return get_users()

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
    """Create a new user with optional Garmin credentials."""
    username = body.get("username")
    if not username:
        raise HTTPException(status_code=400, detail="Missing username")
    result = create_user(
        username=username,
        name=body.get("name", username),
        garmin_email=body.get("email"),
        garmin_password=body.get("password")
    )
    return JSONResponse(result)


@app.get("/api/garmin/sync")
def sync_garmin(target_date: str = None, simulate: bool = False, x_user_id: int = Header(...)) -> JSONResponse:
    """Trigger Garmin sync and return payloads. Optional simulation mode."""
    try:
        # Fetch email/password from DB for this user
        email, password = get_garmin_creds(x_user_id)
        
        # If no target_date, sync Today AND Yesterday to ensure no behavioral gaps
        dates_to_sync = []
        if target_date:
            dates_to_sync = [date.fromisoformat(target_date)]
        else:
            dates_to_sync = [date.today(), date.today() - timedelta(days=1)]
            
        results = []
        for dt in dates_to_sync:
            data = fetch_garmin_data(email, password, dt, simulate=simulate)
            if data and "error" not in data:
                save_garmin_sync(
                    user_id=x_user_id,
                    sync_date=dt.isoformat(),
                    hrv_avg=_safe_extract(data, "hrv", "lastNightAvg"),
                    resting_hr=_safe_extract(data, "rhr", "restingHeartRate"),
                    body_battery=_safe_extract(data, "body_battery", "latestValue"),
                    intensity_minutes=_safe_extract(data, "intensity_minutes", "total") or 0,
                    active_calories=data.get("active_calories", 0),
                    training_load=data.get("training_load", 0.0),
                    sleep_score=_safe_extract(data, "sleep", "score"),
                    stress_avg=_safe_extract(data, "stress", "avg"),
                    steps_total=(data.get("steps") or {}).get("totalSteps"),
                    spo2_avg=(data.get("spo2") or {}).get("latestSpO2"),
                    respiration_avg=(data.get("respiration") or {}).get("latestRespiration"),
                    raw_payload=data
                )
                results.append({"date": dt.isoformat(), "status": "success"})
            else:
                results.append({"date": dt.isoformat(), "status": "error", "message": data.get("error") if data else "Unknown"})
                
        # Post-Sync Ecosystem Tasks: Trigger evaluation for today and yesterday
        try:
            from backend.eval_service import evaluate_past_recommendations
            evaluate_past_recommendations(x_user_id)
        except Exception as system_err:
            print(f"[Garmin Sync] Non-fatal post-sync ecosystem error: {system_err}")
            
        return JSONResponse({"status": "complete", "results": results})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@app.post("/api/persist")
def persist_data():
    """Manually trigger persistence of all models + DB to HF repo."""
    from backend.persist import persist_to_repo
    result = persist_to_repo()
    return JSONResponse(result)


@app.get("/api/recommendations")
def recommendations(x_user_id: int = Header(...), goal: str = "overall_wellness", mode: str = "auto"):
    res = get_coaching_recommendation(x_user_id, goal=goal, force_mode=mode)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    
    # Save for future Evaluation
    try:
        from datetime import date
        today = date.today().isoformat()
        save_recommendation(
            user_id=x_user_id,
            rec_date=today,
            sleep_rec=res["recommendation"]["sleep"],
            exercise_rec=res["recommendation"]["exercise"],
            nutrition_rec=res["recommendation"]["nutrition"],
            expected_hrv=res.get("expected_deltas", {}).get("hrv", 0.0),
            expected_rhr=res.get("expected_deltas", {}).get("resting_hr", 0.0)
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
    return JSONResponse(evaluated)

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

@app.get("/api/admin/data")
def admin_data():
    """Return all users, profiles, syncs, and logs for admin overview."""
    from backend.database import SessionLocal, User, UserProfile, GarminSync, ManualLog
    db = SessionLocal()
    try:
        users = db.query(User).all()
        profiles = db.query(UserProfile).all()
        syncs = db.query(GarminSync).order_by(GarminSync.created_at.desc()).all()
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
                {"id": u.id, "username": u.username, "garmin_email": u.garmin_email or "", "created_at": str(u.created_at)}
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
            return FileResponse(index_path)
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
        return FileResponse(html_path)
        
    # 5. Check for 'trailingSlash: false' patterns 
    # e.g. /dashboard.html
    direct_html_path = os.path.join(static_dir, full_path.strip("/") + ".html")
    if os.path.exists(direct_html_path):
        return FileResponse(direct_html_path)
        
    # 6. Final SPA Fallback: Everything else serves the main index.html
    # This allows client-side routing to take over
    index_fallback = os.path.join(static_dir, "index.html")
    if os.path.exists(index_fallback):
        return FileResponse(index_fallback)
    
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
