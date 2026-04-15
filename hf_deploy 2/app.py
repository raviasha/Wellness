"""FastAPI server exposing WellnessEnv over HTTP for HF Space deployment."""

from __future__ import annotations

import os
from typing import Any

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
    get_users, update_garmin_creds, get_garmin_creds
)
from backend.llm_nutrition import parse_nutrition_text
from backend.calibration import calibrate_user_persona
from backend.inference_service import get_coaching_recommendation
from datetime import date

app = FastAPI(title="Wellness-Outcome OpenEnv", version="1.0.0")

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()

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


@app.get("/api/garmin/sync")
def sync_garmin(target_date: str = None, x_user_id: int = Header(...)) -> JSONResponse:
    """Trigger Garmin sync and return payloads for HRV, Resting HR, and Body Battery."""
    try:
        dt = None
        if target_date:
            dt = date.fromisoformat(target_date)
        else:
            dt = date.today()
            
        # Fetch email/password from DB for this user
        email, password = get_garmin_creds(x_user_id)
        data = fetch_garmin_data(email, password, dt)
        
        # Persist to DB if sync was successful
        if "error" not in data:
            save_garmin_sync(
                user_id=x_user_id,
                sync_date=dt.isoformat(),
                hrv_avg=data.get("hrv", {}).get("lastNightAvg"),
                resting_hr=data.get("rhr", {}).get("restingHeartRate"),
                body_battery=data.get("body_battery", {}).get("latestValue"),
                intensity_minutes=data.get("intensity_minutes", {}).get("total", 0),
                active_calories=data.get("active_calories", 0),
                training_load=data.get("training_load", 0.0),
                raw_payload=data
            )
            
        return JSONResponse(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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


@app.post("/api/logs")
def post_log(body: dict, x_user_id: int = Header(...)):
    """Receives natural language or structured health logs."""
    try:
        log_type = body.get("type", "food")
        raw_input = body.get("input", "")
        value = body.get("value", 0.0)
        
        add_manual_log(x_user_id, date.today().isoformat(), log_type, value, raw_input)
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


@app.post("/api/calibrate")
def calibrate(x_user_id: int = Header(...)):
    """Trigger the LLM-guided calibration of the digital twin."""
    try:
        result = calibrate_user_persona(x_user_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/recommendations")
def recommendations(x_user_id: int = Header(...)):
    res = get_coaching_recommendation(x_user_id)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

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
