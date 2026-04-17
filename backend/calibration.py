import os
import json
import numpy as np
from backend.database import get_recent_history
from wellness_env.personas import ResponseModel

FEATURE_NAMES = ["Intercept", "Sleep (hours)", "Protein (100g)", "Carbs (100g)", "Fat (100g)", "Quality (0-1)", "Intensity Mins (h)", "Active Cals (100s)"]
OUTCOME_NAMES = ["ΔSleep Score", "ΔHRV (ms)", "ΔResting HR (bpm)", "ΔWeight (kg)", "ΔGarmin Stress"]

def calibrate_user_persona(user_id):
    """
    Scientific Linear Regression (Least Squares) for Digital Twin Calibration.
    Maps daily action features to biomarker deltas.
    Returns R², feature names, weights, and calibrated params (NO auto-training).
    """
    history = get_recent_history(user_id, limit=30)
    
    if len(history["syncs"]) < 15:
        return {"error": f"Insufficient data. Need at least 15 days of health history (currently have {len(history['syncs'])})."}
        
    sync_map = {s["sync_date"]: s for s in history["syncs"]}
    log_map = {}
    for l in history["logs"]:
        date = l["log_date"]
        if date not in log_map: log_map[date] = []
        log_map[date].append(l)
        
    sorted_dates = sorted(sync_map.keys())
    
    X = []
    Y = []
    
    # Lagged Regression: Actions on Day T -> Outcome on Day T+1
    import datetime
    today_str = datetime.date.today().isoformat()
    for i in range(len(sorted_dates) - 1):
        date_t = sorted_dates[i]
        date_t_plus_1 = sorted_dates[i+1]
        
        # Retroactive Locking Enforcer: Never train the model if the outcome is still being accumulated (Today!)
        if date_t_plus_1 == today_str:
            continue
            
        sync_t = sync_map[date_t]
        sync_t_plus_1 = sync_map[date_t_plus_1]
        logs_t = log_map.get(date_t, [])
        
        # Outcome: Change between Day T and Day T+1
        delta_sleep_score = sync_t_plus_1.get("sleep_score", 70) - sync_t.get("sleep_score", 70)
        delta_hrv = sync_t_plus_1["hrv_rmssd"] - sync_t["hrv_rmssd"]
        delta_rhr = sync_t_plus_1["resting_hr"] - sync_t["resting_hr"]
        delta_stress = sync_t_plus_1.get("stress_avg", 50) - sync_t.get("stress_avg", 50)
        
        # Features: Actions on Day T (Nutrition Macros via LLM JSON)
        protein_g = carbs_g = fat_g = 0.0
        q_sum = 0.0
        q_count = 0
        import json
        for l in logs_t:
            if l["log_type"] == "food" and l.get("raw_input"):
                try:
                    parsed = json.loads(l["raw_input"])
                    if isinstance(parsed, dict) and "parsed" in parsed:
                        p = parsed["parsed"]
                        protein_g += float(p.get("protein_g", 0))
                        carbs_g += float(p.get("carbs_g", 0))
                        fat_g += float(p.get("fat_g", 0))
                        qmap = {"high_protein": 0.8, "balanced": 1.0, "high_carb": 0.4, "processed": 0.1, "skipped": 0.0}
                        q_sum += qmap.get(p.get("nutrition_type", "balanced"), 1.0)
                        q_count += 1
                except Exception: pass
        quality_score = (q_sum / q_count) if q_count > 0 else 1.0
        
        # Garmin High-Fidelity Features
        intensity_mins = sync_t.get("active_minutes", 0)
        active_cals = sync_t.get("active_calories", 0)
        
        # Sleep Hours Day T (Physical actionable command)
        import json
        raw = {}
        try:
            if sync_t.get("raw_payload"):
                raw = json.loads(sync_t["raw_payload"]) if isinstance(sync_t["raw_payload"], str) else sync_t["raw_payload"]
        except Exception: pass
        sleep_seconds = raw.get("sleep", {}).get("durationInSeconds", 28800)
        sleep_hours = sleep_seconds / 3600.0
        
        # Weight Delta
        w_t = next((l["value"] for l in logs_t if l["log_type"] == "weight"), None)
        w_t_plus_1 = next((l["value"] for l in log_map.get(date_t_plus_1, []) if l["log_type"] == "weight"), None)
        delta_weight = (w_t_plus_1 - w_t) if (w_t and w_t_plus_1) else 0.0

        # Feature Vector: [Intercept, Sleep_h, Prot, Carbs, Fat, Quality, IntensityMins, ActiveCals]
        X.append([1.0, sleep_hours, protein_g/100.0, carbs_g/100.0, fat_g/100.0, quality_score, intensity_mins/60.0, active_cals/100.0])
        Y.append([delta_sleep_score, delta_hrv, delta_rhr, delta_weight, delta_stress])

    if len(X) < 14:
        return {"error": f"Not enough contiguous days with behavioral logs. Need 14 paired samples (currently have {len(X)})."}

    X_mat = np.array(X)
    Y_mat = np.array(Y)
    
    try:
        # Solve Least Squares
        W, residuals, rank, s_vals = np.linalg.lstsq(X_mat, Y_mat, rcond=None)
        
        # Compute R² for each outcome
        Y_pred = X_mat @ W
        ss_res = np.sum((Y_mat - Y_pred) ** 2, axis=0)
        ss_tot = np.sum((Y_mat - np.mean(Y_mat, axis=0)) ** 2, axis=0)
        r2_per_outcome = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
        r2_overall = float(np.mean(r2_per_outcome))
        
        # Weight matrix as readable table
        weights_table = []
        for i, feat in enumerate(FEATURE_NAMES):
            row = {"feature": feat}
            for j, out in enumerate(OUTCOME_NAMES):
                row[out] = round(float(W[i, j]), 4)
            weights_table.append(row)
        
        # Mapping to ResponseModel with physiological bounds (Clamping)
        # Note: W is now 8x5 (8 features, 5 outcomes: SleepScore, HRV, RHR, Weight, GarminStress)
        hrv_sleep = np.clip(W[1, 1], 1.5, 6.0) 
        rhr_sleep = np.clip(W[1, 2], -1.5, 0.0)
        rhr_exercise = np.clip(W[6, 2], -0.5, 0.1) 
        exercise_loss = np.clip(W[7, 3] * 0.1, -0.1, -0.01)
        nutrition_sensitivity = np.clip(W[3, 3] * 0.05, 0.01, 0.05) # Carbs -> Weight
        stress_hrv = np.clip(-W[5, 4], 1.0, 10.0) # Quality -> GarminStress (inverted)

        params = {
            "hrv_sleep_sensitivity": float(hrv_sleep),
            "rhr_sleep_benefit": float(rhr_sleep),
            "body_fat_nutrition_sensitivity": float(nutrition_sensitivity),
            "cortisol_nutrition_stress": float(stress_hrv),
            "vo2_cardio_gain": 0.15,
            "body_fat_exercise_loss": float(exercise_loss),
            "lean_mass_protein_gain": 0.02,
            "energy_nutrition_sensitivity": 8.0,
            "rhr_exercise_benefit": float(rhr_exercise)
        }
        
        output_dir = os.path.join(os.path.dirname(__file__), "..", "models", f"user_{user_id}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "draft_persona.json")
        
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
            
        return {
            "status": "success",
            "method": "least_squares_regression",
            "samples": len(X),
            "features": FEATURE_NAMES,
            "outcomes": OUTCOME_NAMES,
            "weights": weights_table,
            "r2_per_outcome": {OUTCOME_NAMES[i]: round(float(r2_per_outcome[i]), 4) for i in range(len(OUTCOME_NAMES))},
            "r2_overall": round(r2_overall, 4),
            "params": params,
            "persona_path": output_path,
        }
    except Exception as e:
        return {"error": f"Regression computation error: {str(e)}"}

if __name__ == "__main__":
    print(json.dumps(calibrate_user_persona(1), indent=2))
