import json
from datetime import datetime, timedelta
from backend.database import SessionLocal, Recommendation, WearableSync, ManualLog

def evaluate_user_performance(user_id):
    """
    Closes the loop: Compares yesterday's AI recommendation with today's Garmin results.
    Accounts for compliance (did the user follow the advice?).
    """
    db = SessionLocal()
    try:
        # 1. Get yesterday's recommendation
        yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
        rec = db.query(Recommendation).filter(
            Recommendation.user_id == user_id,
            Recommendation.rec_date == yesterday
        ).first()
        
        if not rec:
            return  # No recommendation to evaluate
            
        # 2. Get the actual deltas from Wearable Syncs
        # We need T (yesterday) and T+1 (today)
        today_date = datetime.utcnow().date().isoformat()
        
        sync_t = db.query(WearableSync).filter(WearableSync.user_id == user_id, WearableSync.sync_date == yesterday).first()
        sync_t_plus_1 = db.query(WearableSync).filter(WearableSync.user_id == user_id, WearableSync.sync_date == today_date).first()
        
        if not sync_t or not sync_t_plus_1:
            return # Missing data for delta calculation
            
        actual_hrv_delta = (sync_t_plus_1.hrv_rmssd or 0) - (sync_t.hrv_rmssd or 0)
        actual_rhr_delta = (sync_t_plus_1.resting_hr or 0) - (sync_t.resting_hr or 0)
        
        # 3. Calculate Compliance
        # Check logs for yesterday (T) to see if they matched the recommendation
        logs = db.query(ManualLog).filter(ManualLog.user_id == user_id, ManualLog.log_date == yesterday).all()
        
        compliance_score = 0.0
        compliance_components = 3  # exercise, sleep, nutrition
        
        # Check exercise compliance
        if rec.exercise_rec and rec.exercise_rec != "none":
            has_exercise = any(l.log_type in ("exercise", "intensity_minutes", "workout") for l in logs)
            # Also count wearable active_minutes as implicit exercise evidence
            if has_exercise or (sync_t and (sync_t.active_minutes or 0) >= 15):
                compliance_score += 1.0
        else:
            compliance_score += 1.0  # No exercise prescribed = auto-pass
        
        # Check sleep compliance
        _SLEEP_RANGE = {
            "less_than_6h": (0, 6), "6-7 hours": (6, 7), "7-8 hours": (7, 8),
            "8-9 hours": (8, 9), "more_than_9h": (9, 14),
        }
        if rec.sleep_rec and rec.sleep_rec in _SLEEP_RANGE:
            lo, hi = _SLEEP_RANGE[rec.sleep_rec]
            actual_sleep = sync_t_plus_1.sleep_duration_hours if sync_t_plus_1 and sync_t_plus_1.sleep_duration_hours else None
            if actual_sleep is not None:
                if lo <= actual_sleep <= hi + 0.5:  # half-hour grace
                    compliance_score += 1.0
                elif abs(actual_sleep - (lo + hi) / 2) <= 1.5:
                    compliance_score += 0.5  # partial credit
            else:
                compliance_score += 0.5  # can't penalise without data
        else:
            compliance_score += 1.0
        
        # Check nutrition compliance
        has_food_log = any(l.log_type == "food" for l in logs)
        if has_food_log:
            compliance_score += 1.0
        else:
            compliance_score += 0.5  # no food log = can't verify, partial credit
        
        compliance_score = min(1.0, compliance_score / compliance_components)
        
        # 4. Calculate Fidelity Score
        # How close was the prediction? 
        # (Lower error = higher fidelity)
        hrv_err = abs(actual_hrv_delta - rec.expected_hrv_delta)
        rhr_err = abs(actual_rhr_delta - rec.expected_rhr_delta)
        
        # Normalize error (assume >10ms or >5bpm error is 0 fidelity)
        hrv_fid = max(0, 1.0 - (hrv_err / 10.0))
        rhr_fid = max(0, 1.0 - (rhr_err / 5.0))
        
        fidelity_score = (hrv_fid + rhr_fid) / 2.0
        
        # 5. Update Recommendation Record
        rec.actual_hrv_delta = float(actual_hrv_delta)
        rec.actual_rhr_delta = float(actual_rhr_delta)
        rec.compliance_score = float(compliance_score)
        rec.fidelity_score = float(fidelity_score)
        
        db.commit()
        print(f"[Eval] Evaluated User {user_id} for {yesterday}: Fidelity={fidelity_score:.2f}, Compliance={compliance_score:.2f}")
        
    except Exception as e:
        print(f"[Eval] Error evaluating user {user_id}: {e}")
    finally:
        db.close()
