import json
from datetime import datetime, timedelta
from backend.database import SessionLocal, Recommendation, GarminSync, ManualLog

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
            
        # 2. Get the actual deltas from Garmin Syncs
        # We need T (yesterday) and T+1 (today)
        today_date = datetime.utcnow().date().isoformat()
        
        sync_t = db.query(GarminSync).filter(GarminSync.user_id == user_id, GarminSync.sync_date == yesterday).first()
        sync_t_plus_1 = db.query(GarminSync).filter(GarminSync.user_id == user_id, GarminSync.sync_date == today_date).first()
        
        if not sync_t or not sync_t_plus_1:
            return # Missing data for delta calculation
            
        actual_hrv_delta = sync_t_plus_1.hrv_avg - sync_t.hrv_avg
        actual_rhr_delta = sync_t_plus_1.resting_hr - sync_t.resting_hr
        
        # 3. Calculate Compliance
        # Check logs for yesterday (T) to see if they matched the recommendation
        logs = db.query(ManualLog).filter(ManualLog.user_id == user_id, ManualLog.log_date == yesterday).all()
        
        compliance_score = 1.0
        # Check exercise compliance
        if rec.exercise_rec != "none":
            has_exercise = any(l.log_type == "exercise" or l.log_type == "intensity_minutes" for l in logs)
            if not has_exercise:
                compliance_score -= 0.5
        
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
