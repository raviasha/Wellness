import math
from datetime import datetime, timedelta
from backend.database import SessionLocal, Recommendation, GarminSync, ManualLog

def evaluate_past_recommendations(user_id: int):
    """
    Evaluates past recommendations by comparing expected outcomes to actual outcomes
    (captured in the next day's Garmin Sync). Also computes a basic compliance score
    based on manual log interactions on the day of the prescription.
    """
    db = SessionLocal()
    try:
        # Get recommendations that haven't been assigned a fidelity score yet
        pending_evals = db.query(Recommendation).filter(
            Recommendation.user_id == user_id,
            Recommendation.fidelity_score == None
        ).order_by(Recommendation.rec_date.asc()).all()
        
        for rec in pending_evals:
            try:
                rec_dt = datetime.strptime(rec.rec_date, "%Y-%m-%d")
                next_day_str = (rec_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                
                # Fetch Current (Baseline) and Next (Outcome) Syncs
                curr_sync = db.query(GarminSync).filter(
                    GarminSync.user_id == user_id,
                    GarminSync.sync_date == rec.rec_date
                ).first()
                
                next_sync = db.query(GarminSync).filter(
                    GarminSync.user_id == user_id,
                    GarminSync.sync_date == next_day_str
                ).first()
                
                # 1. Compute Compliance Score
                logs = db.query(ManualLog).filter(
                    ManualLog.user_id == user_id,
                    ManualLog.log_date == rec.rec_date
                ).all()
                
                compliance = 0.0
                logged_types = [l.log_type for l in logs]
                
                # Naive compliance logic: If they interact with required fields, give credit
                if rec.exercise_rec and rec.exercise_rec != "none":
                    if "note" in logged_types or "workout" in logged_types or "stress" in logged_types:
                        compliance += 0.5
                else:
                    compliance += 0.5  # Pass if no exercise was prescribed
                    
                if "food" in logged_types:
                    compliance += 0.5
                    
                compliance = min(1.0, compliance)
                rec.compliance_score = compliance
                
                # 2. Compute Fidelity Score (Requires next day's sync to exist)
                if curr_sync and next_sync:
                    curr_hrv = curr_sync.hrv_avg or 0
                    next_hrv = next_sync.hrv_avg or 0
                    
                    curr_rhr = curr_sync.resting_hr or 0
                    next_rhr = next_sync.resting_hr or 0
                    
                    actual_hrv_delta = next_hrv - curr_hrv
                    actual_rhr_delta = next_rhr - curr_rhr
                    
                    expected_hrv = rec.expected_hrv_delta or 0.0
                    expected_rhr = rec.expected_rhr_delta or 0.0
                    
                    # Error distance
                    hrv_error = abs(actual_hrv_delta - expected_hrv)
                    rhr_error = abs(actual_rhr_delta - expected_rhr)
                    
                    # Scale error to 0-1 using an exponential decay function
                    # If error is 0 -> fidelity is 1.0. If error is 20 -> fidelity is near 0.
                    fidelity = math.exp(-0.05 * (hrv_error + rhr_error))
                    
                    rec.actual_hrv_delta = actual_hrv_delta
                    rec.actual_rhr_delta = actual_rhr_delta
                    rec.fidelity_score = round(fidelity, 4)
                    
            except Exception as loop_e:
                print(f"[EVAL_SERVICE] Warning evaluating record {rec.id}: {loop_e}")

        db.commit()
    except Exception as e:
        print(f"[EVAL_SERVICE] Critical error in evaluate_past_recommendations: {e}")
    finally:
        db.close()
