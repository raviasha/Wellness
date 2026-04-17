import math
import os
import json
from datetime import datetime, timedelta
from backend.database import SessionLocal, Recommendation, WearableSync, ManualLog

def _generate_long_term_impact(rec, expected_deltas):
    """Generate LLM-based long-term healthspan/lifespan impact text for a recommendation."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

        prompt = f"""You are a longevity science expert. Based on these daily wellness recommendations and their expected short-term effects, describe the realistic long-term health impact on healthspan and lifespan in 2-3 concise sentences.

Recommendation:
- Sleep: {rec.sleep_rec}
- Exercise: {rec.exercise_rec}
- Nutrition: {rec.nutrition_rec}

Expected daily effects:
- HRV change: {expected_deltas.get('hrv', 0):+.1f} ms
- Resting HR change: {expected_deltas.get('rhr', 0):+.1f} bpm
- Sleep quality change: {expected_deltas.get('sleep', 0):+.1f}
- Stress change: {expected_deltas.get('stress', 0):+.1f}
- Body composition change: {expected_deltas.get('weight', 0):+.4f}%
- Energy change: {expected_deltas.get('energy', 0):+.1f}

Focus on concrete healthspan/lifespan outcomes. Reference evidence-based longevity research where relevant. Be specific about what trajectory these daily habits create over months and years."""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[EVAL_SERVICE] LLM long-term impact error: {e}")
        return None

def evaluate_past_recommendations(user_id: int):
    """
    Evaluates past recommendations by comparing expected outcomes to actual outcomes
    (captured in the next day's Garmin Sync). Also computes per-input compliance scores
    and generates long-term impact text.
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
                curr_sync = db.query(WearableSync).filter(
                    WearableSync.user_id == user_id,
                    WearableSync.sync_date == rec.rec_date
                ).first()
                
                next_sync = db.query(WearableSync).filter(
                    WearableSync.user_id == user_id,
                    WearableSync.sync_date == next_day_str
                ).first()
                
                # 1. Compute Per-Input Compliance Scores
                logs = db.query(ManualLog).filter(
                    ManualLog.user_id == user_id,
                    ManualLog.log_date == rec.rec_date
                ).all()
                
                logged_types = [l.log_type for l in logs]
                
                # Exercise compliance
                comp_exercise = 0.0
                if rec.exercise_rec and rec.exercise_rec != "none":
                    has_exercise = any(t in ("exercise", "workout", "intensity_minutes") for t in logged_types)
                    if has_exercise or (curr_sync and (curr_sync.active_minutes or 0) >= 15):
                        comp_exercise = 1.0
                else:
                    comp_exercise = 1.0  # No exercise prescribed = auto-pass
                
                # Sleep compliance
                comp_sleep = 0.0
                _SLEEP_RANGE = {
                    "less_than_6h": (0, 6), "6-7 hours": (6, 7), "7-8 hours": (7, 8),
                    "8-9 hours": (8, 9), "more_than_9h": (9, 14),
                }
                if rec.sleep_rec and rec.sleep_rec in _SLEEP_RANGE:
                    lo, hi = _SLEEP_RANGE[rec.sleep_rec]
                    actual_sleep = next_sync.sleep_duration_hours if next_sync and next_sync.sleep_duration_hours else None
                    if actual_sleep is not None:
                        if lo <= actual_sleep <= hi + 0.5:
                            comp_sleep = 1.0
                        elif abs(actual_sleep - (lo + hi) / 2) <= 1.5:
                            comp_sleep = 0.5
                    else:
                        comp_sleep = 0.5  # can't penalise without data
                else:
                    comp_sleep = 1.0
                
                # Nutrition compliance
                comp_nutrition = 0.0
                if "food" in logged_types:
                    comp_nutrition = 1.0
                else:
                    comp_nutrition = 0.5  # no food log = can't verify, partial credit
                
                # Store per-input and aggregate compliance
                rec.compliance_sleep = comp_sleep
                rec.compliance_exercise = comp_exercise
                rec.compliance_nutrition = comp_nutrition
                rec.compliance_score = min(1.0, (comp_exercise + comp_sleep + comp_nutrition) / 3)
                
                # 2. Compute Fidelity Score & All Actual Deltas (Requires next day's sync)
                if curr_sync and next_sync:
                    curr_hrv = curr_sync.hrv_rmssd or 0
                    next_hrv = next_sync.hrv_rmssd or 0
                    curr_rhr = curr_sync.resting_hr or 0
                    next_rhr = next_sync.resting_hr or 0
                    
                    actual_hrv_delta = next_hrv - curr_hrv
                    actual_rhr_delta = next_rhr - curr_rhr
                    
                    # Sleep score delta
                    curr_sleep = curr_sync.sleep_score or 0
                    next_sleep = next_sync.sleep_score or 0
                    actual_sleep_delta = next_sleep - curr_sleep
                    
                    # Stress delta
                    curr_stress = curr_sync.stress_avg or 0
                    next_stress = next_sync.stress_avg or 0
                    actual_stress_delta = next_stress - curr_stress
                    
                    # Weight delta (from manual logs if available)
                    actual_weight_delta = None
                    weight_logs_curr = db.query(ManualLog).filter(
                        ManualLog.user_id == user_id,
                        ManualLog.log_date == rec.rec_date,
                        ManualLog.log_type == "weight"
                    ).first()
                    weight_logs_next = db.query(ManualLog).filter(
                        ManualLog.user_id == user_id,
                        ManualLog.log_date == next_day_str,
                        ManualLog.log_type == "weight"
                    ).first()
                    if weight_logs_curr and weight_logs_next:
                        actual_weight_delta = (weight_logs_next.value or 0) - (weight_logs_curr.value or 0)
                    
                    expected_hrv = rec.expected_hrv_delta or 0.0
                    expected_rhr = rec.expected_rhr_delta or 0.0
                    
                    # Error distance
                    hrv_error = abs(actual_hrv_delta - expected_hrv)
                    rhr_error = abs(actual_rhr_delta - expected_rhr)
                    
                    # Include sleep and stress errors in fidelity if we have expected values
                    sleep_error = abs(actual_sleep_delta - (rec.expected_sleep_delta or 0.0))
                    stress_error = abs(actual_stress_delta - (rec.expected_stress_delta or 0.0))
                    
                    # Directional accuracy bonus: reward correct direction even if magnitude is off
                    dir_correct = 0
                    dir_total = 0
                    for exp_d, act_d in [
                        (expected_hrv, actual_hrv_delta),
                        (expected_rhr, actual_rhr_delta),
                        (rec.expected_sleep_delta, actual_sleep_delta),
                        (rec.expected_stress_delta, actual_stress_delta),
                    ]:
                        if exp_d and abs(exp_d) > 0.01:
                            dir_total += 1
                            if (exp_d > 0 and act_d > 0) or (exp_d < 0 and act_d < 0):
                                dir_correct += 1
                    directional_accuracy = (dir_correct / dir_total) if dir_total > 0 else 0.5
                    
                    # Softer decay (0.02 instead of 0.05) for wearable-scale variance tolerance
                    total_error = hrv_error + rhr_error + sleep_error * 0.5 + stress_error * 0.5
                    magnitude_fidelity = math.exp(-0.02 * total_error)
                    
                    # Blend: 60% magnitude accuracy + 40% directional accuracy
                    fidelity = 0.6 * magnitude_fidelity + 0.4 * directional_accuracy
                    
                    rec.actual_hrv_delta = actual_hrv_delta
                    rec.actual_rhr_delta = actual_rhr_delta
                    rec.actual_sleep_delta = actual_sleep_delta
                    rec.actual_stress_delta = actual_stress_delta
                    rec.actual_weight_delta = actual_weight_delta
                    rec.fidelity_score = round(fidelity, 4)
                
                # 3. Generate long-term impact if not already set
                if not rec.long_term_impact:
                    expected_deltas = {
                        "hrv": rec.expected_hrv_delta or 0,
                        "rhr": rec.expected_rhr_delta or 0,
                        "sleep": rec.expected_sleep_delta or 0,
                        "stress": rec.expected_stress_delta or 0,
                        "weight": rec.expected_weight_delta or 0,
                        "energy": rec.expected_energy_delta or 0,
                    }
                    impact_text = _generate_long_term_impact(rec, expected_deltas)
                    if impact_text:
                        rec.long_term_impact = impact_text
                    
            except Exception as loop_e:
                print(f"[EVAL_SERVICE] Warning evaluating record {rec.id}: {loop_e}")

        db.commit()
    except Exception as e:
        print(f"[EVAL_SERVICE] Critical error in evaluate_past_recommendations: {e}")
    finally:
        db.close()


# Backward-compatible alias (was in backend/evals.py)
evaluate_user_performance = evaluate_past_recommendations
