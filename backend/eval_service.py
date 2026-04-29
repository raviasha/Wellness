import math
import os
import json
from datetime import datetime, timedelta
from backend.database import SessionLocal, Recommendation, WearableSync, ManualLog

# Prescribed activity level → expected active-minutes thresholds (lo, hi)
_ACTIVITY_MINUTES = {
    "rest_day":          (0,   10),
    "light_activity":    (10,  30),
    "moderate_activity": (30,  60),
    "vigorous_activity": (45,  90),
    "high_intensity":    (60, 120),
}

def _activity_level_compliance(activity_rec: str, actual_active_minutes: int) -> float:
    """Score 0–1 how well actual active minutes match the prescribed activity level."""
    mins = actual_active_minutes or 0
    if activity_rec not in _ACTIVITY_MINUTES:
        return 0.5  # unknown prescription — neutral
    lo, hi = _ACTIVITY_MINUTES[activity_rec]
    mid = (lo + hi) / 2
    if lo <= mins <= hi:
        return 1.0
    # Partial credit: within 50% of the range width from an edge
    half_range = max((hi - lo) / 2, 10)
    if mins < lo:
        gap = lo - mins
    else:
        gap = mins - hi
    return max(0.0, round(1.0 - (gap / half_range) * 0.5, 4))

def _generate_long_term_impact(rec, expected_deltas):
    """Generate LLM-based long-term healthspan/lifespan impact text for a recommendation."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

        prompt = f"""You are a longevity science expert. Based on these daily wellness recommendations and their expected short-term effects, describe the realistic long-term health impact on healthspan and lifespan in 2-3 concise sentences.

Recommendation:
- Sleep: {rec.sleep_rec}
- Activity: {rec.activity_rec}

Expected daily effects:
- HRV change: {expected_deltas.get('hrv', 0):+.1f} ms
- Resting HR change: {expected_deltas.get('rhr', 0):+.1f} bpm
- Sleep score change: {expected_deltas.get('sleep', 0):+.1f}
- Stress change: {expected_deltas.get('stress', 0):+.1f}
- Body battery change: {expected_deltas.get('battery', 0):+.1f}

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
                
                # Activity compliance — sport-aware when recommended_sport is available
                comp_activity = 0.0
                if rec.activity_rec == "rest_day":
                    # Rest day: penalise if they exercised significantly
                    actual_mins = (curr_sync.active_minutes or 0) if curr_sync else 0
                    comp_activity = 1.0 if actual_mins <= 15 else max(0.0, 1.0 - (actual_mins - 15) / 60)
                elif rec.activity_rec:
                    if rec.recommended_sport and curr_sync:
                        # Sport-aware: similarity × duration adherence
                        from backend.goal_interpreter import get_sport_compliance
                        actual_sport = curr_sync.exercise_type or ""
                        actual_dur = curr_sync.exercise_duration_minutes or 0
                        target_dur = rec.recommended_duration or 30
                        sport_comp = get_sport_compliance(
                            rec.recommended_sport, actual_sport, actual_dur, target_dur
                        )
                        # Blend: 60% sport specificity + 40% activity level intensity match
                        level_comp = _activity_level_compliance(rec.activity_rec, curr_sync.active_minutes or 0)
                        comp_activity = round(0.6 * sport_comp + 0.4 * level_comp, 4)
                    elif curr_sync:
                        # No sport target — score purely on active-minutes vs prescribed level
                        comp_activity = _activity_level_compliance(rec.activity_rec, curr_sync.active_minutes or 0)
                    else:
                        comp_activity = 0.5  # no data
                
                # Sleep compliance
                comp_sleep = 0.0
                _SLEEP_RANGE = {
                    "less_than_6h": (0, 6), "6_to_7h": (6, 7), "7_to_8h": (7, 8),
                    "8_to_9h": (8, 9), "more_than_9h": (9, 14),
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
                
                # Store per-input and aggregate compliance (2 inputs: sleep + activity)
                rec.compliance_sleep = comp_sleep
                rec.compliance_activity = comp_activity
                rec.compliance_score = min(1.0, (comp_sleep + comp_activity) / 2)
                
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
                    
                    # Body battery delta
                    curr_battery = curr_sync.recovery_score or 0
                    next_battery = next_sync.recovery_score or 0
                    actual_battery_delta = next_battery - curr_battery

                    # Sleep stage quality delta
                    curr_stage = curr_sync.sleep_stage_quality or 0
                    next_stage = next_sync.sleep_stage_quality or 0
                    actual_sleep_stage_delta = next_stage - curr_stage

                    # VO2 Max delta
                    curr_vo2 = curr_sync.vo2_max or 0
                    next_vo2 = next_sync.vo2_max or 0
                    actual_vo2_delta = next_vo2 - curr_vo2
                    
                    expected_hrv = rec.expected_hrv_delta or 0.0
                    expected_rhr = rec.expected_rhr_delta or 0.0
                    
                    # Error distance
                    hrv_error = abs(actual_hrv_delta - expected_hrv)
                    rhr_error = abs(actual_rhr_delta - expected_rhr)
                    
                    # Include all 7 biomarker errors in fidelity
                    sleep_error = abs(actual_sleep_delta - (rec.expected_sleep_delta or 0.0))
                    stress_error = abs(actual_stress_delta - (rec.expected_stress_delta or 0.0))
                    battery_error = abs(actual_battery_delta - (rec.expected_battery_delta or 0.0))
                    sleep_stage_error = abs(actual_sleep_stage_delta - (rec.expected_sleep_stage_delta or 0.0))
                    vo2_error = abs(actual_vo2_delta - (rec.expected_vo2_delta or 0.0))
                    
                    # Directional accuracy bonus: reward correct direction even if magnitude is off
                    dir_correct = 0
                    dir_total = 0
                    for exp_d, act_d in [
                        (expected_hrv, actual_hrv_delta),
                        (expected_rhr, actual_rhr_delta),
                        (rec.expected_sleep_delta, actual_sleep_delta),
                        (rec.expected_stress_delta, actual_stress_delta),
                        (rec.expected_battery_delta, actual_battery_delta),
                        (rec.expected_sleep_stage_delta, actual_sleep_stage_delta),
                        (rec.expected_vo2_delta, actual_vo2_delta),
                    ]:
                        if exp_d and abs(exp_d) > 0.01:
                            dir_total += 1
                            if (exp_d > 0 and act_d > 0) or (exp_d < 0 and act_d < 0):
                                dir_correct += 1
                    directional_accuracy = (dir_correct / dir_total) if dir_total > 0 else 0.5
                    
                    # Softer decay (0.02 instead of 0.05) for wearable-scale variance tolerance
                    total_error = hrv_error + rhr_error + sleep_error * 0.5 + stress_error * 0.5 + battery_error * 0.5 + sleep_stage_error * 0.5 + vo2_error * 0.3
                    magnitude_fidelity = math.exp(-0.02 * total_error)

                    # Blend: 60% magnitude accuracy + 40% directional accuracy.
                    # This is the pure model fidelity — compliance is tracked separately
                    # so the gap can be decomposed in the UI.
                    fidelity = 0.6 * magnitude_fidelity + 0.4 * directional_accuracy
                    
                    rec.actual_hrv_delta = actual_hrv_delta
                    rec.actual_rhr_delta = actual_rhr_delta
                    rec.actual_sleep_delta = actual_sleep_delta
                    rec.actual_stress_delta = actual_stress_delta
                    rec.actual_battery_delta = actual_battery_delta
                    rec.actual_sleep_stage_delta = actual_sleep_stage_delta
                    rec.actual_vo2_delta = actual_vo2_delta
                    rec.fidelity_score = round(fidelity, 4)

                    # Dual-path: compute alt fidelity if alt predictions are stored
                    if rec.expected_deltas_alt:
                        try:
                            alt_d = json.loads(rec.expected_deltas_alt) if isinstance(rec.expected_deltas_alt, str) else rec.expected_deltas_alt
                            alt_hrv_error   = abs(actual_hrv_delta   - float(alt_d.get("hrv",     0.0)))
                            alt_rhr_error   = abs(actual_rhr_delta   - float(alt_d.get("rhr",     0.0)))
                            alt_sleep_error = abs(actual_sleep_delta  - float(alt_d.get("sleep",   0.0)))
                            alt_stress_error= abs(actual_stress_delta - float(alt_d.get("stress",  0.0)))
                            alt_bat_error   = abs(actual_battery_delta- float(alt_d.get("battery", 0.0)))

                            alt_dir_correct = 0
                            alt_dir_total   = 0
                            for exp_d, act_d in [
                                (alt_d.get("hrv",     0), actual_hrv_delta),
                                (alt_d.get("rhr",     0), actual_rhr_delta),
                                (alt_d.get("sleep",   0), actual_sleep_delta),
                                (alt_d.get("stress",  0), actual_stress_delta),
                                (alt_d.get("battery", 0), actual_battery_delta),
                            ]:
                                if exp_d and abs(float(exp_d)) > 0.01:
                                    alt_dir_total += 1
                                    if (float(exp_d) > 0 and act_d > 0) or (float(exp_d) < 0 and act_d < 0):
                                        alt_dir_correct += 1

                            alt_dir_acc = (alt_dir_correct / alt_dir_total) if alt_dir_total > 0 else 0.5
                            alt_total_err = alt_hrv_error + alt_rhr_error + alt_sleep_error * 0.5 + alt_stress_error * 0.5 + alt_bat_error * 0.5
                            alt_mag_fid = math.exp(-0.02 * alt_total_err)
                            rec.fidelity_score_alt = round(0.6 * alt_mag_fid + 0.4 * alt_dir_acc, 4)
                        except Exception as alt_e:
                            print(f"[EVAL_SERVICE] Alt fidelity error rec {rec.id}: {alt_e}")
                
                # 3. Generate long-term impact if not already set
                if not rec.long_term_impact:
                    expected_deltas = {
                        "hrv": rec.expected_hrv_delta or 0,
                        "rhr": rec.expected_rhr_delta or 0,
                        "sleep": rec.expected_sleep_delta or 0,
                        "stress": rec.expected_stress_delta or 0,
                        "battery": rec.expected_battery_delta or 0,
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


def force_recalculate_evals(user_id: int):
    """Re-run compliance + fidelity scoring on ALL recommendation records.

    Regenerates expected deltas for the current active tier using a deterministic
    seed, then re-scores fidelity against actual next-day syncs. This restores
    correct scores after any DB corruption from earlier retroactive rewrites.
    """
    from backend.maturity_config import get_active_tier

    try:
        active_tier = get_active_tier(user_id)
    except Exception:
        active_tier = "rules"

    # Use backtest machinery to get deterministic per-rec expected deltas for
    # the current tier, then write them back to the DB before scoring.
    backtest = backtest_tier_fidelity(user_id, active_tier)

    db = SessionLocal()
    try:
        # Build a lookup of rec_date → backtest predictions
        pred_by_date = {r["rec_date"]: r for r in backtest.get("records", [])}

        all_recs = db.query(Recommendation).filter(
            Recommendation.user_id == user_id,
        ).order_by(Recommendation.rec_date.asc()).all()

        for rec in all_recs:
            try:
                rec_dt = datetime.strptime(rec.rec_date, "%Y-%m-%d")
                next_day_str = (rec_dt + timedelta(days=1)).strftime("%Y-%m-%d")

                curr_sync = db.query(WearableSync).filter(
                    WearableSync.user_id == user_id,
                    WearableSync.sync_date == rec.rec_date
                ).first()
                next_sync = db.query(WearableSync).filter(
                    WearableSync.user_id == user_id,
                    WearableSync.sync_date == next_day_str
                ).first()

                # Restore expected deltas from backtest if available
                bt = pred_by_date.get(rec.rec_date)
                if bt:
                    pred = bt["predicted"]
                    rec.expected_hrv_delta          = pred.get("hrv", rec.expected_hrv_delta)
                    rec.expected_rhr_delta          = pred.get("resting_hr", rec.expected_rhr_delta)
                    rec.expected_sleep_delta        = pred.get("sleep_score", rec.expected_sleep_delta)
                    rec.expected_stress_delta       = pred.get("stress_avg", rec.expected_stress_delta)
                    rec.expected_battery_delta      = pred.get("body_battery", rec.expected_battery_delta)
                    rec.expected_sleep_stage_delta  = pred.get("sleep_stage_quality", rec.expected_sleep_stage_delta)
                    rec.expected_vo2_delta          = pred.get("vo2_max", rec.expected_vo2_delta)

                    # Write fidelity directly from backtest (already computed there)
                    if next_sync:
                        rec.fidelity_score = bt["fidelity"]

                        # Also write actuals while we have both syncs
                        rec.actual_hrv_delta          = bt["actual"]["hrv"]
                        rec.actual_rhr_delta          = bt["actual"]["resting_hr"]
                        rec.actual_sleep_delta        = bt["actual"]["sleep_score"]
                        rec.actual_stress_delta       = bt["actual"]["stress_avg"]
                        rec.actual_battery_delta      = bt["actual"]["body_battery"]
                        rec.actual_sleep_stage_delta  = bt["actual"]["sleep_stage_quality"]
                        rec.actual_vo2_delta          = bt["actual"]["vo2_max"]

                # --- Compliance (sport-aware) ---
                comp_activity = 0.0
                if rec.activity_rec == "rest_day":
                    actual_mins = (curr_sync.active_minutes or 0) if curr_sync else 0
                    comp_activity = 1.0 if actual_mins <= 15 else max(0.0, 1.0 - (actual_mins - 15) / 60)
                elif rec.activity_rec:
                    if rec.recommended_sport and curr_sync:
                        from backend.goal_interpreter import get_sport_compliance
                        actual_sport = curr_sync.exercise_type or ""
                        actual_dur = curr_sync.exercise_duration_minutes or 0
                        target_dur = rec.recommended_duration or 30
                        sport_comp = get_sport_compliance(
                            rec.recommended_sport, actual_sport, actual_dur, target_dur
                        )
                        level_comp = _activity_level_compliance(rec.activity_rec, curr_sync.active_minutes or 0)
                        comp_activity = round(0.6 * sport_comp + 0.4 * level_comp, 4)
                    elif curr_sync:
                        comp_activity = _activity_level_compliance(rec.activity_rec, curr_sync.active_minutes or 0)
                    else:
                        comp_activity = 0.5

                comp_sleep = 0.0
                _SLEEP_RANGE = {
                    "less_than_6h": (0, 6), "6_to_7h": (6, 7), "7_to_8h": (7, 8),
                    "8_to_9h": (8, 9), "more_than_9h": (9, 14),
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
                        comp_sleep = 0.5
                else:
                    comp_sleep = 1.0

                rec.compliance_sleep = comp_sleep
                rec.compliance_activity = comp_activity
                rec.compliance_score = min(1.0, (comp_sleep + comp_activity) / 2)

                # --- Fidelity + Actuals (requires both syncs) ---
                if curr_sync and next_sync:
                    actual_hrv_delta   = (next_sync.hrv_rmssd or 0) - (curr_sync.hrv_rmssd or 0)
                    actual_rhr_delta   = (next_sync.resting_hr or 0) - (curr_sync.resting_hr or 0)
                    actual_sleep_delta = (next_sync.sleep_score or 0) - (curr_sync.sleep_score or 0)
                    actual_stress_delta = (next_sync.stress_avg or 0) - (curr_sync.stress_avg or 0)
                    actual_battery_delta = (next_sync.recovery_score or 0) - (curr_sync.recovery_score or 0)
                    actual_sleep_stage_delta = (next_sync.sleep_stage_quality or 0) - (curr_sync.sleep_stage_quality or 0)
                    actual_vo2_delta = (next_sync.vo2_max or 0) - (curr_sync.vo2_max or 0)

                    expected_hrv = rec.expected_hrv_delta or 0.0
                    expected_rhr = rec.expected_rhr_delta or 0.0

                    hrv_error    = abs(actual_hrv_delta - expected_hrv)
                    rhr_error    = abs(actual_rhr_delta - expected_rhr)
                    sleep_error  = abs(actual_sleep_delta - (rec.expected_sleep_delta or 0.0))
                    stress_error = abs(actual_stress_delta - (rec.expected_stress_delta or 0.0))
                    battery_error = abs(actual_battery_delta - (rec.expected_battery_delta or 0.0))
                    sleep_stage_error = abs(actual_sleep_stage_delta - (rec.expected_sleep_stage_delta or 0.0))
                    vo2_error    = abs(actual_vo2_delta - (rec.expected_vo2_delta or 0.0))

                    dir_correct = dir_total = 0
                    for exp_d, act_d in [
                        (expected_hrv, actual_hrv_delta),
                        (expected_rhr, actual_rhr_delta),
                        (rec.expected_sleep_delta, actual_sleep_delta),
                        (rec.expected_stress_delta, actual_stress_delta),
                        (rec.expected_battery_delta, actual_battery_delta),
                        (rec.expected_sleep_stage_delta, actual_sleep_stage_delta),
                        (rec.expected_vo2_delta, actual_vo2_delta),
                    ]:
                        if exp_d and abs(exp_d) > 0.01:
                            dir_total += 1
                            if (exp_d > 0 and act_d > 0) or (exp_d < 0 and act_d < 0):
                                dir_correct += 1
                    directional_accuracy = (dir_correct / dir_total) if dir_total > 0 else 0.5

                    total_error = hrv_error + rhr_error + sleep_error * 0.5 + stress_error * 0.5 + battery_error * 0.5 + sleep_stage_error * 0.5 + vo2_error * 0.3
                    magnitude_fidelity = math.exp(-0.02 * total_error)
                    fidelity = 0.6 * magnitude_fidelity + 0.4 * directional_accuracy

                    rec.actual_hrv_delta = actual_hrv_delta
                    rec.actual_rhr_delta = actual_rhr_delta
                    rec.actual_sleep_delta = actual_sleep_delta
                    rec.actual_stress_delta = actual_stress_delta
                    rec.actual_battery_delta = actual_battery_delta
                    rec.actual_sleep_stage_delta = actual_sleep_stage_delta
                    rec.actual_vo2_delta = actual_vo2_delta
                    rec.fidelity_score = round(fidelity, 4)

            except Exception as loop_e:
                print(f"[EVAL_SERVICE] force_recalculate warning rec {rec.id}: {loop_e}")

        db.commit()
        return len(all_recs)
    except Exception as e:
        print(f"[EVAL_SERVICE] Critical error in force_recalculate_evals: {e}")
        return 0
    finally:
        db.close()


# Backward-compatible alias (was in backend/evals.py)
evaluate_user_performance = evaluate_past_recommendations


def backtest_tier_fidelity(user_id: int, tier: str) -> dict:
    """
    Retroactively compute what fidelity score any tier would have achieved on
    every historical recommendation. Does NOT write to the DB — read-only backtest.

    The LLM recommendations (sleep_rec, activity_rec) are immutable.
    This function re-runs expected deltas for the requested tier and scores them
    against the already-stored actual deltas.

    Returns:
        {
          "tier": "copula",
          "avg_fidelity": 0.52,
          "records": [{"rec_date": "...", "fidelity": 0.55, ...}, ...]
        }
    """
    VALID_TIERS = {"rules", "copula", "ml_model", "nn"}
    if tier not in VALID_TIERS:
        return {"error": f"Unknown tier '{tier}'. Valid: {VALID_TIERS}"}

    # We need _get_expected_deltas with a forced tier, not the active tier.
    # Build a tier-specific delta function using the same simulator machinery.
    from backend.inference_service import _get_expected_deltas, _get_expected_deltas_from_ml
    from backend.database import get_recent_history
    import os, json

    history = get_recent_history(user_id, limit=60)

    # Resolve which model artefacts are available
    MODELS_ROOT = os.path.join("models")
    persona_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "calibrated_persona.json")
    ml_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "outcome_models.json")
    nn_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "ppo_wellness_v2.pt")

    if tier == "copula" and not os.path.exists(persona_path):
        return {"error": "Copula model not available — run Deep Calibrate first."}
    if tier == "ml_model" and not os.path.exists(ml_model_path):
        return {"error": "ML model not available — run Re-train Models first."}
    if tier == "nn" and not os.path.exists(nn_path):
        return {"error": "NN model not available — at least 90 days required."}

    def _forced_deltas(rec_dict, curr_sync_dict):
        """Get expected deltas for the specified tier regardless of active tier."""
        if tier == "rules":
            # Generic ResponseModel, no persona
            from wellness_env.personas import ResponseModel, PersonaConfig
            from wellness_env.simulator import compute_biomarker_changes
            from wellness_env.models import (
                Action, Biomarkers, Goal,
                SleepDuration, ActivityLevel, BedtimeWindow, ExerciseType, ExerciseDuration
            )
            import random

            sleep_map = {
                "less_than_6h": SleepDuration.VERY_SHORT, "6_to_7h": SleepDuration.SHORT,
                "7_to_8h": SleepDuration.OPTIMAL_LOW, "8_to_9h": SleepDuration.OPTIMAL_HIGH,
                "more_than_9h": SleepDuration.LONG,
            }
            rm = ResponseModel()
            rec_sleep = sleep_map.get(rec_dict.get("sleep", "7_to_8h"), SleepDuration.OPTIMAL_LOW)
            rec_activity = ActivityLevel(rec_dict.get("activity", "moderate_activity"))

            def _sv(key, default):
                return (curr_sync_dict.get(key) or default) if curr_sync_dict else default

            current_bio = Biomarkers(
                resting_hr=_sv("resting_hr", 70), hrv=_sv("hrv_rmssd", 45),
                sleep_score=_sv("sleep_score", 70), stress_avg=_sv("stress_avg", 50),
                body_battery=_sv("recovery_score", 50),
                sleep_stage_quality=_sv("sleep_stage_quality", 35),
                vo2_max=_sv("vo2_max", 40),
            )
            persona = PersonaConfig(
                name=f"user_{user_id}", compliance_rate=1.0, goal=Goal.STRESS_MANAGEMENT,
                sleep_default=rec_sleep, activity_default=rec_activity,
                starting_biomarkers=current_bio, response_model=rm,
            )
            action = Action(sleep=rec_sleep, activity=rec_activity,
                            bedtime=BedtimeWindow.OPTIMAL, exercise_type=ExerciseType.NONE,
                            exercise_duration=ExerciseDuration.NONE)
            deltas = compute_biomarker_changes(action, current_bio, persona, [], random.Random(42))
            return deltas.model_dump()

        elif tier == "copula":
            from wellness_env.personas import ResponseModel, PersonaConfig
            with open(persona_path) as f:
                params = json.load(f)
            rm = ResponseModel(**params)
            from wellness_env.models import (
                Action, Biomarkers, Goal,
                SleepDuration, ActivityLevel, BedtimeWindow, ExerciseType, ExerciseDuration
            )
            from wellness_env.simulator import compute_biomarker_changes
            import random

            sleep_map = {
                "less_than_6h": SleepDuration.VERY_SHORT, "6_to_7h": SleepDuration.SHORT,
                "7_to_8h": SleepDuration.OPTIMAL_LOW, "8_to_9h": SleepDuration.OPTIMAL_HIGH,
                "more_than_9h": SleepDuration.LONG,
            }
            rec_sleep = sleep_map.get(rec_dict.get("sleep", "7_to_8h"), SleepDuration.OPTIMAL_LOW)
            rec_activity = ActivityLevel(rec_dict.get("activity", "moderate_activity"))

            def _sv(key, default):
                return (curr_sync_dict.get(key) or default) if curr_sync_dict else default

            current_bio = Biomarkers(
                resting_hr=_sv("resting_hr", 70), hrv=_sv("hrv_rmssd", 45),
                sleep_score=_sv("sleep_score", 70), stress_avg=_sv("stress_avg", 50),
                body_battery=_sv("recovery_score", 50),
                sleep_stage_quality=_sv("sleep_stage_quality", 35),
                vo2_max=_sv("vo2_max", 40),
            )
            persona = PersonaConfig(
                name=f"user_{user_id}", compliance_rate=1.0, goal=Goal.STRESS_MANAGEMENT,
                sleep_default=rec_sleep, activity_default=rec_activity,
                starting_biomarkers=current_bio, response_model=rm,
            )
            action = Action(sleep=rec_sleep, activity=rec_activity,
                            bedtime=BedtimeWindow.OPTIMAL, exercise_type=ExerciseType.NONE,
                            exercise_duration=ExerciseDuration.NONE)
            deltas = compute_biomarker_changes(action, current_bio, persona, [], random.Random(42))
            return deltas.model_dump()

        elif tier in ("ml_model", "nn"):
            result = _get_expected_deltas_from_ml(user_id, rec_dict, curr_sync_dict, history)
            return result if result else {}

        return {}

    db = SessionLocal()
    try:
        all_recs = db.query(Recommendation).filter(
            Recommendation.user_id == user_id,
        ).order_by(Recommendation.rec_date.asc()).all()

        records = []
        fidelity_scores = []

        for rec in all_recs:
            try:
                rec_dt = datetime.strptime(rec.rec_date, "%Y-%m-%d")
                next_day_str = (rec_dt + timedelta(days=1)).strftime("%Y-%m-%d")

                curr_sync = db.query(WearableSync).filter(
                    WearableSync.user_id == user_id,
                    WearableSync.sync_date == rec.rec_date
                ).first()
                next_sync = db.query(WearableSync).filter(
                    WearableSync.user_id == user_id,
                    WearableSync.sync_date == next_day_str
                ).first()

                if not curr_sync or not next_sync:
                    continue  # can't score without both syncs

                curr_dict = {c.name: getattr(curr_sync, c.name) for c in curr_sync.__table__.columns}
                rec_dict = {
                    "sleep": rec.sleep_rec or "7_to_8h",
                    "activity": rec.activity_rec or "moderate_activity",
                }

                try:
                    pred = _forced_deltas(rec_dict, curr_dict)
                except Exception as e:
                    print(f"[BACKTEST] delta error rec {rec.id}: {e}")
                    continue

                # Score against actuals
                actual = {
                    "hrv":               (next_sync.hrv_rmssd or 0) - (curr_sync.hrv_rmssd or 0),
                    "resting_hr":        (next_sync.resting_hr or 0) - (curr_sync.resting_hr or 0),
                    "sleep_score":       (next_sync.sleep_score or 0) - (curr_sync.sleep_score or 0),
                    "stress_avg":        (next_sync.stress_avg or 0) - (curr_sync.stress_avg or 0),
                    "body_battery":      (next_sync.recovery_score or 0) - (curr_sync.recovery_score or 0),
                    "sleep_stage_quality": (next_sync.sleep_stage_quality or 0) - (curr_sync.sleep_stage_quality or 0),
                    "vo2_max":           (next_sync.vo2_max or 0) - (curr_sync.vo2_max or 0),
                }

                pred_map = {
                    "hrv": pred.get("hrv", 0), "resting_hr": pred.get("resting_hr", 0),
                    "sleep_score": pred.get("sleep_score", 0), "stress_avg": pred.get("stress_avg", 0),
                    "body_battery": pred.get("body_battery", 0),
                    "sleep_stage_quality": pred.get("sleep_stage_quality", 0),
                    "vo2_max": pred.get("vo2_max", 0),
                }
                weights = {"hrv": 1, "resting_hr": 1, "sleep_score": 0.5,
                           "stress_avg": 0.5, "body_battery": 0.5,
                           "sleep_stage_quality": 0.5, "vo2_max": 0.3}

                total_error = sum(abs(actual[k] - pred_map[k]) * w for k, w in weights.items())
                magnitude_fidelity = math.exp(-0.02 * total_error)

                dir_correct = dir_total = 0
                for k in pred_map:
                    e, a = pred_map[k], actual[k]
                    if abs(e) > 0.01:
                        dir_total += 1
                        if (e > 0 and a > 0) or (e < 0 and a < 0):
                            dir_correct += 1
                directional_accuracy = (dir_correct / dir_total) if dir_total > 0 else 0.5

                fidelity = round(0.6 * magnitude_fidelity + 0.4 * directional_accuracy, 4)
                fidelity_scores.append(fidelity)

                records.append({
                    "rec_date": rec.rec_date,
                    "sleep_rec": rec.sleep_rec,
                    "activity_rec": rec.activity_rec,
                    "predicted": pred_map,
                    "actual": actual,
                    "fidelity": fidelity,
                })

            except Exception as loop_e:
                print(f"[BACKTEST] warning rec {rec.id}: {loop_e}")

        avg = round(sum(fidelity_scores) / len(fidelity_scores), 4) if fidelity_scores else 0.0
        return {
            "tier": tier,
            "avg_fidelity": avg,
            "n_records": len(records),
            "records": records,
        }
    finally:
        db.close()
