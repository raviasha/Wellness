import os
import json
import torch
import numpy as np
from openai import OpenAI
from backend.database import get_recent_history, get_user_profile, get_custom_goal
from backend.goal_interpreter import GoalProfile, get_phase_modifiers
from rl_training.ppo_lite import ActorCritic
from wellness_env.models import (
    SleepDuration, ActivityLevel, Goal, Action, Biomarkers,
    BedtimeWindow, ExerciseType, ExerciseDuration,
)
from wellness_env.simulator import compute_biomarker_changes
from wellness_env.personas import PersonaConfig, ResponseModel
import random
import datetime

# Paths
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")

def _get_causal_history(user_id, history_dict, window_days=14):
    """
    Summarizes the Action -> Response history for LLM context.
    Shifts inputs T to match Outcomes T+1.
    """
    syncs = history_dict.get("syncs", [])
    logs = history_dict.get("logs", [])
    
    # Map syncs and logs by date for easy lookup
    sync_map = {s["sync_date"]: s for s in syncs}
    log_map = {}
    for l in logs:
        d = l["log_date"]
        if d not in log_map: log_map[d] = []
        log_map[d].append(l)
        
    # Get sorted unique dates
    all_dates = sorted(list(set(sync_map.keys()) | set(log_map.keys())), reverse=True)
    
    causal_ledger = []
    # We look for Action(T) -> Result(T+1)
    # So for each date D (The Result Day), we look for Action on D-1
    for date_str in all_dates[:window_days]:
        try:
            curr_date = datetime.date.fromisoformat(date_str)
            prev_date_str = (curr_date - datetime.timedelta(days=1)).isoformat()
            
            # Result (Today)
            res = sync_map.get(date_str, {})
            
            # Action (Yesterday)
            act_sync = sync_map.get(prev_date_str, {})
            act_logs = log_map.get(prev_date_str, [])
            
            # Nutrition is not tracked in this system — Garmin-only inputs
            # Extract sleep hours from TODAY'S sync (as it happened last night)
            sleep_h = 0
            if res.get("sleep_duration_hours"):
                sleep_h = round(res["sleep_duration_hours"], 1)
            else:
                try:
                    raw = res.get("raw_payload", {})
                    if isinstance(raw, str): raw = json.loads(raw)
                    sleep_obj = (raw or {}).get("sleep", {})
                    dto = sleep_obj.get("dailySleepDTO", {})
                    duration = sleep_obj.get("durationInSeconds") or \
                               dto.get("sleepDurationInSeconds") or \
                               dto.get("sleepTimeSeconds")
                    if duration: sleep_h = round(duration / 3600, 1)
                except: pass


            record = {
                "date": date_str,
                "biological_outcome": {
                    "hrv": res.get("hrv_rmssd"),
                    "rhr": res.get("resting_hr"),
                    "sleep_score": res.get("sleep_score"),
                    "stress_avg": res.get("stress_avg"),
                    "recovery_score": res.get("recovery_score"),
                },
                "previous_day_behavior": {
                    "sleep_hours": sleep_h,
                    "steps": act_sync.get("steps"),
                    "active_minutes": act_sync.get("active_minutes"),
                    "active_calories": act_sync.get("active_calories"),
                }
            }
            causal_ledger.append(record)
        except: continue
        
    return causal_ledger

# Fidelity levels based on data richness
FIDELITY_LEVELS = {
    0: {"label": "Generic",       "color": "#6b7280", "desc": "No personal data yet. Showing evidence-based defaults."},
    1: {"label": "Basic",         "color": "#f59e0b", "desc": "Some data available. Recommendations partially personalized."},
    2: {"label": "Calibrated",    "color": "#6366f1", "desc": "Regression model calibrated. Using your personal response patterns."},
    2.5: {"label": "ML Model",    "color": "#8b5cf6", "desc": "Ridge regression per biomarker trained on your history. ML-predicted expected outcomes."},
    3: {"label": "AI-Optimized",  "color": "#22c55e", "desc": "Neural network trained on your digital twin. Fully personalized."},
}

# Goal-specific heuristic overrides
GOAL_STRATEGIES = {
    "stress_management": {
        "sleep": "8_to_9h", "activity": "light_activity",
        "focus": "Reduce stress and restore parasympathetic balance for recovery."
    },
    "cardiovascular_fitness": {
        "sleep": "7_to_8h", "activity": "vigorous_activity",
        "focus": "Improve cardiovascular markers (lower RHR, higher HRV) through targeted activity."
    },
    "sleep_optimization": {
        "sleep": "8_to_9h", "activity": "moderate_activity",
        "focus": "Optimise sleep score through consistent bedtime and moderate daytime activity."
    },
    "recovery_energy": {
        "sleep": "8_to_9h", "activity": "light_activity",
        "focus": "Maximise body battery and recovery through rest-focused approach."
    },
    "active_living": {
        "sleep": "7_to_8h", "activity": "moderate_activity",
        "focus": "Build sustainable daily activity habits for long-term health improvement."
    },
}


def _get_fidelity_level(user_id, history):
    """Determine recommendation fidelity based on available data and active tier."""
    user_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "ppo_wellness_v2.pt")
    # Legacy fallback for users whose per-user model exists only as v1
    if not os.path.exists(user_model_path):
        user_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "ppo_wellness_lite.pt")
    persona_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "calibrated_persona.json")
    ml_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "outcome_models.json")

    has_syncs = len(history.get("syncs", [])) > 0
    has_logs = len(history.get("logs", [])) > 0
    has_persona = os.path.exists(persona_path)
    has_model = os.path.exists(user_model_path)
    has_ml = os.path.exists(ml_model_path)

    # Check active tier to promote to 2.5 when user explicitly chose ML path
    active_tier = "rules"
    if has_ml or has_model:
        try:
            from backend.maturity_config import get_active_tier
            active_tier = get_active_tier(user_id)
        except Exception:
            pass

    if has_model and active_tier != "ml_model":
        return 3
    elif has_ml and active_tier == "ml_model":
        return 2.5
    elif has_model:
        return 3
    elif has_persona:
        return 2
    elif has_syncs or has_logs:
        return 1
    else:
        return 0


def get_coaching_recommendation(user_id, goal: str = "overall_wellness", force_mode: str = "auto"):
    """
    Orchestrates the Hybrid Inference with progressive personalization:
    Level 0: Generic evidence-based defaults (tailored to goal)
    Level 1: Basic personalization from available data
    Level 2: Calibrated regression model
    Level 3: Full NN-optimized recommendations
    """
    # 1. Pull History
    history = get_recent_history(user_id, limit=30)
    profile = get_user_profile(user_id)
    
    fidelity = _get_fidelity_level(user_id, history)
    
    if force_mode == "llm":
        fidelity = min(fidelity, 1)
    elif force_mode == "nn":
        # Try to force NN. If no model parameters exist yet, gracefully fall back to 1.
        fidelity = 3 if fidelity > 1 else 1

    fidelity_info = FIDELITY_LEVELS[fidelity]
    
    user_name = profile.get("name", "User") if profile else "User"

    # --- Custom Goal Resolution ---
    custom_goal_data = get_custom_goal(user_id)
    goal_profile = None
    if custom_goal_data and custom_goal_data.get("goal_profile"):
        try:
            goal_profile = GoalProfile.from_dict(custom_goal_data["goal_profile"])
            # Recalculate days_to_target since it may have changed
            if goal_profile.target_date:
                from datetime import date as _date
                days = (_date.fromisoformat(goal_profile.target_date) - _date.today()).days
                goal_profile.days_to_target = max(0, days)
                from backend.goal_interpreter import get_periodization_phase
                goal_profile.periodization_phase = get_periodization_phase(goal_profile.days_to_target)
        except Exception as e:
            print(f"[Inference] Error loading custom goal profile: {e}")
            goal_profile = None

    # Use custom goal focus if available, otherwise fall back to preset strategy
    if goal_profile:
        strategy = {
            "sleep": "8_to_9h" if goal_profile.periodization_phase in ("taper", "event_week") else "7_to_8h",
            "activity": get_phase_modifiers(goal_profile.periodization_phase).get("intensity_cap", "moderate_activity"),
            "focus": goal_profile.focus_summary,
        }
    else:
        strategy = GOAL_STRATEGIES.get(goal, GOAL_STRATEGIES["stress_management"])
    
    # Level 0: No data at all
    if fidelity == 0:
        llm_out = _generate_llm_recommendation(user_name, goal, strategy["focus"], None, None, None, fidelity, goal_profile)
        # Compute expected deltas from heuristic strategy via simulator
        heuristic_rec = {"sleep": strategy["sleep"], "activity": strategy["activity"]}
        expected = _get_expected_deltas(user_id, heuristic_rec, None, None)
        long_term = _generate_long_term_impact_text(llm_out["recommendation"], expected, goal)
        return {
            "status": "success",
            "fidelity": fidelity,
            "fidelity_info": fidelity_info,
            "goal": goal,
            "goal_profile": goal_profile.to_dict() if goal_profile else None,
            "recommendation": llm_out["recommendation"],
            "insight": llm_out["insight"],
            "raw_biomarkers": None,
            "expected_deltas": expected,
            "long_term_impact": long_term,
            "data_points": {"syncs": 0, "logs": 0},
        }
    
    # Level 1+: We have some data
    syncs = history.get("syncs", [])
    latest = syncs[0] if syncs else {}
    prev = syncs[1] if len(syncs) > 1 else latest
    
    biomarkers = {
        "hrv": latest.get("hrv_rmssd"),
        "rhr": latest.get("resting_hr"),
        "battery": latest.get("recovery_score"),
        "sleep_score": latest.get("sleep_score"),
        "stress_avg": latest.get("stress_avg"),
        "sleep_stage_quality": latest.get("sleep_stage_quality"),
        "vo2_max": latest.get("vo2_max"),
    }
    
    # Level 1–2: Autonomous LLM determines prescription
    # Fidelity 2 = copula persona exists but no user NN model — LLM prescribes,
    # copula persona is used automatically in _get_expected_deltas for accurate deltas.
    if fidelity <= 2:
        # Get 14-day causal history for deep context
        causal_history = _get_causal_history(user_id, history, window_days=14)
        
        llm_out = _generate_llm_recommendation(user_name, goal, strategy["focus"], None, biomarkers, causal_history, fidelity, goal_profile)
        # Compute expected deltas from the LLM's recommended actions via simulator
        llm_rec = llm_out.get("recommendation", {})
        sim_rec = {
            "sleep": llm_rec.get("sleep", strategy["sleep"]),
            "activity": llm_rec.get("activity", strategy["activity"]),
        }
        expected = _get_expected_deltas(user_id, sim_rec, latest, history)
        long_term = _generate_long_term_impact_text(llm_out["recommendation"], expected, goal)
        return {
            "status": "success",
            "fidelity": fidelity,
            "fidelity_info": fidelity_info,
            "goal": goal,
            "goal_profile": goal_profile.to_dict() if goal_profile else None,
            "recommendation": llm_out["recommendation"],
            "insight": llm_out["insight"],
            "raw_biomarkers": biomarkers,
            "expected_deltas": expected,
            "long_term_impact": long_term,
            "data_points": {"syncs": len(syncs), "logs": len(history.get("logs", []))},
        }
    
    # Level 2-3: NN Forward Pass
    user_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "ppo_wellness_v2.pt")
    if not os.path.exists(user_model_path):
        user_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "ppo_wellness_lite.pt")

    # Build observation vector (17-dim: 3 meta + 7 biomarkers + 7 deltas)
    def _bv(key, default):
        return (latest.get(key) or default) if latest else default

    def _delta(key, default):
        if latest and prev:
            return (latest.get(key) or default) - (prev.get(key) or default)
        return 0.0

    obs = [
        0.5, 0.8, 0.0,
        # 7 biomarkers
        _bv("resting_hr", 70),
        _bv("hrv_rmssd", 45),
        _bv("sleep_score", 70),
        _bv("stress_avg", 50),
        _bv("recovery_score", 50),
        _bv("sleep_stage_quality", 35),
        _bv("vo2_max", 40),
        # 7 deltas
        _delta("resting_hr", 70),
        _delta("hrv_rmssd", 45),
        _delta("sleep_score", 70),
        _delta("stress_avg", 50),
        _delta("recovery_score", 50),
        _delta("sleep_stage_quality", 35),
        _delta("vo2_max", 40),
    ]

    state_dim = 17
    action_dim = 25  # kept for legacy API; ignored by multi-head PPOLite

    # Use global cache to avoid redundant loads and memory spikes
    global _MODEL_CACHE
    if "_MODEL_CACHE" not in globals():
        globals()["_MODEL_CACHE"] = {}

    cache_key = f"user_{user_id}"
    if cache_key in globals()["_MODEL_CACHE"]:
        policy = globals()["_MODEL_CACHE"][cache_key]
    else:
        policy = ActorCritic(state_dim)
        if os.path.exists(user_model_path):
            try:
                policy.load_state_dict(
                    torch.load(user_model_path, map_location=torch.device('cpu'), weights_only=True)
                )
                globals()["_MODEL_CACHE"][cache_key] = policy
            except Exception as e:
                print(f"[Inference Cache] Error loading model: {e}")
        policy.eval()

    with torch.no_grad():
        state_t = torch.FloatTensor(obs)
        features = policy.backbone(state_t)
        action_indices = [
            int(torch.argmax(head(features)).item())
            for head in policy.actor_heads
        ]

    # Decode 5D action: [sleep, bedtime, activity, exercise_type, exercise_duration]
    sleep_options = list(SleepDuration)
    bedtime_options = list(BedtimeWindow)
    activity_options = list(ActivityLevel)
    exercise_type_options = list(ExerciseType)
    exercise_duration_options = list(ExerciseDuration)

    nn_activity = activity_options[action_indices[2]].value

    # NN heads are independent — enforce action-space consistency:
    # rest_day must mean no exercise (the two heads don't know about each other).
    if nn_activity == "rest_day":
        nn_exercise_type = ExerciseType.NONE.value
        nn_exercise_duration = ExerciseDuration.NONE.value
    else:
        nn_exercise_type = exercise_type_options[action_indices[3]].value
        nn_exercise_duration = exercise_duration_options[action_indices[4]].value
        # Non-rest exercise type but duration decoded as none → default to 30–45 min
        if nn_exercise_type != ExerciseType.NONE.value and nn_exercise_duration == ExerciseDuration.NONE.value:
            nn_exercise_duration = ExerciseDuration.MODERATE.value

    nn_rec = {
        "sleep": sleep_options[action_indices[0]].value,
        "bedtime": bedtime_options[action_indices[1]].value,
        "activity": nn_activity,
        "exercise_type": nn_exercise_type,
        "exercise_duration": nn_exercise_duration,
    }
    
    llm_out = _generate_llm_recommendation(user_name, goal, strategy["focus"], nn_rec, biomarkers, None, fidelity, goal_profile)
    
    # Calculate Expected Deltas for Eval
    expected = _get_expected_deltas(user_id, nn_rec, latest, history)

    # Dual-path: if active tier is ml_model, use ML expected deltas as primary
    # and store rules-based deltas as alt; if NN is active and ML exists, run both.
    ml_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "outcome_models.json")
    inference_path = "nn"
    expected_deltas_alt = None

    if os.path.exists(ml_path):
        ml_expected = _get_expected_deltas_from_ml(user_id, nn_rec, latest, history)
        if ml_expected is not None:
            try:
                from backend.maturity_config import get_active_tier
                active_tier = get_active_tier(user_id)
            except Exception:
                active_tier = "nn"

            if active_tier == "ml_model":
                # ML is primary; NN deltas are alt
                inference_path = "ml_model"
                expected_deltas_alt = expected  # rules/NN deltas stored as alt
                expected = ml_expected
            else:
                # NN is primary; ML deltas are alt for comparison
                inference_path = "nn"
                expected_deltas_alt = ml_expected

    long_term = _generate_long_term_impact_text(llm_out["recommendation"], expected, goal)
    
    return {
        "status": "success",
        "fidelity": fidelity,
        "fidelity_info": fidelity_info,
        "goal": goal,
        "goal_profile": goal_profile.to_dict() if goal_profile else None,
        "recommendation": llm_out["recommendation"],
        "insight": llm_out["insight"],
        "raw_biomarkers": biomarkers,
        "expected_deltas": expected,
        "expected_deltas_alt": expected_deltas_alt,
        "inference_path": inference_path,
        "long_term_impact": long_term,
        "data_points": {"syncs": len(syncs), "logs": len(history.get("logs", []))},
    }


def _get_expected_deltas_from_ml(user_id, rec, latest_sync, history_dict):
    """Compute expected deltas using the per-outcome ML Ridge model suite.

    Returns a dict with the same keys as compute_biomarker_changes().model_dump(),
    or None if the suite is not available.
    """
    try:
        from backend.outcome_models import load_outcome_models, predict_from_suite
        from wellness_env.ml_simulator import _build_feature_vector, _OUTCOME_ORDER, _OUTCOME_TO_FIELD
        from wellness_env.models import Action, SleepDuration, ActivityLevel, BedtimeWindow, ExerciseType, ExerciseDuration, SLEEP_HOURS
        import math, numpy as np

        suite = load_outcome_models(user_id)
        if suite is None:
            return None

        # Map rec strings → Action enums
        sleep_map = {
            "less_than_6h": SleepDuration.VERY_SHORT, "6_to_7h": SleepDuration.SHORT,
            "7_to_8h": SleepDuration.OPTIMAL_LOW, "8_to_9h": SleepDuration.OPTIMAL_HIGH,
            "more_than_9h": SleepDuration.LONG,
        }
        bedtime_map = {b.value: b for b in BedtimeWindow}
        exercise_type_map = {e.value: e for e in ExerciseType}
        exercise_duration_map = {d.value: d for d in ExerciseDuration}

        rec_sleep = sleep_map.get(rec.get("sleep", "7_to_8h"), SleepDuration.OPTIMAL_LOW)
        rec_bedtime = bedtime_map.get(rec.get("bedtime", ""), BedtimeWindow.OPTIMAL)
        rec_activity = ActivityLevel(rec.get("activity", "moderate_activity"))
        rec_exercise_type = exercise_type_map.get(rec.get("exercise_type", ""), ExerciseType.NONE)
        rec_exercise_duration = exercise_duration_map.get(rec.get("exercise_duration", ""), ExerciseDuration.NONE)

        # Build an Action with sleep_hours attribute derived from the sleep enum
        sleep_hours = SLEEP_HOURS.get(rec_sleep, 7.5)
        action = Action(
            sleep=rec_sleep,
            bedtime=rec_bedtime,
            activity=rec_activity,
            exercise_type=rec_exercise_type,
            exercise_duration=rec_exercise_duration,
        )
        # Attach sleep_hours so _build_feature_vector can use it
        action.sleep_hours = sleep_hours

        # Extract yesterday's deltas from history
        syncs = (history_dict or {}).get("syncs", [])
        prev_deltas_dict = {name: 0.0 for name in _OUTCOME_ORDER}
        if len(syncs) >= 2:
            t0 = syncs[1]
            t1 = syncs[0]
            prev_deltas_dict = {
                "delta_sleep_score":         (t1.get("sleep_score", 70) or 70) - (t0.get("sleep_score", 70) or 70),
                "delta_hrv_ms":              (t1.get("hrv_rmssd", 45)   or 45) - (t0.get("hrv_rmssd", 45) or 45),
                "delta_rhr_bpm":             (t1.get("resting_hr", 70)  or 70) - (t0.get("resting_hr", 70) or 70),
                "delta_stress":              (t1.get("stress_avg", 50)  or 50) - (t0.get("stress_avg", 50) or 50),
                "delta_body_battery":        (t1.get("recovery_score", 50) or 50) - (t0.get("recovery_score", 50) or 50),
                "delta_sleep_stage_quality": (t1.get("sleep_stage_quality", 35) or 35) - (t0.get("sleep_stage_quality", 35) or 35),
                "delta_vo2_max":             (t1.get("vo2_max", 40) or 40) - (t0.get("vo2_max", 40) or 40),
            }

        # Assume 100% compliance for the recommendation prediction:
        # we are answering "what happens if the user follows this exactly?"
        # Fidelity scoring later captures real-world deviation from this ideal.
        x_vec = _build_feature_vector(action, prev_deltas_dict, compliance_sleep=1.0, compliance_activity=1.0)
        ml_preds = predict_from_suite(suite, x_vec)

        # Map ML outcome names → Biomarker field names
        _ml_to_bio = {
            "delta_sleep_score":         "sleep_score",
            "delta_hrv_ms":              "hrv",
            "delta_rhr_bpm":             "resting_hr",
            "delta_stress":              "stress_avg",
            "delta_body_battery":        "body_battery",
            "delta_sleep_stage_quality": "sleep_stage_quality",
            "delta_vo2_max":             "vo2_max",
        }
        return {bio: round(ml_preds.get(ml, 0.0), 4) for ml, bio in _ml_to_bio.items()}

    except Exception as e:
        print(f"[ML Expected Deltas] Error for user {user_id}: {e}")
        return None


def _get_expected_deltas(user_id, rec, latest_sync, history_list):
    """Run simulator to predict the outcome of the recommendation."""
    try:
        persona_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "calibrated_persona.json")

        # Only use calibrated persona when the active tier is copula or higher.
        # On the Rules tier, always use generic defaults so the score reflects
        # the rules simulator accuracy — not a copula that may have been fitted
        # and then reverted.
        rm = ResponseModel()
        try:
            from backend.maturity_config import get_active_tier
            active_tier = get_active_tier(user_id)
        except Exception:
            active_tier = "rules"

        if active_tier != "rules" and os.path.exists(persona_path):
            with open(persona_path, "r") as f:
                params = json.load(f)
                rm = ResponseModel(**params)
        
        # Map rec strings to enums
        sleep_map = {"less_than_6h": SleepDuration.VERY_SHORT, "6_to_7h": SleepDuration.SHORT, 
                     "7_to_8h": SleepDuration.OPTIMAL_LOW, "8_to_9h": SleepDuration.OPTIMAL_HIGH, 
                     "more_than_9h": SleepDuration.LONG}
        
        activity_val = rec.get("activity", "moderate_activity")
        
        rec_sleep = sleep_map.get(rec["sleep"], SleepDuration.OPTIMAL_LOW)
        rec_activity = ActivityLevel(activity_val)

        bedtime_map = {b.value: b for b in BedtimeWindow}
        exercise_type_map = {e.value: e for e in ExerciseType}
        exercise_duration_map = {d.value: d for d in ExerciseDuration}
        rec_bedtime = bedtime_map.get(rec.get("bedtime", ""), BedtimeWindow.OPTIMAL)
        rec_exercise_type = exercise_type_map.get(rec.get("exercise_type", ""), ExerciseType.NONE)
        rec_exercise_duration = exercise_duration_map.get(rec.get("exercise_duration", ""), ExerciseDuration.NONE)
        
        def _sv(key, default):
            return (latest_sync.get(key) or default) if latest_sync else default

        current_bio = Biomarkers(
            resting_hr=_sv("resting_hr", 70),
            hrv=_sv("hrv_rmssd", 45),
            sleep_score=_sv("sleep_score", 70),
            stress_avg=_sv("stress_avg", 50),
            body_battery=_sv("recovery_score", 50),
            sleep_stage_quality=_sv("sleep_stage_quality", 35),
            vo2_max=_sv("vo2_max", 40),
        )
                
        persona = PersonaConfig(
            name=f"user_{user_id}",
            compliance_rate=1.0,  # predict outcome assuming perfect compliance ("if you follow this")
            goal=Goal.STRESS_MANAGEMENT,
            sleep_default=rec_sleep,
            activity_default=rec_activity,
            starting_biomarkers=current_bio,
            response_model=rm,
        )
        
        action = Action(
            sleep=rec_sleep,
            bedtime=rec_bedtime,
            activity=rec_activity,
            exercise_type=rec_exercise_type,
            exercise_duration=rec_exercise_duration,
        )
        
        sim_history = []
        for s in (history_list or {}).get("syncs", [])[:7]:
            sim_history.append({
                "actual_action": {"activity": s.get("activity", "moderate_activity"), "sleep": "7_to_8h"}
            })

        # Seed is deterministic from rec content only — NOT history length — so
        # recalculating at any point in time produces the same expected delta.
        date_seed = hash(rec.get("sleep", "") + rec.get("activity", "") + str(user_id))
        deltas = compute_biomarker_changes(action, current_bio, persona, sim_history, random.Random(date_seed))
        return deltas.model_dump()
    except Exception as e:
        print(f"[Expected Deltas] Error computing prediction: {e}")
        return {"hrv": 0.0, "resting_hr": 0.0}

def _generate_long_term_impact_text(recommendation, expected_deltas, goal):
    """Generate LLM-based long-term healthspan/lifespan impact text."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        goal_label = goal.replace("_", " ").title()

        prompt = f"""You are a longevity science expert. Based on these daily wellness recommendations and their expected short-term effects, describe the realistic long-term health impact on healthspan and lifespan in 2-3 concise sentences.

Goal: {goal_label}
Recommendation:
- Sleep: {recommendation.get('sleep', 'N/A')}
- Activity: {recommendation.get('activity', 'N/A')}
- Exercise type: {recommendation.get('exercise_type', 'N/A')}
- Exercise duration: {recommendation.get('exercise_duration', 'N/A')}
- Bedtime: {recommendation.get('bedtime', 'N/A')}

Expected daily effects:
- HRV change: {expected_deltas.get('hrv', 0):+.1f} ms
- Resting HR change: {expected_deltas.get('resting_hr', 0):+.1f} bpm
- Sleep score change: {expected_deltas.get('sleep_score', 0):+.1f}
- Stress change: {expected_deltas.get('stress_avg', 0):+.1f}
- Body battery change: {expected_deltas.get('body_battery', 0):+.1f}
- Sleep stage quality change: {expected_deltas.get('sleep_stage_quality', 0):+.1f}
- VO2 max change: {expected_deltas.get('vo2_max', 0):+.2f} mL/kg/min

Focus on concrete healthspan/lifespan outcomes backed by evidence-based longevity research. Be specific about the trajectory these daily habits create over months and years. Keep it to 2-3 impactful sentences."""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Error] Long-term impact: {e}")
        return None

def _generate_llm_recommendation(user_name, goal, goal_focus, nn_rec, biomarkers, logs, fidelity, goal_profile=None):
    """Generates JSON structured prescriptions (Fidelity 1) or synthesizes NN prescriptions (Fidelity 3).
    
    When goal_profile is provided (custom goal), injects sport-specific context
    so the LLM recommends the exact sport and duration from the goal.
    """
    try:
        from backend.goal_interpreter import get_phase_modifiers
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        
        bio_text = "No biometric data available yet." if not biomarkers else (
            f"- Current HRV: {biomarkers.get('hrv', 'N/A')}\n"
            f"        - Resting HR: {biomarkers.get('rhr', 'N/A')}\n"
            f"        - Body Battery: {biomarkers.get('battery', 'N/A')}\n"
            f"        - Sleep Score: {biomarkers.get('sleep_score', 'N/A')}\n"
            f"        - Stress Avg: {biomarkers.get('stress_avg', 'N/A')}\n"
            f"        - Sleep Stage Quality: {biomarkers.get('sleep_stage_quality', 'N/A')}\n"
            f"        - VO2 Max: {biomarkers.get('vo2_max', 'N/A')}"
        )
        
        # Serialize recent causal history for LLM context
        history_text = "Recent 14-Day Causal History (Action at T -> Result at T+1):\n" + json.dumps(logs, indent=2) if logs else "No historical behavioral data available."
        goal_label = goal.replace("_", " ").title()

        # Sport-specific context block for custom goals
        sport_context = ""
        if goal_profile:
            phase_desc = ""
            try:
                phase_mods = get_phase_modifiers(goal_profile.periodization_phase)
                phase_desc = phase_mods.get("description", "")
            except Exception:
                pass
            sport_context = f"""

CUSTOM GOAL CONTEXT:
- User's goal: "{goal_profile.original_text}"
- Recommended primary activity: play {goal_profile.recommended_sport} for {goal_profile.recommended_duration_minutes} minutes
- Supporting exercises: {', '.join(goal_profile.supporting_exercises) if goal_profile.supporting_exercises else 'none specified'}
- Target date: {goal_profile.target_date or 'ongoing'}
- Days to target: {goal_profile.days_to_target if goal_profile.days_to_target is not None else 'N/A'}
- Current phase: {goal_profile.periodization_phase} — {phase_desc}

IMPORTANT: You MUST recommend "{goal_profile.recommended_sport}" as the primary exercise activity in your recommendation.
The exercise_type should map to the closest available enum (e.g., pickleball → cardio, yoga → flexibility).
In your insight text, reference the specific sport "{goal_profile.recommended_sport}" by name."""
        
        if nn_rec is None:
            # Autonomous AI Mode (fidelity 0–2: LLM prescribes from biomarkers)
            sys_msg = f"""You are an elite virtual health coach. You must strictly output valid JSON.
Your goal is to prescribe the daily routine for {user_name} targeting {goal_label}.
Allowed Activity Enums: [rest_day, light_activity, moderate_activity, vigorous_activity, high_intensity]
Allowed Sleep Enums: [less_than_6h, 6_to_7h, 7_to_8h, 8_to_9h, more_than_9h]
Allowed Bedtime Enums: [before_10pm, 10pm_to_11pm, 11pm_to_midnight, midnight_to_1am, after_1am]
Allowed ExerciseType Enums: [rest, cardio, strength, flexibility, hiit]
Allowed ExerciseDuration Enums: [none, 15_to_30min, 30_to_45min, 45_to_60min, over_60min]

CRITICAL CONSTRAINT: If activity is "rest_day", you MUST set exercise_type to "rest" and exercise_duration to "none". A rest day means NO exercise.
CRITICAL CONSTRAINT: Never prescribe less than 7 hours of sleep (do NOT use "less_than_6h" or "6_to_7h"). Minimum is "7_to_8h".
IMPORTANT: This system tracks ONLY Garmin wearable data. Nutrition is NOT tracked and must NEVER be mentioned.
The only behavioral inputs are: sleep hours, steps, active minutes, and active calories.

Context:
Biometrics Today: {bio_text}
{history_text}
Strategy: {goal_focus}
{sport_context}

DATA GAP INSTRUCTIONS:
- If Garmin sync data is missing for a day (null values), briefly note it only if it meaningfully affects your analysis.
- Do NOT mention nutrition, meals, macros, or diet — they are not part of this system.

CAUSAL ANALYSIS:
- Analyze the history. Look for correlations between Garmin inputs and next-day biomarker responses.
  (e.g., 'Higher active minutes yesterday → lower stress today', 'More sleep → improved HRV')
- Based on these patterns, prescribe the best physiological routine for tomorrow.
- Output JSON exactly like this:
{{
  "recommendation": {{ "sleep": "<enum>", "bedtime": "<enum>", "activity": "<enum>", "exercise_type": "<enum>", "exercise_duration": "<enum>"{', "sport_detail": "' + goal_profile.recommended_sport + ' for ' + str(goal_profile.recommended_duration_minutes) + ' min"' if goal_profile else ''} }},
  "insight": "A 2-sentence analytical coaching insight explaining which Garmin-tracked pattern drove this prescription.{' Reference the specific sport ' + goal_profile.recommended_sport + ' by name.' if goal_profile else ''}"
}}"""
        else:
            # Synthesizer Mode
            sys_msg = f"""You are an elite virtual health coach. You must strictly output valid JSON.
A mathematical Neural Network has prescribed the following optimal routine for {user_name} to target {goal_label}:
- Sleep: {nn_rec['sleep']}
- Activity: {nn_rec['activity']}
- Exercise type: {nn_rec.get('exercise_type', 'N/A')}
- Exercise duration: {nn_rec.get('exercise_duration', 'N/A')}
- Target bedtime: {nn_rec.get('bedtime', 'N/A')}

IMPORTANT: This system tracks ONLY Garmin wearable data. Nutrition is NOT tracked and must NEVER be mentioned.

Context:
Biometrics: {bio_text}
{history_text}
Strategy: {goal_focus}
{sport_context}

You MUST NOT change the Neural Network's prescription. Simply synthesize an empathetic explanation grounded in Garmin biomarkers (HRV, RHR, sleep score, sleep stage quality, body battery, stress, VO2 max where available). Output JSON exactly like this:
{{
  "recommendation": {{ "sleep": "{nn_rec['sleep']}", "activity": "{nn_rec['activity']}", "exercise_type": "{nn_rec.get('exercise_type','none')}", "exercise_duration": "{nn_rec.get('exercise_duration','none')}", "bedtime": "{nn_rec.get('bedtime','optimal')}"{', "sport_detail": "' + goal_profile.recommended_sport + ' for ' + str(goal_profile.recommended_duration_minutes) + ' min"' if goal_profile else ''} }},
  "insight": "A 2-sentence coaching insight explaining the physiological rationale behind this plan using only Garmin-tracked signals.{' Reference ' + goal_profile.recommended_sport + ' specifically.' if goal_profile else ''}"
}}"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": sys_msg}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[LLM Error] Generating JSON Recommendation: {e}")
        if nn_rec:
            return {"recommendation": nn_rec, "insight": f"Focus on {nn_rec.get('activity', 'moderate activity')} with optimized sleep. {goal_focus}"}
        else:
            return {
                "recommendation": {"sleep": "7_to_8h", "activity": "moderate_activity", "exercise_type": "cardio", "exercise_duration": "30_to_45min", "bedtime": "10pm_to_11pm"},
                "insight": "System offline. Defaulting to an active recovery baseline."
            }
