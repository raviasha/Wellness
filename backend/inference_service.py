import os
import json
import torch
import numpy as np
from openai import OpenAI
from backend.database import get_recent_history, get_user_profile
from rl_training.ppo_lite import ActorCritic
from wellness_env.models import SleepDuration, ExerciseType, NutritionType, Goal, Action, Biomarkers
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
            
            # Summary of actions
            meals = []
            macros = {"protein": 0, "carbs": 0, "fat": 0}
            quality_scores = []
            _type_to_score = {"high_protein": 8, "balanced": 7, "high_carb": 5, "processed": 3, "skipped": 0}
            for l in act_logs:
                if l["log_type"] == "food":
                    # NOTE: calorie values are LLM-estimated and unreliable with sparse food descriptions.
                    # They are excluded from the causal context sent to the recommendation LLM.
                    try:
                        raw = json.loads(l["raw_input"]) if isinstance(l["raw_input"], str) else l["raw_input"]
                        if raw.get("parsed"):
                            p = raw["parsed"]
                            macros["protein"] += p.get("protein_g", 0)
                            macros["carbs"] += p.get("carbs_g", 0)
                            macros["fat"] += p.get("fat_g", 0)
                            qs = p.get("quality_score") or _type_to_score.get(p.get("nutrition_type", ""), None)
                            if qs is not None:
                                quality_scores.append(qs)
                        if raw.get("text"):
                            meals.append(raw["text"])
                    except: pass

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
                    "active_calories": act_sync.get("active_calories"),
                    "intensity_min": act_sync.get("active_minutes"),
                    "sleep_hours": sleep_h,
                    "macros": macros,
                    "meals": meals,
                    "nutrition_quality": round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else None
                }
            }
            causal_ledger.append(record)
        except: continue
        
    return causal_ledger

# Fidelity levels based on data richness
FIDELITY_LEVELS = {
    0: {"label": "Generic", "color": "#6b7280", "desc": "No personal data yet. Showing evidence-based defaults."},
    1: {"label": "Basic", "color": "#f59e0b", "desc": "Some data available. Recommendations partially personalized."},
    2: {"label": "Calibrated", "color": "#6366f1", "desc": "Regression model calibrated. Using your personal response patterns."},
    3: {"label": "AI-Optimized", "color": "#22c55e", "desc": "Neural network trained on your digital twin. Fully personalized."},
}

# Goal-specific heuristic overrides
GOAL_STRATEGIES = {
    "overall_wellness": {
        "sleep": "7-8 hours", "exercise": "moderate_cardio", "nutrition": "balanced",
        "focus": "Maintain consistency across all biomarkers for baseline wellness."
    },
    "longevity_optimization": {
        "sleep": "8-9 hours", "exercise": "moderate_cardio", "nutrition": "balanced",
        "focus": "Optimize for lowest sustainable Resting HR and maximum HRV stability."
    },
    "metabolic_health": {
        "sleep": "7-8 hours", "exercise": "hiit", "nutrition": "high_protein",
        "focus": "Improve insulin sensitivity through high-intensity output and caloric management."
    },
    "recovery_focus": {
        "sleep": "8-9 hours", "exercise": "yoga", "nutrition": "balanced",
        "focus": "Prioritize Parasympathetic Nervous System dominance to maximize HRV rebound."
    },
    "muscle_preservation": {
        "sleep": "8-9 hours", "exercise": "strength", "nutrition": "high_protein",
        "focus": "Optimize protein timing and deep sleep to maximize muscle protein synthesis."
    },
}


def _get_fidelity_level(user_id, history):
    """Determine recommendation fidelity based on available data."""
    user_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "ppo_wellness_lite.pt")
    persona_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "calibrated_persona.json")
    
    has_syncs = len(history.get("syncs", [])) > 0
    has_logs = len(history.get("logs", [])) > 0
    has_persona = os.path.exists(persona_path)
    has_model = os.path.exists(user_model_path)
    
    if has_model:
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
    strategy = GOAL_STRATEGIES.get(goal, GOAL_STRATEGIES["overall_wellness"])
    
    # Level 0: No data at all
    if fidelity == 0:
        llm_out = _generate_llm_recommendation(user_name, goal, strategy["focus"], None, None, None, fidelity)
        # Compute expected deltas from heuristic strategy via simulator
        heuristic_rec = {"sleep": strategy["sleep"], "exercise": strategy["exercise"], "nutrition": strategy["nutrition"]}
        expected = _get_expected_deltas(user_id, heuristic_rec, None, None)
        long_term = _generate_long_term_impact_text(llm_out["recommendation"], expected, goal)
        return {
            "status": "success",
            "fidelity": fidelity,
            "fidelity_info": fidelity_info,
            "goal": goal,
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
    }
    
    # Level 1: Autonomous LLM determines prescription
    if fidelity <= 1:
        # Get 14-day causal history for deep context
        causal_history = _get_causal_history(user_id, history, window_days=14)
        
        llm_out = _generate_llm_recommendation(user_name, goal, strategy["focus"], None, biomarkers, causal_history, fidelity)
        # Compute expected deltas from the LLM's recommended actions via simulator
        llm_rec = llm_out.get("recommendation", {})
        sim_rec = {
            "sleep": llm_rec.get("sleep", strategy["sleep"]),
            "exercise": llm_rec.get("exercise", strategy["exercise"]),
            "nutrition": llm_rec.get("nutrition", strategy["nutrition"]),
        }
        expected = _get_expected_deltas(user_id, sim_rec, latest, history)
        long_term = _generate_long_term_impact_text(llm_out["recommendation"], expected, goal)
        return {
            "status": "success",
            "fidelity": fidelity,
            "fidelity_info": fidelity_info,
            "goal": goal,
            "recommendation": llm_out["recommendation"],
            "insight": llm_out["insight"],
            "raw_biomarkers": biomarkers,
            "expected_deltas": expected,
            "long_term_impact": long_term,
            "data_points": {"syncs": len(syncs), "logs": len(history.get("logs", []))},
        }
    
    # Level 2-3: NN Forward Pass
    user_model_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "ppo_wellness_lite.pt")
    
    # Build observation vector
    obs = [
        0.5, 0.8, 0.0,
        latest.get("resting_hr", 70) if latest else 70, 
        latest.get("hrv_rmssd", 45) if latest else 45,
        35.0, 25.0, 55.0, 75.0, 50.0,
        latest.get("recovery_score", 50) if latest else 50,
        (latest.get("resting_hr", 70) - prev.get("resting_hr", 70)) if (latest and prev) else 0,
        (latest.get("hrv_rmssd", 45) - prev.get("hrv_rmssd", 45)) if (latest and prev) else 0,
        0, 0, 0, 0, 0, 0
    ]
    
    state_dim = 19
    action_dim = 150
    
    # Use global cache to avoid redundant loads and memory spikes
    global _MODEL_CACHE
    if "_MODEL_CACHE" not in globals():
        globals()["_MODEL_CACHE"] = {}
        
    cache_key = f"user_{user_id}"
    if cache_key in globals()["_MODEL_CACHE"]:
        policy = globals()["_MODEL_CACHE"][cache_key]
    else:
        policy = ActorCritic(state_dim, action_dim)
        if os.path.exists(user_model_path):
            try:
                policy.load_state_dict(torch.load(user_model_path, map_location=torch.device('cpu'), weights_only=True))
                globals()["_MODEL_CACHE"][cache_key] = policy
            except Exception as e:
                print(f"[Inference Cache] Error loading model: {e}")
        policy.eval()
    
    with torch.no_grad():
        state_t = torch.FloatTensor(obs)
        action_probs = policy.actor(state_t)
        action_idx = torch.argmax(action_probs).item()

    # Decode action
    sleep_options = list(SleepDuration)
    exercise_options = list(ExerciseType)
    nutrition_options = list(NutritionType)
    
    sl_idx = action_idx % len(sleep_options)
    rem = action_idx // len(sleep_options)
    ex_idx = rem % len(exercise_options)
    nu_idx = rem // len(exercise_options)
    
    nn_rec = {
        "sleep": sleep_options[sl_idx].value,
        "exercise": exercise_options[ex_idx].value,
        "nutrition": nutrition_options[nu_idx].value,
    }
    
    llm_out = _generate_llm_recommendation(user_name, goal, strategy["focus"], nn_rec, biomarkers, None, fidelity)
    
    # Calculate Expected Deltas for Eval
    expected = _get_expected_deltas(user_id, nn_rec, latest, history)
    long_term = _generate_long_term_impact_text(llm_out["recommendation"], expected, goal)
    
    return {
        "status": "success",
        "fidelity": fidelity,
        "fidelity_info": fidelity_info,
        "goal": goal,
        "recommendation": llm_out["recommendation"],
        "insight": llm_out["insight"],
        "raw_biomarkers": biomarkers,
        "expected_deltas": expected,
        "long_term_impact": long_term,
        "data_points": {"syncs": len(syncs), "logs": len(history.get("logs", []))},
    }


def _get_expected_deltas(user_id, rec, latest_sync, history_list):
    """Run simulator to predict the outcome of the recommendation."""
    try:
        persona_path = os.path.join(MODELS_ROOT, f"user_{user_id}", "calibrated_persona.json")
        
        # Defaults if no persona exists
        rm = ResponseModel()
        if os.path.exists(persona_path):
            with open(persona_path, "r") as f:
                params = json.load(f)
                rm = ResponseModel(**params)
        
        # Map rec strings to enums for persona defaults
        sleep_map = {"less_than_6h": SleepDuration.VERY_SHORT, "6-7 hours": SleepDuration.SHORT, 
                     "7-8 hours": SleepDuration.OPTIMAL_LOW, "8-9 hours": SleepDuration.OPTIMAL_HIGH, 
                     "more_than_9h": SleepDuration.LONG}
        
        exercise_val = rec.get("exercise", "none")
        exercise_map = {"yoga": "yoga", "light_cardio": "light_cardio", "moderate_cardio": "moderate_cardio",
                        "hiit": "hiit", "strength": "strength", "none": "none"}
        exercise_val = exercise_map.get(exercise_val, exercise_val)
        
        rec_sleep = sleep_map.get(rec["sleep"], SleepDuration.OPTIMAL_LOW)
        rec_exercise = ExerciseType(exercise_val)
        rec_nutrition = NutritionType(rec.get("nutrition", "balanced"))
        
        current_bio = Biomarkers(
            resting_hr=(latest_sync.get("resting_hr") or latest_sync.get("rhr") or 70) if latest_sync else 70,
            hrv=(latest_sync.get("hrv_rmssd") or latest_sync.get("hrv") or latest_sync.get("hrv_avg") or 45) if latest_sync else 45,
            vo2_max=(latest_sync.get("vo2_max") or 35) if latest_sync else 35,
            body_fat_pct=(latest_sync.get("body_fat_pct") or 25) if latest_sync else 25,
            lean_mass_kg=(latest_sync.get("lean_mass_kg") or 55) if latest_sync else 55,
            sleep_efficiency=(latest_sync.get("sleep_efficiency") or latest_sync.get("sleep_score") or 80) if latest_sync else 80,
            cortisol_proxy=(latest_sync.get("cortisol_proxy") or latest_sync.get("stress_avg") or 40) if latest_sync else 40,
            energy_level=(latest_sync.get("energy_level") or latest_sync.get("body_battery") or 50) if latest_sync else 50
        )
                
        persona = PersonaConfig(
            name=f"user_{user_id}",
            compliance_rate=0.8,
            goal=Goal.OVERALL_WELLNESS,
            sleep_default=rec_sleep,
            exercise_default=rec_exercise,
            nutrition_default=rec_nutrition,
            starting_biomarkers=current_bio,
            response_model=rm,
        )
        
        action = Action(
            sleep=rec_sleep,
            exercise=rec_exercise,
            nutrition=rec_nutrition,
        )
        
        sim_history = []
        for s in (history_list or {}).get("syncs", [])[:7]: 
            sim_history.append({
                "actual_action": {"exercise": s.get("exercise", "none"), "sleep": "7_to_8h"}
            })
            
        # Use date-based seed so predictions vary day-to-day (not fixed noise)
        date_seed = hash(rec.get("sleep", "") + rec.get("exercise", "") + str(user_id) + str(len(sim_history)))
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
- Exercise: {recommendation.get('exercise', 'N/A')}
- Nutrition: {recommendation.get('nutrition', 'N/A')}

Expected daily effects:
- HRV change: {expected_deltas.get('hrv', 0):+.1f} ms
- Resting HR change: {expected_deltas.get('resting_hr', 0):+.1f} bpm
- Sleep efficiency change: {expected_deltas.get('sleep_efficiency', 0):+.1f}
- Cortisol/Stress change: {expected_deltas.get('cortisol_proxy', 0):+.1f}
- Body fat change: {expected_deltas.get('body_fat_pct', 0):+.4f}%
- Energy change: {expected_deltas.get('energy_level', 0):+.1f}

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

def _generate_llm_recommendation(user_name, goal, goal_focus, nn_rec, biomarkers, logs, fidelity):
    """Generates JSON structured prescriptions (Fidelity 1) or synthesizes NN prescriptions (Fidelity 3)."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        
        bio_text = "No biometric data available yet." if not biomarkers else f"""
        - Current HRV: {biomarkers.get('hrv', 'N/A')}
        - Resting HR: {biomarkers.get('rhr', 'N/A')}
        - Body Battery: {biomarkers.get('battery', 'N/A')}"""
        
        # Serialize recent causal history for LLM context
        history_text = "Recent 14-Day Causal History (Action at T -> Result at T+1):\n" + json.dumps(logs, indent=2) if logs else "No historical behavioral data available."
        goal_label = goal.replace("_", " ").title()
        
        if fidelity <= 1:
            # Autonomous AI Mode
            sys_msg = f"""You are an elite virtual health coach. You must strictly output valid JSON.
Your goal is to prescribe the daily routine for {user_name} targeting {goal_label}.
Allowed Exercise Enums: [none, light_cardio, moderate_cardio, hiit, strength, yoga]
Allowed Sleep Enums: [less_than_6h, 6-7 hours, 7-8 hours, 8-9 hours, more_than_9h]
Allowed Nutrition Enums: [balanced, high_protein, low_carb, keto, fasting]

Context:
Biometrics Today: {bio_text}
{history_text}
Strategy: {goal_focus}

CRITICAL DATA GAP INSTRUCTIONS:
1. If you see 'None', null, or empty lists in the causal history (e.g., missing meals, 0 sleep hours, or missing sync dates), you MUST:
   - Acknowledge these specific gaps in your 'insight'.
   - Explain how these gaps reduce the certainty of your causal analysis for that specific period.
   - Encourage the user to backfill the data to improve prediction fidelity.

CAUSAL ANALYSIS:
- Analyze the 14-day history. Look for correlations! (e.g., 'If protein is low, HRV drops'). 
- Based on these patterns, prescribe the best physiological routine for tomorrow. 
- Output JSON exactly like this:
{{
  "recommendation": {{ "sleep": "<enum>", "exercise": "<enum>", "nutrition": "<enum>" }},
  "insight": "A 2-sentence analytical coaching insight explaining exactly which historical pattern led to this plan, while explicitly addressing any data gaps if present."
}}"""
        else:
            # Synthesizer Mode
            sys_msg = f"""You are an elite virtual health coach. You must strictly output valid JSON.
A mathematical Neural Network has prescribed the following optimal routine for {user_name} to target {goal_label}:
- Sleep: {nn_rec['sleep']}
- Exercise: {nn_rec['exercise']}
- Nutrition: {nn_rec['nutrition']}

Context:
Biometrics: {bio_text}
{history_text}
Strategy: {goal_focus}

CRITICAL DATA GAP INSTRUCTIONS:
1. If the input history has significant gaps, acknowledge them in your insight as a reason why the NN's prediction may have lower confidence for those days.

You MUST NOT change the Neural Network's prescription. Simply synthesize an empathetic explanation. Output JSON exactly like this:
{{
  "recommendation": {{ "sleep": "{nn_rec['sleep']}", "exercise": "{nn_rec['exercise']}", "nutrition": "{nn_rec['nutrition']}" }},
  "insight": "A 2-sentence coaching insight explaining the physiological why behind this plan and addressing data gaps if relevant."
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
            return {"recommendation": nn_rec, "insight": f"Focus on {nn_rec['exercise']} with {nn_rec['nutrition']} nutrition. {goal_focus}"}
        else:
            return {
                "recommendation": {"sleep": "7-8 hours", "exercise": "yoga", "nutrition": "balanced"}, 
                "insight": "System offline. Defaulting to an active recovery baseline."
            }
