#!/usr/bin/env python3
"""Baseline LLM agent for the Outcome-Based Wellness Simulator.

Runs all 3 tasks sequentially, printing structured stdout.
Uses OpenAI-compatible API via env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from wellness_env import WellnessEnv, Action, Observation
from wellness_env.models import SleepDuration, ExerciseType, NutritionType, Goal
from wellness_env.payoff import _linear_slope, _stddev

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)

SYSTEM_PROMPT = """\
You are a wellness decision system. Follow these EXACT rules to choose actions.

## SLEEP RULES (pick ONE):
1. If cortisol > 65 OR energy < 30 → "8_to_9h"
2. If compliance_rate <= 0.3 → "7_to_8h"
3. If goal is stress_management → "8_to_9h"
4. If goal is athletic_performance AND HRV dropped (hrv_delta < -2) → "8_to_9h"
5. Default → "7_to_8h"

## EXERCISE RULES (pick ONE):
1. If cortisol > 65 OR energy < 30 → "yoga"
2. If compliance_rate <= 0.3 → "light_cardio"
3. If goal is athletic_performance AND HRV dropped (hrv_delta < -2) → "yoga"
4. If goal is athletic_performance → alternate "hiit" on odd days, "strength" on even days
5. If goal is muscle_gain → "strength"
6. If goal is stress_management → "yoga"
7. If goal is weight_loss → "moderate_cardio" if energy > 50, else "light_cardio"
8. Default → "moderate_cardio"

## NUTRITION RULES (pick ONE):
1. If goal is athletic_performance OR muscle_gain → "high_protein"
2. If goal is stress_management → "balanced"
3. If goal is weight_loss → "balanced"
4. Default → "balanced"

## OUTPUT: JSON only, no markdown:
{"sleep": "...", "exercise": "...", "nutrition": "..."}
"""


def build_user_message(obs: Observation, step_num: int) -> str:
    """Build minimal user message with only decision-relevant inputs."""
    b = obs.biomarkers
    d = obs.deltas
    return (
        f"day={step_num} total={obs.total_days} goal={obs.goal.value} "
        f"compliance={obs.compliance_rate}\n"
        f"cortisol={b.cortisol_proxy:.1f} energy={b.energy_level:.1f} "
        f"hrv_delta={d.hrv:+.2f}"
    )


def call_llm(obs: Observation, step_num: int, history_actions: list[dict]) -> Action:
    """Call the LLM and parse the response into an Action."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(obs, step_num)},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=80,
        )

        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        data = json.loads(content)
        return Action(
            sleep=SleepDuration(data["sleep"]),
            exercise=ExerciseType(data["exercise"]),
            nutrition=NutritionType(data["nutrition"]),
        )
    except Exception:
        return _fallback_action(obs)


def _fallback_action(obs: Observation) -> Action:
    """Rule-based fallback when LLM is unavailable.

    Uses biomarker values and day number to make decisions.
    """
    b = obs.biomarkers
    d = obs.deltas
    goal = obs.goal
    day = obs.day

    # Recovery priority: if cortisol is high or energy is low
    if b.cortisol_proxy > 65 or b.energy_level < 30:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.YOGA,
            nutrition=NutritionType.BALANCED,
        )

    # Low compliance: moderate, achievable changes
    if obs.compliance_rate <= 0.3:
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.LIGHT_CARDIO,
            nutrition=NutritionType.BALANCED,
        )

    # Goal-specific strategies
    if goal == Goal.WEIGHT_LOSS:
        exercise = ExerciseType.MODERATE_CARDIO if b.energy_level > 50 else ExerciseType.LIGHT_CARDIO
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=exercise,
            nutrition=NutritionType.BALANCED,
        )
    elif goal == Goal.MUSCLE_GAIN:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.STRENGTH,
            nutrition=NutritionType.HIGH_PROTEIN,
        )
    elif goal == Goal.ATHLETIC_PERFORMANCE:
        if d.hrv < -2:  # HRV dropped → need recovery
            return Action(
                sleep=SleepDuration.OPTIMAL_HIGH,
                exercise=ExerciseType.YOGA,
                nutrition=NutritionType.HIGH_PROTEIN,
            )
        # Alternate intense/recovery to avoid overtraining (threshold=2-3)
        if day % 3 == 0:
            exercise = ExerciseType.YOGA  # recovery day
        elif day % 2 == 1:
            exercise = ExerciseType.HIIT
        else:
            exercise = ExerciseType.STRENGTH
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=exercise,
            nutrition=NutritionType.HIGH_PROTEIN,
        )
    elif goal == Goal.STRESS_MANAGEMENT:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.YOGA,
            nutrition=NutritionType.BALANCED,
        )
    elif goal == Goal.LONGEVITY:
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.MODERATE_CARDIO,
            nutrition=NutritionType.BALANCED,
        )
    else:  # overall_wellness
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.MODERATE_CARDIO,
            nutrition=NutritionType.BALANCED,
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

TASKS = ["single_goal", "multi_outcome", "resistant_adaptation"]


def run_task(env: WellnessEnv, task_name: str, use_llm: bool = True) -> None:
    """Run a single task and print structured stdout."""
    obs = env.reset(task_name)
    config = env._config
    persona = env._persona

    print(
        f"[START] task={task_name} env=wellness-outcome model={MODEL_NAME} "
        f"persona={persona.name} goal={persona.goal.value} "
        f"compliance={persona.compliance_rate} days={config['total_days']}"
    )

    rewards: list[float] = []
    history_actions: list[dict] = []

    for step_num in range(1, config["total_days"] + 1):
        try:
            if use_llm:
                action = call_llm(obs, step_num, history_actions)
            else:
                action = _fallback_action(obs)

            result = env.step(action)

            action_dict = action.model_dump()
            actual_dict = result.info["actual_action"]
            complied = result.info["complied"]
            reward_val = result.reward.total
            rewards.append(reward_val)

            history_actions.append({
                "step": step_num,
                "action": action_dict,
                "actual": actual_dict,
                "complied": complied,
                "reward": reward_val,
            })

            # Compact biomarker snapshot
            b = result.observation.biomarkers
            bio_str = (
                f"rhr={b.resting_hr},hrv={b.hrv},vo2={b.vo2_max},"
                f"bf={b.body_fat_pct},lm={b.lean_mass_kg},"
                f"se={b.sleep_efficiency},cortisol={b.cortisol_proxy},"
                f"energy={b.energy_level}"
            )

            # Compact delta snapshot
            dl = result.observation.deltas
            delta_str = (
                f"rhr={dl.resting_hr:+.3f},hrv={dl.hrv:+.3f},vo2={dl.vo2_max:+.4f},"
                f"bf={dl.body_fat_pct:+.4f},lm={dl.lean_mass_kg:+.4f},"
                f"se={dl.sleep_efficiency:+.3f},cortisol={dl.cortisol_proxy:+.3f},"
                f"energy={dl.energy_level:+.3f}"
            )

            print(
                f"[STEP] step={step_num} "
                f"action={json.dumps(action_dict)} "
                f"reward={reward_val:.2f} "
                f"done={str(result.done).lower()} "
                f"error=null "
                f"actual={json.dumps(actual_dict)} "
                f"complied={str(complied).lower()} "
                f"biomarkers={{{bio_str}}} "
                f"deltas={{{delta_str}}}"
            )

            obs = result.observation
        except Exception as e:
            error_msg = str(e).replace("\n", " ")[:200]
            print(
                f"[STEP] step={step_num} "
                f"action=null "
                f"reward=0.00 "
                f"done=false "
                f"error=\"{error_msg}\""
            )

    # End summary
    score = env.grade()
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    reward_trend = _linear_slope(rewards) if len(rewards) >= 2 else 0.0
    reward_std = _stddev(rewards)
    actual_compliance = (
        sum(1 for h in history_actions if h["complied"]) / len(history_actions)
        if history_actions
        else 0.0
    )
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # Biomarker summary: starting vs ending
    st = env.state()
    bio_start = env._history[0]["biomarkers"] if env._history else {}
    bio_end = st.biomarkers.model_dump() if st.biomarkers else {}

    print(
        f"[END] success=true steps={len(rewards)} "
        f"score={score:.4f} "
        f"rewards={rewards_str} "
        f"task={task_name} avg_reward={avg_reward:.2f} "
        f"reward_trend={reward_trend:+.2f} reward_stddev={reward_std:.2f} "
        f"compliance_rate_actual={actual_compliance:.2f}"
    )
    print()


def main():
    use_llm = bool(OPENAI_API_KEY)
    if not use_llm:
        print(
            "# WARNING: No API key found. Using rule-based fallback agent.",
            file=sys.stderr,
        )

    seed = int(os.environ.get("SEED", "42"))
    env = WellnessEnv(seed=seed)

    for task_name in TASKS:
        try:
            run_task(env, task_name, use_llm=use_llm)
        except Exception:
            print(
                f"[END] success=false steps=0 score=0.0000 rewards= task={task_name}"
            )
            traceback.print_exc(file=sys.stderr)
            print()


if __name__ == "__main__":
    main()
