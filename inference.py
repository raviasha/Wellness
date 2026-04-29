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
from wellness_env.models import SleepDuration, ActivityLevel, Goal
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
Actions are: sleep duration + activity level. No nutrition input.

## SLEEP RULES (pick ONE):
1. If stress_avg > 65 OR body_battery < 30 → "8_to_9h"
2. If compliance_rate <= 0.3 → "7_to_8h"
3. If goal is stress_management → "8_to_9h"
4. If goal is cardiovascular_fitness AND HRV dropped (hrv_delta < -2) → "8_to_9h"
5. Default → "7_to_8h"

## ACTIVITY RULES (pick ONE):
1. If stress_avg > 65 OR body_battery < 30 → "rest_day"
2. If compliance_rate <= 0.3 → "light_activity"
3. If goal is cardiovascular_fitness AND HRV dropped (hrv_delta < -2) → "rest_day"
4. If goal is cardiovascular_fitness → alternate "vigorous_activity" on odd days, "moderate_activity" on even days
5. If goal is stress_management → "light_activity"
6. If goal is active_living → "moderate_activity" if body_battery > 50, else "light_activity"
7. If goal is sleep_optimization → "moderate_activity"
8. Default → "moderate_activity"

## OUTPUT: JSON only, no markdown:
{"sleep": "...", "activity": "..."}
"""


def build_user_message(obs: Observation, step_num: int) -> str:
    """Build minimal user message with only decision-relevant inputs."""
    b = obs.biomarkers
    d = obs.deltas
    return (
        f"day={step_num} total={obs.total_days} goal={obs.goal.value} "
        f"compliance={obs.compliance_rate}\n"
        f"stress_avg={b.stress_avg:.1f} body_battery={b.body_battery:.1f} "
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
            activity=ActivityLevel(data["activity"]),
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

    # Recovery priority: if stress is high or battery is low
    if b.stress_avg > 65 or b.body_battery < 30:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            activity=ActivityLevel.REST_DAY,
        )

    # Low compliance: moderate, achievable changes
    if obs.compliance_rate <= 0.3:
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            activity=ActivityLevel.LIGHT_ACTIVITY,
        )

    # Goal-specific strategies
    if goal == Goal.ACTIVE_LIVING:
        activity = ActivityLevel.MODERATE_ACTIVITY if b.body_battery > 50 else ActivityLevel.LIGHT_ACTIVITY
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            activity=activity,
        )
    elif goal == Goal.CARDIOVASCULAR_FITNESS:
        if d.hrv < -2:  # HRV dropped → need recovery
            return Action(
                sleep=SleepDuration.OPTIMAL_HIGH,
                activity=ActivityLevel.REST_DAY,
            )
        # Alternate intense/recovery to avoid overtraining
        if day % 3 == 0:
            activity = ActivityLevel.REST_DAY
        elif day % 2 == 1:
            activity = ActivityLevel.HIGH_INTENSITY
        else:
            activity = ActivityLevel.VIGOROUS_ACTIVITY
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            activity=activity,
        )
    elif goal == Goal.STRESS_MANAGEMENT:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            activity=ActivityLevel.LIGHT_ACTIVITY,
        )
    elif goal == Goal.SLEEP_OPTIMIZATION:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            activity=ActivityLevel.MODERATE_ACTIVITY,
        )
    elif goal == Goal.RECOVERY_ENERGY:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            activity=ActivityLevel.LIGHT_ACTIVITY,
        )
    else:
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            activity=ActivityLevel.MODERATE_ACTIVITY,
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

TASKS = ["cardiovascular_fitness", "stress_recovery", "sedentary_activation", "sleep_optimization"]


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
                f"rhr={b.resting_hr},hrv={b.hrv},"
                f"sleep_score={b.sleep_score},stress_avg={b.stress_avg},"
                f"body_battery={b.body_battery}"
            )

            # Compact delta snapshot
            dl = result.observation.deltas
            delta_str = (
                f"rhr={dl.resting_hr:+.3f},hrv={dl.hrv:+.3f},"
                f"sleep_score={dl.sleep_score:+.3f},stress_avg={dl.stress_avg:+.3f},"
                f"body_battery={dl.body_battery:+.3f}"
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
