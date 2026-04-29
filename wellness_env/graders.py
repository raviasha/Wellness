"""Task graders — compute 0.0–1.0 scores from episode history.

Unlike the original predefined-payoff graders, these evaluate based on
actual outcome improvements (biomarker deltas), not action quality.

MVP 1: 5 Garmin biomarkers (resting_hr, hrv, sleep_score, stress_avg, body_battery).
4 tasks: cardiovascular_fitness, stress_recovery, sedentary_activation, sleep_optimization.
"""

from __future__ import annotations

from typing import Any

from .payoff import _linear_slope, _stddev

# The 7 Garmin-measured biomarkers
MARKERS = ["resting_hr", "hrv", "sleep_score", "stress_avg", "body_battery",
           "sleep_stage_quality", "vo2_max"]
LOWER_IS_BETTER = {"resting_hr", "stress_avg"}


def _normalize(x: float, lo: float, hi: float) -> float:
    """Normalize x to [0, 1] given expected range [lo, hi]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _get_primary_biomarker_key(history: list[dict[str, Any]]) -> str:
    """Determine the primary biomarker to evaluate based on the task's goal."""
    if not history:
        return "body_battery"
    goal = history[0].get("goal", "stress_management")
    primary_map = {
        "stress_management": "stress_avg",
        "cardiovascular_fitness": "resting_hr",
        "sleep_optimization": "sleep_score",
        "recovery_energy": "body_battery",
        "active_living": "body_battery",
    }
    return primary_map.get(goal, "body_battery")


def _biomarker_breadth(history: list[dict[str, Any]]) -> float:
    """Fraction of the 5 biomarkers that improved from first to last step."""
    if len(history) < 2:
        return 0.0
    improved_count = 0
    for m in MARKERS:
        first_val = history[0]["biomarkers"][m]
        last_val = history[-1]["biomarkers"][m]
        if m in LOWER_IS_BETTER:
            if last_val < first_val - 0.01:
                improved_count += 1
        else:
            if last_val > first_val + 0.01:
                improved_count += 1
    return improved_count / len(MARKERS)


# ---------------------------------------------------------------------------
# Task 1: Cardiovascular Fitness (Easy — 14 days)
# ---------------------------------------------------------------------------

def grade_cardiovascular_fitness(history: list[dict[str, Any]]) -> float:
    """Easy task: optimise RHR and HRV over 14 days.

    Score = 0.6 * normalize(avg_reward)
          + 0.2 * normalize(primary_biomarker_improvement)
          + 0.2 * reward_trend
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)

    primary_key = _get_primary_biomarker_key(history)
    if len(history) >= 2:
        first_val = history[0]["biomarkers"][primary_key]
        last_val = history[-1]["biomarkers"][primary_key]
        if primary_key in LOWER_IS_BETTER:
            improvement = first_val - last_val
        else:
            improvement = last_val - first_val
    else:
        improvement = 0.0

    trend = _linear_slope(rewards) if len(rewards) >= 3 else 0.0

    improvement_scales = {
        "resting_hr": 3.0,
        "hrv": 10.0,
        "sleep_score": 10.0,
        "stress_avg": 10.0,
        "body_battery": 15.0,
    }
    scale = improvement_scales.get(primary_key, 10.0)

    score = (
        0.6 * _normalize(avg_reward, 40.0, 75.0)
        + 0.2 * _normalize(improvement, 0.0, scale)
        + 0.2 * _normalize(trend, -0.5, 1.0)
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 2: Stress Recovery (Medium — 30 days)
# ---------------------------------------------------------------------------

def grade_stress_recovery(history: list[dict[str, Any]]) -> float:
    """Medium task: balance improvements across all 5 biomarkers for stress recovery.

    Score = 0.35 * normalize(avg_reward)
          + 0.25 * biomarker_breadth
          + 0.20 * consistency
          + 0.20 * reward_trend
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)
    breadth = _biomarker_breadth(history)
    consistency = 1.0 - _normalize(_stddev(rewards), 0.0, 30.0)
    trend = _normalize(_linear_slope(rewards), -0.5, 1.0)

    score = (
        0.35 * _normalize(avg_reward, 35.0, 65.0)
        + 0.25 * breadth
        + 0.20 * consistency
        + 0.20 * max(0.0, trend)
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 3: Sedentary Activation (Hard — 30 days, low compliance)
# ---------------------------------------------------------------------------

def grade_sedentary_activation(history: list[dict[str, Any]]) -> float:
    """Hard task: improve outcomes for a sedentary user who barely complies.

    Score = 0.25 * normalize(avg_reward)
          + 0.25 * outcome_improvement (last 7 vs first 7 days)
          + 0.20 * consistency
          + 0.15 * compliance_effectiveness
          + 0.15 * biomarker_breadth
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)

    # Outcome improvement: last 7 vs first 7
    if len(history) >= 14:
        first7_avg = sum(rewards[:7]) / 7.0
        last7_avg = sum(rewards[-7:]) / 7.0
        improvement = last7_avg - first7_avg
    else:
        half = len(history) // 2
        if half > 0:
            first_half = sum(rewards[:half]) / half
            second_half = sum(rewards[half:]) / max(1, len(rewards) - half)
            improvement = second_half - first_half
        else:
            improvement = 0.0

    consistency = 1.0 - _normalize(_stddev(rewards), 0.0, 30.0)

    compliance_events = [h.get("complied", False) for h in history]
    actual_compliance = sum(1 for c in compliance_events if c) / max(1, len(compliance_events))
    configured_compliance = history[0].get("compliance_rate", 0.25) if history else 0.25
    compliance_effectiveness = _normalize(
        actual_compliance - configured_compliance, -0.1, 0.15
    )

    breadth = _biomarker_breadth(history)

    score = (
        0.25 * _normalize(avg_reward, 10.0, 45.0)
        + 0.25 * _normalize(improvement, 0.0, 15.0)
        + 0.20 * consistency
        + 0.15 * compliance_effectiveness
        + 0.15 * breadth
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 4: Sleep Optimization (Medium — 30 days)
# ---------------------------------------------------------------------------

def grade_sleep_optimization(history: list[dict[str, Any]]) -> float:
    """Medium task: improve sleep score and downstream recovery metrics.

    Score = 0.35 * normalize(avg_reward)
          + 0.30 * sleep_score_improvement
          + 0.15 * body_battery_improvement
          + 0.20 * reward_trend
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)

    # Sleep score improvement
    if len(history) >= 2:
        sleep_first = history[0]["biomarkers"]["sleep_score"]
        sleep_last = history[-1]["biomarkers"]["sleep_score"]
        sleep_improvement = sleep_last - sleep_first

        battery_first = history[0]["biomarkers"]["body_battery"]
        battery_last = history[-1]["biomarkers"]["body_battery"]
        battery_improvement = battery_last - battery_first
    else:
        sleep_improvement = 0.0
        battery_improvement = 0.0

    trend = _normalize(_linear_slope(rewards), -0.5, 1.0)

    score = (
        0.35 * _normalize(avg_reward, 35.0, 65.0)
        + 0.30 * _normalize(sleep_improvement, 0.0, 15.0)
        + 0.15 * _normalize(battery_improvement, 0.0, 15.0)
        + 0.20 * max(0.0, trend)
    )
    return round(max(0.0, min(1.0, score)), 4)
