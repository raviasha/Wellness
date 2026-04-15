"""Task graders — compute 0.0–1.0 scores from episode history.

Unlike the original predefined-payoff graders, these evaluate based on
actual outcome improvements (biomarker deltas), not action quality.
"""

from __future__ import annotations

from typing import Any

from .payoff import _linear_slope, _stddev


def _normalize(x: float, lo: float, hi: float) -> float:
    """Normalize x to [0, 1] given expected range [lo, hi]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _get_primary_biomarker_key(history: list[dict[str, Any]]) -> str:
    """Determine the primary biomarker to evaluate based on the task's goal.

    Returns the biomarker delta key that has the highest weight for this goal.
    """
    if not history:
        return "energy_level"
    goal = history[0].get("goal", "overall_wellness")
    primary_map = {
        "weight_loss": "body_fat_pct",
        "muscle_gain": "lean_mass_kg",
        "overall_wellness": "energy_level",
        "longevity": "vo2_max",
        "athletic_performance": "vo2_max",
        "stress_management": "cortisol_proxy",
    }
    return primary_map.get(goal, "energy_level")


# ---------------------------------------------------------------------------
# Task 1: Single-Goal Focus (Easy)
# ---------------------------------------------------------------------------

def grade_single_goal(history: list[dict[str, Any]]) -> float:
    """Easy task: optimize the primary biomarker for the persona's goal.

    Score = 0.6 * normalize(avg_reward, 40, 75)
          + 0.2 * normalize(primary_biomarker_improvement, ...)
          + 0.2 * reward_trend (positive = improving)
    """
    if not history:
        return 0.0

    # Average reward across episode
    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)

    # Primary biomarker cumulative improvement
    primary_key = _get_primary_biomarker_key(history)
    if len(history) >= 2:
        first_val = history[0]["biomarkers"][primary_key]
        last_val = history[-1]["biomarkers"][primary_key]
        # For "lower is better" markers, negate
        if primary_key in {"resting_hr", "body_fat_pct", "cortisol_proxy"}:
            improvement = first_val - last_val
        else:
            improvement = last_val - first_val
    else:
        improvement = 0.0

    # Reward trend
    trend = _linear_slope(rewards) if len(rewards) >= 3 else 0.0

    # Scale improvement based on what's realistic
    improvement_scales = {
        "body_fat_pct": 2.0,      # 2% body fat loss over 14 days = excellent
        "lean_mass_kg": 0.5,      # 0.5 kg lean mass gain over 14 days = excellent
        "vo2_max": 2.0,           # 2 ml/kg/min over 14 days = excellent
        "energy_level": 20.0,     # 20 point energy improvement = excellent
        "cortisol_proxy": 20.0,   # 20 point cortisol drop = excellent
        "resting_hr": 3.0,        # 3 bpm drop over 14 days = excellent
        "hrv": 10.0,              # 10ms HRV improvement = excellent
        "sleep_efficiency": 10.0, # 10% improvement = excellent
    }
    scale = improvement_scales.get(primary_key, 10.0)

    score = (
        0.6 * _normalize(avg_reward, 40.0, 75.0)
        + 0.2 * _normalize(improvement, 0.0, scale)
        + 0.2 * _normalize(trend, -0.5, 1.0)
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 2: Multi-Outcome Balance (Medium)
# ---------------------------------------------------------------------------

def grade_multi_outcome(history: list[dict[str, Any]]) -> float:
    """Medium task: balance improvements across all biomarkers.

    Score = 0.35 * normalize(avg_reward, 40, 75)
          + 0.25 * biomarker_breadth (how many markers improved)
          + 0.20 * consistency (low reward variance)
          + 0.20 * reward_trend
    """
    if not history:
        return 0.0

    rewards = [h["reward_total"] for h in history]
    avg_reward = sum(rewards) / len(rewards)

    # Biomarker breadth: fraction of markers that improved overall
    if len(history) >= 2:
        markers = ["resting_hr", "hrv", "vo2_max", "body_fat_pct",
                    "lean_mass_kg", "sleep_efficiency", "cortisol_proxy", "energy_level"]
        lower_better = {"resting_hr", "body_fat_pct", "cortisol_proxy"}
        improved_count = 0
        for m in markers:
            first_val = history[0]["biomarkers"][m]
            last_val = history[-1]["biomarkers"][m]
            if m in lower_better:
                if last_val < first_val - 0.01:
                    improved_count += 1
            else:
                if last_val > first_val + 0.01:
                    improved_count += 1
        breadth = improved_count / len(markers)
    else:
        breadth = 0.0

    # Consistency — wider range to handle realistic reward variance
    consistency = 1.0 - _normalize(_stddev(rewards), 0.0, 30.0)

    # Trend
    trend = _normalize(_linear_slope(rewards), -0.5, 1.0)

    score = (
        0.35 * _normalize(avg_reward, 35.0, 65.0)
        + 0.25 * breadth
        + 0.20 * consistency
        + 0.20 * max(0.0, trend)
    )
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task 3: Resistant User Adaptation (Hard)
# ---------------------------------------------------------------------------

def grade_resistant_adaptation(history: list[dict[str, Any]]) -> float:
    """Hard task: improve outcomes for a user who barely complies.

    Score = 0.25 * normalize(avg_reward, 35, 65)
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

    # Consistency — wider range to handle realistic reward variance
    consistency = 1.0 - _normalize(_stddev(rewards), 0.0, 30.0)

    # Compliance effectiveness: did the agent achieve compliance above baseline?
    compliance_events = [h.get("complied", False) for h in history]
    actual_compliance = sum(1 for c in compliance_events if c) / max(1, len(compliance_events))
    configured_compliance = history[0].get("compliance_rate", 0.25) if history else 0.25
    # Reward if actual > configured (agent adapted recommendations to increase compliance)
    compliance_effectiveness = _normalize(
        actual_compliance - configured_compliance, -0.1, 0.15
    )

    # Biomarker breadth (same as multi-outcome)
    if len(history) >= 2:
        markers = ["resting_hr", "hrv", "vo2_max", "body_fat_pct",
                    "lean_mass_kg", "sleep_efficiency", "cortisol_proxy", "energy_level"]
        lower_better = {"resting_hr", "body_fat_pct", "cortisol_proxy"}
        improved_count = 0
        for m in markers:
            first_val = history[0]["biomarkers"][m]
            last_val = history[-1]["biomarkers"][m]
            if m in lower_better:
                if last_val < first_val - 0.01:
                    improved_count += 1
            else:
                if last_val > first_val + 0.01:
                    improved_count += 1
        breadth = improved_count / len(markers)
    else:
        breadth = 0.0

    score = (
        0.30 * _normalize(avg_reward, 10.0, 45.0)
        + 0.20 * _normalize(improvement, 0.0, 15.0)
        + 0.20 * consistency
        + 0.15 * compliance_effectiveness
        + 0.15 * breadth
    )
    return round(max(0.0, min(1.0, score)), 4)
