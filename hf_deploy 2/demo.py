#!/usr/bin/env python3
"""
Wellness Simulator Demo — generates visual charts showcasing environment behavior.

Run: python demo.py
Outputs PNG charts to demo_output/ directory. No API key needed.
"""

from __future__ import annotations

import os
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from wellness_env import WellnessEnv, Action
from wellness_env.models import SleepDuration, ExerciseType, NutritionType

OUTPUT_DIR = "demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def optimal_strategy(obs, step):
    """Good agent: adapts to persona, manages fatigue."""
    if obs.compliance_rate <= 0.3:
        # Gentle for resistant users — start small, ramp up
        if step <= 10:
            return Action(sleep=SleepDuration.SHORT, exercise=ExerciseType.LIGHT_CARDIO, nutrition=NutritionType.HIGH_CARB)
        elif step <= 20:
            return Action(sleep=SleepDuration.OPTIMAL_LOW, exercise=ExerciseType.YOGA, nutrition=NutritionType.BALANCED)
        else:
            return Action(sleep=SleepDuration.OPTIMAL_LOW, exercise=ExerciseType.MODERATE_CARDIO, nutrition=NutritionType.BALANCED)
    elif obs.fatigue > 65:
        return Action(sleep=SleepDuration.OPTIMAL_HIGH, exercise=ExerciseType.NONE, nutrition=NutritionType.HIGH_PROTEIN)
    else:
        return Action(sleep=SleepDuration.OPTIMAL_LOW, exercise=ExerciseType.MODERATE_CARDIO, nutrition=NutritionType.BALANCED)


def random_strategy(obs, step):
    """Random agent: picks actions uniformly."""
    import random as _r
    return Action(
        sleep=_r.choice(list(SleepDuration)),
        exercise=_r.choice(list(ExerciseType)),
        nutrition=_r.choice(list(NutritionType)),
    )


def naive_strategy(obs, step):
    """Naive agent: always recommends maximum intensity."""
    return Action(sleep=SleepDuration.OPTIMAL_HIGH, exercise=ExerciseType.HIIT, nutrition=NutritionType.HIGH_PROTEIN)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name, strategy_fn, seed=42):
    env = WellnessEnv(seed=seed)
    obs = env.reset(task_name)
    days = env._config["total_days"]

    data = {
        "rewards": [], "sleep_scores": [], "exercise_scores": [],
        "nutrition_scores": [], "energy": [], "fatigue": [],
        "sleep_debt": [], "complied": [],
    }

    for step in range(1, days + 1):
        action = strategy_fn(obs, step)
        result = env.step(action)
        data["rewards"].append(result.reward.total)
        data["sleep_scores"].append(result.reward.sleep_score)
        data["exercise_scores"].append(result.reward.exercise_score)
        data["nutrition_scores"].append(result.reward.nutrition_score)
        data["energy"].append(result.observation.energy_level)
        data["fatigue"].append(result.observation.fatigue)
        data["sleep_debt"].append(result.observation.sleep_debt)
        data["complied"].append(result.info["complied"])
        obs = result.observation

    data["score"] = env.grade()
    return data


# ---------------------------------------------------------------------------
# Chart 1: Reward Trajectories — 3 Strategies Compared
# ---------------------------------------------------------------------------

def chart_reward_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tasks = [("sleep_focus", 14), ("full_wellness", 30), ("resistant_user", 30)]
    strategies = [
        ("Optimal", optimal_strategy, "#2ecc71"),
        ("Naive (Max Intensity)", naive_strategy, "#e74c3c"),
        ("Random", random_strategy, "#95a5a6"),
    ]

    for ax, (task, days) in zip(axes, tasks):
        x = np.arange(1, days + 1)
        for name, fn, color in strategies:
            data = run_episode(task, fn)
            ax.plot(x, data["rewards"], label=f"{name} (score={data['score']:.2f})", color=color, linewidth=2)
        ax.set_title(task.replace("_", " ").title(), fontsize=14, fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 110)

    fig.suptitle("Reward Trajectories: Agent Strategy Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_reward_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Chart 2: Pillar Radar — End-of-Episode Snapshot
# ---------------------------------------------------------------------------

def chart_pillar_radar():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw=dict(polar=True))
    tasks = ["sleep_focus", "full_wellness", "resistant_user"]
    categories = ["Sleep", "Exercise", "Nutrition", "Energy", "Low Fatigue"]

    for ax, task in zip(axes, tasks):
        data = run_episode(task, optimal_strategy)
        # Average over last 5 days
        n = min(5, len(data["rewards"]))
        values = [
            np.mean(data["sleep_scores"][-n:]),
            np.mean(data["exercise_scores"][-n:]),
            np.mean(data["nutrition_scores"][-n:]),
            np.mean(data["energy"][-n:]),
            100.0 - np.mean(data["fatigue"][-n:]),  # invert fatigue
        ]
        # Normalize to 0-1
        values = [v / 100.0 for v in values]
        values.append(values[0])  # close the polygon

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles.append(angles[0])

        ax.plot(angles, values, "o-", linewidth=2, color="#3498db")
        ax.fill(angles, values, alpha=0.25, color="#3498db")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"{task.replace('_', ' ').title()}\nscore={data['score']:.2f}", fontsize=12, fontweight="bold", pad=20)

    fig.suptitle("Pillar Balance (Last 5 Days, Optimal Agent)", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_pillar_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Chart 3: Compliance Adaptation — Resistant User Deep Dive
# ---------------------------------------------------------------------------

def chart_compliance_adaptation():
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Optimal vs Naive on resistant user
    for name, fn, color in [("Optimal (gradual)", optimal_strategy, "#2ecc71"), ("Naive (max intensity)", naive_strategy, "#e74c3c")]:
        data = run_episode("resistant_user", fn)
        days = np.arange(1, 31)

        axes[0].plot(days, data["rewards"], label=f"{name} — score={data['score']:.2f}", color=color, linewidth=2)

        # Compliance rolling average
        complied_float = [1.0 if c else 0.0 for c in data["complied"]]
        window = 5
        rolling = [np.mean(complied_float[max(0, i - window + 1):i + 1]) for i in range(len(complied_float))]
        axes[1].plot(days, rolling, label=name, color=color, linewidth=2)

    axes[0].set_title("Resistant User: Reward Trajectory", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Reward")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 110)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    axes[1].set_title("Rolling Compliance Rate (5-day window)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Compliance Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Configured rate (0.25)")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Compliance Adaptation: Why Gradual Recommendations Win", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_compliance_adaptation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Chart 4: State Dynamics — Energy, Fatigue, Sleep Debt
# ---------------------------------------------------------------------------

def chart_state_dynamics():
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    data = run_episode("full_wellness", optimal_strategy)
    days = np.arange(1, 31)

    axes[0].plot(days, data["energy"], color="#f39c12", linewidth=2)
    axes[0].fill_between(days, data["energy"], alpha=0.2, color="#f39c12")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Energy Level", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(y=40, color="red", linestyle="--", alpha=0.5, label="Danger zone")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(days, data["fatigue"], color="#e74c3c", linewidth=2)
    axes[1].fill_between(days, data["fatigue"], alpha=0.2, color="#e74c3c")
    axes[1].set_ylabel("Fatigue")
    axes[1].set_title("Fatigue Level", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 105)
    axes[1].axhline(y=80, color="red", linestyle="--", alpha=0.5, label="Overtraining threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(days, data["sleep_debt"], color="#9b59b6", linewidth=2)
    axes[2].fill_between(days, data["sleep_debt"], alpha=0.2, color="#9b59b6")
    axes[2].set_ylabel("Sleep Debt (hrs)")
    axes[2].set_xlabel("Day")
    axes[2].set_title("Accumulated Sleep Debt", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    # Mark compliance
    for i, c in enumerate(data["complied"]):
        if not c:
            for ax in axes:
                ax.axvline(x=i + 1, color="gray", alpha=0.1, linewidth=3)

    fig.suptitle("State Dynamics: Full Wellness Task (Optimal Agent)\nGray lines = non-compliant days",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_state_dynamics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Chart 5: Score Summary Bar Chart
# ---------------------------------------------------------------------------

def chart_score_summary():
    tasks = ["sleep_focus", "full_wellness", "resistant_user"]
    strategies = [
        ("Random", random_strategy),
        ("Naive", naive_strategy),
        ("Optimal", optimal_strategy),
    ]
    colors = ["#95a5a6", "#e74c3c", "#2ecc71"]

    scores = {}
    for name, fn in strategies:
        scores[name] = [run_episode(t, fn)["score"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(strategies):
        bars = ax.bar(x + i * width, scores[name], width, label=name, color=colors[i], edgecolor="white")
        for bar, score in zip(bars, scores[name]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace("_", " ").title() for t in tasks], fontsize=12)
    ax.set_ylabel("Grader Score (0.0 – 1.0)", fontsize=12)
    ax.set_title("Final Scores by Strategy and Task", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "5_score_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating demo charts...\n")
    chart_reward_comparison()
    chart_pillar_radar()
    chart_compliance_adaptation()
    chart_state_dynamics()
    chart_score_summary()
    print(f"\nAll charts saved to {OUTPUT_DIR}/")
    print("\nCharts:")
    print("  1. Reward trajectories — 3 strategies across 3 tasks")
    print("  2. Pillar radar — balanced vs unbalanced performance")
    print("  3. Compliance adaptation — why gradual recs beat aggressive ones")
    print("  4. State dynamics — energy/fatigue/sleep debt over time")
    print("  5. Score summary — final grader scores comparison")


if __name__ == "__main__":
    main()
