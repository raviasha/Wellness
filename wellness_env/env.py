"""WellnessEnv — Outcome-based OpenEnv-compliant environment."""

from __future__ import annotations

import random
import os
import json
import copy
from typing import Any

from .graders import grade_multi_outcome, grade_resistant_adaptation, grade_single_goal
from .models import (
    Action,
    Biomarkers,
    BiomarkerDeltas,
    EnvState,
    Goal,
    Observation,
    OutcomeTrends,
    RewardBreakdown,
    StepResult,
)
from .payoff import _linear_slope, _stddev, compute_reward
from .personas import PERSONAS, PersonaConfig, apply_compliance
from .simulator import apply_deltas, apply_life_event, compute_biomarker_changes


# ---------------------------------------------------------------------------
# Task configs — each task uses a different persona/goal/duration
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "single_goal": {
        "persona": "athletic_performance",
        "total_days": 14,
        "description": "Easy: Optimize VO2 max and athletic performance over 14 days",
    },
    "multi_outcome": {
        "persona": "stress_management",
        "total_days": 30,
        "description": "Medium: Balance all biomarkers for stress management over 30 days",
    },
    "resistant_adaptation": {
        "persona": "weight_loss",
        "total_days": 30,
        "description": "Hard: Help a resistant, low-compliance weight-loss persona improve over 30 days",
    },
    "personal_coaching": {
        "persona": "digital_twin",
        "total_days": 30,
        "description": "Personalized: Optimize your unique biomarkers using your calibrated rules.",
    },
}

GRADERS = {
    "single_goal": grade_single_goal,
    "multi_outcome": grade_multi_outcome,
    "resistant_adaptation": grade_resistant_adaptation,
    "personal_coaching": grade_multi_outcome,
}


class WellnessEnv:
    """Outcome-based wellness simulator implementing the OpenEnv interface.

    Key difference from the original: rewards are driven by measured
    biomarker changes (outcomes), not predefined action-quality scores.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self._task_name: str | None = None
        self._config: dict[str, Any] = {}
        self._persona: PersonaConfig | None = None
        self._day: int = 0
        self._total_days: int = 0
        self._biomarkers: Biomarkers | None = None
        self._prev_deltas: BiomarkerDeltas = BiomarkerDeltas(
            resting_hr=0, hrv=0, vo2_max=0, body_fat_pct=0,
            lean_mass_kg=0, sleep_efficiency=0, cortisol_proxy=0, energy_level=0
        )
        self._history: list[dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False

    # -------------------------------------------------------------------
    # OpenEnv interface
    # -------------------------------------------------------------------

    def reset(self, task_name: str) -> Observation:
        """Initialize a new episode for the given task."""
        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}"
            )

        self._task_name = task_name
        self._config = TASK_CONFIGS[task_name]
        self._persona = copy.deepcopy(PERSONAS[self._config["persona"]])
        
        # Check for calibrated persona if using digital_twin
        if self._config["persona"] == "digital_twin":
            # Check per-user calibrated personas (models/user_*/calibrated_persona.json)
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            cal_paths = []
            # Look for any user-calibrated persona
            if os.path.isdir(models_dir):
                for d in sorted(os.listdir(models_dir)):
                    p = os.path.join(models_dir, d, "calibrated_persona.json")
                    if os.path.exists(p):
                        cal_paths.append(p)
            # Use the most recent one (or could be passed in)
            cal_path = cal_paths[-1] if cal_paths else None
            
            if cal_path:
                try:
                    with open(cal_path, "r") as f:
                        cal_data = json.load(f)
                    for k, v in cal_data.items():
                        if hasattr(self._persona.response_model, k):
                            setattr(self._persona.response_model, k, v)
                except Exception:
                    pass

        self._day = 0
        self._total_days = self._config["total_days"]

        # Clone starting biomarkers
        self._biomarkers = self._persona.starting_biomarkers.model_copy()
        self._prev_deltas = BiomarkerDeltas(
            resting_hr=0, hrv=0, vo2_max=0, body_fat_pct=0,
            lean_mass_kg=0, sleep_efficiency=0, cortisol_proxy=0, energy_level=0
        )
        self._history = []
        self._cumulative_reward = 0.0
        self._done = False

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Apply action, advance one day."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._persona is None or self._biomarkers is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._day += 1

        # 1. Compliance filter
        actual_action, complied = apply_compliance(action, self._persona, self._rng)

        # 2. Life events
        actual_action = apply_life_event(actual_action, self._rng)

        # 3. Compute biomarker changes (the hidden response model)
        deltas = compute_biomarker_changes(
            actual_action, self._biomarkers, self._persona,
            self._history, self._rng,
        )

        # 4. Apply deltas to get new biomarker state
        self._biomarkers = apply_deltas(self._biomarkers, deltas)
        self._prev_deltas = deltas

        # 5. Compute outcome-based reward
        reward = compute_reward(deltas, self._persona.goal, self._biomarkers)

        # 6. Record history
        entry: dict[str, Any] = {
            "day": self._day,
            "goal": self._persona.goal.value,
            "recommended_action": action.model_dump(),
            "actual_action": actual_action.model_dump(),
            "complied": complied,
            "biomarkers": self._biomarkers.model_dump(),
            "deltas": deltas.model_dump(),
            "reward_total": reward.total,
            "compliance_rate": self._persona.compliance_rate,
        }
        self._history.append(entry)
        self._cumulative_reward += reward.total

        # 7. Check done
        self._done = self._day >= self._total_days

        # 8. Build observation
        obs = self._make_observation()

        # 9. Info dict
        info: dict[str, Any] = {
            "recommended_action": action.model_dump(),
            "actual_action": actual_action.model_dump(),
            "complied": complied,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return full internal state for debugging/grading."""
        return EnvState(
            day=self._day,
            total_days=self._total_days,
            goal=self._persona.goal if self._persona else Goal.OVERALL_WELLNESS,
            persona_name=self._persona.name if self._persona else "",
            compliance_rate=self._persona.compliance_rate if self._persona else 0.0,
            biomarkers=self._biomarkers or Biomarkers(
                resting_hr=70, hrv=40, vo2_max=30, body_fat_pct=25,
                lean_mass_kg=55, sleep_efficiency=75, cortisol_proxy=50, energy_level=50
            ),
            history=self._history,
            cumulative_reward=round(self._cumulative_reward, 2),
        )

    def grade(self) -> float:
        """Compute the grader score (0.0–1.0) for the current task."""
        if self._task_name is None:
            raise RuntimeError("No task has been run. Call reset() first.")
        grader = GRADERS[self._task_name]
        return grader(self._history)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        """Build observation from current state."""
        trends = None
        if len(self._history) >= 7:
            trends = self._compute_trends()

        return Observation(
            day=self._day,
            total_days=self._total_days,
            goal=self._persona.goal if self._persona else Goal.OVERALL_WELLNESS,
            biomarkers=self._biomarkers or Biomarkers(
                resting_hr=70, hrv=40, vo2_max=30, body_fat_pct=25,
                lean_mass_kg=55, sleep_efficiency=75, cortisol_proxy=50, energy_level=50
            ),
            deltas=self._prev_deltas,
            trends=trends,
            persona_name=self._persona.name if self._persona else "",
            compliance_rate=self._persona.compliance_rate if self._persona else 0.0,
        )

    def _compute_trends(self) -> OutcomeTrends:
        """Compute 7-day trends for each biomarker and reward."""
        recent = self._history[-7:]
        rewards = [h["reward_total"] for h in recent]

        def _marker_trend(key: str) -> float:
            values = [h["biomarkers"][key] for h in recent]
            return round(_linear_slope(values), 4)

        return OutcomeTrends(
            resting_hr_trend=_marker_trend("resting_hr"),
            hrv_trend=_marker_trend("hrv"),
            vo2_max_trend=_marker_trend("vo2_max"),
            body_fat_trend=_marker_trend("body_fat_pct"),
            lean_mass_trend=_marker_trend("lean_mass_kg"),
            sleep_efficiency_trend=_marker_trend("sleep_efficiency"),
            reward_trend=round(_linear_slope(rewards), 4),
            reward_consistency=round(_stddev(rewards), 4),
        )
