import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json
from wellness_env import WellnessEnv
from wellness_env.models import Action, SleepDuration, ExerciseType, NutritionType, Goal
from wellness_env.personas import ResponseModel

class WellnessGymEnv(gym.Env):
    """Gymnasium wrapper for WellnessEnv allowing integration with tools like Stable Baselines 3."""
    
    def __init__(
        self,
        seed: int = 42,
        task_name: str = "single_goal",
        persona_path: str = None,
        distribution_path: str = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.persona_path = persona_path
        self.distribution_path = distribution_path

        # Load distribution if provided
        distribution = None
        if distribution_path and os.path.exists(distribution_path):
            from backend.distribution_calibration import load_distribution
            distribution = load_distribution(distribution_path)

        self.env = WellnessEnv(
            seed=seed,
            simulator_mode="distribution" if distribution is not None else "rules",
            distribution=distribution,
        )
        
        # Action space: Sleep(5) x Exercise(6) x Nutrition(5) = 150 unique discrete actions
        self.sleep_options = list(SleepDuration)
        self.exercise_options = list(ExerciseType)
        self.nutrition_options = list(NutritionType)
        
        self.action_space = spaces.Discrete(
            len(self.sleep_options) * len(self.exercise_options) * len(self.nutrition_options)
        )
        
        # Observation space: 19 continuous features
        # [day_normalized, compliance_rate, goal_idx, 8x biomarkers, 8x deltas]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        
    def _get_obs(self, obs):
        goals = list(Goal)
        goal_idx = goals.index(obs.goal)
        
        return np.array([
            obs.day / obs.total_days,
            obs.compliance_rate,
            float(goal_idx),
            
            obs.biomarkers.resting_hr,
            obs.biomarkers.hrv,
            obs.biomarkers.vo2_max,
            obs.biomarkers.body_fat_pct,
            obs.biomarkers.lean_mass_kg,
            obs.biomarkers.sleep_efficiency,
            obs.biomarkers.cortisol_proxy,
            obs.biomarkers.energy_level,
            
            obs.deltas.resting_hr,
            obs.deltas.hrv,
            obs.deltas.vo2_max,
            obs.deltas.body_fat_pct,
            obs.deltas.lean_mass_kg,
            obs.deltas.sleep_efficiency,
            obs.deltas.cortisol_proxy,
            obs.deltas.energy_level,
        ], dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.env = WellnessEnv(seed=seed)
        obs = self.env.reset(self.task_name)
        
        # Apply calibrated persona if provided
        if self.persona_path and os.path.exists(self.persona_path):
            try:
                with open(self.persona_path, "r") as f:
                    cal_data = json.load(f)
                for k, v in cal_data.items():
                    if hasattr(self.env._persona.response_model, k):
                        setattr(self.env._persona.response_model, k, v)
            except Exception:
                pass
        
        return self._get_obs(obs), {}
        
    def step(self, action_idx):
        action_idx = int(action_idx)
        sl_idx = action_idx % len(self.sleep_options)
        rem = action_idx // len(self.sleep_options)
        ex_idx = rem % len(self.exercise_options)
        nu_idx = rem // len(self.exercise_options)
        
        action = Action(
            sleep=self.sleep_options[sl_idx],
            exercise=self.exercise_options[ex_idx],
            nutrition=self.nutrition_options[nu_idx]
        )
        
        res = self.env.step(action)
        reward = res.reward.total
        done = res.done
        
        return self._get_obs(res.observation), float(reward), done, False, res.info
