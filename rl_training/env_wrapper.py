import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import json
from wellness_env import WellnessEnv
from wellness_env.models import (
    Action, SleepDuration, ActivityLevel, BedtimeWindow,
    ExerciseType, ExerciseDuration, Goal,
)
from wellness_env.personas import ResponseModel

class WellnessGymEnv(gym.Env):
    """Gymnasium wrapper for WellnessEnv — MultiDiscrete([5,5,5,5,5]) action space."""

    def __init__(
        self,
        seed: int = 42,
        task_name: str = "cardiovascular_fitness",
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

        # --- Action space: 5 independent dimensions ---
        self.sleep_options = list(SleepDuration)
        self.bedtime_options = list(BedtimeWindow)
        self.activity_options = list(ActivityLevel)
        self.exercise_type_options = list(ExerciseType)
        self.exercise_duration_options = list(ExerciseDuration)

        self.action_space = spaces.MultiDiscrete([
            len(self.sleep_options),             # 5: SleepDuration
            len(self.bedtime_options),           # 5: BedtimeWindow
            len(self.activity_options),          # 5: ActivityLevel
            len(self.exercise_type_options),     # 5: ExerciseType
            len(self.exercise_duration_options), # 5: ExerciseDuration
        ])

        # --- Observation space: 17 continuous features ---
        # [day_normalized, compliance_rate, goal_idx,
        #  7x biomarkers, 7x deltas]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
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
            obs.biomarkers.sleep_score,
            obs.biomarkers.stress_avg,
            obs.biomarkers.body_battery,
            obs.biomarkers.sleep_stage_quality,
            obs.biomarkers.vo2_max,

            obs.deltas.resting_hr,
            obs.deltas.hrv,
            obs.deltas.sleep_score,
            obs.deltas.stress_avg,
            obs.deltas.body_battery,
            obs.deltas.sleep_stage_quality,
            obs.deltas.vo2_max,
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

    def step(self, action_array):
        # action_array is a length-5 integer array from MultiDiscrete
        sl_idx  = int(action_array[0])
        bt_idx  = int(action_array[1])
        act_idx = int(action_array[2])
        et_idx  = int(action_array[3])
        ed_idx  = int(action_array[4])

        action = Action(
            sleep=self.sleep_options[sl_idx],
            bedtime=self.bedtime_options[bt_idx],
            activity=self.activity_options[act_idx],
            exercise_type=self.exercise_type_options[et_idx],
            exercise_duration=self.exercise_duration_options[ed_idx],
        )

        res = self.env.step(action)
        reward = res.reward.total
        done = res.done

        return self._get_obs(res.observation), float(reward), done, False, res.info
