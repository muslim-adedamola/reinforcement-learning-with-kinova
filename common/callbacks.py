"""Callbacks shared across the Kinova RL environments.

SaveModelCallback is used by every environment. The curriculum and CSV-logging
callbacks are only used by the dynamic-obstacle environment.
"""

import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveModelCallback(BaseCallback):
    """Periodically checkpoint the model, keyed by timestep."""

    def __init__(self, check_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Saving model at timestep {self.num_timesteps}")
        return True


class TimeBasedCurriculumCallback(BaseCallback):
    """After a fixed number of timesteps, switch all envs to the dynamic-obstacle stage."""

    def __init__(self, switch_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.switch_timesteps = switch_timesteps
        self.switched = False

    def _on_step(self) -> bool:
        if (not self.switched) and (self.num_timesteps >= self.switch_timesteps):
            if self.verbose:
                print(f"[Curriculum] Switching to dynamic obstacles at timestep {self.num_timesteps}")
            self.training_env.env_method("set_curriculum_stage", 1)
            self.switched = True
        return True


class CurriculumBatchCallback(BaseCallback):
    """Log the mean episode reward to CSV, one row per BATCH_SIZE finished episodes."""

    def __init__(self, csv_path, batch_size=100, verbose=1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.batch_rewards = []
        self.batch_index = 0
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode_range", "mean_reward"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self.batch_rewards.append(ep_info["r"])
                if len(self.batch_rewards) >= self.batch_size:
                    self.batch_index += 1
                    start_ep = (self.batch_index - 1) * self.batch_size + 1
                    end_ep = self.batch_index * self.batch_size
                    mean_reward = float(np.mean(self.batch_rewards))
                    with open(self.csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([f"{start_ep}-{end_ep}", f"{mean_reward:.2f}"])
                    self.batch_rewards.clear()
        return True
