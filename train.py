"""Unified training entry point for the three Kinova RL environments.

Usage:
    python train.py --env reaching
    python train.py --env multi_static
    python train.py --env dynamic

Each env keeps its own settings (vec-env type, VecNormalize on/off, callbacks,
timesteps) through configs.py, and writes checkpoints to its own models/ folder.
"""

import os
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from configs import get_config, load_env_class
from common.callbacks import (
    SaveModelCallback,
    TimeBasedCurriculumCallback,
    CurriculumBatchCallback,
)


def build_env(config):
    env_cls = load_env_class(config)
    env_kwargs = dict(config["env_kwargs"])
    # Subproc workers should not open a window; rgb_array is safe for the rest.
    render_mode = None if config["vec_env"] == "subproc" else "rgb_array"

    def make_env():
        return env_cls(render_mode=render_mode, **env_kwargs)

    vec_env_cls = SubprocVecEnv if config["vec_env"] == "subproc" else None
    env = make_vec_env(make_env, n_envs=config["n_envs"], vec_env_cls=vec_env_cls)

    if config["use_vecnormalize"]:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env


def build_callbacks(config, save_path):
    callbacks = [SaveModelCallback(
        check_freq=max(config["total_timesteps"] // 10, 1),
        save_path=save_path,
        verbose=1,
    )]
    if config["curriculum"] is not None:
        callbacks.append(TimeBasedCurriculumCallback(
            switch_timesteps=config["curriculum"]["switch_timesteps"], verbose=1
        ))
    if config["csv_log"]:
        callbacks.append(CurriculumBatchCallback(
            csv_path=os.path.join(config["dir"], "logs", "episode_batches.csv"),
            verbose=1,
        ))
    return callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        choices=["reaching", "multi_static", "dynamic"])
    args = parser.parse_args()

    config = get_config(args.env)
    save_path = os.path.join(config["dir"], "models")
    os.makedirs(save_path, exist_ok=True)

    env = build_env(config)
    callbacks = build_callbacks(config, save_path)

    tb_log = os.path.join(config["dir"], "tensorboard")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_log)

    t0 = time.time()
    model.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=4,
        callback=callbacks,
        progress_bar=True,
    )
    print(f"Training took {(time.time() - t0) / 3600:.2f} hours.")

    model.save(os.path.join(save_path, config["model_name"]))

    # Save normalization stats so eval can reproduce the training input distribution.
    if config["use_vecnormalize"]:
        env.save(os.path.join(save_path, "vec_normalize_stats.pkl"))


if __name__ == "__main__":
    main()
