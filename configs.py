"""Per-environment configuration registry.

Each environment is a distinct formulation with its own observation space,
reward, scene, vec-env setup, and trained models. This registry holds those
differences in one place so train.py and test.py can stay generic.

Notes:
- `use_vecnormalize` is True only for the two environments that were trained
  with VecNormalize. The dynamic env was trained WITHOUT it, so it must stay
  False, otherwise its saved policy is evaluated on the wrong input distribution.
- `total_timesteps` is set explicitly per env rather than derived from
  n_envs * episode_len * n_episodes, which over-counts badly.
- The env class in each env.py is renamed so the three are distinct
  (ReachingEnv, MultiStaticObstacleEnv, DynamicObstacleEnv).

Vec-env choice (read before raising n_envs):
- All three envs use SubprocVecEnv with a modest n_envs (30). Each MuJoCo
  model opens roughly 16 mesh files when it compiles. With DummyVecEnv, every
  env is built in one process, so a high n_envs opens n_envs * 16 file handles
  at once and trips the OS open-file limit (commonly 1024). MuJoCo reports the
  failed mesh open as "resource not found via provider or OS filesystem", which
  looks like a missing file but is actually the handle ceiling. SubprocVecEnv
  spreads the envs across worker processes so no single process hits the limit.
- If you switch back to "dummy" or raise n_envs a lot, raise the limit too
  (e.g. `ulimit -n 65535`) before training.
- The shipped reaching and multi-static policies were originally trained at
  higher counts on DummyVecEnv (reaching ~100, multi-static ~250). The values
  below describe a reproducible training setup, not the exact throughput that
  produced those specific checkpoints.
"""

ENV_CONFIGS = {
    "reaching": {
        "module": "reaching.env",
        "class": "ReachingEnv",
        "dir": "reaching",
        "env_kwargs": {"episode_len": 2500},
        "n_envs": 30,
        "total_timesteps": 50_000_000,
        "use_vecnormalize": True,
        "vec_env": "subproc",
        "model_name": "reaching_ppo",
        "curriculum": None,
        "csv_log": False,
        "viewer": None,
    },
    "multi_static": {
        # The trained 76D policy (multi_static_ppo.zip) is included, but the
        # VecNormalize stats from its run were not preserved. A correct
        # evaluation needs those stats, and they cannot be rebuilt from the
        # policy, so reproducing results means retraining from train.py (which
        # saves stats at the end). For a rough render without stats, run
        # test.py with --no-normalize, which feeds raw observations and does
        # NOT reflect trained performance.
        "module": "multi_static_obstacle.env",
        "class": "MultiStaticObstacleEnv",
        "dir": "multi_static_obstacle",
        "env_kwargs": {"episode_len": 2500},
        "n_envs": 30,
        "total_timesteps": 300_000_000,
        "use_vecnormalize": True,
        "vec_env": "subproc",
        "model_name": "multi_static_ppo",
        "curriculum": None,
        "csv_log": False,
        "viewer": None,
    },
    "dynamic": {
        "module": "dynamic_obstacle.env",
        "class": "DynamicObstacleEnv",
        "dir": "dynamic_obstacle",
        "env_kwargs": {"episode_len": 2500, "curriculum_stage": 0},
        "n_envs": 30,
        "total_timesteps": 100_000_000,
        "use_vecnormalize": False,
        "vec_env": "subproc",
        "model_name": "dynamic_ppo",
        "curriculum": {"switch_timesteps": 40_000_000},
        "csv_log": True,
        "viewer": {
            "distance": 3.0,
            "lookat": [0.0, -0.25, 0.824],
            "elevation": -45,
            "azimuth": 180,
            "hide_graph": True,
        },
    },
}


def get_config(env_name):
    if env_name not in ENV_CONFIGS:
        raise ValueError(
            f"Unknown env '{env_name}'. Choices: {', '.join(ENV_CONFIGS)}"
        )
    return ENV_CONFIGS[env_name]


def load_env_class(config):
    """Import and return the env class named in a config entry."""
    import importlib
    module = importlib.import_module(config["module"])
    return getattr(module, config["class"])
