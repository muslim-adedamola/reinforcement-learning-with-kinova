"""Unified evaluation entry point for the three Kinova RL environments.

Usage:
    python test.py --env reaching     --model reaching/models/reaching_ppo.zip
    python test.py --env multi_static --model multi_static_obstacle/models/multi_static_ppo.zip
    python test.py --env dynamic      --model dynamic_obstacle/models/dynamic_ppo.zip --stage 0

Dynamic env extras (ignored by the other two, which do not define them):
    --goal     7 joint angles to force a specific goal pose
    --state    7 joint angles to force a specific start state
    --obstacle 3 coordinates to force a specific obstacle position

Correctness note:
    reaching and multi_static were trained with VecNormalize(norm_obs=True), so
    eval loads the saved stats and applies them before predict(). dynamic was
    trained without VecNormalize, so no normalization is applied.

    --no-normalize forces raw (unnormalized) observations even for an env that
    was trained with VecNormalize. This is only a rendering sanity check: the
    policy sees observations at a different scale than it trained on, so the
    rollout does NOT reflect trained performance. Use it when the stats file is
    missing and you just want to confirm the scene loads and the arm moves.
"""

import os
import time
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs import get_config, load_env_class


def build_eval_env(config, stage, normalize=True):
    env_cls = load_env_class(config)
    env_kwargs = dict(config["env_kwargs"])
    if "curriculum_stage" in env_kwargs:
        env_kwargs["curriculum_stage"] = stage

    def make_env():
        return env_cls(render_mode="human", **env_kwargs)

    env = DummyVecEnv([make_env])

    if config["use_vecnormalize"] and normalize:
        stats = os.path.join(config["dir"], "models", "vec_normalize_stats.pkl")
        if not os.path.exists(stats):
            raise FileNotFoundError(
                f"Expected VecNormalize stats at {stats}. This env was trained "
                f"with normalization; evaluating without the stats is incorrect. "
                f"Re-run with --no-normalize for a raw-observation render that "
                f"does not reflect trained performance, or retrain to regenerate "
                f"the stats."
            )
        env = VecNormalize.load(stats, env)
        env.training = False
        env.norm_reward = False
    elif config["use_vecnormalize"] and not normalize:
        print(
            "[warning] --no-normalize: feeding raw observations to a policy "
            "trained with VecNormalize. This is a rendering check only and does "
            "not reflect trained performance."
        )
    return env


def unwrap(env):
    """Return the raw env instance inside the vec wrappers."""
    base = env.venv if isinstance(env, VecNormalize) else env
    return base.envs[0]


def apply_viewer(underlying, viewer):
    """Apply a saved camera configuration. No-op if the env has no renderer yet."""
    if viewer is None:
        return
    try:
        underlying.mujoco_renderer._get_viewer("human")
        cam = underlying.mujoco_renderer.viewer.cam
        cam.distance = viewer["distance"]
        cam.lookat[0], cam.lookat[1], cam.lookat[2] = viewer["lookat"]
        cam.elevation = viewer["elevation"]
        cam.azimuth = viewer["azimuth"]
        if viewer.get("hide_graph"):
            underlying.mujoco_renderer.viewer._hide_graph = True
    except Exception as exc:
        print(f"[viewer] skipped camera setup: {exc}")


def apply_special_state(underlying, args):
    """Force a specific goal / start state / obstacle if requested and supported.

    Method names match the env (set_speical_*). Returns True if anything was set.
    """
    changed = False
    if args.goal and hasattr(underlying, "set_speical_goal"):
        underlying.set_speical_goal(args.goal); changed = True
    if args.state and hasattr(underlying, "set_speical_state"):
        underlying.set_speical_state(args.state); changed = True
    if args.obstacle and hasattr(underlying, "set_speical_obstacle"):
        underlying.set_speical_obstacle(args.obstacle); changed = True
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        choices=["reaching", "multi_static", "dynamic"])
    parser.add_argument("--model", required=True, help="Path to the .zip policy")
    parser.add_argument("--stage", type=int, default=0,
                        help="Dynamic env only: 0 static, 1 moving")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Feed raw observations even if the env was trained "
                             "with VecNormalize. Rendering sanity check only; "
                             "does NOT reflect trained performance.")
    parser.add_argument("--goal", type=float, nargs=7, default=None)
    parser.add_argument("--state", type=float, nargs=7, default=None)
    parser.add_argument("--obstacle", type=float, nargs=3, default=None)
    args = parser.parse_args()

    config = get_config(args.env)
    normalize = not args.no_normalize
    env = build_eval_env(config, args.stage, normalize=normalize)
    model = PPO.load(args.model, device="cpu")

    obs = env.reset()
    underlying = unwrap(env)

    apply_viewer(underlying, config.get("viewer"))

    # Forced goal/state/obstacle only make sense without obs normalization.
    # Refresh obs so the first action sees them.
    effective_norm = config["use_vecnormalize"] and normalize
    if apply_special_state(underlying, args) and not effective_norm:
        if hasattr(underlying, "_get_obs"):
            obs = np.array([underlying._get_obs()])

    try:
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if step % 100 == 0:
                print(f"step {step}  reward {float(reward[0]):.3f}")
            if done[0]:
                obs = env.reset()
                break
            time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
