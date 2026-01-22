# Reinforcement Learning with Kinova Gen3

![Demo](assets/a.gif)

## Overview
This project trains a Kinova Gen3 robotic arm in MuJoCo to reach randomly sampled joint-space
goals using Proximal Policy Optimization (PPO).

The task is intentionally minimal and obstacle-free to serve as an introduction to:
- Custom MuJoCo environments
- Reinforcement learning for robotic manipulators
- Stable-Baselines3 PPO workflows

**Key tools:** MuJoCo, Gymnasium, Stable-Baselines3

---

## Level 2: Obstacle & Collision Avoidance (Simulation)

A second stage of this extends the basic joint-space reaching task by introducing:
- obstacle-aware observations
- collision detection (obstacles + table)
- safety-driven reward shaping

👉 View on the branch:
https://github.com/muslim-adedamola/reinforcement-learning-with-kinova/tree/obstacle-and-collision-avoidance

---

## Task Description
- **Robot:** Kinova Gen3 (7-DOF)
- **Objective:** Reach randomly sampled joint-space goal configurations
- **Observation space (21D):**
  - Joint positions (7)
  - Joint velocities (7)
  - Joint error to goal (7)
- **Action space:** Joint control
- **Algorithm:** PPO
- **Training horizons:** 20M and 50M timesteps

---

## Results
- Stable convergence within tens of millions of timesteps
- Smooth joint trajectories encouraged via action regularization
- High success rate in reaching goal configurations

---

## Quick Start

```bash
pip install -r requirements.txt
python test.py
```

To train from scratch:

```bash
python train.py
```

## Structure
- `kinova_model/` - Mujoco MJCF files (robot + scene)
- `kinova_obs_env.py` - Custom Gymnasium Environment
- `train.py` - PPO training script
- `test.py` - Policy rollout and visualization
- `models/` - Saved PPO Checkpoints
- `assets/` - Demo GIF
- `docs/` - detailed explanation

## Documentation
For a full explanation see:

- `docs/explanation.md`

## Limitations
- State Based (non-visual) training
- No obstacle avoidance
- No sim-to-real transfer
