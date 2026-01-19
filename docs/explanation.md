## Quick Start

Clone the repo, install requirements and run 

```bash
python train.py
```
  
You should see a simulation of a Kinova robot on a table trying to reach a goal point.  

## General Overview

This document provides a detailed technical explanation of the environment design,
reward formulation, and training pipeline used to train a Kinova Gen3 robotic arm
with reinforcement learning in MuJoCo.

The project is intentionally minimal and state-based, with the goal of serving as
a clear introduction to building custom MuJoCo environments and training robotic
manipulators using PPO.

![Demo](../assets/a.gif)

---

## 1. Environment Overview

The task is formulated as a joint-space goal-reaching problem for a 7-DOF Kinova
Gen3 robotic arm. At the beginning of each episode, a random joint configuration
(goal) is sampled within valid joint limits. The agent must learn a control policy
that drives the robot joints from their current configuration to the sampled goal.

The environment is implemented as a custom Gymnasium environment and inherits
from `MujocoEnv`. MuJoCo is used for physics simulation, while Stable-Baselines3
is used for PPO-based training.

The task does not include perception or obstacle avoidance and is purely state-based.

---

## 2. MuJoCo Scene and Robot Model

The MuJoCo model files are located in the `kinova_model/` directory and are written
in MJCF format.

The scene includes:
- The Kinova Gen3 robot mounted on a table
- A green sphere indicating the target end-effector position
- A red sphere representing an obstacle (currently inactive)

Although obstacle geometry exists in the scene for future extensions, obstacles
are not used in the current learning objective. The task focuses solely on joint-space
goal reaching.

The robot’s kinematic structure and joint limits are defined in the MJCF files,
while task-specific logic is handled in the Python environment.

---

## 3. Observation and Action Spaces

### Observation Space

The observation space is a 21-dimensional continuous vector composed of:
- Joint positions (7)
- Joint velocities (7)
- Difference between current joint positions and goal joint angles (7)

This representation provides the agent with full state information required to
perform the task. The observation space is defined as a `Box` space with unbounded
limits, since the values are real-valued.

### Action Space

The action space is continuous and corresponds to control inputs applied at the
seven robot joints. Action regularization is later applied through the reward
function to encourage smooth and safe motion.

---

## 4. Goal Sampling and Workspace Constraints

At the start of each episode, goal joint angles are generated randomly.

For each joint:
- If limits are defined in the MJCF model, those limits are used
- Otherwise, default limits of `[-π, π]` are applied

The sampled joint angles are passed to a forward kinematics function to compute
the corresponding end-effector position.

Since the robot is mounted on a table, the computed end-effector position is
adjusted by the table height. The resulting Cartesian position is then checked
against predefined workspace bounds to ensure the goal is physically reachable.

If the position lies outside the workspace, a new set of joint angles is sampled.
This process continues until a valid goal is found.

Once a valid goal is selected:
- The goal joint angles are stored
- The green target sphere in the MuJoCo scene is positioned at the corresponding
  end-effector location to visually indicate the target

---

## 5. Reward Function Design

The reward function is designed to encourage accurate, smooth, and stable motion.

### Goal Reward
If the agent reaches the goal, defined as all joint errors being less than 0.01,
a positive reward of +10 is assigned. A counter is also incremented to track how
often goals are reached during training.

### Distance Penalty
If the goal is not reached, the reward is computed as the negative Euclidean
distance between the current joint positions and the goal joint angles. This
encourages the agent to minimize joint-space error.

### Action Penalty
To promote smooth control and avoid aggressive joint movements, the squared
action values are summed and added as a penalty term.

The final reward is a scaled combination of the distance penalty and action
penalty.

---

## 6. Episode Termination and Reset Logic

An episode is terminated under the following conditions:
- The goal is reached
- Invalid values (e.g., NaNs or infinities) appear in the observation

An episode is truncated if the agent fails to reach the goal within the maximum
episode length.

When a new episode begins, the environment is reset by:
- Resetting the step counter
- Resetting joint positions and velocities
- Randomizing the goal configuration
- Updating the visual target sphere

---

## 7. Training Pipeline

Training is performed using Proximal Policy Optimization (PPO) from the
Stable-Baselines3 library.

Key aspects of the training setup include:
- Vectorized environments for parallel rollout collection
- Observation and reward normalization to stabilize learning
- Periodic checkpoint saving using a custom callback

### Model Saving
A custom `SaveModelCallback` class is implemented to save model checkpoints at
user-defined intervals (e.g., every 500,000 or 10 million timesteps). This allows
training to be resumed or analyzed at intermediate stages.

If a previously saved model exists, training continues from that checkpoint;
otherwise, training starts from scratch.

Training statistics are logged using TensorBoard for monitoring loss curves,
reward trends, and learning stability.

---

## 8. Testing and Evaluation

The `test.py` script is used to evaluate trained models.

The script:
- Loads a trained PPO policy
- Instantiates the environment with `render_mode="human"`
- Runs multiple rollout episodes
- Uses the trained policy to predict actions at each timestep

Printed diagnostic values are included to observe how joint positions evolve
relative to the goal during execution.

---

## 9. Note:

Should you encounter an error “AttributeError: '***mujoco.\_structs.MjData' object has no attribute 'solver\_iter***'”, when trying to run the simulation, Here is the [solution](https://github.com/Farama-Foundation/Gymnasium/pull/746) 


**References/Helpful Guides**

1. [https://gymnasium.farama.org/environments/mujoco/](https://gymnasium.farama.org/environments/mujoco/)  
2. [https://gymnasium.farama.org/environments/mujoco/pusher/](https://gymnasium.farama.org/environments/mujoco/pusher/)  
3. [https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/pusher\_v4.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/pusher_v4.py)  
4. [https://mujoco.readthedocs.io/en/latest/overview.html](https://mujoco.readthedocs.io/en/latest/overview.html)  
5. [https://github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)  
6. [https://github.com/Farama-Foundation/Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)  
7. [https://github.com/techstartucalgary/RoboticArm](https://github.com/techstartucalgary/RoboticArm)  
8. [https://stable-baselines.readthedocs.io/en/master/guide/custom\_env.html](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)  
9. [https://roboticseabass.com/2020/08/15/introduction-to-deep-reinforcement-learning/](https://roboticseabass.com/2020/08/15/introduction-to-deep-reinforcement-learning/)  
10. https://roboticseabass.com/2020/08/02/an-intuitive-guide-to-reinforcement-learning/
11. [https://github.com/denisgriaznov/CustomMuJoCoEnviromentForRL](https://github.com/denisgriaznov/CustomMuJoCoEnviromentForRL)
12. [Mujoco Menagerie](https://github.com/google-deepmind/mujoco_menagerie)


***MJCF files are obtained from [Mujoco Menagerie](https://github.com/google-deepmind/mujoco_menagerie).***
