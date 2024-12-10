import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from kinova_fk import ForwardKinematics
from stable_baselines3.common.callbacks import BaseCallback
import os
import mujoco


class KinovaObsEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes" : [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps" : 100
    }


    def __init__(self, episode_len=2000, reset_noise_scale=1e-2, **kwargs):
        utils.EzPickle.__init__(self, reset_noise_scale, **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            os.path.abspath("kinova_model/scene.xml"),
            5,
            observation_space=observation_space,
            **kwargs
            )
        
        self.episode_len = episode_len
        self.step_number = 0
        self.goal_reached_count = 0
        self.goal_angles = np.zeros(7)
        self.fk = ForwardKinematics()

        #workspace limit of robot
        self.workspace_limits = {
            'x' : (-0.7, 0.7),
            'y' : (-0.7, 0.7),
            'z' : (0.8, 1.2)
        }

        #self.print_model_info()

    #print model info
    def print_model_info(self):
        print("qpos shape:", self.data.qpos.shape)
        print("qvel shape:", self.data.qvel.shape)
        print("ctrl shape:", self.data.ctrl.shape)
        print("Sample qpos:", self.data.qpos[:10])
        print("Sample qvel:", self.data.qvel[:10])

        a = [self.model.name_jntadr[i] for i in range(self.model.njnt)]
        b = [self.model.name_bodyadr[i] for i in range(self.model.nbody)]
        n_obj = self.model.njnt
        m_obj = self.model.nbody

        # Print joint and body names
        id2name = {i: None for i in range(n_obj)}
        name2id = {}

        id2name2 = {j: None for j in range(m_obj)}
        name2id2 = {}

        for count in a:
            name = self.model.names[count:].split(b"\x00")[0].decode()
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert 0 <= obj_id < n_obj and id2name[obj_id] is None
            name2id[name] = obj_id
            id2name[obj_id] = name
        
        for count2 in b:
            name = self.model.names[count2:].split(b"\x00")[0].decode()
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert 0 <= obj_id < m_obj and id2name2[obj_id] is None
            name2id2[name] = obj_id
            id2name2[obj_id] = name

        print("Joint names:", tuple(id2name[id] for id in sorted(name2id.values())))
        print("Joint name to ID mapping:", name2id)
        print("Joint ID to name mapping:", id2name)

        print("Body names:", tuple(id2name2[id] for id in sorted(name2id.values())))
        print("Body name to ID mapping:", name2id2)
        print("Body ID to name mapping:", id2name2)


    #get observations
    def _get_obs(self):
        
        return np.concatenate([
            self.data.qpos.flat[:7],
            self.data.qvel.flat[:7],
            self.data.qpos.flat[:7] - self.goal_angles
            ])


    #set positions of sphere randomly
    def _set_sphere_position(self):
        # Ensure the correct type for the entity is used
        body_name = "sphere"
    
        # Get the body ID by name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
        if body_id == -1:
            raise ValueError(f"Body named '{body_name}' not found in the model.")

        # Generate a new position within workspace limits
        x = np.random.uniform(self.workspace_limits['x'][0], self.workspace_limits['x'][1])
        y = np.random.uniform(self.workspace_limits['y'][0], self.workspace_limits['y'][1])
        z = np.random.uniform(self.workspace_limits['z'][0], self.workspace_limits['z'][1])
        new_position = np.array([x, y, z])

        # Set the new position for the body
        self.model.body_pos[body_id] = new_position


    def _set_goal_pose(self):
        while True:
            joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
            angles = []

            for joint_name in joints:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id == -1:
                    raise ValueError(f"Joint named '{joint_name}' not found in the model.")
                low_limit = self.model.jnt_range[joint_id, 0] if self.model.jnt_limited[joint_id] else -np.pi
                high_limit = self.model.jnt_range[joint_id, 1] if self.model.jnt_limited[joint_id] else np.pi
                random_angle = np.random.uniform(low_limit, high_limit)
                angles.append(random_angle)

            self.goal_angles = np.array(angles)
            T = self.fk.forward_kinematics(self.goal_angles)
            T_table_offset = np.array([
                [1, 0, 0, 0.0],
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.6],
                [0, 0, 0, 1]
            ])
            T = T_table_offset @ T
            x_goal, y_goal, z_goal = T[0, 3], T[1, 3], T[2, 3]

            if (self.workspace_limits['x'][0] <= x_goal <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y_goal <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z_goal <= self.workspace_limits['z'][1]):

                goal_position = np.array([x_goal, y_goal, z_goal])
                self._label_goal_pose(goal_position)
                return self.goal_angles, goal_position


    #set random goal position for cartesian space
    def _label_goal_pose(self, position):
        # Target marker
        goal_marker_name = "target"
        goal_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_marker_name)
    
        if goal_marker_id == -1:
            raise ValueError(f"Body named '{goal_marker_name}' not found in the model.")
    
        self.model.body_pos[goal_marker_id] = position


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        self.step_number += 1

        observation = self._get_obs()

        # Check if observation contains only finite values
        is_finite = np.isfinite(observation).all()
        current_joint_positions = observation[:7]

        # Check if current joint positions are close to the goal
        goal_reached = np.allclose(current_joint_positions, self.goal_angles, atol=1e-2) 

        if goal_reached:
            self.goal_reached_count += 1
            reward = 10
        else:       
            vec_1 = current_joint_positions - self.goal_angles
            reward_dist = -np.linalg.norm(vec_1)
            reward_ctrl = -np.square(action).sum()
            reward = 0.5 * reward_dist + 0.1 * reward_ctrl

        done = not is_finite or goal_reached
        truncated = self.step_number > self.episode_len


        if self.render_mode == "human":
            self.render()
        return observation, reward, done, truncated, {}

    

    #reset, restart simulation
    def reset_model(self):
        self.step_number = 0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale       

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
  
        self.set_state(qpos, qvel)
        self._set_goal_pose()
        
        self._set_sphere_position()

        observation = self._get_obs()
        return observation


#save model at intervals
class SaveModelCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Saving model at timestep {self.num_timesteps}")
        return True
