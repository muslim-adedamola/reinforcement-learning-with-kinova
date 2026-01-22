import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import c_func
from stable_baselines3.common.callbacks import BaseCallback
import os
import mujoco


class KinovaObsEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100
    }

    def __init__(self, episode_len=2000, reset_noise_scale=1e-2, **kwargs):
        utils.EzPickle.__init__(self, reset_noise_scale, **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(76,), dtype=np.float64)
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
        self.obstacle_collision = False
        self.table_collision = False
        self.min_distance_to_obstacle_spheres = ([0.5, 0.5, 0.5, 0.5])
        self.collision_flags = np.zeros(4, dtype=bool) 
        self.previous_link_distances = np.zeros((4, 8))
        self.current_link_distances = np.zeros((4, 8))  # 4 obstacles, 8 links
        self.delta_link_distances = np.zeros((4, 8)) 
        #self.link_sphere_positions = np.zeros((8, 3))  #positions of the 7 link spheres

        # Workspace limits of the robot
        self.workspace_limits = {
            'x': (-0.7, 0.7),
            'y': (-0.7, 0.7),
            'z': (0.8, 1.2)
        }

        self.obstacle_sphere_positions = np.zeros((4, 3))  # Positions of the 4 obstacle spheres

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


    def _get_obs(self):
        current_joint_positions = self.data.qpos.flat[:7]
        current_joint_velocities = self.data.qvel.flat[:7]
        goal_position_diff = current_joint_positions - self.goal_angles

        end_effector_position, goal_position_fk = self._compute_end_effector_pos(current_joint_positions, self.goal_angles)
    
        dist_to_goal = np.linalg.norm(end_effector_position - goal_position_fk)

        return np.concatenate([
            current_joint_positions,  
            current_joint_velocities,  
            goal_position_diff,
            self.obstacle_sphere_positions.flatten(),
            self.min_distance_to_obstacle_spheres,   
            self.collision_flags,  
            np.array([self.obstacle_collision], dtype=np.float64),
            np.array([self.table_collision], dtype=np.float64),
            self.delta_link_distances.flatten(),
            np.array([dist_to_goal]),
    ])


    def _set_sphere_positions(self):
        #minimum distance of obstacles from base of robot since base can't move
        min_distance_from_base = 0.2

        # Set positions of the 4 obstacle spheres randomly within the workspace limits
        for i in range(4):
            while True:
                x = np.random.uniform(self.workspace_limits['x'][0], self.workspace_limits['x'][1])
                y = np.random.uniform(self.workspace_limits['y'][0], self.workspace_limits['y'][1])
                z = np.random.uniform(self.workspace_limits['z'][0], self.workspace_limits['z'][1])
                obstacle_position = np.array([x, y, z])

                base_position = np.array([0, 0, 0.6])

                distance_to_base = np.linalg.norm(obstacle_position - base_position) #check distance of obstacle from base

                if distance_to_base > min_distance_from_base:
                    self.obstacle_sphere_positions[i] = obstacle_position
                    # Set the new position for the body (obstacle)
                    sphere_name = f"sphere{i + 1}"  # Unique names for each obstacle
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, sphere_name)
                    self.model.body_pos[body_id] = self.obstacle_sphere_positions[i]
                    break


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
            T = c_func.casadi_f0(np.array(self.goal_angles).reshape(1, -1))
            x_goal, y_goal, z_goal = T[0][21], T[0][22], T[0][23]
            z_goal = z_goal + 0.5   #to account for table's height as it wasn't in c_func

            if (self.workspace_limits['x'][0] <= x_goal <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y_goal <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z_goal <= self.workspace_limits['z'][1]):

                goal_position = np.array([x_goal, y_goal, z_goal])
                self._label_goal_pose(goal_position)
                return self.goal_angles, goal_position


    def _label_goal_pose(self, position):
        # Target marker
        goal_marker_name = "target"
        goal_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_marker_name)
    
        if goal_marker_id == -1:
            raise ValueError(f"Body named '{goal_marker_name}' not found in the model.")
    
        self.model.body_pos[goal_marker_id] = position


    def _calculate_joint_positions(self, current_joint_positions):
        joint_positions = np.zeros((8, 3))

        current_joint_positions = np.array(current_joint_positions)
        sphere_positions = c_func.casadi_f0(np.array(current_joint_positions).reshape(1, -1))

        sphere_positions = sphere_positions[0]

        count = 0

        for i in range(len(joint_positions)):
            for j in range(joint_positions.shape[1]):
                joint_positions[i, j] = sphere_positions[count]
                count += 1

        joint_positions[:, -1] += 0.5   # to account for height of table as it wasn't in c_func
    
        return joint_positions


    def _update_collision_info(self, joint_positions):
        ########################################################################
        # for x in range(len(joint_positions)):
        #     self.link_sphere_positions[x] = joint_positions[x]

        #     # Set the new position for the body (obstacle)
        #     sphere_name = f"link_sphere{x + 1}"
        #     body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, sphere_name)
        #     self.model.body_pos[body_id] = self.link_sphere_positions[x]
        ###########################################################################

        # Compute current link distances
        for i in range(4):  # For each obstacle
            for j in range(8):  # For each link
                self.current_link_distances[i, j] = np.linalg.norm(joint_positions[j] - self.obstacle_sphere_positions[i])

        # Compute delta distances before updating previous distances
        self.delta_link_distances = self.current_link_distances - self.previous_link_distances

        # Update previous distances
        self.previous_link_distances = np.copy(self.current_link_distances)

        # Collision detection
        for i in range(4):
            self.min_distance_to_obstacle_spheres[i] = np.min(self.current_link_distances[i])
            self.collision_flags[i] = self.min_distance_to_obstacle_spheres[i] < 0.05

        self.obstacle_collision = np.any(self.collision_flags)
        if self.obstacle_collision:
            print("Collision with obstacle++++++")

        #for table collision
        table_height = 0.5
        table_min_distance = 0.05

        for i in range(8):
            link_pos = joint_positions[i]
            distance_to_table = link_pos[2] - table_height   # only z coordinate is relevant

            if distance_to_table < table_min_distance:
                self.table_collision = True
                break
            #     print("Collision with table---------")
            #     break
            # else:
            #     print("no collision with table")

        return

    #calculate end_effector position at any given instant
    def _compute_end_effector_pos(self, current_joint_positions, goal_angles):

        A1 = c_func.casadi_f0(np.array(current_joint_positions).reshape(1, -1))
        current_end_effector_position = np.array([A1[0][21], A1[0][22], A1[0][23] + 0.5])

        A2 = c_func.casadi_f0(np.array(goal_angles).reshape(1, -1))
        goal_position_end_eff = np.array([A2[0][21], A2[0][22], A2[0][23] + 0.5])

        return current_end_effector_position, goal_position_end_eff
    

    def _compute_reward(self, goal_reached, current_joint_positions, action):
        if self.obstacle_collision:
            return -500

        if self.table_collision:
            return -400
        
        if goal_reached:
            self.goal_reached_count += 1
            print("goal reached")
            return 200

        vec_1 = current_joint_positions - self.goal_angles
        reward_dist = -np.linalg.norm(vec_1)

        current_end_effector_position, goal_position_fk  = self._compute_end_effector_pos(current_joint_positions, self.goal_angles)

        distance_to_goal = np.linalg.norm(current_end_effector_position - goal_position_fk)

        reward_goal = -distance_to_goal

        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_goal + 0.1 * reward_ctrl

        # #near collision penalties
        # for i in range(4):  # For each obstacle
        #     if self.min_distance_to_obstacle_spheres[i] < 0.1:  # Near-collision threshold
        #         reward -= (1.0 - self.min_distance_to_obstacle_spheres[i]) * 100

        for i in range(4):  # For each obstacle
            for j in range(8):  # For each link
                if self.delta_link_distances[i, j] < 0:  # Improvement threshold
                    reward += 15 * self.delta_link_distances[i, j]

        return reward

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.step_number += 1

        # Update collision info before constructing the observation
        joint_positions = self._calculate_joint_positions(self.data.qpos.flat[:7])
        self._update_collision_info(joint_positions)

        # Get observation
        observation = self._get_obs()

        # Compute reward
        current_joint_positions = observation[:7]
        current_end_effector, goal_end_effector = self._compute_end_effector_pos(current_joint_positions, self.goal_angles)
        goal_reached = np.allclose(current_joint_positions, self.goal_angles, atol=1e-2) or np.allclose(current_end_effector, goal_end_effector, atol=1e-2)
        reward = self._compute_reward(goal_reached, current_joint_positions, action)

        # Check termination conditions
        done = not np.isfinite(observation).all() or goal_reached or self.obstacle_collision or self.table_collision
        truncated = self.step_number > self.episode_len

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, truncated, {}


    def reset_model(self):
        self.step_number = 0
        self._set_sphere_positions()
        self._set_goal_pose()
        self.obstacle_collision = False
        self.table_collision = False
        self.previous_link_distances = np.zeros((4, 8))
        self.delta_link_distances = np.zeros((4, 8))
        self.collision_flags = np.zeros(4, dtype=bool)

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale       

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        joint_positions = self._calculate_joint_positions(self.data.qpos.flat[:7])
        self._update_collision_info(joint_positions)
        
        observation = self._get_obs()
        return observation


# SaveModelCallback class remains the same
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
