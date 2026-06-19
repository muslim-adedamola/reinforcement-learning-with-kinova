import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from c_func import casadi_f0 as get_testpoint_pose
from stable_baselines3.common.callbacks import BaseCallback
import os
import mujoco


class DynamicObstacleEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100
    }

    def __init__(self, episode_len=2000, reset_noise_scale=1e-2, curriculum_stage=0, random_mix_prob=0.5, **kwargs):     
        utils.EzPickle.__init__(self, reset_noise_scale, curriculum_stage, random_mix_prob, **kwargs)       

        self.curriculum_stage = curriculum_stage
        self.random_mix_prob = random_mix_prob
        observation_space = Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64)
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            os.path.abspath("kinova_model_base/scene.xml"),
            5, # frame_skip: Number of MuJoCo simulation steps per gym `step()`. mj.step is 2ms
            observation_space=observation_space,
            **kwargs
        )
        self._init_curriculum()


    def _init_curriculum(self):
        self.episode_len = getattr(self, 'episode_len', 2500)
        self.step_number = 0
        self.goal_reached_count = 0
        self.goal_angles = np.zeros(7)
        self.obstacle_collision = False
        self.distance_to_obstacle = 2.5
        self.robot_spehere_radius = [0.12,0.12,0.12,0.12,0.06,0.06,0.10]
        self.workspace_limits = {
            'x': (-0.7, 0.7), 'y': (-0.7, 0.7), 'z': (1.5, 1.8)
        }
        # determine obstacle mode per episode
        self.is_moving_obst = False

        #reward function parameters
        self.rho = 3 #2 was used in paper
        self.eta = 0.05
        self.delta = 0.1
        self.distance_threshold = 0.35
        self.k = 1 #5 was used in the paper


    def set_curriculum_stage(self, stage: int):
        """Allow callback to advance curriculum."""
        self.curriculum_stage = stage


    def _choose_obstacle_mode(self):
        if self.curriculum_stage == 0:
            self.is_moving_obst = False
        elif self.curriculum_stage == 1:
            self.is_moving_obst = True
        else:  # mixed
            self.is_moving_obst = (np.random.rand() < self.random_mix_prob)


    #to print model info for debugging and for info about environment
    def print_model_info(self):
        print("qpos shape:", self.data.qpos.shape)
        print("qvel shape:", self.data.qvel.shape)
        print("ctrl shape:", self.data.ctrl.shape)
        print("ctrl array is:", self.data.ctrl)
        print("model nv is:", self.model.nv)
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


    #get observations from environment
    def _get_obs(self):
        current_joint_positions = self.data.qpos.flat[:7]
        current_joint_velocities = self.data.qvel.flat[:7]

        goal_position_diff = current_joint_positions - self.goal_angles
    
        T1 = get_testpoint_pose(np.array(current_joint_positions).reshape(1, -1))

        end_effector_position = np.array([T1[0][21], T1[0][22], T1[0][23] + 1.2])

        T2 = get_testpoint_pose(np.array(self.goal_angles).reshape(1, -1))
    
        goal_position_fk = np.array([T2[0][21], T2[0][22], T2[0][23] + 1.2])
    
        dist_to_goal = np.linalg.norm(end_effector_position - goal_position_fk)

        return np.concatenate([
            current_joint_positions,  
            current_joint_velocities,  
            goal_position_diff,
            np.array([self.distance_to_obstacle]),
            np.array([self.obstacle_collision], dtype=np.float64),
            np.array([dist_to_goal])  
        ])


    #set positions of sphere randomly
    def _set_sphere_position(self):
        obstacle_marker_name = "sphere1"
        obstacle_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obstacle_marker_name)

        if obstacle_marker_id == -1:
            raise ValueError(f"Body named '{obstacle_marker_name}' not found in the model.")
        
        # Generate a new position within workspace limits
        x = np.random.uniform(self.workspace_limits['x'][0], self.workspace_limits['x'][1]) 
        y = np.random.uniform(self.workspace_limits['y'][0], self.workspace_limits['y'][1])
        z = np.random.uniform(self.workspace_limits['z'][0], self.workspace_limits['z'][1])
        new_position = np.array([x, y, z])

        # Set the new position for the body
        self.model.body_pos[obstacle_marker_id] = new_position


    #simulate motion of moving sphere
    def _set_sphere_motion(self, step_number):
        obstacle_marker_name = "sphere1"
        obstacle_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obstacle_marker_name)

        if obstacle_marker_id == -1:
            raise ValueError(f"Body named '{obstacle_marker_name}' not found in the model.")

        #movement speed and limits for x and y
        speed = 0.005 
        x_range = (-1, 1)
        y_range = (-1, 1)
        fixed_height = 1.7

        # Compute new x and y positions as a function of step_number
        x_position = x_range[0] + (step_number * speed) % (x_range[1] - x_range[0])
        y_position = y_range[0] + (step_number * speed) % (y_range[1] - y_range[0])
        
        # Update the obstacle position
        self.model.body_pos[obstacle_marker_id] = [x_position, y_position, fixed_height]


    #set the goal pose
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
            T = get_testpoint_pose(np.array(self.goal_angles).reshape(1, -1))
            x_goal, y_goal, z_goal = T[0][21], T[0][22], T[0][23]
            z_goal = z_goal + 1.2

            if (self.workspace_limits['x'][0] <= x_goal <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y_goal <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z_goal <= self.workspace_limits['z'][1]):

                goal_position = np.array([x_goal, y_goal, z_goal])
                self._label_goal_pose(goal_position)
                return self.goal_angles, goal_position


    def set_speical_goal(self, special_goal):
        self.goal_angles = np.array(special_goal)
        T = get_testpoint_pose(np.array(self.goal_angles).reshape(1, -1))
        x_goal, y_goal, z_goal = T[0][21], T[0][22], T[0][23]
        z_goal = z_goal + 1.2
        goal_position = np.array([x_goal, y_goal, z_goal])
        self._label_goal_pose(goal_position)
        return self.goal_angles, goal_position

    def set_speical_obstacle(self, speical_obstacle):
        obstacle_marker_name = "sphere1"
        obstacle_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obstacle_marker_name)

        if obstacle_marker_id == -1:
            raise ValueError(f"Body named '{obstacle_marker_name}' not found in the model.")
        
        # Update the obstacle position
        self.model.body_pos[obstacle_marker_id] = [speical_obstacle[0], speical_obstacle[1], speical_obstacle[2]]
    
    def set_speical_state(self, q_state):
        for i in range(7):
            self.data.qpos[i] = q_state[i]

    def viewer_setup(self):
        #self.viewer.cam.trackbodyid = 0         # id of the body to track ()
        self.viewer.cam.distance = 3.0         # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] = 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = -0.25
        self.viewer.cam.lookat[2] = 0.824
        self.viewer.cam.elevation = -45           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 180              # camera rotation around the camera's vertical axis

    #label goal pose
    def _label_goal_pose(self, position):
        # Target marker
        goal_marker_name = "target"
        goal_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_marker_name)
    
        if goal_marker_id == -1:
            raise ValueError(f"Body named '{goal_marker_name}' not found in the model.")
    
        self.model.body_pos[goal_marker_id] = position


    #calculate test point positions on robot
    def _calculate_joint_positions(self, current_joint_positions):
        joint_positions = np.zeros((8, 3))

        current_joint_positions = np.array(current_joint_positions)
        sphere_positions = get_testpoint_pose(np.array(current_joint_positions).reshape(1, -1))

        sphere_positions = sphere_positions[0]

        count = 0

        for i in range(len(joint_positions)):
            for j in range(joint_positions.shape[1]):
                joint_positions[i, j] = sphere_positions[count]
                count += 1

        joint_positions[:, -1] += 1.2   # to account for height of table as it wasn't in c_func
        
        return joint_positions


    #calculate ssm
    def calculate_ssm(self):
        obstacle_pos = self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere1")]
        test_point_positions = []

        current_joint_positions = self.data.qpos[:7]
        test_point_positions = np.array(get_testpoint_pose(np.array(current_joint_positions).reshape(1, -1))).reshape(8, 3)
        test_point_positions = test_point_positions + np.array([0.0, 0.0, 1.2]) # add table offset

        #compute ssm using the caluclated test_point_velocities
        min_dist  = 10000.0;      
        for t in range(1, 8):
            dist = np.array(test_point_positions[t] - obstacle_pos)
            dist = np.linalg.norm(dist) - 0.05 - self.robot_spehere_radius[t-1]
            if min_dist>dist: min_dist = dist

        if min_dist < 0.0:
            self.obstacle_collision = True
        return min_dist


    def _compute_reward(self, goal_reached, current_joint_positions, distance_to_obstacle, joint_velocities, check_closeness):
        vec_1 = current_joint_positions - self.goal_angles

        avoid_reward = -(1 / (1 + distance_to_obstacle))**self.rho
        
        if goal_reached:
            done_reward = 500
            self.goal_reached_count += 1
        elif self.obstacle_collision:
            done_reward = -200
        elif check_closeness:
            done_reward = 200
        else:
            done_reward = 0

        target_reward = -np.linalg.norm(vec_1)

        reward_task = target_reward + avoid_reward + done_reward

        reward_energy = -np.sum(np.tanh(np.linalg.norm(joint_velocities)))

        x_value = (distance_to_obstacle - self.distance_threshold) / self.k

        beta = (np.tanh(x_value) + 1) / 2

        total_reward = (reward_task * beta) + ((1 - beta) * reward_energy)

        return total_reward 


    def step(self, action):
        if self.is_moving_obst:
            self._set_sphere_motion(self.step_number)

        self.do_simulation(action, self.frame_skip)

        self.step_number += 1

        self.distance_to_obstacle = self.calculate_ssm()
        #print('Distance to obstacle:', self.distance_to_obstacle)

        observation = self._get_obs()

        is_finite = np.isfinite(observation).all()
        current_joint_positions = observation[:7]
        goal_reached = np.allclose(current_joint_positions, self.goal_angles, atol=5e-2)
        check_closeness = np.allclose(current_joint_positions, self.goal_angles, atol=0.1)

        reward = self._compute_reward(goal_reached, current_joint_positions, self.distance_to_obstacle, joint_velocities=observation[7:14], check_closeness=check_closeness)
        done = not is_finite or goal_reached
        truncated = self.step_number > self.episode_len

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, truncated, {}


    def reset_model(self):
        self.step_number = 0
        self.obstacle_collision = False
        self.distance_to_obstacle = 2.5
        # choose static/dynamic based on curriculum
        self._choose_obstacle_mode()

        # set qpos, qvel
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(noise_low, noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(noise_low, noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        self._set_goal_pose()
        if self.is_moving_obst:
            self._set_sphere_motion(self.step_number)
        else:
            self._set_sphere_position()

        return self._get_obs()


# SaveModelCallback
class SaveModelCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Saving model at timestep {self.num_timesteps}")
        return True


# ####################################################################
# def main():
#     #Create an instance of the environment
#     env = KinovaObsEnv()

#     #Reset the environment to initialize the state
#     env.reset_model()
#     print("*******************************************")

#     env._set_goal_pose()
    
#     env.print_model_info()
    
#     #Print the indices of the joint positions
#     #print("Joint position indices:", joint_indices)

#     print("*******************************************")
    
#     #Perform a step to see the output
#     action = np.zeros(env.action_space.shape) 
#     observation, reward, done, truncated, _ = env.step(action)
#     print("Observation after step:", observation)
#     print("Reward:", reward)
#     print("Done:", done)
#     print("Truncated:", truncated)
#     print('Action', action)
#     print(env.action_space.shape)

# if __name__ == "__main__":
#     main()
