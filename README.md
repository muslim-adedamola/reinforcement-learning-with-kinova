# reinforcement-learning-with-kinova
## An intro for newbies.
![Demo](assets/a.gif)


**Documentation**

**To catch up very quickly with the theory, maths, terms, and the concept, I recommend reading reference 10 first.**

**To start from scratch,** 

1. Create and activate a conda environment or any virtual environment of your choice.   
2. Do “pip install \-r requirements.txt” to install all dependencies including the Mujoco software.  
3. Do “python test.py” or “python3 test.py” depending on if you have python3 and if it is linked as the default on your os.  
4. You should see a simulation of a Kinova robot on a table trying to achieve a set of joint angles and also trying to reach a goal point as seen in the attached sample videos in the folder.  
5. Should you encounter an error “AttributeError: '***mujoco.\_structs.MjData' object has no attribute 'solver\_iter***'”, when trying to run the simulation, Here is the [solution](https://github.com/Farama-Foundation/Gymnasium/pull/746) 

**Models**  
kinova\_obs\_trainPPO20M.zip —\> Model trained on 20 million timesteps  
kinova\_obs\_trainPPO50M.zip —\> Model trained on 50 million timesteps 

**Quick-Readme.txt**  
Provides a very quick overview; Training of Kinova Robot to reach any random goal position in the absence of obstacles

**Folders**

**Kinova\_model**: Contains the model of the kinova robot and scene in mujoco (mjcf) format. The robot is spawned in the scene.xml  
Also contained in the scene are the obstacles and sphere indicating the goal point.

**Models:** Contains and saves the model during and after training.

**PPO\_kinova\_obs\_env\_tensorboard:** Contains the events file for logging and training information. This is used by **Tensorboard** which displays the loss and other parameter curves during training.

**Files**

**Requirements.txt:** contains versions of dependencies, packages, libraries needed to set up the project. It is advised to use a virtual environment when setting up to avoid conflict with package versions  
**Sample Videos:** demonstration of results.

**Kinova\_fk.py:** The forward kinematics code of the Kinova robot. Has a constructor which automatically sets the DH parameters which are used in calculating the homogeneous transformation matrices starting from the base till the end effector. The function **forward\_kinematics()** takes a list of the 7 joint angles as input and returns the position of the end effector. Code is documented for other information.

**Kinova\_obs\_env.py:** Defines, builds, and customizes the simulation environment. It inherits from the base MujocoEnv environment. Also has a constructor which sets the episode length, observation space, (observation space is 21; given that we are observing the joint positions of robot (7) \+ velocity of the joints (7) \+ the difference between the joint positions at any instant and the goal angles set for the joints (7)). Observation space is **Box** since we are dealing with real values and have boundaries of negative infinity to positive infinity. The constructor also sets the environment via **scene.xml.**

The workspace of the kinova robot is defined (set to be above the dimensions of the table; the model of the table is a simple shape and can be found in scene.xml).

*def print\_model\_info*: Prints information about the joints, joint IDs, bodies, bodies IDs, shape of joint positions, joint velocities, and actuators. I included this as it was useful during debugging and to make sure I was addressing the correct joint/body when necessary. It involves iterating the number of joints and bodies present in the environment, putting it in a list, indexing it and extracting the names and values.

*def get\_obs*: According to the name, it gets the observation (which can be interpreted as the state of the agent) from the environment. For this simple task, the state/observation are the *“joint positions”,* “*joint velocities*”, and the *“difference between the joint positions and the target and goal angles after the agent takes a step”*.  

*def set\_sphere\_position:* This method is intended to set the position of one red sphere representing the obstacle in the environment. The sphere obstacle has been created in the scene.xml file, and this function takes thi sphere, gets the body ID from the name, and sets its 3D position within the workspace limits previously defined in the constructor.

*def \_label\_goal\_pose*: This function takes a 3D position as argument and sets the position of a tiny green sphere named target in the scene.xml file to the position argument. Similar to the approach used in *set\_sphere\_position*, the name of the target sphere is given, the ID extracted and used in setting the position.   

*def set\_goal\_pose:* This function takes the 7 joints of the kinova robot, puts it in a list, and iterates over the 7 joints. While iterating, it checks if each joint has predefined limits in the gen3.xml file. If not, it sets the lower and upper limits to \-np.pi and np.pi. After setting these limits, it then generates a random angle within these limits. The angles generated for the 7 joints become the goal angles to be achieved/reached by the robot. 

These goal angles are then passed to the forward kinematics so the ideal end effector position can be returned. Once we have the ideal/goal end effector position (calculated by the goal angles), we pre-multiply with the height of the table scene since the robot sits on the table therefore elevating its Z-position. The XYZ position of the end effector (remember, it is for the goal angles) is then checked if it is within the workspace limits. If it is, the *label\_goal\_pose* function is called, and the green sphere target is given this position, to help us indicate where the end effector should be when the goal angles are achieved by the robot. If the XYZ position is not within the workspace limits, the loop starts again and the loop only ends/breaks when the XYZ position is within the workspace limits. 

*def step*: In this method, the agent takes a step, that is, performs an action, and the step number is incremented. After this step is taken, the state/observation of the agent is obtained (which now contains the current joint positions, joint velocities and the difference between the goal angles and current joint angles.), from the *get\_obs()* method. The state values are checked to ensure there are no infinite values, and the current joint positions are extracted from the observation. Then the *goal\_reached* condition is defined; it checks if the *current\_joint\_positions* are close to the *goal\_angles.* Confirms if the difference between each corresponding joint position and goal angle is less than 0.01.   
**Reward:** If a goal is reached, a counter is incremented (to allow us to check for the number of times the robot reached goals during training) and a reward of 10 is applied. Otherwise, we calculate the difference between each joint position and goal angle, and the Euclidean distance of this vector difference. We make this negative since we want to close the difference between the current joint positions and goal angles. A Euclidean distance of 13 means they are 13 units apart, since we want to reduce this distance, we add a negative sign. Euclidean distance of \-9 means they are close compared to \-20. We also penalize the control action (taken by the joints) by squaring the actions of the 7 joints and adding them, to ensure smooth, safe motion of motors at the 7 joints. We then scale these rewards and add them.   
There is a *done* condition to check if the episode is done. The episode is complete if either a goal is reached or the observation contains invalid values. The episode is truncated if the agent does not reach the goal point and the number of steps taken by the agent exceeds episode length.

*def reset\_model:* This basically resets the environment and state once a new episode begins. It resets the step\_number counter to 0, the joint positions and velocities, and also randomizes the position of the obstacle and goal sphere in the environment.

*Class SaveModelCallback:* Also a custom defined class to create the functionality of saving the models at a defined timestep when training. Say, you are training for 20 million timesteps byt would like to save the models at intervals 500,000 timesteps or 10 million timesteps, this class gives the functionality. It inherits from BaseCallback from stable\_baselines3 library, and has a constructor which checks the frequency of save and the *save\_path.* Also an *on\_step* function which checks the timestep and interval to confirm when to save the model.

**Test.py:** Loads the trained model and creates an instance of the customized Kinova environment and sets render mode to “human”, for visualization. Resets the environment for fresh values and runs the simulation loop for a number of times. The trained model is used to predict and give an action which is passed and used by the agent to take a step in the environment. If the episode ends or is truncated, the environment is reset. The print values are for feedback and to see how the goal angles and joint position values at any given instant.

**Train.py:** Training script, sets the required training variables such as *episode length, render\_mode, save\_path and save\_freq and model\_save\_name.* An instance of the Kinova environment is created by *make\_env* function, and this environment is vectorized to allow for parallel training simultaneously. [See](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) for more info/details on vectorization. For easier training and to prevent huge values, vectorized environments are normalized. A callback is created to check for save\_frequency (what interval to save the models at). The *if-else loop* checks if a saved model exists. If it does, it continues training with the additional timesteps using the weights and parameters of the existing model. If not, a new training is started from scratch with the set timesteps. We get the number of times the goal is reached, to allow us count during training, set time variables to allow us count duration of training, and save the model when training is complete. The PPO algorithm with its default parameters is used for training. The arguments used for training include “MlpPolicy” which means we are using an actor-critic method that uses neural networks. [See](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) for more info on arguments of PPO algorithm

**NB:** The methods ***reset\_model, step, get\_obs*** are initialized in the original MujocoEnv and must be completed/defined/customized for your use. Other functions are custom to ensure other features/capabilities are met.

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

***If you have any questions, contact me***

***Also, special thanks to Artemiy, my lab colleague who made this work easy to start by providing DH parameters and transformation matrices of Kinova Gen3 Robot. He also provided the integrated version of Scene.xml, Kinova Gen3.xml and the Robotiq Gripper in Mujoco formats (mjcf) all from [Mujoco Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Special thanks to them too***
