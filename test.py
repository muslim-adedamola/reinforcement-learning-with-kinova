from kinova_obs_env import KinovaObsEnv
from stable_baselines3 import PPO
import time

# Initialize the environment
test_env = KinovaObsEnv(render_mode="human")

# Load the pre-trained model
model = PPO.load("./kinova_obs_trainPPO50M.zip")   #change to "kinova_obs_trainPPO20M.zip" to test model trained on 20M timesteps

# Reset the environment
observation = test_env.reset_model()

try:
    # Run the simulation loop
    for step in range(20000):
        action, _states = model.predict(observation)
        observation, reward, done, truncated, info = test_env.step(action)
                
        current_joint_positions = observation[:7]
        angle_difference = current_joint_positions - test_env.goal_angles

        if step % 100 == 0:
            print("*****************************")
            print(f"Goal Angles: {test_env.goal_angles}")
            print(f"Current Joint Angles: {current_joint_positions}")
            print(f"Step: {step}, Angle Difference: {angle_difference}")
            print(reward)

        if done or truncated:
            observation = test_env.reset()
            break

        time.sleep(0.05)
finally:
    test_env.close()

