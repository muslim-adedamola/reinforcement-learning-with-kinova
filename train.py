import os
import time
from kinova_obs_env import KinovaObsEnv, SaveModelCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize 

n_envs = 250
n_episodes = 400
episode_len = 2500 
render_mode = "rgb_array"
save_freq = 50000000 // n_envs
save_path = "/home/arms/muslim_kinova_rl/models"  
model_save_name = "kinova_obs_1"

def make_env():
    return KinovaObsEnv(episode_len=episode_len, render_mode=render_mode)

env = make_vec_env(make_env, n_envs)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Create a callback for saving the model
save_callback = SaveModelCallback(check_freq=save_freq, save_path=save_path, verbose=1)

# Check if a saved model exists
if os.path.exists(f"{save_path}/{model_save_name}.zip"):
    print("Loading existing model...")
    model = PPO.load(f"{save_path}/{model_save_name}", env=env)

    start_time = time.time()
    
    additional_timesteps = 50000000  # Adjust as needed
    model.learn(total_timesteps=additional_timesteps, log_interval=4, callback=save_callback, progress_bar=True)

else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_kinova_obs_env_tensorboard/")

    start_time = time.time()

    total_timesteps = 300_000_000 
    model.learn(total_timesteps=total_timesteps, log_interval=4, callback=save_callback, progress_bar=True)

# After loading or training, continue with post-training tasks
goal_reaches = env.get_attr('goal_reached_count')
print(f"Total goals reached during training: {sum(goal_reaches)}")

end_time = time.time()
training_duration_seconds = end_time - start_time
training_duration_hours = training_duration_seconds / 3600 
print(f"Training took {training_duration_hours:.2f} hours.")

# Save the final model
model.save(f"{save_path}/{model_save_name}")

# Save the vector normalization statistics
if isinstance(env, VecNormalize):
    env.save("vec_normalize_stats.pkl")
