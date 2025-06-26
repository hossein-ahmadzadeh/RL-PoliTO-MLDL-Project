import os
import gym
import wandb
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *  # Triggers UDR via reset_model()

BEST_PARAMS = {
    "learning_rate": 0.0007081485369506237,
    "clip_range": 0.26785576292805646,
    "ent_coef": 0.01776300171237951,
    "n_steps": 2048,
    "batch_size": 128,
    "gamma": 0.987060729890745,
    "gae_lambda": 0.9779077684516388,
}

TOTAL_TIMESTEPS = 3_000_000
SAVE_PATH = "PPO_UDR_MODEL"
MODEL_NAME = "ppo_udr_source.zip"
LOG_DIR = "PPO/logs/ppo_udr_source"

class WandUDRCallback(BaseCallback):
    def __init__(self, print_every=1000, verbose=1):
        super().__init__(verbose)
        self.print_every = print_every
        self.episode_rewards = []
        self.recent_rewards = []
        self.smoothed = []
        self.variance = []
        self.cumulative_sim = 0.0
        self.cumulative_wall = 0.0
        self.cumulative_sim_times = []
        self.cumulative_wall_times = []
        self.wall_times = []
        self.episode_lengths = []
        self._episode_start_time = None
        self.window = 100
        os.makedirs(LOG_DIR, exist_ok=True)

    def _on_step(self) -> bool:
        if self._episode_start_time is None:
            self._episode_start_time = time.time()

        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep:
                reward = ep["r"]
                steps = ep["l"]
                sim_time = steps * 0.008
                wall_time = time.time() - self._episode_start_time
                self._episode_start_time = None

                self.episode_rewards.append(reward)
                self.recent_rewards.append(reward)
                self.wall_times.append(wall_time)
                self.episode_lengths.append(sim_time)
                self.cumulative_sim += sim_time
                self.cumulative_wall += wall_time
                self.cumulative_sim_times.append(self.cumulative_sim)
                self.cumulative_wall_times.append(self.cumulative_wall)

                if len(self.recent_rewards) > self.window:
                    self.recent_rewards.pop(0)

                smoothed = np.mean(self.recent_rewards) if len(self.recent_rewards) == self.window else None
                var = np.var(self.recent_rewards) if smoothed is not None else None

                if smoothed is not None:
                    self.smoothed.append(smoothed)
                    self.variance.append(var)

                wandb.log({
                    "reward_per_episode": reward,
                    "rolling_reward_mean_100": smoothed,
                    "rolling_reward_var_100": var,
                    "cumulative_sim_time": self.cumulative_sim,
                    "cumulative_wall_time": self.cumulative_wall
                }, step=self.num_timesteps)

                if self.verbose and len(self.episode_rewards) % self.print_every == 0:
                    print(f"--- Episode {len(self.episode_rewards)} ---")
                    print(f"  Reward: {reward:.2f}")
                    print(f"  Smoothed: {smoothed:.2f}" if smoothed else "")
                    print(f"  Sim Time: {sim_time:.2f}s | Wall Time: {wall_time:.2f}s")
        return True

    def _on_training_end(self) -> None:
        np.save(os.path.join(LOG_DIR, 'episode_rewards.npy'), self.episode_rewards)
        np.save(os.path.join(LOG_DIR, 'rolling_reward_mean_100.npy'), self.smoothed)
        np.save(os.path.join(LOG_DIR, 'rolling_reward_var_100.npy'), self.variance)
        np.save(os.path.join(LOG_DIR, 'cumulative_sim_time.npy'), self.cumulative_sim_times)
        np.save(os.path.join(LOG_DIR, 'cumulative_wall_time.npy'), self.cumulative_wall_times)
        np.save(os.path.join(LOG_DIR, 'episode_times_simulated.npy'), self.episode_lengths)
        np.save(os.path.join(LOG_DIR, 'episode_times_wallclock.npy'), self.wall_times)

# === Init W&B ===
wandb.init(
    project="PPO_UDR_Hopper",
    name="ppo_udr_source_train",
    config=BEST_PARAMS
)

# === Env Setup ===
env = Monitor(gym.make("CustomHopper-source-v0"))

# === Callbacks ===
eval_callback = EvalCallback(env, n_eval_episodes=50, eval_freq=50000,
                             best_model_save_path=SAVE_PATH, verbose=0)
wandb_callback = WandUDRCallback()
callback = CallbackList([eval_callback, wandb_callback])

# === Train PPO ===
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=BEST_PARAMS["learning_rate"],
    clip_range=BEST_PARAMS["clip_range"],
    ent_coef=BEST_PARAMS["ent_coef"],
    n_steps=BEST_PARAMS["n_steps"],
    batch_size=BEST_PARAMS["batch_size"],
    gamma=BEST_PARAMS["gamma"],
    gae_lambda=BEST_PARAMS["gae_lambda"],
    verbose=0
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# === Save ===
os.makedirs(SAVE_PATH, exist_ok=True)
model.save(os.path.join(SAVE_PATH, MODEL_NAME))

# === Final Evaluation ===
mean, std = evaluate_policy(model, env, n_eval_episodes=50)
wandb.log({"final_mean_reward": mean, "final_std_reward": std})
wandb.finish()
