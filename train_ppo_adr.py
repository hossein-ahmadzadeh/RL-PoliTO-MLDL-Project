# train_ppo_adr_sweep.py

import os
import time
import gym
import wandb
import numpy as np
import argparse

# Ensure custom_hopper.py is in ./env/ and the folder has an __init__.py file
from env.custom_hopper import CustomHopper
from ADR import AutomaticDomainRandomization, ADRCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

# --- Constants and Hyperparameters ---
BEST_PARAMS = {
    "learning_rate": 0.0007081485369506237,
    "clip_range": 0.26785576292805646,
    "ent_coef": 0.01776300171237951,
    "n_steps": 2048,
    "batch_size": 128,
    "gamma": 0.987060729890745,
    "gae_lambda": 0.9779077684516388,
}

THIGH_MEAN_MASS = 3.92699082
LEG_MEAN_MASS = 2.71433605
FOOT_MEAN_MASS = 5.0893801

# --- W&B Sweep Configuration ---
sweep_configuration = {
    "method": "grid",
    "name": "ADR-Delta-Sweep",
    "metric": {"name": "eval/mean_reward", "goal": "maximize"},
    "parameters": {
        "delta": {"values": [0.02, 0.05, 0.1]},
    }
}

class DetailedLoggingCallback(BaseCallback):
    """Logs detailed episode data to W&B for plotting."""
    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.recent_rewards = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                if "episode" in info:
                    ep_info = info["episode"]
                    reward, length = ep_info["r"], ep_info["l"]
                    
                    self.recent_rewards.append(reward)
                    if len(self.recent_rewards) > self.window_size: self.recent_rewards.pop(0)
                    
                    log_data = {"reward/episode_reward": reward, "episode_length": length}
                    if len(self.recent_rewards) == self.window_size:
                        log_data["reward/rolling_mean_100"] = np.mean(self.recent_rewards)
                    
                    wandb.log(log_data, step=self.num_timesteps)
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    n_envs = os.cpu_count() or 1
    parser.add_argument('--p-b', default=0.5, type=float, help='Probability of testing a boundary.')
    parser.add_argument('--m', default=50, type=int, help='ADR buffer size.')
    parser.add_argument('--low-th', default=1000, type=int, help='Lower performance threshold.')
    parser.add_argument('--high-th', default=1500, type=int, help='Upper performance threshold.')
    parser.add_argument('--n-envs', default=n_envs, type=int, help='Number of parallel environments.')
    parser.add_argument('--timesteps', default=3_000_000, type=int, help='Total training timesteps.')
    parser.add_argument('--save-path', default='./models_ADR_Sweep/', type=str, help='Path to save models.')
    parser.add_argument('--device', default='auto', type=str, help='Device to use (cpu, cuda).')
    return parser.parse_args()

def objective(config=None):
    with wandb.init(config=config):
        config = wandb.config
        args = parse_args()

        # --- Environment Setup ---
        # CRITICAL: Use 'CustomHopper-adr-v0' and wrap in VecMonitor
        train_env = make_vec_env("CustomHopper-adr-v0", n_envs=args.n_envs, vec_env_cls=DummyVecEnv)
        train_env = VecMonitor(train_env)
        eval_env = gym.make("CustomHopper-target-v0")

        # --- ADR Handler ---
        init_params = {"thigh": THIGH_MEAN_MASS, "leg": LEG_MEAN_MASS, "foot": FOOT_MEAN_MASS}
        handlerADR = AutomaticDomainRandomization(
            init_params, p_b=args.p_b, m=args.m, delta=config.delta,
            thresholds=[args.low_th, args.high_th]
        )

        # --- Callbacks ---
        delta_save_path = os.path.join(args.save_path, f"delta_{config.delta}")
        best_model_path = os.path.join(delta_save_path, "best_model")
        
        adr_callback = ADRCallback(handlerADR=handlerADR)
        logging_callback = DetailedLoggingCallback()
        # EvalCallback logs to wandb automatically under "eval/" prefix
        eval_callback = EvalCallback(
            eval_env, best_model_save_path=best_model_path,
            log_path=delta_save_path, eval_freq=max(25000 // args.n_envs, 1),
            n_eval_episodes=50, deterministic=True,
        )
        callbacks = CallbackList([adr_callback, logging_callback, eval_callback])

        # --- Model ---
        model = PPO('MlpPolicy', train_env, verbose=0, device=args.device, **BEST_PARAMS)
        
        # --- Training ---
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

def main():
    # Login to W&B
    wandb.login()

    # Create the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Hopper_ADR_Final_Sweep")
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=objective, count=3) # Runs for all 3 delta values

    # --- Final Evaluation After Sweep ---
    print("\n\n--- Final Evaluation on Target Environment ---")
    args = parse_args()
    for delta in sweep_configuration["parameters"]["delta"]["values"]:
        model_path = os.path.join(args.save_path, f"delta_{delta}", "best_model", "best_model.zip")
        if os.path.exists(model_path):
            print(f"Evaluating model for delta = {delta}")
            test_env = gym.make('CustomHopper-target-v0')
            model = PPO.load(model_path)
            mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=100, render=False, warn=False)
            print(f"[s-t] Mean Reward (delta={delta}): {mean_reward:.2f} +/- {std_reward:.2f}")
        else:
            print(f"Model for delta = {delta} not found at {model_path}")

if __name__ == '__main__':
    main()