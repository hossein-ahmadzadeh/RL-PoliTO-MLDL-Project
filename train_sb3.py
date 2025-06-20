"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *

from stable_baselines3 import PPO  # or SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import torch
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to train/predict on')
    return parser.parse_args()


class LoggerWithStop(BaseCallback):
    def __init__(self, max_episodes=10_000, log_dir='PPO/logs', verbose=1, print_every=1000):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wall_clock_times = []
        self.smoothed = []
        self.variance = []
        self.recent_rewards = []
        self.window_size = 100
        self._episode_start_time = None
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self._episode_start_time is None:
            self._episode_start_time = time.time()

        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep:
                reward = ep['r']
                steps = ep['l']
                sim_time = steps * 0.008
                wall_time = time.time() - self._episode_start_time
                self._episode_start_time = None

                # Log all
                self.episode_rewards.append(reward)
                self.episode_lengths.append(sim_time)
                self.wall_clock_times.append(wall_time)

                self.recent_rewards.append(reward)
                if len(self.recent_rewards) > self.window_size:
                    self.recent_rewards.pop(0)

                # Logging stats
                episode_num = len(self.episode_rewards)
                if len(self.recent_rewards) == self.window_size:
                    smoothed = np.mean(self.recent_rewards)
                    var = np.var(self.recent_rewards)
                    self.smoothed.append(smoothed)
                    self.variance.append(var)
                else:
                    smoothed = None
                    var = None

                # Print summary every N episodes
                if self.verbose and episode_num % self.print_every == 0:
                    print(f"--- Episode {episode_num}/{self.max_episodes} ---")
                    print(f"  Reward: {reward:.2f}")
                    if smoothed is not None:
                        print(f"  Smoothed (100): {smoothed:.2f} | Variance: {var:.2f}")
                    print(f"  Sim Time: {sim_time:.2f}s | Wall Time: {wall_time:.2f}s")
                    print("-" * 40)

                # Stop condition
                if episode_num >= self.max_episodes:
                    print(f"Reached {self.max_episodes} episodes. Stopping training.")
                    return False
        return True

    def _on_training_end(self) -> None:
        np.save(os.path.join(self.log_dir, 'episode_rewards.npy'), np.array(self.episode_rewards))
        np.save(os.path.join(self.log_dir, 'episode_times_simulated.npy'), np.array(self.episode_lengths))
        np.save(os.path.join(self.log_dir, 'episode_times_wallclock.npy'), np.array(self.wall_clock_times))
        np.save(os.path.join(self.log_dir, 'episode_rewards_smoothed_100.npy'), np.array(self.smoothed))
        np.save(os.path.join(self.log_dir, 'episode_rewards_variance_100.npy'), np.array(self.variance))
        if self.verbose:
            print(f"Saved logs to {self.log_dir}")



def main():
    args = parse_args()
    device = args.device

    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')


    print("=== Source Environment ===")
    print("State space:", source_env.observation_space)
    print("Action space:", source_env.action_space)
    print("Dynamics parameters:", source_env.get_parameters())

    print("\n=== Target Environment ===")
    print("State space:", target_env.observation_space)
    print("Action space:", target_env.action_space)
    print("Dynamics parameters:", target_env.get_parameters())

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    # Env setup
    train_env = Monitor(gym.make('CustomHopper-source-v0'))
    eval_env = Monitor(gym.make('CustomHopper-target-v0'))

    # Callbacks
    episode_cb = LoggerWithStop(
        max_episodes=10_000,
        verbose=1,
        print_every=1000
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='PPO/model',
        log_path='PPO/logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model = PPO("MlpPolicy", train_env, verbose=1, device=device)

    model.learn(
        total_timesteps=int(1e9),
        callback=CallbackList([eval_cb, episode_cb])
    )

    # Evaluation
    obs = eval_env.reset()
    total_reward = 0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        if done:
            break
    print("Total eval reward:", total_reward)


if __name__ == '__main__':
    main()