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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to train/predict on')
    return parser.parse_args()


class EpisodeLogger(BaseCallback):
    def __init__(self, log_dir='PPO/logs', verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.smoothed = []
        self.variance = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            ep = info.get('episode')
            if ep:
                self.episode_rewards.append(ep['r'])
                self.episode_lengths.append(ep['l'])

                window = self.episode_rewards[-100:]
                if len(window) == 100:
                    self.smoothed.append(np.mean(window))
                    self.variance.append(np.var(window))
        return True

    def _on_training_end(self) -> None:
        np.save(os.path.join(self.log_dir, 'episode_rewards.npy'), np.array(self.episode_rewards))
        np.save(os.path.join(self.log_dir, 'episode_times.npy'), np.array(self.episode_lengths))
        np.save(os.path.join(self.log_dir, 'episode_rewards_smoothed_npy.npy'), np.array(self.smoothed))
        np.save(os.path.join(self.log_dir, 'episode_rewards_variance_100.npy'), np.array(self.variance))
        if self.verbose:
            print(f"Saved logs to {self.log_dir}")




def main():
    args = parse_args()
    device = args.device

    train_env = gym.make('CustomHopper-source-v0')
    eval_env = gym.make('CustomHopper-target-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    # Env setup
    train_env = Monitor(gym.make('CustomHopper-source-v0'))
    eval_env = Monitor(gym.make('CustomHopper-target-v0'))

    # Callbacks
    eval_cb = EvalCallback(eval_env,
        best_model_save_path='PPO/model',
        log_path='PPO/logs/',
        eval_freq=5000, deterministic=True, render=False)
    episode_cb = EpisodeLogger(verbose=1)

    model = PPO("MlpPolicy", train_env, verbose=1, device=device)
    model.learn(total_timesteps=1_000_000,
                callback=CallbackList([eval_cb, episode_cb]))
    model.save('PPO/model/model_sb3_ppo.mdl')

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