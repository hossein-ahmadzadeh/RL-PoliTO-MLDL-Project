# train_sb3_sac.py
import argparse, os
import gym
import numpy as np
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
import time

class EpisodeLogger(BaseCallback):
    def __init__(self, log_dir='SAC/logs', verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []      # Simulated time
        self.wall_clock_times = []     # Real time (wall clock)
        self.smoothed = []
        self.variance = []
        self.recent_rewards = []
        self.window_size = 100
        self._episode_start_time = None

    def _on_step(self) -> bool:

        # Start timing on first step of an episode
        if self._episode_start_time is None:
            self._episode_start_time = time.time()

        for epinfo in self.locals.get('infos', []):
            ep = epinfo.get('episode')
            if ep:
                r, l = ep['r'], ep['l']
                self.episode_rewards.append(r)
                self.episode_lengths.append(l * 0.008) # Simulated time per step

                # Add wall-clock time
                end_time = time.time()
                wall_clock_duration = end_time - self._episode_start_time
                self.wall_clock_times.append(wall_clock_duration)
                self._episode_start_time = None  # Reset for next episode

                # Manually update rolling window like in train.py
                self.recent_rewards.append(r)
                if len(self.recent_rewards) > self.window_size:
                    self.recent_rewards.pop(0)
                
                if len(self.recent_rewards) == 100:
                    self.smoothed.append(np.mean(self.recent_rewards))
                    self.variance.append(np.var(self.recent_rewards))
        return True

    def _on_training_end(self) -> None:
        np.save(os.path.join(self.log_dir, 'episode_rewards.npy'), np.array(self.episode_rewards))
        np.save(os.path.join(self.log_dir, 'episode_times_simulated.npy'), np.array(self.episode_lengths))
        np.save(os.path.join(self.log_dir, 'episode_times_wallclock.npy'), np.array(self.wall_clock_times))
        np.save(os.path.join(self.log_dir, 'episode_rewards_smoothed_100.npy'), np.array(self.smoothed))
        np.save(os.path.join(self.log_dir, 'episode_rewards_variance_100.npy'), np.array(self.variance))
        if self.verbose:
            print(f"Logs saved to {self.log_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'], help='Device for training')
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device

    train_env = Monitor(gym.make('CustomHopper-source-v0'))
    eval_env = Monitor(gym.make('CustomHopper-target-v0'))

    print('Env params:', train_env.observation_space, train_env.action_space, train_env.env.get_parameters())

    episode_cb = EpisodeLogger(verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='SAC/model',
        log_path='SAC/logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model = SAC("MlpPolicy", train_env, verbose=1, device=device)
    model.learn(
        total_timesteps=1_000_000_000,  # very large; will be stopped early
        callback=CallbackList([episode_cb, stop_cb, eval_cb])
    )
    model.save('SAC/model/model_sb3_sac.mdl')

    # Quick eval
    obs = eval_env.reset()
    total = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = eval_env.step(action)
        total += r
    print("Final SAC eval reward:", total)

if __name__ == '__main__':
    main()