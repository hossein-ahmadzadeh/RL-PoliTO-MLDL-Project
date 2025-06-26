import os
import gym
import time
import wandb
import numpy as np
import argparse
from ADR import AutomaticDomainRandomization, ADRCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

LOG_DIR = "./logs_ADR"

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to train on')
    return parser.parse_args()


class WandADRCallback(EvalCallback):
    def __init__(self, print_every=1000, verbose=1):
        super().__init__(Monitor(gym.make("CustomHopper-adr-v0")), n_eval_episodes=50, eval_freq=50000,
                         best_model_save_path="./best_eval_adr/", deterministic=True, render=False)
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


def main():
    args = parse_args()
    wandb.init(project="PPO_ADR_Hopper")

    env_id = "CustomHopper-adr-v0"
    train_env = make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv)
    test_env = gym.make(env_id)

    init_params = {"thigh": THIGH_MEAN_MASS, "leg": LEG_MEAN_MASS, "foot": FOOT_MEAN_MASS}
    adr_handler = AutomaticDomainRandomization(
        init_params=init_params,
        p_b=0.5,
        m=50,
        delta=0.05,
        thresholds=[1000, 1500]
    )

    train_env.set_attr("bounds", adr_handler.get_bounds())

    eval_callback = WandADRCallback()
    adr_callback = ADRCallback(
        handlerADR=adr_handler,
        vec_env=train_env,
        eval_callback=eval_callback,
        n_envs=1,
        verbose=1,
        save_freq=50000,
        save_path="./models_ADR"
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=BEST_PARAMS["learning_rate"],
        clip_range=BEST_PARAMS["clip_range"],
        ent_coef=BEST_PARAMS["ent_coef"],
        n_steps=BEST_PARAMS["n_steps"],
        batch_size=BEST_PARAMS["batch_size"],
        gamma=BEST_PARAMS["gamma"],
        gae_lambda=BEST_PARAMS["gae_lambda"],
        device=args.device
    )

    model.learn(total_timesteps=3_000_000, callback=CallbackList([adr_callback, eval_callback]), progress_bar=True)

    wandb.finish()


if __name__ == '__main__':
    main()
