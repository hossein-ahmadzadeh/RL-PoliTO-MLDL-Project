import gym
import os
import time
import wandb
import numpy as np
import argparse
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', default=3000000, type=int)
    parser.add_argument('--save-freq', default=50000, type=int)
    parser.add_argument('--best-model-path', default='./BestModelTuning/model', type=str)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Training device')
    return parser.parse_args()

args = parse_args()

# === Hyperparameter Sweep Configuration ===
sweep_configuration = {
    "method": "random",
    "name": "ppo_hopper_sweep",
    "metric": {"name": "mean_reward", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 5e-4, "max": 1e-3},
        "clip_range": {"min": 0.25, "max": 0.35},
        "entropy_coefficient": {"min": 0.005, "max": 0.02},
        "gamma": {"min": 0.97, "max": 0.99},
        "gae_lambda": {"min": 0.95, "max": 0.99},
        "n_steps": {"values": [2048, 4096]},
        "batch_size": {"values": [64, 128]}
    }
}

# === W&B and Logger Callback ===
class WandCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if any(self.locals["dones"]):
            wandb.log({
                "reward": self.locals['infos'][0]['episode']['r'],
                "step": self.num_timesteps
            }, step=self.num_timesteps)
        return True

class LoggerWithStop(BaseCallback):
    def __init__(self, max_episodes=10000, log_dir='PPO/logs', print_every=1000, verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.print_every = print_every
        self.episode_rewards = []
        self.smoothed = []
        self.variance = []
        self.recent_rewards = []

        self.wall_times = []
        self.episode_lengths = []
        self.cumulative_sim_times = []
        self.cumulative_wall_times = []

        self._episode_start_time = None
        self.cumulative_sim = 0.0
        self.cumulative_wall = 0.0
        self.window_size = 100

    def _on_step(self) -> bool:
        if len(self.episode_rewards) >= self.max_episodes:
            pass

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

                self.cumulative_sim += sim_time
                self.cumulative_wall += wall_time

                self.episode_rewards.append(reward)
                self.episode_lengths.append(sim_time)
                self.wall_times.append(wall_time)
                self.cumulative_sim_times.append(self.cumulative_sim)
                self.cumulative_wall_times.append(self.cumulative_wall)
                self.recent_rewards.append(reward)

                if len(self.recent_rewards) > self.window_size:
                    self.recent_rewards.pop(0)

                smoothed = np.mean(self.recent_rewards) if len(self.recent_rewards) == self.window_size else None
                var = np.var(self.recent_rewards) if smoothed is not None else None
                if smoothed is not None:
                    self.smoothed.append(smoothed)
                    self.variance.append(var)

                wandb.log({
                    "episode": len(self.episode_rewards),
                    "reward_per_episode": reward,
                    "episode_length": steps,
                    "sim_time_per_episode": sim_time,
                    "wall_time_per_episode": wall_time,
                    "cumulative_sim_time": self.cumulative_sim,
                    "cumulative_wall_time": self.cumulative_wall,
                    "rolling_reward_mean_100": smoothed,
                    "rolling_reward_var_100": var
                }, step=self.num_timesteps)

                if self.verbose and len(self.episode_rewards) % self.print_every == 0:
                    print(f"--- Episode {len(self.episode_rewards)} ---")
                    print(f"  Reward: {reward:.2f}")
                    print(f"  Smoothed: {smoothed:.2f}" if smoothed else "")
                    print(f"  Sim Time: {sim_time:.2f}s | Wall Time: {wall_time:.2f}s")
        return True

    def _on_training_end(self):
        np.save(os.path.join(self.log_dir, 'episode_rewards.npy'), self.episode_rewards)
        np.save(os.path.join(self.log_dir, 'episode_times_simulated.npy'), self.episode_lengths)
        np.save(os.path.join(self.log_dir, 'episode_times_wallclock.npy'), self.wall_times)
        np.save(os.path.join(self.log_dir, 'rolling_reward_mean_100.npy'), self.smoothed)
        np.save(os.path.join(self.log_dir, 'rolling_reward_var_100.npy'), self.variance)
        np.save(os.path.join(self.log_dir, 'cumulative_sim_time.npy'), self.cumulative_sim_times)
        np.save(os.path.join(self.log_dir, 'cumulative_wall_time.npy'), self.cumulative_wall_times)


# === Run Tracking ===
countt = {'co': 0}
best_params = {
    "source": {'best_mean': -float('inf'), "run_id_suffix": ""},
    "target": {'best_mean': -float('inf'), "run_id_suffix": ""}
}


def objective(envname):
    countt["co"] += 1
    is_source = envname == "CustomHopper-source-v0"

    wandb.init(
        project="RL-Hopper-PPO-Tuning",
        group="CustomHopper-source-v0-Sweep" if is_source else "CustomHopper-target-v0-Sweep",
        name=f"{envname}-Run-{countt['co']}"
    )

    # Sampled hyperparameters
    lr = wandb.config.learning_rate
    ent = wandb.config.entropy_coefficient
    clip = wandb.config.clip_range
    gamma = wandb.config.gamma
    gae_lambda = wandb.config.gae_lambda
    n_steps = wandb.config.n_steps
    batch_size = wandb.config.batch_size

    env = Monitor(gym.make(envname))
    run_id = f"{countt['co']}_{envname}"

    logger_cb = LoggerWithStop(log_dir=f'PPO/logs/{run_id}')
    wandb_cb = WandCallback()
    eval_cb = EvalCallback(env, n_eval_episodes=50, eval_freq=args.save_freq,
                           best_model_save_path=os.path.join(args.best_model_path, run_id), verbose=0)
    callback = CallbackList([eval_cb, wandb_cb, logger_cb])

    model = PPO("MlpPolicy", env,
                learning_rate=lr, ent_coef=ent, clip_range=clip,
                gamma=gamma, gae_lambda=gae_lambda,
                n_steps=n_steps, batch_size=batch_size,
                verbose=0, device=args.device)

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model = PPO.load(os.path.join(args.best_model_path, run_id, "best_model.zip"))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

    key = "source" if is_source else "target"
    if mean_reward > best_params[key]["best_mean"]:
        best_params[key].update({
            "learning_rate": lr,
            "clip_range": clip,
            "entropy_coefficient": ent,
            "best_mean": mean_reward,
            "best_std": std_reward,
            "run_id_suffix": run_id
        })

    wandb.finish()
    return mean_reward, std_reward


def main():
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="RL-Hopper-PPO-Tuning")
    wandb.agent(sweep_id, function=lambda: objective("CustomHopper-source-v0"), count=10)
    wandb.agent(sweep_id, function=lambda: objective("CustomHopper-target-v0"), count=10)

    def test_model(env_id, run_id):
        path = os.path.join(args.best_model_path, run_id, "best_model.zip")
        env = Monitor(gym.make(env_id))
        model = PPO.load(path)
        return evaluate_policy(model, env, n_eval_episodes=50)

    print("\n--- Final Evaluations ---")
    ss = test_model("CustomHopper-source-v0", best_params["source"]["run_id_suffix"])
    st = test_model("CustomHopper-target-v0", best_params["source"]["run_id_suffix"])
    tt = test_model("CustomHopper-target-v0", best_params["target"]["run_id_suffix"])

    print(f"[s‑s] mean_reward: {ss[0]:.2f} ± {ss[1]:.2f}")
    print(f"[s‑t] mean_reward: {st[0]:.2f} ± {st[1]:.2f}")
    print(f"[t‑t] mean_reward: {tt[0]:.2f} ± {tt[1]:.2f}")

    with open("final_results.txt", "w") as f:
        for domain in best_params:
            f.write(f"{domain.upper()} BEST PARAMS:\n")
            for k, v in best_params[domain].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("=== Final Evaluations ===\n")
        f.write(f"[s→s]: {ss[0]:.2f} ± {ss[1]:.2f}\n")
        f.write(f"[s→t]: {st[0]:.2f} ± {st[1]:.2f}\n")
        f.write(f"[t→t]: {tt[0]:.2f} ± {tt[1]:.2f}\n")


if __name__ == "__main__":
    main()
