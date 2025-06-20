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
    parser.add_argument('--timesteps', default=5000000, type=int)
    parser.add_argument('--save-freq', default=50000, type=int)
    parser.add_argument('--best-model-path', default='./BestModelTuning/model', type=str)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Training device')
    return parser.parse_args()

args = parse_args()

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

class WandCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if any(self.locals["dones"]):
            wandb.log({
                "reward": self.locals['infos'][0]['episode']['r'],
                "step": self.num_timesteps
            })
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
        self.window_size = 100
        self._episode_start_time = None

    def _on_step(self) -> bool:
        if len(self.episode_rewards) >= self.max_episodes:
            return True

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

                self.episode_rewards.append(reward)
                self.episode_lengths.append(sim_time)
                self.wall_times.append(wall_time)
                self.recent_rewards.append(reward)

                if len(self.recent_rewards) > self.window_size:
                    self.recent_rewards.pop(0)

                smoothed = np.mean(self.recent_rewards) if len(self.recent_rewards) == self.window_size else None
                var = np.var(self.recent_rewards) if smoothed is not None else None

                if smoothed is not None:
                    self.smoothed.append(smoothed)
                    self.variance.append(var)

                log_data = {
                    "episode_reward": reward,
                    "sim_time": sim_time,
                    "wall_time": wall_time,
                    "episode_num": len(self.episode_rewards)
                }
                if smoothed is not None:
                    log_data["reward_smoothed"] = smoothed
                    log_data["reward_variance"] = var
                wandb.log(log_data)

                if self.verbose and len(self.episode_rewards) % self.print_every == 0:
                    print(f"--- Episode {len(self.episode_rewards)} ---")
                    print(f"  Reward: {reward:.2f}")
                    print(f"  Smoothed: {smoothed:.2f}" if smoothed else "")
                    print(f"  Sim Time: {sim_time:.2f}s | Wall Time: {wall_time:.2f}s")
                    print("-" * 40)
        return True

    def _on_training_end(self):
        np.save(os.path.join(self.log_dir, 'episode_rewards.npy'), self.episode_rewards)
        np.save(os.path.join(self.log_dir, 'episode_times_simulated.npy'), self.episode_lengths)
        np.save(os.path.join(self.log_dir, 'episode_times_wallclock.npy'), self.wall_times)
        np.save(os.path.join(self.log_dir, 'episode_rewards_smoothed_100.npy'), self.smoothed)
        np.save(os.path.join(self.log_dir, 'episode_rewards_variance_100.npy'), self.variance)

best_params = {
    "source": {'best_mean' : -float('inf')},
    "target": {'best_mean' : -float('inf')}
}

def objective(envname):
    wandb.init(project="F_MldlRLproject_TuningPPO_Source_Target")

    lr = wandb.config.learning_rate
    ent = wandb.config.entropy_coefficient
    clip = wandb.config.clip_range
    gamma = wandb.config.gamma
    gae_lambda = wandb.config.gae_lambda
    n_steps = wandb.config.n_steps
    batch_size = wandb.config.batch_size

    train_env = Monitor(gym.make(envname))
    logger_cb = LoggerWithStop(log_dir=f'PPO/logs/{envname}')
    wandb_cb = WandCallback()

    # Save always to a fixed folder
    save_path = os.path.join(args.best_model_path, f"best_{envname}/")
    eval_cb = EvalCallback(train_env, n_eval_episodes=50, eval_freq=args.save_freq,
                           best_model_save_path=save_path, verbose=0)
    callback = CallbackList([eval_cb, wandb_cb, logger_cb])

    model = PPO("MlpPolicy", train_env,
                learning_rate=lr,
                ent_coef=ent,
                clip_range=clip,
                gamma=gamma,
                gae_lambda=gae_lambda,
                n_steps=n_steps,
                batch_size=batch_size,
                verbose=0,
                device=args.device)

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model = PPO.load(os.path.join(save_path, "best_model.zip"))
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=50)
    wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

    key = "source" if envname == "CustomHopper-source-v0" else "target"
    if mean_reward > best_params[key]["best_mean"]:
        best_params[key] = {
            "learning_rate": lr,
            "clip_range": clip,
            "entropy_coefficient": ent,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "best_mean": mean_reward,
            "best_std": std_reward
        }

    wandb.finish()
    return mean_reward, std_reward

def main():
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="F_MldlRLproject_TuningPPO_Source_Target")

    wandb.agent(sweep_id, function=lambda: objective("CustomHopper-source-v0"), count=10)
    print("Best params [source]:", best_params)

    wandb.agent(sweep_id, function=lambda: objective("CustomHopper-target-v0"), count=10)
    print("Best params [target]:", best_params)

    def test_model(env_id):
        path = os.path.join(args.best_model_path, f"best_{env_id}/best_model.zip")
        test_env = Monitor(gym.make(env_id))
        model = PPO.load(path)
        return evaluate_policy(model, test_env, n_eval_episodes=50)

    print("\n--- Final Evaluations ---")
    ss = test_model("CustomHopper-source-v0")
    st = test_model("CustomHopper-target-v0")
    tt = test_model("CustomHopper-target-v0")

    print(f"[s-s] mean_reward: {ss[0]:.2f} ± {ss[1]:.2f}")
    print(f"[s-t] mean_reward: {st[0]:.2f} ± {st[1]:.2f}")
    print(f"[t-t] mean_reward: {tt[0]:.2f} ± {tt[1]:.2f}")

    with open("final_results.txt", "w") as f:
        f.write("=== Best Hyperparameters ===\n")
        for domain in best_params:
            f.write(f"{domain.upper()}:\n")
            for k, v in best_params[domain].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("=== Final Evaluations ===\n")
        f.write(f"source→source (s→s): {ss[0]:.2f} ± {ss[1]:.2f}\n")
        f.write(f"source→target (s→t) [lower bound]: {st[0]:.2f} ± {st[1]:.2f}\n")
        f.write(f"target→target (t→t) [upper bound]: {tt[0]:.2f} ± {tt[1]:.2f}\n")

if __name__ == "__main__":
    main()
