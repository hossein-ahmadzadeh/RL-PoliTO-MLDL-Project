import numpy as np
import matplotlib.pyplot as plt
import os

# === Paths ===
log_dir = "PPO/logs"
output_dir = "PPO/Result"
os.makedirs(output_dir, exist_ok=True)

# === Load .npy logs ===
episode_rewards = np.load(os.path.join(log_dir, "episode_rewards.npy"))
episode_times = np.load(os.path.join(log_dir, "episode_times.npy"))
smoothed_rewards = np.load(os.path.join(log_dir, "episode_rewards_smoothed_npy.npy"))
reward_variance = np.load(os.path.join(log_dir, "episode_rewards_variance_100.npy"))

# === Load .npz evaluation results ===
eval_data = np.load(os.path.join(log_dir, "evaluations.npz"))
eval_timesteps = eval_data["timesteps"]
eval_rewards = eval_data["results"]  # shape: (n_eval_points, n_eval_episodes)
eval_mean_rewards = np.mean(eval_rewards, axis=1)

# === Helper function ===
def save_plot(x, y, title, ylabel, filename, color="blue"):
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, color=color)
    plt.xlabel("Episode" if "episode" in filename else "Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.close()

# === Plotting ===
episodes = np.arange(1, len(episode_rewards) + 1)
save_plot(episodes, episode_rewards, "Raw Episode Rewards", "Reward", "episode_rewards", "green")
save_plot(episodes, episode_times, "Episode Time (duration)", "Time (s)", "episode_times", "orange")
save_plot(np.arange(1, len(smoothed_rewards) + 1), smoothed_rewards, "Smoothed Rewards (window=100)", "Reward", "episode_rewards_smoothed", "blue")
save_plot(np.arange(1, len(reward_variance) + 1), reward_variance, "Reward Variance (window=100)", "Variance", "episode_rewards_variance", "purple")
save_plot(eval_timesteps, eval_mean_rewards, "Evaluation Reward Mean vs Timesteps", "Eval Reward", "evaluation_mean_rewards", "crimson")