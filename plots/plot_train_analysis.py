import numpy as np
import matplotlib.pyplot as plt
import os

# === Configurable model name ===
model_name = "model_actor_critic_norm_tanh_entropy"  # Change this to switch models

# === Paths ===
analysis_dir = f"analysis/{model_name}"
output_dir = f"report/{model_name}/images/train"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
returns = np.load(f"{analysis_dir}/episode_rewards.npy")
losses = np.load(f"{analysis_dir}/losses.npy")
times = np.load(f"{analysis_dir}/episode_times.npy")
rolling_var = np.load(f"{analysis_dir}/episode_rewards_variance_100.npy")
adv_var_log = np.load(f"{analysis_dir}/advantages_variance_log.npy")
td_var_log = np.load(f"{analysis_dir}/td_target_variance_log.npy")
smoothed_rewards = np.load(f"{analysis_dir}/episode_rewards_smoothed_100.npy")

episodes = np.arange(1, len(returns) + 1)

# === Plot helper ===
def save_plot(x, y, title, ylabel, filename, color='blue'):
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, color=color)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

# === Plot: Rewards ===
save_plot(episodes, returns, "Episode Reward (raw)", "Reward", "returns_raw")
save_plot(episodes, smoothed_rewards, "Smoothed Reward (window=100)", "Smoothed Reward", "returns_smoothed_100")
save_plot(episodes, np.cumsum(returns) / episodes, "Smoothed Reward (avg)", "Average Reward", "returns_avg")
save_plot(episodes, np.cumsum(returns), "Cumulative Reward", "Total Reward", "returns_cumulative")

# === Plot: Loss ===
save_plot(episodes, losses, "Episode Loss (raw)", "Loss", "losses_raw")
save_plot(episodes, np.cumsum(losses) / episodes, "Smoothed Loss (avg)", "Average Loss", "losses_avg")
save_plot(episodes, np.cumsum(losses), "Cumulative Loss", "Total Loss", "losses_cumulative")

# === Plot: Time ===
save_plot(episodes, times, "Episode Time (raw)", "Time (s)", "times_raw")
save_plot(episodes, np.cumsum(times) / episodes, "Smoothed Time (avg)", "Avg Time per Episode", "times_avg")
save_plot(episodes, np.cumsum(times), "Cumulative Time", "Total Time (s)", "times_cumulative")

# === Plot: Reward Variance ===
save_plot(np.arange(1, len(rolling_var) + 1), rolling_var, "Reward Variance (rolling window)", "Variance", "episode_rewards_variance_rolling", color="purple")

# === Plot: Advantage Variance ===
save_plot(np.arange(1, len(adv_var_log) + 1), adv_var_log, "Advantage Variance (per episode)", "Variance", "advantages_variance_log", color="darkred")

# === Plot: TD Target Variance ===
save_plot(np.arange(1, len(td_var_log) + 1), td_var_log, "TD Target Variance (per episode)", "Variance", "td_target_variance_log", color="olive")
