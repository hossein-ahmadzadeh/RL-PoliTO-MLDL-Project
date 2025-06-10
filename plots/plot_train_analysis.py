import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = "report/reinforce_nobaseline_nonnorm_tanh_action/images/train"
os.makedirs(output_dir, exist_ok=True)

# Load data
returns = np.load("analysis/returns_per_episode_reinforce_nobaseline_nonnorm_tanh_action.npy")
losses = np.load("analysis/losses_per_episode_reinforce_nobaseline_nonnorm_tanh_action.npy")
times = np.load("analysis/episode_times_reinforce_nobaseline_nonnorm_tanh_action.npy")
variances=np.load("analysis/variances_per_episode_reinforce_nobaseline_nonnorm_tanh_action.npy")


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_and_save(x, y, title, ylabel, filename, label=""):
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, label=label)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()


# --- Returns ---
plot_and_save(np.arange(len(returns)), returns, "Returns per Episode", "Return", "returns_raw")
plot_and_save(np.arange(len(returns) - 99), moving_average(returns, 100), "Smoothed Returns (avg over 100 episodes)", "Average Return", "returns_avg")
plot_and_save(np.arange(len(returns)), np.cumsum(returns), "Cumulative Returns", "Cumulative Return", "returns_cumulative")

# --- Losses ---
plot_and_save(np.arange(len(losses)), losses, "Loss per Episode", "Loss", "losses_raw")
plot_and_save(np.arange(len(losses) - 99), moving_average(losses, 100), "Smoothed Loss (avg over 100 episodes)", "Average Loss", "losses_avg")
plot_and_save(np.arange(len(losses)), np.cumsum(losses), "Cumulative Loss", "Cumulative Loss", "losses_cumulative")

# --- Episode Times ---
plot_and_save(np.arange(len(times)), times, "Episode Time per Episode", "Time (s)", "times_raw")
plot_and_save(np.arange(len(times) - 99), moving_average(times, 100), "Smoothed Time (avg over 100 episodes)", "Average Time (s)", "times_avg")
plot_and_save(np.arange(len(times)), np.cumsum(times), "Cumulative Time", "Total Time (s)", "times_cumulative")

import matplotlib.pyplot as plt

# --- Variances ---
episodes = np.arange(0, len(variances) * 1000, 1000)  # فواصل 1000 اپیزود
plot_and_save(episodes, variances, "Variance of Returns Over Episodes (REINFORCE on Hopper)", "Variance", "variances_raw")
plot_and_save(episodes[:-99], moving_average(variances, 100), "Smoothed Variance (avg over 100 episodes)", "Average Variance", "variances_avg")
plot_and_save(episodes, np.cumsum(variances), "Cumulative Variance", "Cumulative Variance", "variances_cumulative")