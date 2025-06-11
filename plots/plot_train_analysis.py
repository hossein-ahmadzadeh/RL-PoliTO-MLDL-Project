import os
import numpy as np
import matplotlib.pyplot as plt

model_name = "te"
log_dir = f"logs/{model_name}"
analysis_dir = f"analysis/{model_name}"
output_dir = f"report/{model_name}/images/train"
os.makedirs(output_dir, exist_ok=True)

window_size = 100

def moving_average(data, window_size):
    if data is None or len(data) < window_size:
        print(f"[WARNING] moving_average skipped (len={len(data) if data is not None else 'None'})")
        return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



def plot_and_save(y, title, ylabel, filename, xlabel="Episode", x=None, label=None):
    plt.figure(figsize=(12, 6))
    if x is None:
        x = np.arange(len(y))
    plt.plot(x, y, label=label if label else None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

# --------- Load Data ---------
returns = np.load(f"{analysis_dir}/returns_per_episode.npy")
losses = np.load(f"{analysis_dir}/losses_per_episode.npy")
times = np.load(f"{analysis_dir}/episode_times.npy")
variances = np.load(f"{analysis_dir}/variances_episode_return_window.npy")
agent_returns_variances_per_episode = np.load(f"{analysis_dir}/agent_returns_variances.npy")

mu_log = np.load(f"{log_dir}/mu_log.npy", allow_pickle=True)
sigma_log = np.load(f"{log_dir}/sigma_log.npy", allow_pickle=True)
entropy_log = np.load(f"{log_dir}/entropy_log.npy")
returns_mean = np.load(f"{log_dir}/agent_returns_mean_log_per_episode.npy")
returns_std = np.load(f"{log_dir}/agent_returns_std_log_per_episode.npy")
advantages_mean = np.load(f"{log_dir}/agent_advantages_mean_log_per_episode.npy")
advantages_std = np.load(f"{log_dir}/agent_advantages_std_log_per_episode.npy")


plot_and_save(returns, "Returns per Episode", "Return", "returns_raw")
plot_and_save(moving_average(returns, window_size), "Smoothed Returns (avg over 100 episodes)", "Average Return", "returns_avg", x=np.arange(len(returns) - window_size + 1))
plot_and_save(np.cumsum(returns), "Cumulative Returns", "Cumulative Return", "returns_cumulative")

plot_and_save(losses, "Loss per Episode", "Loss", "losses_raw")
plot_and_save(moving_average(losses, window_size), "Smoothed Loss (avg over 100 episodes)", "Average Loss", "losses_avg", x=np.arange(len(losses) - window_size + 1))
plot_and_save(np.cumsum(losses), "Cumulative Loss", "Cumulative Loss", "losses_cumulative")

plot_and_save(times, "Episode Time per Episode", "Time (s)", "times_raw")
plot_and_save(moving_average(times, window_size), "Smoothed Time (avg over 100 episodes)", "Average Time (s)", "times_avg", x=np.arange(len(times) - window_size + 1))
plot_and_save(np.cumsum(times), "Cumulative Time", "Total Time (s)", "times_cumulative")

# --------- Variance ---------
episodes = np.arange(0, len(variances) * 1000, 1000)
plot_and_save(variances, "Variance of Returns Over Episodes", "Variance", "variances_raw", x=episodes)
plot_and_save(moving_average(variances, window_size), "Smoothed Variance (avg over 100 episodes)", "Average Variance", "variances_avg", x=episodes[:-window_size + 1])
plot_and_save(np.cumsum(variances), "Cumulative Variance", "Cumulative Variance", "variances_cumulative", x=episodes)

# --------- Moving Window Variance (از train) ---------
plot_and_save(variances, f'Variance of Total Episode Returns (Window {window_size})', 'Variance', 'returns_variance_window')

# --------- Rewards ---------
plot_and_save(returns, "Episode Rewards Over Time", "Total Reward", "episode_rewards")

# --------- Entropy ---------
plot_and_save(entropy_log, "Entropy over Steps", "Entropy", "entropy")

# --------- Agent Variance Logs ---------
plot_and_save(agent_returns_variances_per_episode, "Variance of Discounted Returns (Agent)", "Variance", "agent_returns_variance")
plot_and_save(advantages_mean, "Mean Advantage (Agent)", "Mean Advantage", "agent_advantage_mean")
plot_and_save(advantages_std, "Std Advantage (Agent)", "Std Advantage", "agent_advantage_std")


mu_log = np.array(mu_log)
sigma_log = np.array(sigma_log)

for i in range(mu_log.shape[1]):
    plot_and_save(mu_log[:, i], f"μ Component {i}", f"μ[{i}]", f"mu_{i}")

for i in range(sigma_log.shape[1]):
    plot_and_save(sigma_log[:, i], f"σ Component {i}", f"σ[{i}]", f"sigma_{i}")

print(f"✅ All plots saved to: {output_dir}")
