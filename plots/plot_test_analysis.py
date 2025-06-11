import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
out_dir = "report/model_reinforce_baseline_nonorm/images/test"
os.makedirs(out_dir, exist_ok=True)

# Load test data
returns = np.load("analysis/model_reinforce_baselinee/returns_per_episode_reinforce_beseline.npy")

times = np.load("analysis//model_reinforce_baselinee/episode_times_reinforce_baseline.npy")


episodes = np.arange(1, len(returns) + 1)

# Plot helper
def save_plot(x, y, title, ylabel, filename, color='blue'):
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, color=color)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}.png", dpi=300)
    plt.close()

# Test Returns
save_plot(episodes, returns, "Test Returns per Episode", "Return", "test_returns")
save_plot(episodes, np.cumsum(returns)/episodes, "Test Average Return", "Average Return", "test_returns_avg")
save_plot(episodes, np.cumsum(returns), "Test Cumulative Return", "Cumulative Return", "test_returns_cumulative")

# Test Times
save_plot(episodes, times, "Test Time per Episode", "Time (s)", "test_times")
save_plot(episodes, np.cumsum(times)/episodes, "Test Average Time", "Average Time (s)", "test_times_avg")
save_plot(episodes, np.cumsum(times), "Test Cumulative Time", "Cumulative Time (s)", "test_times_cumulative")

