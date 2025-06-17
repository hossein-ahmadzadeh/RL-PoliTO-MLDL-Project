import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
model_name = "REINFORCE"
analysis_dir = f"analysis/{model_name}"
out_dir = f"report/{model_name}/images/test"
os.makedirs(out_dir, exist_ok=True)

def safe_load(path):
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None
    return np.load(path)

# Load test data

returns = safe_load(f"{analysis_dir}/returns_per_episode.npy")
times = safe_load(f"{analysis_dir}/episode_times.npy")


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

if returns is not None and times is not None:
    episodes = np.arange(1, len(returns) + 1)

    # Test Returns
    save_plot(episodes, returns, "Test Returns per Episode", "Return", "test_returns")
    save_plot(episodes, np.cumsum(returns)/episodes, "Test Average Return", "Average Return", "test_returns_avg")
    save_plot(episodes, np.cumsum(returns), "Test Cumulative Return", "Cumulative Return", "test_returns_cumulative")

    # Test Times
    save_plot(episodes, times, "Test Time per Episode", "Time (s)", "test_times")
    save_plot(episodes, np.cumsum(times)/episodes, "Test Average Time", "Average Time (s)", "test_times_avg")
    save_plot(episodes, np.cumsum(times), "Test Cumulative Time", "Cumulative Time (s)", "test_times_cumulative")

    print(f"✅ Test plots saved to: {out_dir}")
else:
    print("❌ One or more required files are missing.")