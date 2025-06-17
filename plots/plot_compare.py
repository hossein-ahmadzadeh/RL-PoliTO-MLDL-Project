import numpy as np
import matplotlib.pyplot as plt
import os

# === Model names to compare ===
model_names = [
    "model_reinforce_with_baseline",
    "model_reinforce_with_baseline_norm",
    "model_reinforce_with_baseline_tanh",
    "model_reinforce_with_baseline_norm_tanh"
]

# === Directories ===
log_base = "logs"
analysis_base = "analysis"
output_dir = "report/comparison/images"
os.makedirs(output_dir, exist_ok=True)

# === Plot helper ===
def plot_comparison(data_dict, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    for model_name, data in data_dict.items():
        x = np.arange(1, len(data) + 1)
        plt.plot(x, data, label=model_name)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

# === Load and store data ===
def load_avg(arr):
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

def collect_data():
    data = {
        "entropy": {},
        "losses_avg": {},
        "returns_avg": {},
        "returns_smoothed_100": {},
        "times_avg": {},
        "episode_rewards_variance": {}
    }

    for model_name in model_names:
        try:
            entropy = np.load(f"{log_base}/{model_name}/entropy_log.npy")
            losses = np.load(f"{analysis_base}/{model_name}/losses.npy")
            returns = np.load(f"{analysis_base}/{model_name}/episode_rewards.npy")
            times = np.load(f"{analysis_base}/{model_name}/episode_times.npy")
            smoothed = np.load(f"{analysis_base}/{model_name}/episode_rewards_smoothed_100.npy")
            variance = np.load(f"{analysis_base}/{model_name}/episode_rewards_variance_100.npy")

            data["entropy"][model_name] = entropy
            data["losses_avg"][model_name] = load_avg(losses)
            data["returns_avg"][model_name] = load_avg(returns)
            data["returns_smoothed_100"][model_name] = smoothed
            data["times_avg"][model_name] = load_avg(times)
            data["episode_rewards_variance"][model_name] = variance

        except Exception as e:
            print(f"[WARNING] Failed to load model {model_name}: {e}")

    return data

# === Run comparison ===
data = collect_data()

plot_comparison(data["entropy"], "Policy Entropy Over Time", "Entropy", "compare_entropy")
plot_comparison(data["losses_avg"], "Average Loss per Episode", "Loss", "compare_losses_avg")
plot_comparison(data["returns_avg"], "Average Return per Episode", "Return", "compare_returns_avg")
plot_comparison(data["returns_smoothed_100"], "Smoothed Return (Window=100)", "Smoothed Return", "compare_returns_smoothed_100")
plot_comparison(data["times_avg"], "Average Time per Episode", "Time (s)", "compare_times_avg")
plot_comparison(data["episode_rewards_variance"], "Rolling Reward Variance (Window=100)", "Variance", "compare_episode_rewards_variance")