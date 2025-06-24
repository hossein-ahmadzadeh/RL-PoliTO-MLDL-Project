import numpy as np
import matplotlib.pyplot as plt
import os

# === Ensure save folder exists ===
save_dir = "UniformDomainRandomization"
os.makedirs(save_dir, exist_ok=True)

# === File mappings ===
data_files = {
    "Target → Target [Upper Bound]": {
        "rewards": "PPO_BEST_MODEL_TEST_Target/returns_test_best_model.npy",
        "times": "PPO_BEST_MODEL_TEST_Target/times_test_best_model.npy",
        "color": "#1f77b4"  # blue
    },
    "Source → Target [Lower Bound]": {
        "rewards": "PPO_BEST_MODEL_TEST_NO_UDR/returns_test_best_model.npy",
        "times": "PPO_BEST_MODEL_TEST_NO_UDR/times_test_best_model.npy",
        "color": "#ff7f0e"  # orange
    },
    "Source → Target [UDR]": {
        "rewards": "PPO_BEST_MODEL_TEST_UDR/returns_test_best_model.npy",
        "times": "PPO_BEST_MODEL_TEST_UDR/times_test_best_model.npy",
        "color": "#2ca02c"  # green
    }
}

# === Running average helper ===
def running_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# === Load all data ===
for key, paths in data_files.items():
    paths["reward_vals"] = np.load(paths["rewards"])
    paths["time_vals"] = np.load(paths["times"])

# === Plot 1: Reward vs Episode ===
plt.figure(figsize=(10, 5))
for key, data in data_files.items():
    plt.plot(data["reward_vals"], label=key, color=data["color"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.legend()
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}/reward_vs_episode.png")

# === Plot 2: Smoothed Reward with aligned x-axis ===
plt.figure(figsize=(10, 5))
for key, data in data_files.items():
    smoothed = running_average(data["reward_vals"])
    x = np.arange(len(data["reward_vals"]))[len(data["reward_vals"]) - len(smoothed):]
    plt.plot(x, smoothed, label=key, color=data["color"])
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed Reward per Episode (Running Avg = 10)")
plt.legend()
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}/reward_smoothed_vs_episode.png")

# === Plot 3: Mean Reward Bar ===
labels = list(data_files.keys())
means = [np.mean(data["reward_vals"]) for data in data_files.values()]
colors = [data["color"] for data in data_files.values()]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, means, color=colors)
for bar, val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 5, f"{val:.1f}", ha='center')
plt.ylabel("Mean Reward")
plt.title("Mean Reward over 50 Episodes")
plt.xticks(rotation=10)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{save_dir}/reward_mean_bar.png")

# === Plot 4: Reward Std Deviation Bar ===
stds = [np.std(data["reward_vals"]) for data in data_files.values()]
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, stds, color=colors)
for bar, val in zip(bars, stds):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 2, f'{val:.1f}', ha='center')
plt.ylabel("Std Deviation")
plt.title("Reward Std Deviation (50 Episodes)")
plt.xticks(rotation=10)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{save_dir}/reward_std_bar.png")

# === Plot 5: Time vs Episode ===
plt.figure(figsize=(10, 5))
for key, data in data_files.items():
    plt.plot(data["time_vals"], label=key, color=data["color"])
plt.xlabel("Episode")
plt.ylabel("Wall Time (s)")
plt.title("Episode Duration")
plt.legend()
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}/time_vs_episode.png")

# === Plot 6: Smoothed Time with aligned x-axis ===
plt.figure(figsize=(10, 5))
for key, data in data_files.items():
    smoothed = running_average(data["time_vals"])
    x = np.arange(len(data["time_vals"]))[len(data["time_vals"]) - len(smoothed):]
    plt.plot(x, smoothed, label=key, color=data["color"])
plt.xlabel("Episode")
plt.ylabel("Smoothed Time (s)")
plt.title("Smoothed Episode Duration (Running Avg = 10)")
plt.legend()
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}/time_smoothed_vs_episode.png")

print("✅ All updated plots saved with aligned episode indices in 'UniformDomainRandomization/'")
