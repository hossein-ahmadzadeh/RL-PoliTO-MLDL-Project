import numpy as np
import matplotlib.pyplot as plt
import os

# === Settings ===
model_name = "model_reinforce_nobaseline_norm_tanh_action"
file_stub = model_name.replace("model_", "")
analysis_dir = f"analysis/{model_name}"
out_dir = f"report/{model_name}/images/train"
os.makedirs(out_dir, exist_ok=True)

# === Load data ===
returns = np.load(f"{analysis_dir}/returns_per_episode_{file_stub}.npy")
losses = np.load(f"{analysis_dir}/losses_per_episode_{file_stub}.npy")
times = np.load(f"{analysis_dir}/episode_times_{file_stub}.npy")
variances = np.load(f"{analysis_dir}/returns_variance_100.npy")

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
    plt.savefig(f"{out_dir}/{filename}.png", dpi=300)
    plt.close()

# === Plot: Returns ===
save_plot(episodes, returns, "Episode Return (raw)", "Return", "returns_raw")
save_plot(episodes, np.cumsum(returns) / episodes, "Smoothed Return (avg)", "Average Return", "returns_avg")
save_plot(episodes, np.cumsum(returns), "Cumulative Return", "Total Return", "returns_cumulative")

# === Plot: Loss ===
save_plot(episodes, losses, "Episode Loss (raw)", "Loss", "losses_raw")
save_plot(episodes, np.cumsum(losses) / episodes, "Smoothed Loss (avg)", "Average Loss", "losses_avg")
save_plot(episodes, np.cumsum(losses), "Cumulative Loss", "Total Loss", "losses_cumulative")

# === Plot: Time ===
save_plot(episodes, times, "Episode Time (raw)", "Time (s)", "times_raw")
save_plot(episodes, np.cumsum(times) / episodes, "Smoothed Time (avg)", "Avg Time per Episode", "times_avg")
save_plot(episodes, np.cumsum(times), "Cumulative Time", "Total Time (s)", "times_cumulative")

# === Plot: Variance every 100 episodes ===
x_var = np.arange(1, len(variances) + 1) * 100
save_plot(x_var, variances, "Variance of Return (every 100 episodes)", "Return Variance", "returns_variance_100", color="purple")
