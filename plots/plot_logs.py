import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

# === Model & Directory Settings ===
model_name = "model_reinforce_with_baseline"  # <------------------- Updated to your working model
base_log_dir = "logs"
base_report_dir = "report"

log_dir = os.path.join(base_log_dir, model_name)
output_dir = os.path.join(base_report_dir, model_name, "images", "log")
os.makedirs(output_dir, exist_ok=True)

# === Load available logs ===
mu_log = np.load(f"{log_dir}/mu_log.npy")
sigma_log = np.load(f"{log_dir}/sigma_log.npy")
actions_log = np.load(f"{log_dir}/actions_log.npy")
entropy_log = np.load(f"{log_dir}/entropy_log.npy")
discounted_returns_mean_log = np.load(f"{log_dir}/discounted_returns_mean_log.npy")
discounted_returns_std_log = np.load(f"{log_dir}/discounted_returns_std_log.npy")
discounted_returns_variance_log = np.load(f"{log_dir}/discounted_returns_variance_log.npy")
advantages_mean_log = np.load(f"{log_dir}/advantages_mean_log.npy")
advantages_std_log = np.load(f"{log_dir}/advantages_std_log.npy")
advantages_variance_log = np.load(f"{log_dir}/advantages_variance_log.npy")

# === Plot helpers ===
def plot_3d_lines(data, title, ylabel, filename):
    data = np.array(data)
    x = np.arange(len(data))
    plt.figure(figsize=(12, 6))
    plt.plot(x, data[:, 0], label=f"{ylabel}[0]", color="red")
    plt.plot(x, data[:, 1], label=f"{ylabel}[1]", color="green")
    plt.plot(x, data[:, 2], label=f"{ylabel}[2]", color="blue")
    plt.xlabel("Episode" if "mu" in filename or "sigma" in filename else "Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.close()

def plot_averaged_actions(actions, avg_every=1000):
    actions = np.array(actions)
    num_chunks = len(actions) // avg_every
    avg_actions = np.array([
        actions[i*avg_every:(i+1)*avg_every].mean(axis=0)
        for i in range(num_chunks)
    ])
    x = np.arange(len(avg_actions)) * avg_every
    plt.figure(figsize=(12, 6))
    plt.plot(x, avg_actions[:, 0], label="Action[0]", color="red")
    plt.plot(x, avg_actions[:, 1], label="Action[1]", color="green")
    plt.plot(x, avg_actions[:, 2], label="Action[2]", color="blue")
    plt.xlabel(f"Step (avg every {avg_every})")
    plt.ylabel("Avg Action")
    plt.title("Average Sampled Actions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"actions_log_avg_{avg_every}.png"), dpi=300)
    plt.close()

# === Plotting ===
plot_3d_lines(mu_log, "Policy Mean (\u03bc) per Episode", "\u03bc", "mu_log")
plot_3d_lines(sigma_log, "Policy Std Dev (\u03c3) per Episode", "\u03c3", "sigma_log")
plot_3d_lines(actions_log, "Sampled Actions per Step", "Action", "actions_log")

# Entropy
plt.figure(figsize=(12, 5))
plt.plot(entropy_log, color='purple')
plt.xlabel("Step")
plt.ylabel("Entropy")
plt.title("Policy Entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "entropy_log.png"), dpi=300)
plt.close()

# Averaged actions
plot_averaged_actions(actions_log, avg_every=1000)

# Discounted returns stats
plt.figure(figsize=(12, 5))
plt.plot(discounted_returns_mean_log, label="Mean of Discounted Return", color='orange')
plt.xlabel("Episode")
plt.ylabel("Mean")
plt.title("Mean of Discounted Return")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "discounted_returns_mean_log.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(discounted_returns_std_log, label="Std Dev of Discounted Return", color='teal')
plt.xlabel("Episode")
plt.ylabel("Std Dev")
plt.title("Std Dev of Discounted Return")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "discounted_returns_std_log.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(discounted_returns_variance_log, label="Variance of Discounted Return", color='brown')
plt.xlabel("Episode")
plt.ylabel("Variance")
plt.title("Variance of Discounted Return")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "discounted_returns_variance_log.png"), dpi=300)
plt.close()

# Advantages stats
plt.figure(figsize=(12, 5))
plt.plot(advantages_mean_log, label="Advantage Mean", color='crimson')
plt.xlabel("Episode")
plt.ylabel("Mean")
plt.title("Mean of Advantage per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "advantages_mean_log.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(advantages_std_log, label="Advantage Std Dev", color='blue')
plt.xlabel("Episode")
plt.ylabel("Std Dev")
plt.title("Std Dev of Advantage per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "advantages_std_log.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(advantages_variance_log, label="Advantage Variance", color='darkgreen')
plt.xlabel("Episode")
plt.ylabel("Variance")
plt.title("Variance of Advantage per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "advantages_variance_log.png"), dpi=300)
plt.close()
