import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

# === Paths ===
model_name = "model_reinforce_with_actor_critic_norm_tanh_action"
log_dir = f"logs/{model_name}"
output_dir = f"report/{model_name}/images/log"
os.makedirs(output_dir, exist_ok=True)

# === Load logs ===
mu_log = np.load(f"{log_dir}/mu_log.npy")
sigma_log = np.load(f"{log_dir}/sigma_log.npy")
actions_log = np.load(f"{log_dir}/actions_log.npy")
entropy_log = np.load(f"{log_dir}/entropy_log.npy")
td_target_mean_log = np.load(f"{log_dir}/td_target_mean_log.npy")
td_target_std_log = np.load(f"{log_dir}/td_target_std_log.npy")
advantages_mean_log = np.load(f"{log_dir}/advantages_mean_log.npy")
advantages_std_log = np.load(f"{log_dir}/advantages_std_log.npy")
advantages_log = np.load(f"{log_dir}/advantages_log.npy", allow_pickle=True)
td_target_log = np.load(f"{log_dir}/td_target_log.npy", allow_pickle=True)

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
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
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
    plt.savefig(f"{output_dir}/actions_log_avg_{avg_every}.png", dpi=300)
    plt.close()

# === Plotting ===
plot_3d_lines(mu_log, "Policy Mean (μ) per Episode", "μ", "mu_log")
plot_3d_lines(sigma_log, "Policy Std Dev (σ) per Episode", "σ", "sigma_log")
plot_3d_lines(actions_log, "Sampled Actions per Step", "Action", "actions_log")

# Entropy
plt.figure(figsize=(12, 5))
plt.plot(entropy_log, color='purple')
plt.xlabel("Step")
plt.ylabel("Entropy")
plt.title("Policy Entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/entropy_log.png", dpi=300)
plt.close()

# Averaged actions
plot_averaged_actions(actions_log, avg_every=1000)

# TD Target stats
plt.figure(figsize=(12, 5))
plt.plot(td_target_mean_log, label="Mean of TD Target", color='orange')
plt.xlabel("Episode")
plt.ylabel("Mean")
plt.title("Mean of TD Target")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/td_target_mean_log.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(td_target_std_log, label="Std Dev of TD Target", color='teal')
plt.xlabel("Episode")
plt.ylabel("Std Dev")
plt.title("Std Dev of TD Target")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/td_target_std_log.png", dpi=300)
plt.close()

# Advantage stats
plt.figure(figsize=(12, 5))
plt.plot(advantages_mean_log, label="Advantage Mean", color='red')
plt.xlabel("Episode")
plt.ylabel("Mean")
plt.title("Mean of Advantage per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/advantages_mean_log.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(advantages_std_log, label="Advantage Std Dev", color='blue')
plt.xlabel("Episode")
plt.ylabel("Std Dev")
plt.title("Std Dev of Advantage per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/advantages_std_log.png", dpi=300)
plt.close()

# === Plot: TD Target Values per Episode ===
plt.figure(figsize=(12, 5))
for i, td in enumerate(td_target_log):
    plt.plot(td, color='orange', alpha=0.1)  # Light transparency
plt.xlabel("Time Step")
plt.ylabel("TD Target")
plt.title("TD Target Values per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/td_target_log.png", dpi=300)
plt.close()

# === Plot: Advantage Values per Episode ===
plt.figure(figsize=(12, 5))
for i, adv in enumerate(advantages_log):
    plt.plot(adv, color='crimson', alpha=0.1)
plt.xlabel("Time Step")
plt.ylabel("Advantage")
plt.title("Advantage Values per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/advantages_log.png", dpi=300)
plt.close()
