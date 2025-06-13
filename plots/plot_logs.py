import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

# -------- path-------
model_name = "REINFORCE"  
log_dir = f"logs/{model_name}"
output_dir = f"report/{model_name}/images/log"
os.makedirs(output_dir, exist_ok=True)


def safe_load(path):
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None
    return np.load(path, allow_pickle=True)

mu_log = safe_load(f"{log_dir}/mu_log.npy")
sigma_log = safe_load(f"{log_dir}/sigma_log.npy")
entropy_log = safe_load(f"{log_dir}/entropy_log.npy")



def plot_3d_lines(data, title, ylabel, filename):
    if data is None:
        return
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


def plot_averaged_actions(actions, avg_every=10000):
    if actions is None:
        return
    actions = np.array(actions)
    num_chunks = len(actions) // avg_every
    avg_actions = np.array([
        actions[i * avg_every:(i + 1) * avg_every].mean(axis=0)
        for i in range(num_chunks)
    ])
    x = np.arange(len(avg_actions)) * avg_every

    plt.figure(figsize=(12, 6))
    plt.plot(x, avg_actions[:, 0], label="Action[0]", color="red")
    plt.plot(x, avg_actions[:, 1], label="Action[1]", color="green")
    plt.plot(x, avg_actions[:, 2], label="Action[2]", color="blue")
    plt.xlabel(f"Step (averaged every {avg_every})")
    plt.ylabel("Average Action")
    plt.title(f"Average Sampled Actions (every {avg_every} steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actions_log_avg_{avg_every}.png", dpi=300)
    plt.close()


plot_3d_lines(mu_log, "Policy Mean (μ) per Episode", "μ", "mu_log")
plot_3d_lines(sigma_log, "Policy Std Dev (σ) per Episode", "σ", "sigma_log")

if entropy_log is not None:
    plt.figure(figsize=(12, 5))
    plt.plot(entropy_log, color='purple', linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.title("Policy Entropy over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/entropy_log.png", dpi=300)
    plt.close()

print(f"✅ All log plots saved to: {output_dir}")
