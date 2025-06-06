import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl


mpl.rcParams['agg.path.chunksize'] = 10000

# Create output directory
os.makedirs("report/reinforce_nobaseline_nonnorm_tanh_action/images/log", exist_ok=True)

# Load logs
mu_log = np.load("logs/mu_log_tanh_action.npy")
sigma_log = np.load("logs/sigma_log_tanh_action.npy")
actions_log = np.load("logs/actions_log_tanh_action.npy")
entropy_log = np.load("logs/entropy_log_tanh_action.npy")

# Function to plot 3 separate curves in one plot
def plot_3d_lines(data, title, ylabel, filename):
    x = np.arange(len(data))
    data = np.array(data)
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
    plt.savefig(f"report/reinforce_nobaseline_nonnorm_tanh_action/images/log/{filename}.png", dpi=300)
    plt.close()


# Function to compute and plot moving averages for actions
def plot_averaged_actions(actions, avg_every=10000):
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
    plt.xlabel(f"Step (averaged every {avg_every})")
    plt.ylabel("Average Action")
    plt.title(f"Average Sampled Actions (every {avg_every} steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"report/reinforce_nobaseline_nonnorm_tanh_action/images/log/actions_log_avg_{avg_every}.png", dpi=300)
    plt.close()



# μ per episode
plot_3d_lines(mu_log, "Policy Mean (μ) per Episode", "μ", "mu_log")

# σ per episode
plot_3d_lines(sigma_log, "Policy Std Dev (σ) per Episode", "σ", "sigma_log")

# actions per step
plot_3d_lines(actions_log, "Sampled Actions per Step", "Action", "actions_log")

# entropy per step
plt.figure(figsize=(12, 5))
plt.plot(entropy_log, color='purple', linewidth=1)
plt.xlabel("Step")
plt.ylabel("Entropy")
plt.title("Policy Entropy over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("report/reinforce_nobaseline_nonnorm_tanh_action/images/log/entropy_log.png", dpi=300)
plt.close()

# Plot smoothed/averaged actions
plot_averaged_actions(actions_log, avg_every=10000)  # or try 1000 / 100000 too
