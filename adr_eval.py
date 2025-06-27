import os
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.custom_hopper import *  # Custom Hopper env registration

# Models excluding UDR
MODELS = {
    "Target → Target [Upper Bound]": "./BestModelTuning/model/15_CustomHopper-target-v0/best_model.zip",
    "Source → Target [Lower Bound]": "./BestModelTuning/model/9_CustomHopper-source-v0/best_model.zip",
    "ADR Δ=0.02": "./best_models/best_eval_ADR0.02/best_model.zip",
    "ADR Δ=0.05": "./best_models/best_eval_ADR0.05/best_model.zip",
    "ADR Δ=0.1": "./best_models/best_eval_ADR0.1/best_model.zip",
}

# Consistent color palette
COLOR_MAP = {
    "Target → Target [Upper Bound]": "#1f77b4",  # Blue
    "Source → Target [Lower Bound]": "#ff7f0e",  # Orange
    "ADR Δ=0.02": "#d62728",                     # Red
    "ADR Δ=0.05": "#9467bd",                     # Purple
    "ADR Δ=0.1": "#8c564b",                      # Brown
}

# Output directories
SAVE_DIR = os.path.join("LAST", "No_UDR")
LOG_DIR = os.path.join(SAVE_DIR, "log")
IMG_DIR = os.path.join(SAVE_DIR, "images")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

EPISODES = 50
DEVICE = "cuda"

def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def evaluate_model(name, path):
    print(f"\nEvaluating {name}")
    env = gym.make('CustomHopper-target-v0')
    model = PPO.load(path, device=DEVICE)

    rewards, times = [], []
    for ep in range(EPISODES):
        obs = env.reset()
        done, total_reward = False, 0
        start = time.time()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        times.append(time.time() - start)
        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

    tag = name.replace(" ", "_").replace("→", "to").replace("[", "").replace("]", "").replace(".", "")
    np.save(os.path.join(LOG_DIR, f"returns_test_{tag}.npy"), np.array(rewards))
    np.save(os.path.join(LOG_DIR, f"times_test_{tag}.npy"), np.array(times))

    return name, rewards

# Run evaluation for each model
results = [evaluate_model(name, path) for name, path in MODELS.items()]

# === Line Plot (Smoothed Reward per Episode) ===
plt.figure(figsize=(12, 6))
for name, rewards in results:
    smoothed = smooth(rewards)
    plt.plot(range(10, 10 + len(smoothed)), smoothed, label=name, color=COLOR_MAP[name])
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed Reward per Episode (No UDR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "reward_vs_episode_smoothed.png"))
plt.close()

# === Bar Plot: Mean Reward ===
plt.figure(figsize=(10, 5))
names = [name for name, _ in results]
means = [np.mean(r) for _, r in results]
bars = plt.bar(names, means, color=[COLOR_MAP[name] for name in names])
for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{mean:.1f}",
             ha='center', va='bottom', fontsize=9)
plt.ylabel("Mean Reward")
plt.xticks(rotation=30, ha="right")
plt.title("Mean Reward per Model (No UDR)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "reward_mean_bar.png"))
plt.close()

# === Bar Plot: Std Reward ===
plt.figure(figsize=(10, 5))
stds = [np.std(r) for _, r in results]
bars = plt.bar(names, stds, color=[COLOR_MAP[name] for name in names])
for bar, std in zip(bars, stds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{std:.1f}",
             ha='center', va='bottom', fontsize=9)
plt.ylabel("Std Reward")
plt.xticks(rotation=30, ha="right")
plt.title("Standard Deviation of Reward per Model (No UDR)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "reward_std_bar.png"))
plt.close()

print("\nEvaluation without UDR completed. Results saved in 'LAST/No_UDR/'")
