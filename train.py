import argparse
import torch
import gym
import numpy as np
import os
import time
import wandb

from env.custom_hopper import *
from agent import Agent, Policy 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int)
    parser.add_argument('--print-every', default=1000, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model-name', default='ActorCritic', type=str)
    return parser.parse_args()

args = parse_args()

# === Paths ===
model_name = args.model_name
log_dir = f"logs/{model_name}"
analysis_dir = f"analysis/{model_name}"
model_path = f"models/{model_name}.mdl"

# === Create directories ===
os.makedirs(log_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

def main():
    # === Initialize wandb ===
    wandb.init(project="RL-Hopper-All", name=model_name)

    env = gym.make('CustomHopper-source-v0')
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    training_rewards_per_episode = []
    times_per_episode = []
    sim_times_per_episode = []
    cumulative_wall_time = []
    cumulative_sim_time = []
    losses_per_episode = []

    smoothed_training_rewards = []
    training_reward_variance_window = []
    recent_training_rewards_window = []
    window_size = 100

    for episode in range(args.n_episodes):
        start_time = time.time()
        done = False
        train_reward = 0
        state = env.reset()

        episode_steps = 0
        while not done:
            action, action_probabilities = agent.get_action(state)
            previous_state = state
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            agent.store_outcome(previous_state, state, action_probabilities, reward, done)
            train_reward += reward
            episode_steps += 1

        loss = agent.update_policy()
        losses_per_episode.append(loss)

        sim_duration = episode_steps * 0.008
        end_time = time.time()
        wall_time = end_time - start_time

        training_rewards_per_episode.append(train_reward)
        times_per_episode.append(wall_time)
        sim_times_per_episode.append(sim_duration)

        total_wall = (cumulative_wall_time[-1] if cumulative_wall_time else 0) + wall_time
        total_sim = (cumulative_sim_time[-1] if cumulative_sim_time else 0) + sim_duration
        cumulative_wall_time.append(total_wall)
        cumulative_sim_time.append(total_sim)

        recent_training_rewards_window.append(train_reward)
        if len(recent_training_rewards_window) > window_size:
            recent_training_rewards_window.pop(0)

        mean_reward, var_reward = None, None
        if len(recent_training_rewards_window) == window_size:
            mean_reward = np.mean(recent_training_rewards_window)
            var_reward = np.var(recent_training_rewards_window)
            smoothed_training_rewards.append(mean_reward)
            training_reward_variance_window.append(var_reward)
            wandb.log({
                "rolling_reward_mean_100": mean_reward,
                "rolling_reward_var_100": var_reward
            }, step=episode + 1)

        # === wandb logging ===
        wandb.log({
            "episode": episode + 1,
            "reward_per_episode": train_reward,
            "episode_length": episode_steps,
            "sim_time_per_episode": sim_duration,
            "wall_time_per_episode": wall_time,
            "cumulative_sim_time": total_sim,
            "cumulative_wall_time": total_wall
        }, step=episode + 1)

        if (episode + 1) % args.print_every == 0:
            print(f"--- Episode {episode+1}/{args.n_episodes} ---")
            print(f"  Reward: {train_reward:.2f} | Smoothed: {mean_reward:.2f} | Variance: {var_reward:.2f}")
            print(f"  Time: {wall_time:.2f}s (Sim: {sim_duration:.2f}s)")
            print("-" * 40)

    # === Save logs ===
    np.save(f"{log_dir}/mu_log.npy", np.array(agent.mu_log))
    np.save(f"{log_dir}/sigma_log.npy", np.array(agent.sigma_log))
    np.save(f"{log_dir}/entropy_log.npy", np.array(agent.entropy_log))

    np.save(f"{log_dir}/td_target_mean_log.npy", np.array(agent.td_target_mean_log))
    np.save(f"{log_dir}/td_target_std_log.npy", np.array(agent.td_target_std_log))
    np.save(f"{log_dir}/td_target_variance_log.npy", np.array(agent.td_target_variance_log))
    
    np.save(f"{log_dir}/advantages_mean_log.npy", np.array(agent.advantages_mean_log))
    np.save(f"{log_dir}/advantages_std_log.npy", np.array(agent.advantages_std_log))
    np.save(f"{log_dir}/advantages_variance_log.npy", np.array(agent.advantages_variance_log))

    np.save(f"{analysis_dir}/episode_times_wallclock.npy", np.array(times_per_episode))
    np.save(f"{analysis_dir}/episode_times_simulated.npy", np.array(sim_times_per_episode))
    np.save(f"{analysis_dir}/episode_rewards.npy", np.array(training_rewards_per_episode))
    np.save(f"{analysis_dir}/losses.npy", np.array(losses_per_episode))
    np.save(f"{analysis_dir}/episode_rewards_smoothed_100.npy", np.array(smoothed_training_rewards))
    np.save(f"{analysis_dir}/episode_rewards_variance_100.npy", np.array(training_reward_variance_window))
    np.save(f"{analysis_dir}/cumulative_wall_time.npy", np.array(cumulative_wall_time))
    np.save(f"{analysis_dir}/cumulative_sim_time.npy", np.array(cumulative_sim_time))

    # === Save model ===
    torch.save(agent.policy.state_dict(), model_path)
    wandb.save(model_path)
    wandb.finish()

if __name__ == '__main__':
    main()
