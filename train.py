import argparse
import torch
import gym
import numpy as np
import os
import time

from env.custom_hopper import *
from agent import Agent, Policy 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int)
    parser.add_argument('--print-every', default=5000, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--model-name', default='model_actor_critic_norm_tanh_entropy', type=str)
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
	env = gym.make('CustomHopper-source-v0')
	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

	all_rewards, episode_times, losses = [], [], []
	smoothed_returns, returns_var_per_window, returns_window = [], [], []
	window_size = 100

	for episode in range(args.n_episodes):
		start_time = time.time()
		done = False
		train_reward = 0
		state = env.reset()

		while not done:
			action, action_probabilities = agent.get_action(state)
			previous_state = state
			state, reward, done, _ = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			train_reward += reward

		loss = agent.update_policy()
		losses.append(loss)

		end_time = time.time()
		all_rewards.append(train_reward)
		episode_times.append(end_time - start_time)

		if (episode + 1) % args.print_every == 0:
			avg_return = np.mean(returns_window) if returns_window else 0
			var_return = np.var(returns_window) if returns_window else 0
			print(f"--- Episode {episode+1}/{args.n_episodes} ---")
			print(f"  Reward: {train_reward:.2f} | Smoothed: {avg_return:.2f} | Variance: {var_return:.2f}")
			print(f"  Loss: {loss:.4f} | Time: {end_time - start_time:.2f}s")
			print("-" * 40)

		returns_window.append(train_reward)
		if len(returns_window) > window_size:
			returns_window.pop(0)

		smoothed_returns.append(np.mean(returns_window))
		returns_var_per_window.append(np.var(returns_window))

	# === Save logs ===
	np.save(f"{log_dir}/mu_log.npy", np.array(agent.mu_log))
	np.save(f"{log_dir}/sigma_log.npy", np.array(agent.sigma_log))
	np.save(f"{log_dir}/actions_log.npy", np.array(agent.actions_log))
	np.save(f"{log_dir}/entropy_log.npy", np.array(agent.entropy_log))

	np.save(f"{log_dir}/advantages_mean_log.npy", np.array(agent.advantages_mean_log))
	np.save(f"{log_dir}/advantages_std_log.npy", np.array(agent.advantages_std_log))
	np.save(f"{log_dir}/td_target_mean_log.npy", np.array(agent.td_target_mean_log))
	np.save(f"{log_dir}/td_target_std_log.npy", np.array(agent.td_target_std_log))

	np.save(f"{log_dir}/advantages_log.npy", np.array(agent.advantages_log, dtype=object))
	np.save(f"{log_dir}/td_target_log.npy", np.array(agent.td_target_log, dtype=object))

	np.save(f"{analysis_dir}/episode_times.npy", np.array(episode_times))
	np.save(f"{analysis_dir}/episode_rewards.npy", np.array(all_rewards))
	np.save(f"{analysis_dir}/losses.npy", np.array(losses))
	np.save(f"{analysis_dir}/episode_rewards_smoothed_100.npy", np.array(smoothed_returns))
	np.save(f"{analysis_dir}/episode_rewards_variance_100.npy", np.array(returns_var_per_window))
	np.save(f"{analysis_dir}/advantages_variance_log.npy", np.array(agent.advantages_variance_log))
	np.save(f"{analysis_dir}/td_target_variance_log.npy", np.array(agent.td_target_variance_log))

	# === Save model ===
	torch.save(agent.policy.state_dict(), model_path)

if __name__ == '__main__':
	main()
