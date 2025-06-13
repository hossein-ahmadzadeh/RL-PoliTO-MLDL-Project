"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy 

import numpy as np
import os
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=5000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #


	all_returns = []	# 🏆 Store returns per episode
	episode_times = []  # ⏱ Store training time per episode
	losses = []  # Track loss per episode
	# Logs for the moving window variance of episode returns
	variances_episode_return_window =  []      # Stores the moving window variance of returns
	returns_window = [] # A buffer to hold returns for the current window
	window_size = 100   # The size of the moving windo
	# Logs retrieved from agent.py (metrics calculated inside agent.update_polic
	agent_returns_mean_log_per_episode = []
	agent_returns_std_log_per_episode = []
	agent_advantages_mean_log_per_episode = []
	agent_advantages_std_log_per_episode = []
	agent_returns_variances_per_episode = [] # Variance of discounted returns (not advantages)
# Stores variance of discounted returns (from Agent.update_policy)
# Agent's internal logs (mu, sigma, entropy) that are appended in get_action
	agent_mu_log = []
	agent_sigma_log = []
	agent_entropy_log = []


	for episode in range(args.n_episodes):
		start_time = time.time()
		
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		loss = agent.update_policy()
		losses.append(loss)

		end_time = time.time() # End timer for the episode
		current_episode_time = end_time - start_time
		all_returns.append(train_reward) # Add current episode's total return to the list
		episode_times.append(current_episode_time)
		 # --- Collect Metrics for Logging ---
        # Collect agent's internal logs for this episode (last value)
		agent_returns_mean_log_per_episode.append(agent.returns_mean_log[-1] if agent.returns_mean_log else 0.0)
		agent_returns_std_log_per_episode.append(agent.returns_std_log[-1] if agent.returns_std_log else 0.0)
		agent_advantages_mean_log_per_episode.append(agent.advantages_mean_log[-1] if agent.advantages_mean_log else 0.0)
		agent_advantages_std_log_per_episode.append(agent.advantages_std_log[-1])
		 # This is the variance of discounted returns (not advantages) from the agent
		agent_returns_variances_per_episode.append(agent.returns_variance_log[-1] if agent.returns_variance_log else 0.0)

        # Append all collected mu, sigma, entropy from agent (these are per-step, not per-episode)
		agent_mu_log.extend(agent.mu_log)
		agent_sigma_log.extend(agent.sigma_log)
		agent_entropy_log.extend(agent.entropy_log)
        
        # Clear agent's internal per-step logs after copying them
		agent.mu_log = []
		agent.sigma_log = []
		agent.entropy_log = []

		 # Add current episode's time to the list
		returns_window.append(train_reward)
		if len(returns_window) > window_size:
			returns_window.pop(0) # Remove the oldest return if window size is exceeded
        
        # Calculate variance only if there are enough samples in the window (at least 2)
		if len(returns_window) > 1:
			current_variance_window = np.var(returns_window)
			variances_episode_return_window.append(current_variance_window)
		else:
			variances_episode_return_window.append(0.0) # If not enough data, set variance to 0.0

        # Retrieve and store the variance logged by the agent (from returns_pg, now named 'returns')
		if agent.advantages_variance_log:
			agent_returns_variances_per_episode.append(agent.advantages_variance_log[-1])
		else:
			agent_returns_variances_per_episode.append(0.0)


		# Log each 5000 episodes 
		if (episode + 1) % args.print_every == 0:
			print(f'--- Episode {episode + 1}/{args.n_episodes} ---')
			print(f'  Total Episode Return: {train_reward:.2f}')
			print(f'  Average Return (last {min(window_size, len(all_returns))} episodes): {np.mean(returns_window):.2f}') 
			print(f'  Variance of Total Episode Returns (last {min(window_size, len(returns_window))} episodes): {variances_episode_return_window[-1]:.2f}')
            # Print the new variance from agent.py
			print(f'  Variance of Advantages (from Agent): {agent_returns_variances_per_episode[-1]:.2f}')
			print(f'  Policy Loss: {loss:.4f}' if loss is not None else '  Policy Loss: None')
			print(f'  Episode Time: {current_episode_time:.4f} sec')
			print("-" * 30)

	model_name = "REINFORCE-b"
	# Define paths
	log_dir = os.path.join("logs", model_name)
	analysis_dir = os.path.join("analysis", model_name)
	model_path = os.path.join("models", f"{model_name}.mdl")

	# Ensure all subdirectories exist
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(analysis_dir, exist_ok=True)
	os.makedirs("models", exist_ok=True)

		
	

	np.save(f"{log_dir}/mu_log.npy", np.array(agent_mu_log))
	np.save(f"{log_dir}/sigma_log.npy", np.array(agent_sigma_log))
	np.save(f"{log_dir}/entropy_log.npy", np.array(agent_entropy_log))

	np.save(f"{analysis_dir}/episode_times.npy", np.array(episode_times))
	np.save(f"{analysis_dir}/returns_per_episode.npy", np.array(all_returns))
	np.save(f"{analysis_dir}/losses_per_episode.npy", np.array(losses))
	np.save(f"{analysis_dir}/variances_episode_return_window.npy", np.array(variances_episode_return_window))
	np.save(f"{analysis_dir}/agent_returns_variances.npy", np.array(agent_returns_variances_per_episode))

	
	np.save(f"{log_dir}/agent_returns_mean_log_per_episode.npy", np.array(agent_returns_mean_log_per_episode))
	np.save(f"{log_dir}/agent_returns_std_log_per_episode.npy", np.array(agent_returns_std_log_per_episode))
	np.save(f"{log_dir}/agent_advantages_mean_log_per_episode.npy", np.array(agent_advantages_mean_log_per_episode))
	np.save(f"{log_dir}/agent_advantages_std_log_per_episode.npy", np.array(agent_advantages_std_log_per_episode))

	
	torch.save(agent.policy.state_dict(), model_path)


	

if __name__ == '__main__':
	main()