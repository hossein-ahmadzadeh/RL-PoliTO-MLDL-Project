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
    parser.add_argument('--n-episodes', default=50000, type=int, help='Number of training episodes')
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

    # TASK 2 and 3: interleave data collection to policy updates
    
	all_ad = []	# 🏆 Store returns per episode
	all_m_ad=[]
	
	episode_times = []  # ⏱ Store training time per episode
	ad_mean_window=[]
	losses = []  # Track loss per episode
	# Logs for the moving window variance of episode returns
	advantage_window = [] # A buffer to hold returns for the current window
	window_size = 100   # The size of the moving windo
	# Logs retrieved from agent.py (metrics calculated inside agent.update_polic

	
	agent_ad_variances_per_episode = [] # Variance of discounted returns (not advantages)

	variances_episode_ad_window =  []  

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
		
		
		loss, returns_variance,mean_advantage = agent.update_policy()
		losses.append(loss)

		agent_ad_variances_per_episode.append(returns_variance)

		end_time = time.time() # End timer for the episode
		current_episode_time = end_time - start_time

		all_ad.append(train_reward)
		all_m_ad.append(mean_advantage)

		 # Add current episode's total return to the list
		episode_times.append(current_episode_time)

		 # --- Collect Metrics for Logging ---
        # Collect agent's internal logs for this episode (last value)


        # Append all collected mu, sigma, entropy from agent (these are per-step, not per-episode)
		agent_mu_log.extend(agent.mu_log)
		agent_sigma_log.extend(agent.sigma_log)
		agent_entropy_log.extend(agent.entropy_log)
        
        # Clear agent's internal per-step logs after copying them
		agent.mu_log = []
		agent.sigma_log = []
		agent.entropy_log = []

		 # Add current episode's time to the list
		advantage_window.append(train_reward)
		ad_mean_window.append(mean_advantage)

		if len(advantage_window) > window_size:
			advantage_window.pop(0) # Remove the oldest return if window size is exceeded
        
        # Calculate variance only if there are enough samples in the window (at least 2)
		if len(advantage_window) > 1:
			current_variance_window = np.var(advantage_window)
		else:
			current_variance_window = 0.0 # If not enough data, set variance to 0.0
		variances_episode_ad_window.append(current_variance_window)
		# Log each 5000 episodes 
		if (episode + 1) % args.print_every == 0:
			print(f'--- Episode {episode + 1}/{args.n_episodes} ---')
			print(f'  Total Episode Return: {train_reward:.2f}')
			print(f'  Average Return (last {min(window_size, len(all_ad))} episodes): {np.mean(advantage_window):.2f}') 
			print(f'  Variance of Total Episode advantage (last {min(window_size, len(advantage_window))} episodes): {variances_episode_ad_window[-1]:.2f}')
            # Print the new variance from agent.py
			print(f'  Variance of Advantages (from Agent): {agent_ad_variances_per_episode[-1]:.2f}')
			print(f'  Policy Loss: {loss:.4f}' if loss is not None else '  Policy Loss: None')
			print(f'  Episode Time: {current_episode_time:.4f} sec')
			print("-" * 30)

	model_name = "AC-G"
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

	np.save(f"{log_dir}/advantages_mean_log.npy", np.array(agent.advantages_mean_log))
	np.save(f"{log_dir}/advantages_std_log.npy", np.array(agent.advantages_std_log))
	np.save(f"{log_dir}/td_target_mean_log.npy", np.array(agent.td_target_mean_log))
	np.save(f"{log_dir}/td_target_std_log.npy", np.array(agent.td_target_std_log))
	np.save(f"{analysis_dir}/advantages_variance_log.npy", np.array(agent.advantages_variance_log))
	np.save(f"{analysis_dir}/td_target_variance_log.npy", np.array(agent.td_target_variance_log))

	np.save(f"{log_dir}/advantages_log.npy", np.array(agent.advantages_log, dtype=object))
	np.save(f"{log_dir}/td_target_log.npy", np.array(agent.td_target_log, dtype=object))

	np.save(f"{analysis_dir}/episode_times.npy", np.array(episode_times))
	np.save(f"{analysis_dir}/advantage_per_episode.npy", np.array(all_ad))
	np.save(f"{analysis_dir}/advantage_mean.npy", np.array(all_m_ad))
	np.save(f"{analysis_dir}/losses_per_episode.npy", np.array(losses))
	np.save(f"{analysis_dir}/variances_episode_advantage_window.npy", np.array(variances_episode_ad_window))
	np.save(f"{analysis_dir}/agent_advantage_variances.npy", np.array(agent_ad_variances_per_episode))
	

	torch.save(agent.policy.state_dict(), model_path)


	

if __name__ == '__main__':
	main()