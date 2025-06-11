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
    parser.add_argument('--n-episodes', default=1000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=500, type=int, help='Print info every <> episodes')
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
	variances = []      # Stores the moving window variance of returns
	returns_window = [] # A buffer to hold returns for the current window
	window_size = 100   # The size of the moving windo
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
		episode_times.append(current_episode_time) # Add current episode's time to the list
		returns_window.append(train_reward) # Add current episode's return to the window
		if len(returns_window) > window_size:
			returns_window.pop(0) # Remove the oldest return if the window size is exceeded
        
        # Calculate variance only if there are enough samples in the window (at least 2)
		if len(returns_window) > 1:
			current_variance_window = np.var(returns_window)
			variances.append(current_variance_window)
		else:
			variances.append(0.0) # If not enough data, set variance to 0 or NaN
		

		# Log each 5000 episodes 
		if (episode + 1) % args.print_every == 0:
			print(f'--- Episode {episode + 1}/{args.n_episodes} ---')
			print(f'  Total Episode Return: {train_reward:.2f}')
            # Print average return over the current window
			print(f'  Average Return (last {min(window_size, len(all_returns))} episodes): {np.mean(returns_window):.2f}') 
            # Print variance of returns over the current window
			print(f'  Variance of Returns (last {min(window_size, len(returns_window))} episodes): {variances[-1]:.2f}') 
			print(f'  Policy Loss: {loss:.4f}')
			print(f'  Episode Time: {current_episode_time:.4f} sec')
			print("-" * 30)

	model_name="testttttt"
	# Ensure directories exist
	os.makedirs("models", exist_ok=True)
	os.makedirs("logs", exist_ok=True)
	os.makedirs("analysis", exist_ok=True)

	# Save logs
	np.save(f"logs/log.npy",     np.array(agent.mu_log))
	np.save(f"logs/sigma_log.npy",  np.array(agent.sigma_log))
	np.save(f"logs/entropy_log.npy",  np.array(agent.entropy_log))

	# Save episode times
	np.save(f"analysis/episode_times_{model_name}.npy", np.array(episode_times))
	# Save returns
	np.save(f"analysis/returns_per_episode_{model_name}.npy", np.array(all_returns))
	# Save losses
	np.save(f"analysis/losses_per_episode_{model_name}.npy", np.array(losses))
	# Save variances
	np.save(f"analysis/variances_per_episode_{model_name}.npy", np.array(variances)) 


	# Save model
	torch.save(agent.policy.state_dict(), f"models/{model_name}.mdl")

	

if __name__ == '__main__':
	main()