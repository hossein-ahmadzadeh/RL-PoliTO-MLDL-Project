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

	smoothed_returns = []     # To store rolling average return (score)
	returns_var_per_window = []  # To store rolling variance of return
	returns_window = []       # Buffer for rolling statistics
	window_size = 100         # Choose 100 for balance

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

		end_time = time.time()
		all_returns.append(train_reward)
		episode_times.append(end_time - start_time)
		

		# Log each 5000 episodes 
		if (episode+1)%args.print_every == 0:
			print(f"--- Episode {episode+1}/{args.n_episodes} ---")
			print(f"  Episode Return: {train_reward:.2f}")
			print(f"  Smoothed Return (last {len(returns_window)}): {avg_return:.2f}")
			print(f"  Return Variance (last {len(returns_window)}): {returns_var_per_window[-1]:.2f}")
			print(f"  Loss: {loss:.4f}")
			print(f"  Time: {end_time - start_time:.2f}s")
			print("-" * 40)
		
		# Update window
		returns_window.append(train_reward)
		if len(returns_window) > window_size:
			returns_window.pop(0)

		# Compute and store rolling stats
		avg_return = np.mean(returns_window)
		smoothed_returns.append(avg_return)
		returns_var_per_window.append(np.var(returns_window))


	# Ensure directories exist
	os.makedirs("models", exist_ok=True)
	os.makedirs("logs/model_reinforce_nobaseline_norm_tanh_action", exist_ok=True)
	os.makedirs("analysis/model_reinforce_nobaseline_norm_tanh_action", exist_ok=True)

	# Save logs
	np.save("logs/model_reinforce_nobaseline_norm_tanh_action/mu_log_tanh_action.npy", np.array(agent.mu_log))
	np.save("logs/model_reinforce_nobaseline_norm_tanh_action/sigma_log_tanh_action.npy", np.array(agent.sigma_log))
	np.save("logs/model_reinforce_nobaseline_norm_tanh_action/actions_log_tanh_action.npy", np.array(agent.actions_log))
	np.save("logs/model_reinforce_nobaseline_norm_tanh_action/entropy_log_tanh_action.npy", np.array(agent.entropy_log))
	np.save("logs/model_reinforce_nobaseline_norm_tanh_action/returns_mean_log.npy", np.array(agent.returns_mean_log))
	np.save("logs/model_reinforce_nobaseline_norm_tanh_action/returns_std_log.npy", np.array(agent.returns_std_log))


	# Save episode times
	np.save("analysis/model_reinforce_nobaseline_norm_tanh_action/episode_times_reinforce_nobaseline_norm_tanh_action.npy", np.array(episode_times))
	# Save returns
	np.save("analysis/model_reinforce_nobaseline_norm_tanh_action/returns_per_episode_reinforce_nobaseline_norm_tanh_action.npy", np.array(all_returns))
	# Save losses
	np.save("analysis/model_reinforce_nobaseline_norm_tanh_action/losses_per_episode_reinforce_nobaseline_norm_tanh_action.npy", np.array(losses))

	np.save("analysis/model_reinforce_nobaseline_norm_tanh_action/returns_smoothed_100.npy", np.array(smoothed_returns))
	np.save("analysis/model_reinforce_nobaseline_norm_tanh_action/returns_variance_100.npy", np.array(returns_var_per_window))
	

	# Save model
	torch.save(agent.policy.state_dict(), "models/model_reinforce_nobaseline_norm_tanh_action.mdl")

	

if __name__ == '__main__':
	main()