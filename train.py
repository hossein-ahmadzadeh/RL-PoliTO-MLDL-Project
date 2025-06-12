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


	all_rewards = []	# 🏆 Store rewards (scores) per episode
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
		all_rewards.append(train_reward)
		episode_times.append(end_time - start_time)
		

		# Log each 5000 episodes 
		if (episode+1)%args.print_every == 0:
			print(f"--- Episode {episode+1}/{args.n_episodes} ---")
			print(f"  Episode Reward: {train_reward:.2f}")
			print(f"  Smoothed Reward (last {len(returns_window)}): {avg_return:.2f}")
			print(f"  Reward Variance (last {len(returns_window)}): {returns_var_per_window[-1]:.2f}")
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
	os.makedirs("logs/model_reinforce_with_actor_critic_norm_tanh_action", exist_ok=True)
	os.makedirs("analysis/model_reinforce_with_actor_critic_norm_tanh_action", exist_ok=True)

	# Save logs
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/mu_log.npy", np.array(agent.mu_log))
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/sigma_log.npy", np.array(agent.sigma_log))
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/actions_log.npy", np.array(agent.actions_log))
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/entropy_log.npy", np.array(agent.entropy_log))

	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/advantages_mean_log.npy", np.array(agent.advantages_mean_log))
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/advantages_std_log.npy", np.array(agent.advantages_std_log))
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/td_target_mean_log.npy", np.array(agent.td_target_mean_log))
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/td_target_std_log.npy", np.array(agent.td_target_std_log))

	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/advantages_log.npy", np.array(agent.advantages_log, dtype=object))
	np.save("logs/model_reinforce_with_actor_critic_norm_tanh_action/td_target_log.npy", np.array(agent.td_target_log, dtype=object))


	# Save episode times
	np.save("analysis/model_reinforce_with_actor_critic_norm_tanh_action/episode_times.npy", np.array(episode_times))
	# Save returns
	np.save("analysis/model_reinforce_with_actor_critic_norm_tanh_action/episode_rewards.npy", np.array(all_rewards))
	# Save losses
	np.save("analysis/model_reinforce_with_actor_critic_norm_tanh_action/losses.npy", np.array(losses))

	np.save("analysis/model_reinforce_with_actor_critic_norm_tanh_action/episode_rewards_smoothed_100.npy", np.array(smoothed_returns))
	np.save("analysis/model_reinforce_with_actor_critic_norm_tanh_action/episode_rewards_variance_100.npy", np.array(returns_var_per_window))

	np.save("analysis/model_reinforce_with_actor_critic_norm_tanh_action/advantages_variance_log.npy", np.array(agent.advantages_variance_log))
	np.save("analysis/model_reinforce_with_actor_critic_norm_tanh_action/td_target_variance_log.npy", np.array(agent.td_target_variance_log))



	# Save model
	torch.save(agent.policy.state_dict(), "models/model_reinforce_with_actor_critic_norm_tanh_action.mdl")

	

if __name__ == '__main__':
	main()