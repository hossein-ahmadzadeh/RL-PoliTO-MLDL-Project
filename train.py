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
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	# Ensure directories exist
	os.makedirs("models", exist_ok=True)
	os.makedirs("logs", exist_ok=True)
	os.makedirs("analysis", exist_ok=True)

	# Save logs
	np.save("logs/mu_log.npy",     np.array(agent.mu_log))
	np.save("logs/sigma_log.npy",  np.array(agent.sigma_log))
	np.save("logs/actions_log.npy", np.array(agent.actions_log))
	np.save("logs/entropy_log.npy",  np.array(agent.entropy_log))

	# Save episode times
	np.save("analysis/episode_times_reinforce_nobaseline_nonnorm.npy", np.array(episode_times))
	# Save returns
	np.save("analysis/returns_per_episode_reinforce_nobaseline_nonnorm.npy", np.array(all_returns))
	# Save losses
	np.save("analysis/losses_per_episode_reinforce_nobaseline_nonnorm.npy", np.array(losses))


	# Save model
	torch.save(agent.policy.state_dict(), "models/model_reinforce_nobaseline_nonnorm.mdl")

	

if __name__ == '__main__':
	main()