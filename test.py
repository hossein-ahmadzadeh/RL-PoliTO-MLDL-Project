"""Test an RL agent on the OpenAI Gym Hopper environment"""
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
    parser.add_argument('--model', default="models/REINFORCE-b.mdl", type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	agent = Agent(policy, device=args.device)

	# Initialize lists to store returns and times
	test_returns = []
	test_times = []

	for episode in range(args.episodes):
		start_time = time.time()
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward
		
		test_returns.append(test_reward)
		end_time = time.time()
		test_times.append(end_time - start_time)

		print(f"Episode: {episode} | Return: {test_reward}")


	os.makedirs("test_analysis", exist_ok=True)
	
	# Save the test returns and times
	
	
	model_name = os.path.splitext(os.path.basename(args.model))[0]

	
	output_dir = "test_analysis"
	os.makedirs(output_dir, exist_ok=True)


	np.save(f"{output_dir}/returns_test_{model_name}.npy", np.array(test_returns))
	np.save(f"{output_dir}/times_test_{model_name}.npy", np.array(test_times))

	print(f"✅ Test results saved to: {output_dir} as returns_test_{model_name}.npy and times_test_{model_name}.npy")


if __name__ == '__main__':
	main()