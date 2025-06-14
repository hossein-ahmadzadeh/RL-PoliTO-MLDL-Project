"""Test an RL agent on the OpenAI Gym Hopper environment"""
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
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()

model_name = "model_actor_critic_norm_tanh_entropy"	 # Change this to switch models
model_path = args.model or f"models/{model_name}.mdl"
test_output_dir = f"test_analysis/{model_name}"
os.makedirs(test_output_dir, exist_ok=True)

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    policy = Policy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path), strict=True)

    agent = Agent(policy, device=args.device)

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
        test_times.append(time.time() - start_time)

        print(f"Episode: {episode} | Return: {test_reward:.2f}")

    # Save returns and durations
    np.save(f"{test_output_dir}/returns_test_{model_name}.npy", np.array(test_returns))
    np.save(f"{test_output_dir}/times_test_{model_name}.npy", np.array(test_times))


if __name__ == '__main__':
    main()
