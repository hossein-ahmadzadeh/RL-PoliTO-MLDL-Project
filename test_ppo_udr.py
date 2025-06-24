"""Test a trained SB3 PPO agent on the Hopper environment"""
import argparse
import gym
import numpy as np
import os
import time

from env.custom_hopper import *  # CustomHopper-source/target
from stable_baselines3 import PPO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='BestModelTuning/model/15_CustomHopper-target-v0/best_model.zip', type=str, help='Path to PPO model file')
    parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulation')
    return parser.parse_args()

def main():
    args = parse_args()

    # Choose environment
    env = gym.make('CustomHopper-target-v0')  # ← use 'source' if testing on source env

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)
    print('Dynamics parameters:', env.get_parameters())

    model = PPO.load(args.model_path, device=args.device)

    test_returns = []
    test_times = []

    for episode in range(args.episodes):
        done = False
        obs = env.reset()
        total_reward = 0
        start_time = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if args.render:
                env.render()

        test_returns.append(total_reward)
        test_times.append(time.time() - start_time)

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

    # Save test results
    test_log_dir = "PPO_BEST_MODEL_TEST_Target"
    os.makedirs(test_log_dir, exist_ok=True)

    # Clean tag (e.g. ppo_udr_source from ppo_udr_source.zip)
    model_tag = os.path.splitext(os.path.basename(args.model_path))[0]

    np.save(f"{test_log_dir}/returns_test_{model_tag}.npy", np.array(test_returns))
    np.save(f"{test_log_dir}/times_test_{model_tag}.npy", np.array(test_times))


if __name__ == "__main__":
    main()
