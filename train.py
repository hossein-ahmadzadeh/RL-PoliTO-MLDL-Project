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
    parser.add_argument('--n-episodes', default=10000, type=int)
    parser.add_argument('--print-every', default=1000, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--model-name', default='model_actor_critic_simple_norm_tanh', type=str)
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

	# ----------------------------------------------------------------------------------------------- #
	# Logs per episode
	training_rewards_per_episode = []       # Total reward collected in each episode
	times_per_episode = []                  # Duration of each episode
	simulated_times_per_episode = []        # Simulated time per episode (assuming 0.008s per step)
	losses_per_episode = []                 # Policy loss recorded per episode

	# Rolling window metrics for smoothing and stability analysis
	smoothed_training_rewards = []          # Moving average of recent episode rewards
	training_reward_variance_window = []    # Variance of rewards over recent episodes (training stability)
	recent_training_rewards_window = []     # Buffer holding the most recent N episode rewards

	window_size = 100  # Number of episodes to include in the rolling window
	# ----------------------------------------------------------------------------------------------- #


	for episode in range(args.n_episodes):
		start_time = time.time()
		done = False
		train_reward = 0
		step_count = 0  
		state = env.reset()

		while not done:
			action, action_probabilities = agent.get_action(state)
			previous_state = state
			state, reward, done, _ = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			train_reward += reward
			step_count += 1  # <- Count the steps

		loss = agent.update_policy()
		losses_per_episode.append(loss)

		end_time = time.time()
		training_rewards_per_episode.append(train_reward)
		times_per_episode.append(end_time - start_time)
		simulated_times_per_episode.append(step_count * 0.008)  # <- Add this for sim time


		# -------------------------------------------------------- #
		# Update recent reward buffer
		recent_training_rewards_window.append(train_reward)
		if len(recent_training_rewards_window) > window_size:
			recent_training_rewards_window.pop(0)
		# -------------------------------------------------------- #


		# -------------------------------------------------------------- #
		# Compute smoothed reward and variance only when window is full
		if len(recent_training_rewards_window) == window_size:
			mean_reward = np.mean(recent_training_rewards_window)
			var_reward = np.var(recent_training_rewards_window)

			smoothed_training_rewards.append(mean_reward)
			training_reward_variance_window.append(var_reward)
		# -------------------------------------------------------------- #

		if (episode + 1) % args.print_every == 0:
			print(f"--- Episode {episode+1}/{args.n_episodes} ---")
			print(f"  Reward: {train_reward:.2f} | Smoothed: {mean_reward:.2f} | Variance: {var_reward:.2f}")
			print(f"  Loss: {loss:.2f} | Time: {end_time - start_time:.2f}s")
			print("-" * 40)


	# === Save logs ===
	np.save(f"{log_dir}/mu_log.npy", np.array(agent.mu_log))
	np.save(f"{log_dir}/sigma_log.npy", np.array(agent.sigma_log))
	np.save(f"{log_dir}/actions_log.npy", np.array(agent.actions_log))
	np.save(f"{log_dir}/entropy_log.npy", np.array(agent.entropy_log))

	np.save(f"{log_dir}/discounted_returns_mean_log.npy", np.array(agent.discounted_returns_mean_log))
	np.save(f"{log_dir}/discounted_returns_std_log.npy", np.array(agent.discounted_returns_std_log))
	np.save(f"{log_dir}/discounted_returns_variance_log.npy", np.array(agent.discounted_returns_variance_log))

	np.save(f"{log_dir}/advantages_mean_log.npy", np.array(agent.advantages_mean_log))
	np.save(f"{log_dir}/advantages_std_log.npy", np.array(agent.advantages_std_log))
	np.save(f"{log_dir}/advantages_variance_log.npy", np.array(agent.advantages_variance_log))

	np.save(f"{analysis_dir}/episode_times.npy", np.array(times_per_episode))
	np.save(f"{analysis_dir}/episode_times_simulated.npy", np.array(simulated_times_per_episode))
	np.save(f"{analysis_dir}/episode_rewards.npy", np.array(training_rewards_per_episode))
	np.save(f"{analysis_dir}/losses.npy", np.array(losses_per_episode))
	np.save(f"{analysis_dir}/episode_rewards_smoothed_100.npy", np.array(smoothed_training_rewards))
	np.save(f"{analysis_dir}/episode_rewards_variance_100.npy", np.array(training_reward_variance_window))

	# === Save model ===
	torch.save(agent.policy.state_dict(), model_path)

if __name__ == '__main__':
	main()
