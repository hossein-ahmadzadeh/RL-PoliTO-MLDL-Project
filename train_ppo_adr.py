from env.custom_hopper import *
from ADR import AutomaticDomainRandomization, ADRCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
wandb.login()
import gym
import argparse
import os

# === Best Params You Provided ===
BEST_PARAMS = {
    "learning_rate": 0.0007081485369506237,
    "clip_range": 0.26785576292805646,
    "ent_coef": 0.01776300171237951,
    "n_steps": 2048,
    "batch_size": 128,
    "gamma": 0.987060729890745,
    "gae_lambda": 0.9779077684516388,
}

THIGH_MEAN_MASS = 3.92699082
LEG_MEAN_MASS = 2.71433605
FOOT_MEAN_MASS = 5.0893801

sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric":
            {"name": "mean_reward", "goal": "maximize"},
        "parameters": {
            "delta":{"values": [0.02, 0.05 ,0.1]},
        }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p-b', default=0.50, type=float, help='Probability of boundary testing')
    parser.add_argument('--m', default=50, type=int, help='Reward buffer size per boundary')
    parser.add_argument('--low-th', default=1000, type=int, help='Lower reward threshold')
    parser.add_argument('--high-th', default=1500, type=int, help='Upper reward threshold')
    parser.add_argument('--n-envs', default=os.cpu_count(), type=int, help='Number of parallel environments')
    parser.add_argument('--timesteps', default=3_000_000, type=int, help='Total training timesteps')
    parser.add_argument('--save-path', default='./models_ADR_Final/', type=str, help='Where to save intermediate models')
    parser.add_argument('--save-freq', default=50_000, type=int, help='How often to save models')
    parser.add_argument('--best-model-path', default='./best_models/', type=str, help='Best model output directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Training device')
    return parser.parse_args()

args = parse_args()

global adr_callback
global best_params
best_params={'best_std': 0, 'best_mean' : 0, "best_delta":0}

def objective(envname):
	wandb.init(project="MldlRLproject_Tuning_ADR")
	delta = wandb.config.delta
	timesteps = args.timesteps


	init_params = {"thigh": THIGH_MEAN_MASS,  "leg": LEG_MEAN_MASS, "foot": FOOT_MEAN_MASS}

	handlerADR = AutomaticDomainRandomization(init_params, p_b=args.p_b, m=args.m, delta=delta, thresholds=[args.low_th, args.high_th])
		
	train_env = make_vec_env('CustomHopper-adr-v0', n_envs=args.n_envs, vec_env_cls=DummyVecEnv)
	train_env.set_attr(attr_name="bounds", value=handlerADR.get_bounds())
		
	test_env = gym.make('CustomHopper-adr-v0')
	eval_callback = EvalCallback(eval_env=test_env, n_eval_episodes=50, eval_freq = args.save_freq, deterministic=True, render=False, best_model_save_path=args.best_model_path+"best_eval_ADR"+str(delta)+"/", warn=False) 
	adr_callback = ADRCallback(handlerADR, train_env, eval_callback, n_envs=args.n_envs, verbose=0, save_freq=args.save_freq, save_path=args.save_path)
	callbacks = CallbackList([adr_callback, eval_callback]) 	
	
	model = PPO(
        'MlpPolicy',
        train_env,
        verbose=0,
        **BEST_PARAMS
    )
	
	model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
	mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=50, render=False)

	if mean_reward > best_params['best_mean']:
				best_params['best_mean'] = mean_reward
				best_params['best_std'] = std_reward
				best_params['best_delta'] = delta

	wandb.log({"mean_reward": mean_reward, "std_reward":std_reward})
	return mean_reward, std_reward

sweep_id = wandb.sweep(sweep=sweep_configuration, project="MldlRLproject_Tuning_ADR")
wandb.agent(sweep_id, function=lambda:objective("CustomHopper-adr-v0"))
print("Best distributions [source]: ",best_params)
wandb.finish()

def main():

    for delta in ["0.02", "0.05", "0.1"]:
        test_env = gym.make('CustomHopper-target-v0')
        test_env = Monitor(test_env)
        model = PPO.load("./best_eval_ADR"+delta+"/best_model")
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, render=False, warn=False)
        print(f"[s-t] mean_reward ADR delta ="+delta+":{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
	main()