
"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import wandb
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--timesteps', default=int(500000), type=int, help='Timesteps ')
	parser.add_argument('--save-freq', default=int(50000), type=int, help='Frequency of saving the model')
	parser.add_argument('--best-model-path', default='./BestModelTuning/model', type=str , help='Path for the best model found so far')
	return parser.parse_args()

args = parse_args()

sweep_configuration = {
         "method": "random",
         "name": "sweep",
         "metric":
             {"name": "mean_reward", "goal": "maximize"},
         "parameters": {
             "learning_rate": {"min": 5e-4, "max": 1e-3},
             "clip_range": {"min": 0.25, "max": 0.35},
             "entropy_coefficient": {"min": 0.005, "max": 0.02}
         }
}
class WandCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(WandCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Check if we are at the start of a new episode
        done_array = self.locals["dones"]
        if any(done_array):
            wandb.log({"reward": self.locals['infos'][0]['episode']['r'], "step": self.num_timesteps} )
        return True
    
global countt
countt={'co': 0}
global best_params
best_params={
     "source": {'learning_rate':0.0 ,'clip_range':0.0,'entropy_coefficient':0.0, 'best_std': 0.0, 'best_mean' : -float('inf'), "run_id_suffix": ""},
    "target": {'learning_rate':0.0 ,'clip_range':0.0,'entropy_coefficient':0.0, 'best_std': 0.0, 'best_mean' : -float('inf'), "run_id_suffix": ""}
}

def objective(envname):
    wandb.init(project="F_MldlRLproject_TuningPPO_Source_Target")
    countt["co"]=countt["co"] +1

    #sampling from intervals
    learning_rate = wandb.config.learning_rate
    entropy_coeff = wandb.config.entropy_coefficient
    clip_range = wandb.config.clip_range

    train_env = gym.make(envname)
    train_env = Monitor(train_env)
    wand_callback = WandCallback(envname)
    eval_callback = EvalCallback(eval_env = train_env, n_eval_episodes=50, eval_freq=args.save_freq, best_model_save_path=args.best_model_path+str(countt["co"])+"_"+envname+"/", verbose = 0)
    callback = CallbackList([eval_callback, wand_callback])
    model = PPO("MlpPolicy", train_env, learning_rate =learning_rate,
                                         ent_coef = entropy_coeff, 
                                         clip_range = clip_range, 
                                         verbose=0)
    model.learn(total_timesteps = args.timesteps, progress_bar=True, callback=callback)
    #Load the best model found by the EvalCallback
    model = PPO.load(args.best_model_path+str(countt["co"])+"_"+envname+"/best_model.zip")

    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=50)
    wandb.log({"mean_reward": mean_reward, "std_reward":std_reward})
    if envname == "CustomHopper-source-v0":
        current_env_data = best_params["source"]
    else: # envname == "CustomHopper-target-v0"
        current_env_data = best_params["target"]

    if mean_reward > current_env_data["best_mean"]:
        current_env_data["learning_rate"] = learning_rate # استفاده از local vars
        current_env_data["clip_range"] = clip_range
        current_env_data["entropy_coefficient"] = entropy_coeff
        current_env_data["best_mean"] = mean_reward
        current_env_data["best_std"] = std_reward
        current_env_data["run_id_suffix"] = str(countt["co"]) + "_" + envname + "/" # ذخیره بخش ساب‌فیکس مسیر

    wandb.finish()
    return mean_reward, std_reward

sweep_id = wandb.sweep(sweep=sweep_configuration, project="F_MldlRLproject_TuningPPO_Source_Target")
wandb.agent(sweep_id, function=lambda: objective("CustomHopper-source-v0"), count=10)
print("Best params [source]: ",best_params)


wandb.agent(sweep_id, function=lambda: objective("CustomHopper-target-v0"), count=10)
print("Best params [target]: ",best_params)

wandb.finish()

def main():
    envname = "CustomHopper-source-v0"
    test_env = gym.make(envname)
    test_env = Monitor(test_env)
    model = PPO.load(args.best_model_path+best_params["countsource"]+"_"+envname+"/best_model.zip")
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, render=False, warn=False)
    print(f" [s-s] mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}") #source -> source
    
    envname = "CustomHopper-target-v0"
    test_env = gym.make(envname)
    test_env = Monitor(test_env)
    model = PPO.load(args.best_model_path+best_params["countsource"]+"_"+"CustomHopper-source-v0"+"/best_model.zip")
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, render=False, warn=False)
    print(f" [s-t]mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}") #source -> target

    test_env = gym.make(envname)
    test_env = Monitor(test_env)
    model = PPO.load(args.best_model_path+best_params["counttarget"]+"_"+envname+"+/best_model.zip")
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, render=False, warn=False)
    print(f" [t-t]mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}") #target -> target

if __name__ == '__main__':
    main()
