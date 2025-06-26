import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb

EPS = 2e-15

class AutomaticDomainRandomization():
    def __init__(self, init_params: dict, p_b=0.5, m=20, thresholds=[1000, 1400]) -> None:
        self.init_params = init_params
        self.thresholds = thresholds
        self.m = m
        # === FIX 1: Store the fixed torso mass separately ===
        self.fixed_torso_mass = self.init_params.pop('torso')
        self.bounds = self._init_bounds()
        self.p_b = p_b
        self.thigh_mass = None
        self.leg_mass = None
        self.foot_mass = None
        self.rewards = []
        self.partOfBody = ['thigh', 'leg', 'foot']
        self.buffer = {
            "thigh_low": [], "thigh_high": [],
            "leg_low": [], "leg_high": [],
            "foot_low": [], "foot_high": []
        }
        self.keys = list(self.buffer.keys())
        self.performances = {k: [] for k in self.buffer.keys()}
        self.increments = []
        self.delta_choices = [0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.12]

    def _init_bounds(self):
        return {
            "thigh_low": self.init_params['thigh'] - EPS,
            "thigh_high": self.init_params['thigh'] + EPS,
            "leg_low": self.init_params['leg'] - EPS,
            "leg_high": self.init_params['leg'] + EPS,
            "foot_low": self.init_params['foot'] - EPS,
            "foot_high": self.init_params['foot'] + EPS
        }

    def compute_entropy(self):
        ranges = [abs(self.bounds[f'{p}_high'] - self.bounds[f'{p}_low']) for p in self.partOfBody]
        return np.log(ranges).mean()

    def append_reward_bodypart(self, partOfBody: str, reward: float):
        self.buffer[partOfBody].append(reward)

    def evaluate_performance(self, partOfBody: str):
        perf = np.mean(self.buffer[partOfBody])
        self.buffer[partOfBody].clear()
        return perf

    def get_adaptive_delta(self, partOfBody: str):
        performances = self.performances[partOfBody]
        if len(performances) < 3:
            return 0.01
        recent = performances[-1]
        if recent <= 800:
            delta_range = self.delta_choices[0:2]  # very poor
        elif recent <= 1000:
            delta_range = self.delta_choices[1:3]  # poor
        elif recent <= 1200:
            delta_range = self.delta_choices[2:4]  # moderate
        elif recent <= 1400:
            delta_range = self.delta_choices[3:5]  # good
        elif recent <= 1600:
            delta_range = self.delta_choices[4:6]  # great
        else:
            delta_range = self.delta_choices[5:7]  # ultra great
        delta = np.random.choice(delta_range)
        delta += np.clip(np.random.normal(0, 0.0001), -0.0002, 0.0002)
        return max(0.001, delta)

    def update_ADR(self, partOfBody: str):
        if len(self.buffer[partOfBody]) >= self.m:
            highOrLow = partOfBody.split('_')[1]
            performance = self.evaluate_performance(partOfBody)
            self.performances[partOfBody].append(performance)
            delta = self.get_adaptive_delta(partOfBody)
            if performance >= self.thresholds[1]:
                if highOrLow == "high":
                    self.increase(partOfBody, delta)
                else:
                    self.decrease(partOfBody, delta)
            elif performance <= self.thresholds[0]:
                if highOrLow == "low":
                    self.increase(partOfBody, delta)
                else:
                    self.decrease(partOfBody, delta)
            wandb.log({"bound": self.get_bounds()})

    def get_bounds(self):
        return self.bounds

    def random_masses(self):
        thighVal = np.random.uniform(self.bounds["thigh_low"], self.bounds["thigh_high"])
        footVal = np.random.uniform(self.bounds["foot_low"], self.bounds["foot_high"])
        legVal = np.random.uniform(self.bounds["leg_low"], self.bounds["leg_high"])
        bodyParts = {"thigh": thighVal, "foot": footVal, "leg": legVal}
        pb = np.random.uniform(0, 1)
        randomCompletePart = self.set_random_parameter(bodyParts)
        part = randomCompletePart + ("_low" if pb < self.p_b else "_high")
        bodyParts[randomCompletePart] = self.bounds[part]

        full_masses = [
            self.fixed_torso_mass, 
            bodyParts["thigh"], 
            bodyParts["leg"], 
            bodyParts["foot"]
        ]

        return full_masses, part

    def evaluate(self, reward, randomCompletePart):
        self.append_reward_bodypart(randomCompletePart, reward)
        self.update_ADR(randomCompletePart)

    def increase(self, partOfBody: str, delta: float):
        part = partOfBody.split('_')[0]
        high_bound = f"{part}_high"
        highOrLow = partOfBody.split('_')[1]
        new_bound = self.bounds[partOfBody] + delta
        if highOrLow == 'low' and new_bound > self.bounds[high_bound]:
            new_bound = self.bounds[partOfBody]
        self.bounds[partOfBody] = new_bound

    def decrease(self, partOfBody: str, delta: float):
        part = partOfBody.split('_')[0]
        low_bound = f"{part}_low"
        highOrLow = partOfBody.split('_')[1]
        new_bound = self.bounds[partOfBody] - delta
        if highOrLow == 'high' and new_bound < self.bounds[low_bound]:
            new_bound = self.bounds[partOfBody]
        if new_bound <= 0.0:
            new_bound = self.bounds[partOfBody]
        self.bounds[partOfBody] = new_bound

    def set_random_parameter(self, bodyParts):
        return np.random.choice(list(bodyParts.keys()))


class ADRCallback(BaseCallback):
    def __init__(self, handlerADR: AutomaticDomainRandomization, vec_env, eval_callback,
                 verbose=0, save_freq=1000, save_path: str = './models', name_prefix: str = 'adr_model'):
        super(ADRCallback, self).__init__(verbose)
        self.adr = handlerADR
        self.vec_env = vec_env  
        self.eval_callback = eval_callback
        self.bound_used = None
        self.n_episodes = 0
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            wandb.log({
                "step": self.num_timesteps,
                "entropy": self.adr.compute_entropy()
            })

            if done:
                self.n_episodes += 1
                reward = info['episode']['r']
                wandb.log({
                    "reward": reward,
                    "step": self.num_timesteps
                })

                # === ADR Evaluation and Bound Updates ===
                if self.bound_used is not None:
                    self.adr.evaluate(reward, self.bound_used)

                # === Sample New Random Masses + Update Env ===
                env_params, self.bound_used = self.adr.random_masses()
                self.vec_env.env_method('set_parameters', env_params, indices=0)

                # === Log All Bounds ===
                wandb.log({
                    "thigh_low": self.adr.bounds["thigh_low"],
                    "thigh_high": self.adr.bounds["thigh_high"],
                    "leg_low": self.adr.bounds["leg_low"],
                    "leg_high": self.adr.bounds["leg_high"],
                    "foot_low": self.adr.bounds["foot_low"],
                    "foot_high": self.adr.bounds["foot_high"],
                    "thigh_range": self.adr.bounds["thigh_high"] - self.adr.bounds["thigh_low"],
                    "leg_range": self.adr.bounds["leg_high"] - self.adr.bounds["leg_low"],
                    "foot_range": self.adr.bounds["foot_high"] - self.adr.bounds["foot_low"]
                }, step=self.num_timesteps)

        return True
