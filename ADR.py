# ADR.py

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb

EPS = 1e-8 # Use a small epsilon to avoid zero-width or negative distributions

class AutomaticDomainRandomization():
    """
    Implements the Automatic Domain Randomization algorithm.
    It dynamically adjusts the distribution of physics parameters (masses)
    of the environment based on the agent's performance.
    """
    def __init__(self, init_params: dict, p_b=0.5, m=50, delta=0.05, thresholds=[1000, 1600]):
        self.init_params = init_params
        self.thresholds = thresholds
        self.delta = delta
        self.m = m  # Data buffer size for performance evaluation
        self.p_b = p_b # Probability of sampling a boundary value
        self.bounds = self._init_bounds()
        self.partOfBody = ['thigh', 'leg', 'foot']
        
        # Buffers to store rewards for boundary values
        self.buffer = {f"{part}_{bound}": [] for part in self.partOfBody for bound in ["low", "high"]}
        
        # Store historical performance for analysis
        self.performances = {key: [] for key in self.buffer.keys()}

    def _init_bounds(self):
        """Initializes randomization bounds as a narrow distribution around initial parameters."""
        return {
            "thigh_low": self.init_params['thigh'] - EPS, "thigh_high": self.init_params['thigh'] + EPS,
            "leg_low": self.init_params['leg'] - EPS, "leg_high": self.init_params['leg'] + EPS,
            "foot_low": self.init_params['foot'] - EPS, "foot_high": self.init_params['foot'] + EPS
        }

    def compute_entropy(self):
        """Computes the entropy of the current randomization distribution."""
        ranges = [
            self.bounds['thigh_high'] - self.bounds['thigh_low'],
            self.bounds['leg_high'] - self.bounds['leg_low'],
            self.bounds['foot_high'] - self.bounds['foot_low']
        ]
        entropy = np.log(np.array(ranges) + EPS).mean()
        return entropy
    
    def update_ADR(self, partOfBody: str, current_timestep: int):
        """Updates ADR bounds if the buffer for a given boundary is full."""
        if len(self.buffer[partOfBody]) >= self.m:
            body_part, high_or_low = partOfBody.split('_')
            
            performance = np.mean(self.buffer[partOfBody])
            self.buffer[partOfBody].clear()
            self.performances[partOfBody].append(performance)
            wandb.log({f"adr_performance/{partOfBody}": performance}, step=current_timestep)

            if performance >= self.thresholds[1]:  # High performance -> expand
                if high_or_low == "high": self.increase(partOfBody)
                else: self.decrease(partOfBody)
            elif performance <= self.thresholds[0]:  # Low performance -> contract
                if high_or_low == "low": self.increase(partOfBody)
                else: self.decrease(partOfBody)
        
        wandb.log({f"bounds/{k}": v for k, v in self.bounds.items()}, step=current_timestep)
            
    def get_bounds(self):
        return self.bounds
            
    def random_masses(self):
        """Samples new masses, replacing one with a boundary value for evaluation."""
        thigh_val = np.random.uniform(self.bounds["thigh_low"], self.bounds["thigh_high"])
        leg_val = np.random.uniform(self.bounds["leg_low"], self.bounds["leg_high"])
        foot_val = np.random.uniform(self.bounds["foot_low"], self.bounds["foot_high"])
        body_parts_values = {"thigh": thigh_val, "leg": leg_val, "foot": foot_val}

        random_part = np.random.choice(self.partOfBody)
        boundary_to_test = f"{random_part}_{'low' if np.random.rand() < self.p_b else 'high'}"
        
        body_parts_values[random_part] = self.bounds[boundary_to_test]
        return list(body_parts_values.values()), boundary_to_test

    def evaluate(self, reward: float, boundary_tested: str, current_timestep: int):
        """Stores reward for the tested boundary and triggers an update."""
        self.buffer[boundary_tested].append(reward)
        self.update_ADR(partOfBody=boundary_tested, current_timestep=current_timestep)
             
    def increase(self, partOfBody: str):
        self.bounds[partOfBody] += self.delta
        part_type, _ = partOfBody.split('_')
        low_bound_key, high_bound_key = f"{part_type}_low", f"{part_type}_high"
        if self.bounds[low_bound_key] > self.bounds[high_bound_key]:
            self.bounds[low_bound_key] = self.bounds[high_bound_key]

    def decrease(self, partOfBody: str):
        self.bounds[partOfBody] -= self.delta
        part_type, _ = partOfBody.split('_')
        low_bound_key, high_bound_key = f"{part_type}_low", f"{part_type}_high"
        self.bounds[partOfBody] = max(self.bounds[partOfBody], EPS)
        if self.bounds[low_bound_key] > self.bounds[high_bound_key]:
            self.bounds[high_bound_key] = self.bounds[low_bound_key]

class ADRCallback(BaseCallback):
    """Callback to orchestrate Automatic Domain Randomization during training."""
    def __init__(self, handlerADR: AutomaticDomainRandomization, verbose=0):
        super(ADRCallback, self).__init__(verbose)
        self.adr = handlerADR
        self.bounds_used = [None] * self.training_env.num_envs

    def _on_step(self):
        wandb.log({"adr_entropy": self.adr.compute_entropy()}, step=self.num_timesteps)
        
        for i in range(self.training_env.num_envs):
            if self.locals['dones'][i]:
                info = self.locals['infos'][i]
                boundary_tested = self.bounds_used[i]
                if boundary_tested is not None:
                    episode_reward = info['episode']['r']
                    self.adr.evaluate(episode_reward, boundary_tested, self.num_timesteps)
                
                env_params, self.bounds_used[i] = self.adr.random_masses()
                self.training_env.env_method('set_parameters', env_params, indices=[i])
        return True