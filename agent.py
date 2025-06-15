import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        
        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.b = 20

        self.mu_log = []     # Stores [μ1, μ2, μ3] per episode
        self.sigma_log = []  # Stores [σ1, σ2, σ3] per episode
        self.entropy_log = [] # Stores entropy of the action distribution per step
        self.actions_log = []
        
        self.returns_log = []  # برای ذخیره همه‌ی advantageها
        self.returns_mean_log = [] # Log mean of advantages
        self.returns_std_log = []  # Log std of advantages
        self.returns_variance_log = [] # Stores variance of advantage terms
        

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # TASK 2:
        #   - compute discounted returns
          # 1. Compute discounted returns (G_t)
        returns = discount_rewards(rewards, gamma=self.gamma)
    
        # 2. Compute Advantages (A_t = G_t - b) using the constant baseline
        # You correctly had `returns = returns - self.b` here, which computes the advantage.
        # Let's explicitly call it advantages to be clear.
        returns = returns - self.b
        self.returns_log.append(returns.detach().cpu().numpy().tolist())  # ذخیره‌ی همه مقادیر

        # Log mean and std of unnormalized advantages
        self.returns_mean_log.append(returns.mean().item())
        self.returns_std_log.append(returns.std().item())

        # Calculate and log variance of unnormalized advantages (after baseline)
        if len(returns) > 1: # Ensure there's enough data to calculate variance
            returns_variance = torch.var(returns).item()
            
        else:
            returns_variance=0.0 # Append 0 if variance cannot be computed
            
       # 3. Normalize Advantages (A_t_normalized = (A_t - mean(A)) / std(A))
        # This is the "whitening" step for advantages.
        if returns.std() > 1e-6:  # Avoid division by near-zero std
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
             #If std is too small, just subtract the mean (centering)
            returns = returns - returns.mean()
       
        #   - compute policy gradient loss function given actions and returns
        loss = -(action_log_probs * returns).mean()

        #   - compute gradients and step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      

        return loss.item(), returns_variance# Return the loss for monitoring   


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

        
              # 🧠 Entropy of the policy
            entropy = normal_dist.entropy().sum().item()
            self.entropy_log.append(entropy)


            # Logging for monitoring Mean (μ) and Standard Deviation (σ) and actions (a)
            mu = normal_dist.mean.detach().cpu().numpy()
            sigma = normal_dist.stddev.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()

        
             # ⚠️ Monitor μ and σ only on first state per episode (optional via a flag)
            if len(self.states) == 0:  # first step of the episode
                if np.any(np.isnan(mu)) or np.any(np.isinf(mu)) or np.any(np.abs(mu) > 100) or np.any(sigma > 5):
                    print(f"[WARNING] Unusual μ or σ -> μ: {mu}, σ: {sigma}")
                self.mu_log.append(mu.tolist())
                self.sigma_log.append(sigma.tolist())
            
               # Save all sampled actions
            self.actions_log.append(action_np.tolist())  # save full [a1, a2, a3] list

            
            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)