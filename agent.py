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

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

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

        # Ensure sigma is in a reasonable range
        sigma = self.sigma_activation(self.sigma) + 1e-6
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        # TASK 3: forward in the critic network

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)
        
        return normal_dist, state_value

class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.mu_log = []         # Stores [μ1, μ2, μ3] per episode
        self.sigma_log = []      # Stores [σ1, σ2, σ3] per episode
        self.actions_log = []    # Stores [a1, a2, a3] per step
        self.entropy_log = []    # Stores entropy of the action distribution per step

        self.td_target_log = []
        self.td_target_variance_log = []
        self.td_target_mean_log = []  # Log mean of returns
        self.td_target_std_log = []   # Log std of returns

        self.advantages_log = []
        self.advantages_variance_log = []
        self.advantages_mean_log = [] # Log mean of advantages
        self.advantages_std_log = []  # Log std of advantages

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # === Get state-values from critic ===
        with torch.no_grad():
            _, next_state_values = self.policy(next_states)
            
        _,state_values = self.policy(states)

        state_values = state_values.detach()
        next_state_values = next_state_values.detach()

        # Compute TD targets
        td_target = rewards + self.gamma * next_state_values * (1 - done) - state_values
        # === Compute Advantage: A(s) = G - V(s) ===
         # 🌟 محاسبه Generalised Advantage Estimation (GAE) 🌟
        advantages = torch.zeros_like(td_target)
        gae = 0.0 # مقداردهی اولیه به GAE (این برای انباشت استفاده می‌شود)
         # 🔄 حلقه معکوس برای محاسبه GAE در طول اپیزود 🔄
        for t in reversed(range(len(td_target))):
            # اگر اپیزود در زمان t به پایان رسیده باشد، gae برای گام بعدی صفر می‌شود.
            gae = td_target[t] + self.gamma * self.lambda_gae * gae * (1 - done[t])
            advantages[t] = gae

        advantages_mean = advantages.mean().item()
        advantages_std = advantages.std().item()

        td_target_mean = td_target.mean().item()
        td_target_std = td_target.std().item()

        # Log mean and std of unnormalized advantages
        self.td_target_log.append(td_target.detach().cpu().numpy())
        self.td_target_mean_log.append(td_target_mean)
        self.td_target_std_log.append(td_target_std)
        self.td_target_variance_log.append(np.var(td_target.detach().cpu().numpy()))

        self.advantages_log.append(advantages.detach().cpu().numpy())
        self.advantages_mean_log.append(advantages_mean)
        self.advantages_std_log.append(advantages_std)
        # Calculate and log variance of unnormalized advantages (after baseline)
        if len(advantages) > 1: # Ensure there's enough data to calculate variance
            advantages_variance = torch.var(advantages).item()  
        else:
            advantages_variance = 0.0 # Append 0 if variance cannot be computed
        self.advantages_variance_log.append(advantages_variance)  

       # 3. Normalize Advantages (A_t_normalized = (A_t - mean(A)) / std(A))
        # This is the "whitening" step for advantages.
        #if advantages.std() > 1e-6:  # Avoid division by near-zero std
           # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #else:
             #If std is too small, just subtract the mean (centering)
            #advantages = advantages - advantages.mean()
        # === Actor loss ===
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Critic loss (regression: V(s) ≈ TD target)
        critic_loss = F.mse_loss(state_values, td_target.detach())

        # === Total loss: actor + critic ===
        # Note: We can also use a weighted sum if we want to balance actor and critic losses 
        # === Total loss: actor + 0.5 * critic ===
        loss = actor_loss +  critic_loss 
        #   - compute gradients and step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer

        return loss.item(), advantages_variance ,advantages_mean# Return the loss for monitoring   

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()
            
            action_log_prob = normal_dist.log_prob(action).sum()

            # Entropy of the policy
            entropy = normal_dist.entropy().sum().item()
            self.entropy_log.append(entropy)

            # Logging for monitoring Mean (μ) and Standard Deviation (σ) and actions (a)
            mu = normal_dist.mean.detach().cpu().numpy()
            sigma = normal_dist.stddev.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()

            # Monitor μ and σ only on first state per episode (optional via a flag)
            if len(self.states) == 0:  # first step of the episode
                if np.any(np.isnan(mu)) or np.any(np.isinf(mu)) or np.any(np.abs(mu) > 100) or np.any(sigma > 5)or np.any(sigma < 0.3):
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
   