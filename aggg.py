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
    def init(self, state_space, action_space):
        super().init()
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

    # Initialize weights of the network with a normal distribution, set biases to zero
    def init_weights(self):
         for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    # Returns a multivariate normal distribution over the action space
    def forward(self, x):
            x_actor = self.tanh(self.fc1_actor(x))
            x_actor = self.tanh(self.fc2_actor(x_actor))
            action_mean = self.fc3_actor_mean(x_actor)

            sigma = self.sigma_activation(self.sigma)
            normal_dist = Normal(action_mean, sigma)

            return normal_dist

        """
              Critic network
        """
        # TASK 3: critic network for actor-critic algorithm

class Critic(torch.nn.Module):
        def init(self, state_space):
            super().init()
            self.state_space = state_space
            self.hidden = 64
            self.tanh = torch.nn.Tanh()

            # critic network
            self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
            self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
            self.fc3_critic = torch.nn.Linear(self.hidden, 1)

            self.init_weights()

        # Initialize weights of the network with a normal distribution, set biases to zero
        def init_weights(self):
            for m in self.modules():
                if type(m) is torch.nn.Linear:
                    torch.nn.init.normal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

        def forward(self, x):
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))
            value_estimate = self.fc3_critic(x_critic)
            return value_estimate.T

class Agent(object):
   def init(self, policy, critic, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = critic.to(self.train_device)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.b = 20 

        self.mu_log = []     # Stores [μ1, μ2, μ3] per episode
        self.sigma_log = []  # Stores [σ1, σ2, σ3] per episode
        self.actions_log = [] # Stores [a1, a2, a3] per step
        self.entropy_log = [] # Stores entropy of the action distribution per step
       
        self.advantages_log = []
        self.advantages_variance_log = [] # Stores variance of advantage terms
        self.advantages_mean_log = [] # Log mean of advantages
        self.advantages_std_log = []  # Log std of advantagesself.td_target_log = []
        self.td_target_variance_log = []
        self.td_target_mean_log = []  # Log mean of returns
        self.td_target_std_log = []   # Log std of returns



   def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []
       
         # === Get state-values from critic ===
        with torch.no_grad():
            _, next_state_values = self.critic(next_states).detach()

        _, state_values = self.critic(states)

        # Compute TD targets
        td_target = rewards + self.gamma * next_state_values * (1 - done)
        
        self.td_target_log.append(td_target.detach().cpu().numpy())
        self.td_target_variance_log.append(np.var(td_target.detach().cpu().numpy()))
        self.td_target_mean_log.append(td_target.mean().item())
        self.td_target_std_log.append(td_target.std().item())

        # === Compute Advantage: A(s) = G - V(s) ===
        advantages = td_target - state_values

        # Log mean and std of unnormalized advantages
        self.advantages_log.append(advantages.detach().cpu().numpy())
        self.advantages_mean_log.append(advantages.mean().item())
        self.advantages_std_log.append(advantages.std().item())
        
        # Calculate and log variance of unnormalized advantages (after baseline)
        if len(advantages) > 1: # Ensure there's enough data to calculate variance
            advantages_variance = torch.var(advantages).item()
            self.advantages_variance_log.append(advantages_variance)
        else:
            self.advantages_variance_log.append(0.0) # Append 0 if variance cannot be computed
            
       # 3. Normalize Advantages (A_t_normalized = (A_t - mean(A)) / std(A))
        # This is the "whitening" step for advantages.
        if advantages.std() > 1e-6:  # Avoid division by near-zero std
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # If std is too small, just subtract the mean (centering)
            advantages = advantages - advantages.mean()
       
         # === Actor loss ===
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Critic loss (regression: V(s) ≈ TD target)
        critic_loss = F.mse_loss(state_values, td_target.detach())

        # === Total loss: actor + critic ===
        # Note: We can also use a weighted sum if we want to balance actor and critic losses 
        # === Total loss: actor + 0.5 * critic ===
        loss = actor_loss  * critic_loss

        #   - compute gradients and step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer

        return loss.item() # Return the loss for monitoring   


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