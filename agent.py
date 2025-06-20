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

        sigma = self.sigma_activation(self.sigma) 
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
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
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.mu_log = []         # Stores [μ1, μ2, μ3] per episode
        self.sigma_log = []      # Stores [σ1, σ2, σ3] per episode
        self.entropy_log = []    # Stores entropy of the action distribution per step

        self.td_target_variance_log = []
        self.td_target_mean_log = []
        self.td_target_std_log = [] 

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

        _, state_values = self.policy(states)


        # Compute TD targets
        td_target = rewards + self.gamma * next_state_values * (1 - done)
        

        # === Compute Advantage: A(s) = G - V(s) ===
        advantages = td_target - state_values


        advantages_mean = advantages.mean().item()
        advantages_std = advantages.std().item()


        self.td_target_variance_log.append(np.var(td_target.detach().cpu().numpy()))
        self.td_target_mean_log.append(td_target.mean().item())
        self.td_target_std_log.append(td_target.std().item())

        self.advantages_variance_log.append(np.var(advantages.detach().cpu().numpy()))
        self.advantages_mean_log.append(advantages_mean)
        self.advantages_std_log.append(advantages_std)


        if advantages_std > 1e-6:  # Avoid division by near-zero std
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        else:
            advantages = advantages - advantages_mean  # Only subtract mean if std is too small
        

        # === Actor loss ===
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Critic loss (regression: V(s) ≈ TD target)
        critic_loss = F.mse_loss(state_values, td_target.detach())

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item() # Return the loss for monitoring   


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            # -----------------------NO SQUASH---------------------- #
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            # ------------------------------------------------------ #


            entropy = normal_dist.entropy().sum().item()
            self.entropy_log.append(entropy)


            # Logging for monitoring Mean (μ) and Standard Deviation (σ) and actions (a)
            mu = normal_dist.mean.detach().cpu().numpy()
            sigma = normal_dist.stddev.detach().cpu().numpy()

            if len(self.states) == 0:  # first step of the episode
                self.mu_log.append(mu.tolist())
                self.sigma_log.append(sigma.tolist())

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)