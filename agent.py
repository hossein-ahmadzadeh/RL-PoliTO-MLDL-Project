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

        sigma = self.sigma_activation(self.sigma)
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
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        # self.b = 2 # Constant baseline for advantage calculation

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
            _, state_values = self.policy(states)
            _, next_state_values = self.policy(next_states)


        # Compute TD targets
        td_target = rewards + self.gamma * next_state_values * (1 - done)
        

        # === Compute Advantage: A(s) = G - V(s) ===
        advantages = td_target - state_values
        advantages_mean = advantages.mean().item()
        advantages_std = advantages.std().item()


        if advantages_std > 1e-6:  # Avoid division by near-zero std
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        else:
            advantages = advantages - advantages_mean  # Only subtract mean if std is too small


        # === Logging for analysis ===
        self.td_target_log.append(td_target.detach().cpu().numpy())
        self.td_target_variance_log.append(np.var(td_target.cpu().numpy()))
        self.td_target_mean_log.append(td_target.mean().item())
        self.td_target_std_log.append(td_target.std().item())

        self.advantages_log.append(advantages.detach().cpu().numpy())
        self.advantages_variance_log.append(np.var(advantages.cpu().numpy()))
        self.advantages_mean_log.append(advantages_mean)
        self.advantages_std_log.append(advantages_std)
        

        # === Actor loss ===
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Critic loss (regression: V(s) ≈ TD target)
        critic_loss = F.mse_loss(state_values, td_target.detach())

        # === Total loss: actor + critic ===
        # Note: We can also use a weighted sum if we want to balance actor and critic losses 
        # === Total loss: actor + 0.5 * critic ===
        loss = actor_loss + 0.5 * critic_loss

        #   - compute gradients and step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        return loss.item() # Return the loss for monitoring   


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            u = normal_dist.sample()
            action = torch.tanh(u)  # squash into [-1, 1]

            log_prob = normal_dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
            action_log_prob = log_prob.sum()


            # 🧠 Entropy of the policy
            entropy = normal_dist.entropy().sum().item()
            self.entropy_log.append(entropy)


            # Logging for monitoring Mean (μ) and Standard Deviation (σ) and actions (a)
            mu = normal_dist.mean.detach().cpu().numpy()
            sigma = normal_dist.stddev.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()


            # ⚠️ Monitor μ and σ only on first state per episode (optional via a flag)
            if len(self.states) == 0:  # first step of the episode
                if np.any(np.isnan(mu)) or np.any(np.isinf(mu)) or np.any(np.abs(mu) > 100) or np.any(sigma > 5)or np.any(sigma < 0.3):
                    print(f"[WARNING] Unusual μ or σ -> μ: {mu}, σ: {sigma}")
                self.mu_log.append(mu.tolist())
                self.sigma_log.append(sigma.tolist())
            

            # Save all sampled actions
            self.actions_log.append(action_np.tolist())  # save full [a1, a2, a3] list

            # ⚠️ Warn only if out-of-bounds
            if np.any(np.abs(action_np) > 1.0):
                print(f"[WARNING] Out-of-bounds action: {action_np}")


            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
