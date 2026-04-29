import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Number of independent action dimensions (one per policy head)
N_HEADS = 5
HEAD_SIZE = 5  # each dimension has 5 choices


class ActorCritic(nn.Module):
    """Multi-head ActorCritic for MultiDiscrete([5,5,5,5,5]) action space.

    Architecture
    ------------
    Shared backbone:  Linear(state_dim→128) → Tanh → Linear(128→128) → Tanh
    Actor heads (×5): Linear(128→5) → Softmax  (one per action dimension)
    Critic:           Linear(128→1)
    """

    def __init__(self, state_dim: int, n_heads: int = N_HEADS, head_size: int = HEAD_SIZE):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        # Independent policy heads
        self.actor_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, head_size), nn.Softmax(dim=-1))
            for _ in range(n_heads)
        ])

        # Critic value head
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """Sample one action per head; return stacked actions and sum of log-probs."""
        state = torch.from_numpy(state).float()
        features = self.backbone(state)

        actions = []
        logprobs = []
        for head in self.actor_heads:
            probs = head(features)
            dist = Categorical(probs)
            a = dist.sample()
            actions.append(a)
            logprobs.append(dist.log_prob(a))

        # actions: (n_heads,) int tensor
        action_tensor = torch.stack(actions)            # (n_heads,)
        logprob_sum = torch.stack(logprobs).sum()       # scalar
        state_val = self.critic(features)
        return action_tensor.detach(), logprob_sum.detach(), state_val.detach()

    def evaluate(self, states, actions):
        """Evaluate stored actions under the current policy.

        Parameters
        ----------
        states  : (batch, state_dim)
        actions : (batch, n_heads)

        Returns
        -------
        logprob_sum  : (batch,)  — sum over heads
        state_values : (batch, 1)
        entropy_sum  : (batch,)  — sum over heads
        """
        features = self.backbone(states)
        logprob_sum = torch.zeros(states.shape[0], device=states.device)
        entropy_sum = torch.zeros(states.shape[0], device=states.device)

        for i, head in enumerate(self.actor_heads):
            probs = head(features)
            dist = Categorical(probs)
            logprob_sum += dist.log_prob(actions[:, i])
            entropy_sum += dist.entropy()

        state_values = self.critic(features)
        return logprob_sum, state_values, entropy_sum


class RolloutBuffer:
    def __init__(self):
        self.actions = []    # each entry: (n_heads,) tensor
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPOLite:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        """action_dim is ignored (kept for API compat) — heads are fixed at N_HEADS × HEAD_SIZE."""
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.backbone.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_heads.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])

        self.policy_old = ActorCritic(state_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """Return numpy array of shape (n_heads,) with sampled sub-actions."""
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(torch.from_numpy(state).float())
        self.buffer.actions.append(action)          # (n_heads,) int tensor
        self.buffer.logprobs.append(action_logprob) # scalar

        return action.numpy()

    def update(self):
        # Monte Carlo estimate of returns
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.stack(self.buffer.actions).detach()  # (batch, n_heads)
        old_logprobs = torch.stack(self.buffer.logprobs).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.policy.load_state_dict(torch.load(checkpoint_path, weights_only=True))
