"""Hierarchical DIAYN: Meta-controller over pre-trained skills."""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from networks import MetaController, MetaQNetwork, DiscretePolicy
from replay_buffer import ReplayBuffer


@dataclass
class HierarchicalConfig:
    """Config for hierarchical training."""
    num_skills: int = 8
    skill_duration: int = 8           # steps per skill before re-selection (6-10 good)
    meta_hidden_dim: int = 128
    meta_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 50000
    goal_dim: int = 2                 # (x, y) normalized goal position
    meta_obs_dim: int = 6             # (x, y) + direction one-hot (4D)
    entropy_coef: float = 0.1         # entropy bonus for exploration
    device: str = "mps"


class HierarchicalBuffer:
    """Replay buffer for hierarchical transitions (state, goal, skill, reward, next_state, done)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, goal, skill, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, goal, skill, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, goals, skills, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(goals), np.array(skills),
                np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class HierarchicalAgent:
    """Meta-controller that selects from pre-trained DIAYN skills."""

    def __init__(self, config: HierarchicalConfig, obs_dim: int, num_actions: int,
                 pretrained_policy: DiscretePolicy):
        self.config = config
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_skills = config.num_skills
        self.device = torch.device(config.device)

        # Freeze pre-trained low-level policy
        self.low_level_policy = pretrained_policy.to(self.device)
        for param in self.low_level_policy.parameters():
            param.requires_grad = False
        self.low_level_policy.eval()

        # Meta-controller networks (use simplified state: agent_pos + goal = 4D)
        meta_input_dim = config.meta_obs_dim + config.goal_dim
        self.meta_policy = MetaController(
            obs_dim=config.meta_obs_dim,
            goal_dim=config.goal_dim,
            hidden_dim=config.meta_hidden_dim,
            num_skills=config.num_skills
        ).to(self.device)

        self.meta_q1 = MetaQNetwork(
            obs_dim=config.meta_obs_dim,
            goal_dim=config.goal_dim,
            hidden_dim=config.meta_hidden_dim,
            num_skills=config.num_skills
        ).to(self.device)

        self.meta_q2 = MetaQNetwork(
            obs_dim=config.meta_obs_dim,
            goal_dim=config.goal_dim,
            hidden_dim=config.meta_hidden_dim,
            num_skills=config.num_skills
        ).to(self.device)

        # Target networks
        self.meta_q1_target = copy.deepcopy(self.meta_q1)
        self.meta_q2_target = copy.deepcopy(self.meta_q2)
        for p in self.meta_q1_target.parameters():
            p.requires_grad = False
        for p in self.meta_q2_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.meta_policy_optimizer = torch.optim.Adam(
            self.meta_policy.parameters(), lr=config.meta_lr
        )
        self.meta_q1_optimizer = torch.optim.Adam(
            self.meta_q1.parameters(), lr=config.meta_lr
        )
        self.meta_q2_optimizer = torch.optim.Adam(
            self.meta_q2.parameters(), lr=config.meta_lr
        )

        # Replay buffer
        self.buffer = HierarchicalBuffer(config.buffer_size)

        # Current skill tracking
        self.current_skill = None
        self.steps_with_skill = 0

    def select_skill(self, state: np.ndarray, goal: np.ndarray, deterministic: bool = False) -> int:
        """Select skill using meta-controller."""
        state_t = torch.FloatTensor(state).to(self.device)
        goal_t = torch.FloatTensor(goal).to(self.device)
        return self.meta_policy.select_skill(state_t, goal_t, deterministic)

    def select_action(self, state: np.ndarray, skill: int, deterministic: bool = False) -> int:
        """Select action using frozen low-level policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        skill_onehot = F.one_hot(
            torch.tensor([skill], device=self.device), self.num_skills
        ).float()

        with torch.no_grad():
            logits = self.low_level_policy(state_t, skill_onehot)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
        return action

    def should_reselect_skill(self) -> bool:
        """Check if it's time to select a new skill."""
        return self.steps_with_skill >= self.config.skill_duration

    def step_skill_counter(self):
        """Increment skill step counter."""
        self.steps_with_skill += 1

    def reset_skill_counter(self):
        """Reset skill counter for new skill selection."""
        self.steps_with_skill = 0

    def store_transition(self, state, goal, skill, reward, next_state, done):
        """Store hierarchical transition."""
        self.buffer.push(state, goal, skill, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """Update meta-controller."""
        if len(self.buffer) < self.config.batch_size:
            return {}

        states, goals, skills, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        skills = torch.LongTensor(skills).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Update Q-networks
        with torch.no_grad():
            next_skill_probs = self.meta_policy.get_skill_probs(next_states, goals)
            next_q1 = self.meta_q1_target(next_states, goals)
            next_q2 = self.meta_q2_target(next_states, goals)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_skill_probs * next_q).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.config.gamma * next_v

        current_q1 = self.meta_q1(states, goals).gather(1, skills.unsqueeze(-1)).squeeze(-1)
        current_q2 = self.meta_q2(states, goals).gather(1, skills.unsqueeze(-1)).squeeze(-1)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.meta_q1_optimizer.zero_grad()
        q1_loss.backward()
        self.meta_q1_optimizer.step()

        self.meta_q2_optimizer.zero_grad()
        q2_loss.backward()
        self.meta_q2_optimizer.step()

        # Update policy with entropy bonus
        skill_probs = self.meta_policy.get_skill_probs(states, goals)
        q1 = self.meta_q1(states, goals)
        q2 = self.meta_q2(states, goals)
        q = torch.min(q1, q2)

        # Entropy bonus for exploration
        log_probs = torch.log(skill_probs + 1e-8)
        entropy = -(skill_probs * log_probs).sum(dim=-1).mean()

        policy_loss = -(skill_probs * q).sum(dim=-1).mean() - self.config.entropy_coef * entropy

        self.meta_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.meta_policy_optimizer.step()

        # Soft update targets
        for target, source in [(self.meta_q1_target, self.meta_q1),
                               (self.meta_q2_target, self.meta_q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.config.tau * sp.data + (1 - self.config.tau) * tp.data)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
        }

    def save(self, path: str):
        torch.save({
            'meta_policy': self.meta_policy.state_dict(),
            'meta_q1': self.meta_q1.state_dict(),
            'meta_q2': self.meta_q2.state_dict(),
            'meta_q1_target': self.meta_q1_target.state_dict(),
            'meta_q2_target': self.meta_q2_target.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.meta_policy.load_state_dict(checkpoint['meta_policy'])
        self.meta_q1.load_state_dict(checkpoint['meta_q1'])
        self.meta_q2.load_state_dict(checkpoint['meta_q2'])
        self.meta_q1_target.load_state_dict(checkpoint['meta_q1_target'])
        self.meta_q2_target.load_state_dict(checkpoint['meta_q2_target'])
