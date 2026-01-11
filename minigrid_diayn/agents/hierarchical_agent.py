"""Hierarchical agent: meta-controller over pre-trained DIAYN skills."""

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from core import HierarchicalConfig
from networks import MetaController, MetaQNetwork, DiscretePolicy


class HierarchicalBuffer:
    """Replay buffer for (state, goal, skill, reward, next_state, done) transitions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, goal, skill, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, goal, skill, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return tuple(np.array(x) for x in zip(*batch))

    def __len__(self):
        return len(self.buffer)


class HierarchicalAgent:
    """Meta-controller that selects skills to reach goals."""

    def __init__(
        self,
        config: HierarchicalConfig,
        obs_dim: int,
        num_actions: int,
        pretrained_policy: DiscretePolicy,
        pretrained_encoder=None,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_skills = config.num_skills
        self.device = torch.device(config.device)

        # Freeze low-level policy and optional encoder
        self.low_level_policy = pretrained_policy.to(self.device)
        for p in self.low_level_policy.parameters():
            p.requires_grad = False
        self.low_level_policy.eval()
        self.low_level_encoder = None
        if pretrained_encoder is not None:
            self.low_level_encoder = pretrained_encoder.to(self.device)
            for p in self.low_level_encoder.parameters():
                p.requires_grad = False
            self.low_level_encoder.eval()

        # Meta-controller networks
        self.meta_policy = MetaController(
            config.meta_obs_dim, config.goal_dim, config.meta_hidden_dim, config.num_skills
        ).to(self.device)

        self.meta_q1 = MetaQNetwork(
            config.meta_obs_dim, config.goal_dim, config.meta_hidden_dim, config.num_skills
        ).to(self.device)
        self.meta_q2 = MetaQNetwork(
            config.meta_obs_dim, config.goal_dim, config.meta_hidden_dim, config.num_skills
        ).to(self.device)

        self.meta_q1_target = copy.deepcopy(self.meta_q1)
        self.meta_q2_target = copy.deepcopy(self.meta_q2)
        for p in self.meta_q1_target.parameters():
            p.requires_grad = False
        for p in self.meta_q2_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.meta_policy_opt = torch.optim.Adam(self.meta_policy.parameters(), lr=config.meta_lr)
        self.meta_q1_opt = torch.optim.Adam(self.meta_q1.parameters(), lr=config.meta_lr)
        self.meta_q2_opt = torch.optim.Adam(self.meta_q2.parameters(), lr=config.meta_lr)

        # Buffer
        self.buffer = HierarchicalBuffer(config.buffer_size)

        # Skill tracking
        self.current_skill = None
        self.steps_with_skill = 0

    def select_skill(self, state: np.ndarray, goal: np.ndarray, deterministic: bool = False) -> int:
        state_t = torch.FloatTensor(state).to(self.device)
        goal_t = torch.FloatTensor(goal).to(self.device)
        return self.meta_policy.select_skill(state_t, goal_t, deterministic)

    def select_action(self, state: np.ndarray, skill: int, deterministic: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        skill_oh = F.one_hot(torch.tensor([skill], device=self.device), self.num_skills).float()

        with torch.no_grad():
            if self.low_level_encoder is not None:
                state_t = self.low_level_encoder(state_t)
            logits = self.low_level_policy(state_t, skill_oh)
            if deterministic:
                return torch.argmax(logits, dim=-1).item()
            return torch.distributions.Categorical(logits=logits).sample().item()

    def should_reselect_skill(self) -> bool:
        return self.steps_with_skill >= self.config.skill_duration

    def step_skill_counter(self):
        self.steps_with_skill += 1

    def reset_skill_counter(self):
        self.steps_with_skill = 0

    def store_transition(self, state, goal, skill, reward, next_state, done):
        self.buffer.push(state, goal, skill, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        # Require minimum buffer size for stable training
        min_buffer = max(self.config.batch_size, 256)
        if len(self.buffer) < min_buffer:
            return {}

        states, goals, skills, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        skills = torch.LongTensor(skills).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Update Q-networks with soft value target (includes entropy)
        with torch.no_grad():
            next_probs = self.meta_policy.get_skill_probs(next_states, goals)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q = torch.min(
                self.meta_q1_target(next_states, goals),
                self.meta_q2_target(next_states, goals)
            )
            # Soft value: V(s') = E[Q(s',z) - alpha * log pi(z|s')]
            next_v = (next_probs * (next_q - self.config.entropy_coef * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.config.gamma * next_v

        q1 = self.meta_q1(states, goals).gather(1, skills.unsqueeze(-1)).squeeze(-1)
        q2 = self.meta_q2(states, goals).gather(1, skills.unsqueeze(-1)).squeeze(-1)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.meta_q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_q1.parameters(), max_norm=1.0)
        self.meta_q1_opt.step()

        self.meta_q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_q2.parameters(), max_norm=1.0)
        self.meta_q2_opt.step()

        # Update policy (discrete SAC)
        probs = self.meta_policy.get_skill_probs(states, goals)
        log_probs = torch.log(probs + 1e-8)

        # Q-values must be detached to prevent gradients flowing through Q-networks
        with torch.no_grad():
            q = torch.min(self.meta_q1(states, goals), self.meta_q2(states, goals))

        # Discrete SAC policy loss: minimize E[pi * (alpha*log_pi - Q)]
        policy_loss = (probs * (self.config.entropy_coef * log_probs - q)).sum(dim=-1).mean()

        self.meta_policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), max_norm=1.0)
        self.meta_policy_opt.step()

        # Soft update targets
        tau = self.config.tau
        for tp, p in zip(self.meta_q1_target.parameters(), self.meta_q1.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(self.meta_q2_target.parameters(), self.meta_q2.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return {"q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(), "policy_loss": policy_loss.item()}

    def save(self, path: str):
        torch.save({
            "meta_policy": self.meta_policy.state_dict(),
            "meta_q1": self.meta_q1.state_dict(),
            "meta_q2": self.meta_q2.state_dict(),
            "meta_q1_target": self.meta_q1_target.state_dict(),
            "meta_q2_target": self.meta_q2_target.state_dict(),
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.meta_policy.load_state_dict(ckpt["meta_policy"])
        self.meta_q1.load_state_dict(ckpt["meta_q1"])
        self.meta_q2.load_state_dict(ckpt["meta_q2"])
        self.meta_q1_target.load_state_dict(ckpt["meta_q1_target"])
        self.meta_q2_target.load_state_dict(ckpt["meta_q2_target"])
