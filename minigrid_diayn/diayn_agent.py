"""
DIAYN Agent for discrete action spaces (MiniGrid).

Implements Diversity is All You Need algorithm with:
- Discrete policy (Categorical distribution)
- Twin Q-networks for stability
- Automatic entropy tuning
- Pseudo-reward computation
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, Tuple, Optional

from networks import DiscretePolicy, QNetwork, Discriminator
from replay_buffer import ReplayBuffer
from config import DIAYNConfig


class DIAYNAgent:
    """
    DIAYN Agent for discovering diverse skills without external rewards.

    The agent learns skills by maximizing mutual information between
    skills and states: I(S; Z) = H(Z) - H(Z|S)

    This is achieved by:
    1. Training a discriminator q(z|s) to predict skill from state
    2. Using pseudo-reward r = log q(z|s') - log p(z) to train policy
    3. Policy tries to reach states that are distinguishable by skill
    """

    def __init__(self, config: DIAYNConfig, obs_dim: int, num_actions: int):
        """
        Initialize DIAYN agent.

        Args:
            config: Configuration dataclass
            obs_dim: Dimension of observation space
            num_actions: Number of discrete actions
        """
        self.config = config
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_skills = config.num_skills
        self.device = torch.device(config.device)

        # ==================== Networks ====================
        # Policy network (categorical for discrete actions)
        self.policy = DiscretePolicy(
            obs_dim=obs_dim,
            skill_dim=config.num_skills,
            hidden_dim=config.hidden_dim,
            num_actions=num_actions
        ).to(self.device)

        # Twin Q-networks
        self.q_network1 = QNetwork(
            obs_dim=obs_dim,
            skill_dim=config.num_skills,
            hidden_dim=config.hidden_dim,
            num_actions=num_actions
        ).to(self.device)

        self.q_network2 = QNetwork(
            obs_dim=obs_dim,
            skill_dim=config.num_skills,
            hidden_dim=config.hidden_dim,
            num_actions=num_actions
        ).to(self.device)

        # Target networks
        self.q_target1 = copy.deepcopy(self.q_network1)
        self.q_target2 = copy.deepcopy(self.q_network2)

        # Freeze target networks
        for param in self.q_target1.parameters():
            param.requires_grad = False
        for param in self.q_target2.parameters():
            param.requires_grad = False

        # Discriminator (uses 6-dim input: x, y, direction one-hot)
        self.discriminator = Discriminator(
            obs_dim=config.disc_obs_dim,
            hidden_dim=config.hidden_dim,
            num_skills=config.num_skills
        ).to(self.device)

        # ==================== Optimizers ====================
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr_policy
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q_network1.parameters(), lr=config.lr_critic
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q_network2.parameters(), lr=config.lr_critic
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=config.lr_discriminator
        )

        # ==================== Entropy Tuning ====================
        # NOTE: DIAYN paper uses FIXED alpha = 0.1 (Section 3.2, Appendix C)
        # Auto-tuning is disabled by default for DIAYN
        if config.auto_entropy_tuning:
            # Target entropy = 0.5 * log(num_actions) heuristic
            self.target_entropy = 0.5 * np.log(num_actions)
            self.log_alpha = torch.tensor(
                np.log(config.alpha),
                dtype=torch.float32,  # MPS requires float32
                requires_grad=True,
                device=self.device
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=config.lr_alpha
            )
            self.alpha = self.log_alpha.exp().item()
        else:
            # Use fixed alpha from config (DIAYN default: 0.1)
            self.alpha = config.alpha
            self.log_alpha = None
            self.alpha_optimizer = None

        # ==================== Replay Buffer ====================
        self.replay_buffer = ReplayBuffer(capacity=config.buffer_size)

        # ==================== DIAYN Specific ====================
        # Uniform prior: p(z) = 1/num_skills
        self.log_skill_prior = -np.log(config.num_skills)

        # Reward clipping for stability
        self.reward_clip = config.reward_clip

        # Discriminator observation dimension (for 6-dim input)
        self.disc_obs_dim = config.disc_obs_dim

    def select_action(
        self,
        state: np.ndarray,
        skill: int,
        deterministic: bool = False
    ) -> int:
        """
        Select action given state and skill.

        Args:
            state: Current observation (numpy array)
            skill: Skill index (0 to num_skills-1)
            deterministic: If True, use argmax action

        Returns:
            action: Action to take (int)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        skill_onehot = F.one_hot(
            torch.tensor([skill], device=self.device), self.num_skills
        ).float()

        with torch.no_grad():
            logits = self.policy(state_tensor, skill_onehot)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

        return action

    def get_discriminator_obs(self, info: dict) -> np.ndarray:
        """
        Extract 6-dim discriminator observation: (x, y, direction_onehot).

        Position is normalized to [0, 1] by dividing by grid size (7).
        Direction is one-hot encoded (4 dims).

        Args:
            info: Environment info dict containing 'agent_pos' and 'agent_dir'

        Returns:
            6-dim numpy array: [x/7, y/7, dir_0, dir_1, dir_2, dir_3]
        """
        pos = info['agent_pos']
        direction = info['agent_dir']

        # Normalize position to [0, 1]
        x_norm = pos[0] / 7.0
        y_norm = pos[1] / 7.0

        # One-hot encode direction
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[direction] = 1.0

        return np.array([x_norm, y_norm, dir_onehot[0], dir_onehot[1],
                         dir_onehot[2], dir_onehot[3]], dtype=np.float32)

    def compute_pseudo_reward(self, info: dict, skill: int) -> float:
        """
        Compute DIAYN pseudo-reward: r = log q(z|s') - log p(z)

        Uses 6-dim discriminator observation (x, y, direction) instead of full state.

        This reward encourages:
        - States that are easily identifiable by skill (high log q(z|s'))
        - Using all skills (entropy term -log p(z))

        Args:
            info: Environment info dict containing 'agent_pos' and 'agent_dir'
            skill: Skill index used

        Returns:
            pseudo_reward: Intrinsic reward (float)
        """
        disc_obs = self.get_discriminator_obs(info)
        disc_obs_tensor = torch.FloatTensor(disc_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.discriminator.get_log_probs(disc_obs_tensor)
            log_q_z_given_s = log_probs[0, skill].item()

        # r = log q(z|s') - log p(z)
        # log p(z) = log(1/k) = -log(k)
        pseudo_reward = log_q_z_given_s - self.log_skill_prior

        # Clip for stability
        pseudo_reward = np.clip(pseudo_reward, -self.reward_clip, self.reward_clip)

        return pseudo_reward

    def update_discriminator(self, batch_size: int) -> Tuple[float, float]:
        """
        Update discriminator to predict skill from 6-dim discriminator observation.

        Args:
            batch_size: Number of samples for training

        Returns:
            loss: Discriminator loss
            accuracy: Classification accuracy
        """
        if not self.replay_buffer.is_ready(batch_size):
            return 0.0, 0.0

        # Sample batch (now includes disc_next_obs)
        states, actions, rewards, next_states, dones, skills, disc_next_obs = \
            self.replay_buffer.sample(batch_size)

        # Convert to tensors - use disc_next_obs instead of next_states
        disc_obs_tensor = torch.FloatTensor(disc_next_obs).to(self.device)
        skills_tensor = torch.LongTensor(skills).to(self.device)

        # Forward pass: predict skill from 6-dim disc_obs
        logits = self.discriminator(disc_obs_tensor)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, skills_tensor)

        # Update
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.discriminator_optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == skills_tensor).float().mean().item()

        return loss.item(), accuracy

    def update_critic(self, batch_size: int) -> Tuple[float, float]:
        """
        Update twin Q-networks using Bellman equation.

        Args:
            batch_size: Number of samples for training

        Returns:
            q1_loss: Loss for first Q-network
            q2_loss: Loss for second Q-network
        """
        if not self.replay_buffer.is_ready(batch_size):
            return 0.0, 0.0

        # Sample batch (ignore disc_next_obs for critic update)
        states, actions, rewards, next_states, dones, skills, _ = \
            self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        skills_onehot = F.one_hot(
            torch.LongTensor(skills), self.num_skills
        ).float().to(self.device)

        # Compute target Q-value
        with torch.no_grad():
            # Get action probabilities for next states
            next_action_probs = self.policy.get_action_probs(next_states, skills_onehot)

            # Get Q-values from target networks
            next_q1 = self.q_target1(next_states, skills_onehot)
            next_q2 = self.q_target2(next_states, skills_onehot)
            next_q = torch.min(next_q1, next_q2)

            # Compute expected Q-value with entropy
            # V(s') = E_a[Q(s',a) - α log π(a|s')]
            next_log_probs = torch.log(next_action_probs + 1e-8)
            next_v = (next_action_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)

            # Bellman backup
            target_q = rewards + (1 - dones) * self.config.gamma * next_v

        # Update Q-network 1
        current_q1 = self.q_network1.get_q_value(states, skills_onehot, actions)
        q1_loss = F.mse_loss(current_q1, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network1.parameters(), max_norm=1.0)
        self.q1_optimizer.step()

        # Update Q-network 2
        current_q2 = self.q_network2.get_q_value(states, skills_onehot, actions)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network2.parameters(), max_norm=1.0)
        self.q2_optimizer.step()

        return q1_loss.item(), q2_loss.item()

    def update_policy(self, batch_size: int) -> Tuple[float, float, float]:
        """
        Update policy to maximize Q-values and entropy.

        Args:
            batch_size: Number of samples for training

        Returns:
            policy_loss: Policy loss
            alpha_loss: Entropy coefficient loss
            entropy: Average entropy
        """
        if not self.replay_buffer.is_ready(batch_size):
            return 0.0, 0.0, 0.0

        # Sample batch (ignore disc_next_obs for policy update)
        states, actions, rewards, next_states, dones, skills, _ = \
            self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        skills_onehot = F.one_hot(
            torch.LongTensor(skills), self.num_skills
        ).float().to(self.device)

        # Get action probabilities
        action_probs = self.policy.get_action_probs(states, skills_onehot)
        log_probs = torch.log(action_probs + 1e-8)

        # Get Q-values
        q1 = self.q_network1(states, skills_onehot)
        q2 = self.q_network2(states, skills_onehot)
        q = torch.min(q1, q2)

        # Policy loss: minimize -E[Q - α log π]
        # = maximize E[Q] + α H(π)
        policy_loss = (action_probs * (self.alpha * log_probs - q)).sum(dim=-1).mean()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update alpha (entropy coefficient)
        alpha_loss = 0.0
        if self.log_alpha is not None:
            # Entropy of current policy: H(π) = -Σ π(a|s) log π(a|s)
            entropy = -(action_probs * log_probs).sum(dim=-1)

            # Alpha loss: J(α) = α * (H(π) - H_target)
            # We want: if H(π) < H_target → increase α (encourage exploration)
            #          if H(π) > H_target → decrease α (less exploration needed)
            # Loss = α * (H_target - H(π)), minimize this
            alpha_loss = (self.log_alpha.exp() * (self.target_entropy - entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
            entropy = entropy.mean().item()
        else:
            entropy = -(action_probs * log_probs).sum(dim=-1).mean().item()

        return policy_loss.item(), alpha_loss, entropy

    def soft_update_targets(self):
        """Soft update target networks: θ_target = τ*θ + (1-τ)*θ_target"""
        tau = self.config.tau

        for target_param, param in zip(self.q_target1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.q_target2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_step(self, batch_size: int) -> Dict[str, float]:
        """
        Perform one complete training step.

        Order of updates:
        1. Discriminator (learn to identify skills)
        2. Critics (learn Q-values)
        3. Policy (improve action selection)
        4. Target networks (soft update)

        Args:
            batch_size: Number of samples for training

        Returns:
            metrics: Dictionary of training metrics
        """
        # Update discriminator
        disc_loss, disc_acc = self.update_discriminator(batch_size)

        # Update critics
        q1_loss, q2_loss = self.update_critic(batch_size)

        # Update policy
        policy_loss, alpha_loss, entropy = self.update_policy(batch_size)

        # Soft update targets
        self.soft_update_targets()

        return {
            'discriminator_loss': disc_loss,
            'discriminator_accuracy': disc_acc,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'policy_loss': policy_loss,
            'alpha_loss': alpha_loss,
            'entropy': entropy,
            'alpha': self.alpha
        }

    def save(self, path: str):
        """Save agent state to file."""
        torch.save({
            'policy': self.policy.state_dict(),
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'q_target1': self.q_target1.state_dict(),
            'q_target2': self.q_target2.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.log_alpha is not None else None,
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['policy'])
        self.q_network1.load_state_dict(checkpoint['q_network1'])
        self.q_network2.load_state_dict(checkpoint['q_network2'])
        self.q_target1.load_state_dict(checkpoint['q_target1'])
        self.q_target2.load_state_dict(checkpoint['q_target2'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

        if checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
