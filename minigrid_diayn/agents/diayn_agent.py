"""DIAYN Agent for unsupervised skill discovery in MiniGrid.

Implements Diversity Is All You Need (DIAYN) with:
- CNN encoder for visual grid observations
- Categorical policy for discrete actions
- State-based discriminator for skill classification
- Policy gradient with entropy regularization
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core import DIAYNConfig, ReplayBuffer
from networks import DiscretePolicy, StateDiscriminator, PositionDiscriminator, GridEncoder, PartialGridEncoder


class DIAYNAgent:
    """DIAYN agent for discrete grid-world environments.

    Architecture:
    - Encoder: CNN that maps grid observation → feature vector (64-dim)
    - Policy: MLP that maps (features + skill one-hot) → action logits
    - Discriminator: MLP that maps encoded state → skill classification

    The discriminator provides intrinsic rewards that encourage each skill
    to visit distinguishable states, promoting behavioral diversity.
    """

    def __init__(self, config: DIAYNConfig, obs_dim: int, num_actions: int, grid_size: int,
                 partial_obs: bool = False):
        self.config = config
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_skills = config.num_skills
        self.grid_size = grid_size
        self.partial_obs = partial_obs
        self.device = torch.device(config.device)
        self.feature_dim = config.feature_dim

        # CNN Encoder
        if partial_obs:
            self.encoder = PartialGridEncoder(config.feature_dim).to(self.device)
        else:
            self.encoder = GridEncoder(grid_size, config.feature_dim).to(self.device)

        # Policy takes encoded features + skill one-hot
        self.policy = DiscretePolicy(
            config.feature_dim, config.num_skills, config.hidden_dim, num_actions
        ).to(self.device)

        # Discriminator - state-based (default) or position-based
        self.discriminator_type = getattr(config, 'discriminator_type', 'state')
        if self.discriminator_type == "position":
            # Position-based: classifies skill from (x,y) coordinates only
            self.discriminator = PositionDiscriminator(
                hidden_dim=256,
                num_skills=config.num_skills,
                geometric_init=True  # Sector-based initialization for faster learning
            ).to(self.device)
        else:
            # State-based: classifies skill from encoded visual features
            self.discriminator = StateDiscriminator(
                input_dim=config.feature_dim,
                num_skills=config.num_skills,
                hidden_dim=256
            ).to(self.device)

        # Optimizers with weight decay for regularization
        self.policy_optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=config.lr_policy,
            weight_decay=1e-5
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.lr_discriminator,
            weight_decay=1e-5
        )

        # Learning rate schedulers for gradual decay
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=1000
        )
        self.discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.discriminator_optimizer, T_max=1000
        )

        # Entropy coefficient for exploration bonus
        self.entropy_coef = config.entropy_coef

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)

    def select_action(self, state: np.ndarray, skill: int, deterministic: bool = False) -> int:
        """Select action using current policy.

        Dropout remains active during action selection to add exploration noise.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        skill_oh = F.one_hot(torch.tensor([skill], device=self.device), self.num_skills).float()

        with torch.no_grad():
            features = self.encoder(state_t)
            logits = self.policy(features, skill_oh)
            logits = torch.clamp(logits, -10, 10)

            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()

        return action

    def update(self, batch_size: int, train_discriminator: bool = True) -> Tuple[float, float, float, float]:
        """Single update step for both discriminator and policy.

        Args:
            batch_size: Batch size for training
            train_discriminator: If False, skip discriminator update (policy still trains with current disc)

        Returns:
            (disc_loss, disc_acc, policy_loss, entropy)
        """
        if not self.replay_buffer.is_ready(batch_size):
            return 0.0, 0.0, 0.0, 0.0

        # Sample batch
        states, actions, _, next_states, dones, skills, positions = self.replay_buffer.sample(batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        skills_t = torch.LongTensor(skills).to(self.device)
        skills_oh = F.one_hot(skills_t, self.num_skills).float()
        positions_t = torch.FloatTensor(positions).to(self.device)

        # ========== DISCRIMINATOR UPDATE ==========
        # Encode states (detach for discriminator - don't train encoder through disc)
        states_enc = self.encoder(states_t)
        next_states_enc = self.encoder(next_states_t).detach()

        # Discriminator input depends on type
        if self.discriminator_type == "position":
            disc_input = positions_t  # Use normalized (x, y) position
        else:
            disc_input = next_states_enc  # Use encoded state features

        disc_logits = self.discriminator(disc_input)
        disc_loss = F.cross_entropy(disc_logits, skills_t)

        # Only update discriminator if requested
        if train_discriminator:
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
            self.discriminator_optimizer.step()

        # Discriminator accuracy
        with torch.no_grad():
            disc_acc = (disc_logits.argmax(dim=-1) == skills_t).float().mean().item()

        # ========== POLICY UPDATE ==========
        # Re-encode (encoder is trained through policy)
        states_enc = self.encoder(states_t)
        next_states_enc = self.encoder(next_states_t).detach()

        # Get action distribution
        logits = self.policy(states_enc, skills_oh)
        logits = torch.clamp(logits, -10, 10)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Entropy for exploration
        entropy = -(probs * log_probs).sum(dim=-1)

        # Compute intrinsic reward using current discriminator: r = log q(z|s)
        with torch.no_grad():
            # Use same input as discriminator training
            if self.discriminator_type == "position":
                reward_input = positions_t
            else:
                reward_input = next_states_enc
            pred_probs = F.softmax(self.discriminator(reward_input), dim=-1)
            log_pred_probs = torch.log(pred_probs + 1e-6)
            # Dot product with skill one-hot: sum(log_q * skill_oh)
            intrinsic_reward = (log_pred_probs * skills_oh).sum(dim=-1)

            # Add log(num_skills) to center rewards around zero
            # Reward is positive when discriminator is better than random guessing
            log_prior = np.log(self.num_skills)
            intrinsic_reward = intrinsic_reward + log_prior

        # Policy gradient: maximize expected intrinsic reward
        action_log_probs = log_probs.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(action_log_probs * intrinsic_reward).mean()

        # Add entropy bonus
        entropy_loss = -self.entropy_coef * entropy.mean()
        total_policy_loss = policy_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.policy.parameters()), 0.5
        )
        self.policy_optimizer.step()

        return disc_loss.item(), disc_acc, policy_loss.item(), entropy.mean().item()

    def compute_intrinsic_reward(self, next_state: np.ndarray, skill: int) -> float:
        """Compute intrinsic reward for logging/diagnostics.

        Args:
            next_state: Next observation
            skill: Skill index

        Returns:
            Intrinsic reward = log q(z|s)
        """
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        skill_oh = F.one_hot(torch.tensor([skill], device=self.device), self.num_skills).float()

        with torch.no_grad():
            encoded = self.encoder(next_state_t)
            probs = F.softmax(self.discriminator(encoded), dim=-1)
            log_probs = torch.log(probs + 1e-6)
            reward = (log_probs * skill_oh).sum(dim=-1).item()

        return reward

    def compute_pseudo_reward_with_diagnostics(self, next_state: np.ndarray, position: np.ndarray, skill: int) -> dict:
        """Compute reward and diagnostics (for compatibility with existing training loop).

        Args:
            next_state: Next observation
            position: Normalized agent position (used for position-based discriminator)
            skill: Skill index

        Returns:
            Dict with reward, prob_correct, predicted_skill, etc.
        """
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        position_t = torch.FloatTensor(position).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Choose discriminator input based on type
            if self.discriminator_type == "position":
                disc_input = position_t
            else:
                disc_input = self.encoder(next_state_t)

            logits = self.discriminator(disc_input)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-6)

            # Intrinsic reward = log q(z|s)
            reward = log_probs[0, skill].item()
            prob_correct = probs[0, skill].item()
            predicted_skill = logits.argmax(dim=-1).item()
            max_prob = probs.max().item()

        return {
            "reward": reward,
            "prob_correct": prob_correct,
            "predicted_skill": predicted_skill,
            "actual_skill": skill,
            "max_prob": max_prob,
            "is_correct": predicted_skill == skill,
        }

    def step_schedulers(self):
        """Step learning rate schedulers."""
        self.policy_scheduler.step()
        self.discriminator_scheduler.step()

    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "grid_size": self.grid_size,
            "feature_dim": self.feature_dim,
            "partial_obs": self.partial_obs,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.grid_size = ckpt.get("grid_size", self.grid_size)
        self.feature_dim = ckpt.get("feature_dim", self.feature_dim)

    @classmethod
    def from_checkpoint(cls, path: str, config: DIAYNConfig = None):
        """Load agent from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if config is None:
            config = DIAYNConfig()
            if "feature_dim" in ckpt:
                config.feature_dim = ckpt["feature_dim"]

        partial_obs = ckpt.get("partial_obs", False)

        agent = cls(
            config,
            obs_dim=ckpt["obs_dim"],
            num_actions=ckpt["num_actions"],
            grid_size=ckpt["grid_size"],
            partial_obs=partial_obs,
        )
        agent.load(path)
        return agent
