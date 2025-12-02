import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
from collections import deque
import random

from my_neural_nets import Policy, Discriminator, Critic
from replay_buffer import ReplayBuffer


class DIAYN_SAC:
    """
    DIAYN (Diversity is All You Need) algorithm with SAC as the base RL algorithm.
    
    Discovers diverse skills without external rewards by maximizing mutual information
    between skills and states: I(S;Z) = H(Z) - H(Z|S)
    """
    
    def __init__(
        self,
        state_dim=17,
        action_dim=6,
        skill_dim=10,
        hidden_dim=256,
        lr_policy=3e-4,
        lr_critic=3e-4,
        lr_discriminator=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device='cpu'
    ):
        """
        Initialize DIAYN-SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            skill_dim: Number of skills to discover
            hidden_dim: Hidden layer dimension for all networks
            lr_policy: Learning rate for policy
            lr_critic: Learning rate for critics
            lr_discriminator: Learning rate for discriminator
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Initial entropy coefficient (auto-tuned)
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # ==================== Initialize Networks ====================
        # Policy network
        self.policy = Policy(
            input_dim=state_dim,
            skill_dim=skill_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        ).to(device)
        
        # Twin Q-networks (critics)
        self.critic1 = Critic(
            observation_dim=state_dim,
            action_dim=action_dim,
            skill_dim=skill_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.critic2 = Critic(
            observation_dim=state_dim,
            action_dim=action_dim,
            skill_dim=skill_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Discriminator network
        self.discriminator = Discriminator(
            observation_dim=state_dim,
            hidden_dim=hidden_dim,
            skill_dim=skill_dim
        ).to(device)
        
        # ==================== Target Networks ====================
        # Create target networks as copies (only for critics)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Freeze target networks (no gradient computation)
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # ==================== Optimizers ====================
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=lr_policy
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), 
            lr=lr_critic
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), 
            lr=lr_critic
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=lr_discriminator
        )
        
        # ==================== Automatic Entropy Tuning ====================
        self.alpha = alpha
        self.target_entropy = -action_dim  # Heuristic: -dim(A)
        self.log_alpha = torch.tensor(
            np.log(alpha), 
            requires_grad=True, 
            device=device
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], 
            lr=lr_policy
        )
        
        # ==================== Replay Buffer ====================
        self.replay_buffer = ReplayBuffer(capacity=1000000)
        
        # ==================== DIAYN Specific ====================
        # Uniform prior over skills: p(z) = 1/k
        self.log_skill_prior = -np.log(skill_dim)
    
    def select_action(self, state, skill, deterministic=False):
        """
        Select an action given state and skill.
        
        Args:
            state: Current state observation (numpy array)
            skill: Skill index (int, 0 to skill_dim-1)
            deterministic: If True, use mean action (for evaluation)
                          If False, sample from policy (for training)
        
        Returns:
            action: Action to take (numpy array)
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Create skill one-hot vector
        skill_onehot = torch.zeros(1, self.skill_dim).to(self.device)
        skill_onehot[0, skill] = 1.0
        
        # Get action from policy
        with torch.no_grad():
            if deterministic:
                # Use mean action (no exploration)
                _, _, mean_action = self.policy.sample(state, skill_onehot)
                action = mean_action
            else:
                # Sample action (with exploration)
                action, _, _ = self.policy.sample(state, skill_onehot)
        
        # Convert to numpy
        return action.cpu().numpy()[0]
    
    def compute_pseudo_reward(self, next_state, skill):
        """
        Compute DIAYN pseudo-reward: r(s,z) = log q(z|s') - log p(z)
        
        This encourages:
        - States to be discriminative of skills (maximize log q(z|s'))
        - Exploration across all skills (entropy term -log p(z))
        
        Args:
            next_state: Next state observation (numpy array)
            skill: Skill index used
        
        Returns:
            pseudo_reward: Intrinsic reward (float)
        """
        # Convert to tensor
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get discriminator predictions
            logits = self.discriminator(next_state_tensor)
            
            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probability for current skill: log q(z|s')
            log_q_z_given_s = log_probs[0, skill].item()
        
        # Compute pseudo-reward: log q(z|s') - log p(z)
        # log p(z) = log(1/k) = -log(k) for uniform prior
        pseudo_reward = log_q_z_given_s - self.log_skill_prior
        
        return pseudo_reward
    
    def update_discriminator(self, batch_size=256):
        """
        Update discriminator to predict skill from next state.
        
        The discriminator learns to identify which skill produced which state,
        enabling the computation of pseudo-rewards.
        
        Args:
            batch_size: Number of samples for training
            
        Returns:
            loss: Discriminator loss (float)
            accuracy: Classification accuracy (float)
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, skills = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        next_states = torch.FloatTensor(next_states).to(self.device)
        skills = torch.LongTensor(skills).to(self.device)
        
        # Forward pass: predict skill from next_state
        logits = self.discriminator(next_states)
        
        # Cross-entropy loss (classification task)
        loss = F.cross_entropy(logits, skills)
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        
        # Compute accuracy for logging
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == skills).float().mean().item()
        
        return loss.item(), accuracy
    
    def update_critic(self, batch_size=256):
        """
        Update twin Q-networks using Bellman equation.
        
        Uses double Q-learning with target networks for stability.
        
        Args:
            batch_size: Number of samples for training
            
        Returns:
            critic1_loss: Loss for first critic (float)
            critic2_loss: Loss for second critic (float)
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, skills = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Create skill one-hot vectors
        skills_onehot = torch.zeros(batch_size, self.skill_dim).to(self.device)
        for i, skill in enumerate(skills):
            skills_onehot[i, skill] = 1.0
        
        # ==================== Compute Target Q-value ====================
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs, _ = self.policy.sample(next_states, skills_onehot)
            
            # Compute Q-values from target networks
            target_q1 = self.critic1_target(next_states, next_actions, skills_onehot)
            target_q2 = self.critic2_target(next_states, next_actions, skills_onehot)
            
            # Take minimum (avoid overestimation)
            target_q = torch.min(target_q1, target_q2)
            
            # Subtract entropy term (SAC's entropy regularization)
            target_q = target_q - self.alpha * next_log_probs
            
            # Bellman backup: r + γ * (min_Q - α * log_π)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # ==================== Update Critic 1 ====================
        current_q1 = self.critic1(states, actions, skills_onehot)
        critic1_loss = F.mse_loss(current_q1, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # ==================== Update Critic 2 ====================
        current_q2 = self.critic2(states, actions, skills_onehot)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return critic1_loss.item(), critic2_loss.item()
    
    def update_policy(self, batch_size=256):
        """
        Update policy network to maximize Q-values and entropy.
        
        SAC objective: maximize Q(s,a,z) + α * H(π(·|s,z))
        Also updates entropy coefficient α automatically.
        
        Args:
            batch_size: Number of samples for training
            
        Returns:
            policy_loss: Policy loss (float)
            alpha_loss: Entropy coefficient loss (float)
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, skills = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        
        # Create skill one-hot vectors
        skills_onehot = torch.zeros(batch_size, self.skill_dim).to(self.device)
        for i, skill in enumerate(skills):
            skills_onehot[i, skill] = 1.0
        
        # ==================== Update Policy ====================
        # Sample actions from current policy
        new_actions, log_probs, _ = self.policy.sample(states, skills_onehot)
        
        # Compute Q-values for sampled actions
        q1 = self.critic1(states, new_actions, skills_onehot)
        q2 = self.critic2(states, new_actions, skills_onehot)
        q = torch.min(q1, q2)
        
        # Policy loss: maximize Q - α * log_π
        # (Negative because we're minimizing the loss)
        policy_loss = (self.alpha * log_probs - q).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ==================== Update Alpha (Entropy Coefficient) ====================
        # Automatically adjust α to maintain target entropy
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update alpha value
        self.alpha = self.log_alpha.exp().item()
        
        return policy_loss.item(), alpha_loss.item()
    
    def soft_update_target_networks(self):
        """
        Soft update of target networks: θ_target = τ*θ + (1-τ)*θ_target
        
        This slowly moves target networks toward current networks,
        providing stable training targets.
        """
        # Update critic1 target
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        # Update critic2 target
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train_step(self, batch_size=256):
        """
        Perform one complete training step (update all networks).
        
        Order of updates:
        1. Update discriminator (learn to identify skills)
        2. Update critics (learn Q-values)
        3. Update policy (improve action selection)
        4. Soft update target networks
        
        Args:
            batch_size: Number of samples for training
            
        Returns:
            metrics: Dictionary containing all losses and metrics
        """
        # Update discriminator
        disc_loss, disc_acc = self.update_discriminator(batch_size)
        
        # Update critics
        critic1_loss, critic2_loss = self.update_critic(batch_size)
        
        # Update policy
        policy_loss, alpha_loss = self.update_policy(batch_size)
        
        # Soft update target networks
        self.soft_update_target_networks()
        
        # Return all metrics for logging
        return {
            'discriminator_loss': disc_loss,
            'discriminator_accuracy': disc_acc,
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'policy_loss': policy_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }
