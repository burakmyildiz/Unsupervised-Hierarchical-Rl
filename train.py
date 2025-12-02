"""
Training script for DIAYN (Diversity is All You Need) algorithm.

This script trains an agent to discover diverse skills without external rewards
in continuous control environments like HalfCheetah or Ant.
"""

import gymnasium as gym
import torch
import numpy as np
import random
from dıayn_sac import DIAYN_SAC


def train_diayn(
    env_name='HalfCheetah-v5',
    num_episodes=1000,
    max_steps_per_episode=1000,
    num_skills=10,
    batch_size=256,
    updates_per_step=1,
    start_training_steps=1000,
    eval_interval=50,
    device='cpu',
    render_mode=None,
    seed=42
):
    """
    Main training loop for DIAYN.
    
    Args:
        env_name: Gymnasium environment name
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        num_skills: Number of skills to discover
        batch_size: Batch size for network updates
        updates_per_step: Number of gradient updates per environment step
        start_training_steps: Start training after this many steps (random exploration)
        eval_interval: Evaluate skills every N episodes
        device: Device to use ('cpu' or 'cuda')
        render_mode: Visualization mode (None, 'human', or 'rgb_array')
        seed: Random seed for reproducibility
    
    Returns:
        agent: Trained DIAYN_SAC agent
        episode_rewards: List of episode rewards
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # Set environment seed
    env.reset(seed=seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print("=" * 80)
    print("DIAYN Training Configuration")
    print("=" * 80)
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Number of skills: {num_skills}")
    print(f"Device: {device}")
    print(f"Total episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Batch size: {batch_size}")
    print(f"Random exploration steps: {start_training_steps}")
    print("=" * 80)
    
    # Initialize DIAYN-SAC agent
    agent = DIAYN_SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        skill_dim=num_skills,
        hidden_dim=256,
        lr_policy=3e-4,
        lr_critic=3e-4,
        lr_discriminator=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device=device
    )
    
    # Training statistics
    episode_rewards = []
    episode_pseudo_rewards = []
    discriminator_accuracies = []
    total_steps = 0
    
    print("\nStarting training...")
    print("=" * 80)
    
    # Main training loop
    for episode in range(num_episodes):
        # Sample random skill for this episode
        skill = np.random.randint(0, num_skills)
        
        # Reset environment
        state, _ = env.reset()
        
        # Episode statistics
        episode_reward = 0.0
        episode_pseudo_reward = 0.0
        episode_steps = 0
        
        # Episode loop
        for step in range(max_steps_per_episode):
            # Select action
            if total_steps < start_training_steps:
                # Random actions for initial exploration
                action = env.action_space.sample()
            else:
                # Select action from policy
                action = agent.select_action(state, skill, deterministic=False)
            
            # Execute action in environment
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Compute DIAYN pseudo-reward
            pseudo_reward = agent.compute_pseudo_reward(next_state, skill)
            
            # Store transition in replay buffer
            agent.replay_buffer.push(
                state, 
                action, 
                pseudo_reward, 
                next_state, 
                done, 
                skill
            )
            
            # Update statistics
            episode_reward += env_reward  # External reward (for monitoring only)
            episode_pseudo_reward += pseudo_reward  # DIAYN reward (actual training signal)
            episode_steps += 1
            total_steps += 1
            
            # Train networks
            if total_steps >= start_training_steps:
                for _ in range(updates_per_step):
                    train_metrics = agent.train_step(batch_size)
            
            # Move to next state
            state = next_state
            
            # Check if episode is done
            if done:
                break
        
        # Store episode statistics
        episode_rewards.append(episode_pseudo_reward)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Skill: {skill}")
            print(f"  Steps: {episode_steps}")
            print(f"  Pseudo Reward: {episode_pseudo_reward:.2f}")
            print(f"  Avg Pseudo Reward (last 10): {avg_reward:.2f}")
            print(f"  Environment Reward: {episode_reward:.2f}")
            print(f"  Total Steps: {total_steps}")
            
            # Print training metrics if available
            if total_steps >= start_training_steps:
                print(f"  Discriminator Loss: {train_metrics['discriminator_loss']:.4f}")
                print(f"  Discriminator Accuracy: {train_metrics['discriminator_accuracy']:.4f}")
                print(f"  Critic1 Loss: {train_metrics['critic1_loss']:.4f}")
                print(f"  Critic2 Loss: {train_metrics['critic2_loss']:.4f}")
                print(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
                print(f"  Alpha: {train_metrics['alpha']:.4f}")
        
        # Evaluate skills periodically
        if (episode + 1) % eval_interval == 0 and total_steps >= start_training_steps:
            print("\n" + "=" * 80)
            print(f"SKILL EVALUATION AT EPISODE {episode + 1}")
            print("=" * 80)
            evaluate_skills(env, agent, num_skills, num_episodes=3, max_steps=500)
            print("=" * 80)
    
    # Close environment
    env.close()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Final average pseudo-reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    
    return agent, episode_rewards


def evaluate_skills(env, agent, num_skills, num_episodes=5, max_steps=500):
    """
    Evaluate learned skills.
    
    This function runs each skill deterministically and measures:
    - Average environment reward
    - Average displacement (for locomotion tasks)
    - Skill diversity
    
    Args:
        env: Gymnasium environment
        agent: Trained DIAYN_SAC agent
        num_skills: Number of skills to evaluate
        num_episodes: Number of episodes per skill
        max_steps: Maximum steps per episode
    """
    print(f"\nEvaluating {num_skills} skills...")
    
    skill_rewards = []
    skill_positions = []
    
    for skill in range(num_skills):
        episode_rewards = []
        episode_displacements = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            
            # Track position for displacement calculation
            if hasattr(env.unwrapped, 'data'):
                # For MuJoCo environments
                initial_pos = env.unwrapped.data.qpos[0] if len(env.unwrapped.data.qpos) > 0 else 0
            else:
                # Fallback: use first element of state
                initial_pos = state[0]
            
            # Run episode
            for step in range(max_steps):
                # Use deterministic action (no exploration)
                action = agent.select_action(state, skill, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Calculate final position
            if hasattr(env.unwrapped, 'data'):
                final_pos = env.unwrapped.data.qpos[0] if len(env.unwrapped.data.qpos) > 0 else 0
            else:
                final_pos = state[0]
            
            displacement = final_pos - initial_pos
            
            episode_rewards.append(total_reward)
            episode_displacements.append(displacement)
        
        # Calculate averages for this skill
        avg_reward = np.mean(episode_rewards)
        avg_displacement = np.mean(episode_displacements)
        
        skill_rewards.append(avg_reward)
        skill_positions.append(avg_displacement)
        
        print(f"  Skill {skill:2d}: "
              f"Avg Reward: {avg_reward:8.2f}, "
              f"Avg Displacement: {avg_displacement:8.2f}")
    
    # Calculate and print diversity metrics
    reward_std = np.std(skill_rewards)
    position_std = np.std(skill_positions)
    
    print(f"\n  Diversity Metrics:")
    print(f"    Reward std: {reward_std:.2f}")
    print(f"    Displacement std: {position_std:.2f}")
    
    return skill_rewards, skill_positions


def save_agent(agent, filepath):
    """
    Save trained agent to disk.
    
    Args:
        agent: DIAYN_SAC agent to save
        filepath: Path to save the agent
    """
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'critic1_state_dict': agent.critic1.state_dict(),
        'critic2_state_dict': agent.critic2.state_dict(),
        'discriminator_state_dict': agent.discriminator.state_dict(),
        'policy_optimizer': agent.policy_optimizer.state_dict(),
        'critic1_optimizer': agent.critic1_optimizer.state_dict(),
        'critic2_optimizer': agent.critic2_optimizer.state_dict(),
        'discriminator_optimizer': agent.discriminator_optimizer.state_dict(),
    }, filepath)
    print(f"Agent saved to {filepath}")


def load_agent(agent, filepath, device='cpu'):
    """
    Load trained agent from disk.
    
    Args:
        agent: DIAYN_SAC agent to load weights into
        filepath: Path to load the agent from
        device: Device to load tensors to
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
    agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
    agent.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
    agent.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
    agent.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
    agent.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    
    print(f"Agent loaded from {filepath}")


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Train DIAYN
    agent, rewards = train_diayn(
        env_name='Ant-v5',      # Environment (HalfCheetah-v5, Ant-v5, Walker2d-v5, etc.)
        num_episodes=1000,               # Total training episodes
        max_steps_per_episode=1000,      # Max steps per episode
        num_skills=10,                   # Number of skills to discover
        batch_size=256,                  # Batch size for training
        updates_per_step=1,              # Gradient updates per environment step
        start_training_steps=1000,       # Random exploration before training starts
        eval_interval=50,                # Evaluate skills every N episodes
        device=device,                   # CPU or CUDA
        render_mode='human',                # Set to 'human' to visualize (slower)
        seed=42                          # Random seed
    )
    
    # Save trained agent
    save_agent(agent, 'diayn_agent.pth')
    
    print("\n✅ Training complete! Agent saved to 'diayn_agent.pth'")

