"""
Inference script for trained DIAYN agents.

This script allows you to:
1. Load a trained agent
2. Visualize individual skills
3. Compare multiple skills
4. Record videos of skills
5. Analyze skill behaviors
"""

import gymnasium as gym
import torch
import numpy as np
import argparse
from dıayn_sac import DIAYN_SAC


def load_trained_agent(checkpoint_path, state_dim, action_dim, skill_dim, device='cpu'):
    """
    Load a trained DIAYN agent from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint (.pth file)
        state_dim: State dimension
        action_dim: Action dimension
        skill_dim: Number of skills
        device: Device to load model on
    
    Returns:
        agent: Loaded DIAYN_SAC agent
    """
    print(f"Loading agent from {checkpoint_path}...")
    
    # Create agent with same architecture
    agent = DIAYN_SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        skill_dim=skill_dim,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load network weights
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
    agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
    agent.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Set to evaluation mode
    agent.policy.eval()
    agent.discriminator.eval()
    
    print("✅ Agent loaded successfully!")
    return agent


def run_skill(env, agent, skill, num_episodes=1, max_steps=1000, render=True):
    """
    Run a specific skill in the environment.
    
    Args:
        env: Gymnasium environment
        agent: Trained DIAYN_SAC agent
        skill: Skill index to run
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether environment is rendering
    
    Returns:
        rewards: List of episode rewards
        displacements: List of displacements (for locomotion tasks)
        trajectories: List of state trajectories
    """
    rewards = []
    displacements = []
    trajectories = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        trajectory = [state.copy()]
        
        # Get initial position
        initial_pos = state[0] if len(state) > 0 else 0
        
        for step in range(max_steps):
            # Select action deterministically (no exploration)
            action = agent.select_action(state, skill, deterministic=True)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update statistics
            episode_reward += reward
            trajectory.append(next_state.copy())
            state = next_state
            
            if done:
                break
        
        # Calculate displacement
        final_pos = state[0] if len(state) > 0 else 0
        displacement = final_pos - initial_pos
        
        rewards.append(episode_reward)
        displacements.append(displacement)
        trajectories.append(np.array(trajectory))
        
        if render:
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Displacement = {displacement:.2f}")
    
    return rewards, displacements, trajectories


def visualize_all_skills(env, agent, num_skills, episodes_per_skill=1, max_steps=1000):
    """
    Visualize all learned skills sequentially.
    
    Args:
        env: Gymnasium environment (with render_mode='human')
        agent: Trained DIAYN_SAC agent
        num_skills: Number of skills to visualize
        episodes_per_skill: Episodes to run per skill
        max_steps: Maximum steps per episode
    """
    print("\n" + "=" * 60)
    print("Visualizing All Skills")
    print("=" * 60)
    
    all_rewards = []
    all_displacements = []
    
    for skill in range(num_skills):
        print(f"\n{'='*60}")
        print(f"Skill {skill}")
        print(f"{'='*60}")
        
        rewards, displacements, _ = run_skill(
            env, agent, skill, 
            num_episodes=episodes_per_skill,
            max_steps=max_steps,
            render=True
        )
        
        avg_reward = np.mean(rewards)
        avg_displacement = np.mean(displacements)
        
        all_rewards.append(avg_reward)
        all_displacements.append(avg_displacement)
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Displacement: {avg_displacement:.2f}")
        
        # Wait for user input to continue to next skill
        if skill < num_skills - 1:
            input("\nPress Enter to see next skill...")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary of All Skills")
    print("=" * 60)
    
    for skill in range(num_skills):
        print(f"Skill {skill:2d}: "
              f"Reward = {all_rewards[skill]:8.2f}, "
              f"Displacement = {all_displacements[skill]:8.2f}")
    
    print(f"\nDiversity Metrics:")
    print(f"  Reward std: {np.std(all_rewards):.2f}")
    print(f"  Displacement std: {np.std(all_displacements):.2f}")


def compare_skills(env, agent, skills_to_compare, num_episodes=5, max_steps=1000):
    """
    Compare specific skills side by side.
    
    Args:
        env: Gymnasium environment
        agent: Trained DIAYN_SAC agent
        skills_to_compare: List of skill indices to compare
        num_episodes: Episodes per skill
        max_steps: Maximum steps per episode
    """
    print("\n" + "=" * 60)
    print(f"Comparing Skills: {skills_to_compare}")
    print("=" * 60)
    
    results = {}
    
    for skill in skills_to_compare:
        print(f"\nRunning Skill {skill}...")
        rewards, displacements, trajectories = run_skill(
            env, agent, skill,
            num_episodes=num_episodes,
            max_steps=max_steps,
            render=False
        )
        
        results[skill] = {
            'rewards': rewards,
            'displacements': displacements,
            'trajectories': trajectories,
            'avg_reward': np.mean(rewards),
            'avg_displacement': np.mean(displacements),
            'std_displacement': np.std(displacements)
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"{'Skill':<10} {'Avg Reward':<15} {'Avg Displacement':<20} {'Std Displacement':<20}")
    print("-" * 65)
    
    for skill in skills_to_compare:
        r = results[skill]
        print(f"{skill:<10} {r['avg_reward']:<15.2f} {r['avg_displacement']:<20.2f} {r['std_displacement']:<20.2f}")
    
    return results


def analyze_skill_diversity(env, agent, num_skills, num_episodes=10, max_steps=1000):
    """
    Analyze the diversity of learned skills.
    
    Args:
        env: Gymnasium environment
        agent: Trained DIAYN_SAC agent
        num_skills: Number of skills
        num_episodes: Episodes per skill
        max_steps: Maximum steps per episode
    
    Returns:
        analysis: Dictionary with diversity metrics
    """
    print("\n" + "=" * 60)
    print("Analyzing Skill Diversity")
    print("=" * 60)
    
    skill_data = []
    
    for skill in range(num_skills):
        print(f"Evaluating skill {skill}...")
        rewards, displacements, trajectories = run_skill(
            env, agent, skill,
            num_episodes=num_episodes,
            max_steps=max_steps,
            render=False
        )
        
        skill_data.append({
            'skill': skill,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_displacement': np.mean(displacements),
            'std_displacement': np.std(displacements),
            'min_displacement': np.min(displacements),
            'max_displacement': np.max(displacements)
        })
    
    # Calculate diversity metrics
    all_rewards = [s['avg_reward'] for s in skill_data]
    all_displacements = [s['avg_displacement'] for s in skill_data]
    
    analysis = {
        'reward_mean': np.mean(all_rewards),
        'reward_std': np.std(all_rewards),
        'displacement_mean': np.mean(all_displacements),
        'displacement_std': np.std(all_displacements),
        'displacement_range': np.max(all_displacements) - np.min(all_displacements),
        'skill_data': skill_data
    }
    
    # Print analysis
    print("\n" + "=" * 60)
    print("Diversity Analysis")
    print("=" * 60)
    print(f"\nOverall Statistics:")
    print(f"  Reward Mean: {analysis['reward_mean']:.2f}")
    print(f"  Reward Std: {analysis['reward_std']:.2f}")
    print(f"  Displacement Mean: {analysis['displacement_mean']:.2f}")
    print(f"  Displacement Std: {analysis['displacement_std']:.2f}")
    print(f"  Displacement Range: {analysis['displacement_range']:.2f}")
    
    print(f"\nPer-Skill Statistics:")
    print(f"{'Skill':<8} {'Avg Reward':<12} {'Avg Disp':<12} {'Min Disp':<12} {'Max Disp':<12}")
    print("-" * 60)
    for s in skill_data:
        print(f"{s['skill']:<8} {s['avg_reward']:<12.2f} {s['avg_displacement']:<12.2f} "
              f"{s['min_displacement']:<12.2f} {s['max_displacement']:<12.2f}")
    
    return analysis


def interactive_mode(env, agent, num_skills):
    """
    Interactive mode to test skills manually.
    
    Args:
        env: Gymnasium environment
        agent: Trained DIAYN_SAC agent
        num_skills: Number of available skills
    """
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print(f"Available skills: 0 to {num_skills - 1}")
    print("Commands:")
    print("  - Enter skill number (0-9) to run that skill")
    print("  - 'all' to see all skills")
    print("  - 'quit' or 'q' to exit")
    print("=" * 60)
    
    while True:
        command = input("\nEnter command: ").strip().lower()
        
        if command in ['quit', 'q', 'exit']:
            print("Exiting interactive mode.")
            break
        
        elif command == 'all':
            visualize_all_skills(env, agent, num_skills, episodes_per_skill=1)
        
        elif command.isdigit():
            skill = int(command)
            if 0 <= skill < num_skills:
                print(f"\nRunning Skill {skill}...")
                run_skill(env, agent, skill, num_episodes=1, max_steps=1000, render=True)
            else:
                print(f"Invalid skill number. Must be between 0 and {num_skills - 1}")
        
        else:
            print("Invalid command. Try a skill number, 'all', or 'quit'")


def main():
    """Main function for inference."""
    parser = argparse.ArgumentParser(description='DIAYN Inference Script')
    parser.add_argument('--checkpoint', type=str, default='diayn_agent.pth',
                        help='Path to trained agent checkpoint')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5',
                        help='Environment name')
    parser.add_argument('--num-skills', type=int, default=10,
                        help='Number of skills')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'visualize', 'compare', 'analyze', 'single'],
                        help='Inference mode')
    parser.add_argument('--skill', type=int, default=0,
                        help='Skill to run (for single mode)')
    parser.add_argument('--skills', type=int, nargs='+', default=[0, 1, 2],
                        help='Skills to compare (for compare mode)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes per skill')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    
    args = parser.parse_args()
    
    # Create environment
    render_mode = None if args.no_render else 'human'
    env = gym.make(args.env, render_mode=render_mode)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print("\n" + "=" * 60)
    print("DIAYN Inference")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Number of skills: {args.num_skills}")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    
    # Load trained agent
    agent = load_trained_agent(
        args.checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        skill_dim=args.num_skills,
        device=args.device
    )
    
    # Run selected mode
    if args.mode == 'interactive':
        interactive_mode(env, agent, args.num_skills)
    
    elif args.mode == 'visualize':
        visualize_all_skills(env, agent, args.num_skills, 
                           episodes_per_skill=args.episodes,
                           max_steps=args.max_steps)
    
    elif args.mode == 'compare':
        compare_skills(env, agent, args.skills,
                      num_episodes=args.episodes,
                      max_steps=args.max_steps)
    
    elif args.mode == 'analyze':
        analyze_skill_diversity(env, agent, args.num_skills,
                               num_episodes=args.episodes,
                               max_steps=args.max_steps)
    
    elif args.mode == 'single':
        print(f"\nRunning Skill {args.skill}")
        run_skill(env, agent, args.skill,
                 num_episodes=args.episodes,
                 max_steps=args.max_steps,
                 render=True)
    
    env.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

