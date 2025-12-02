from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    Stores transitions and samples random batches for training.
    """
    
    def __init__(self, capacity=1000000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, skill):
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received (pseudo-reward for DIAYN)
            next_state: Next state
            done: Whether episode ended
            skill: Skill index used
        """
        self.buffer.append((state, action, reward, next_state, done, skill))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones, skills)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, skills = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(skills)
        )
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)
