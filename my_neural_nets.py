import torch
import torch.nn as nn
import math
import os
import numpy



class Policy(nn.Module):


    def __init__(self,input_dim = 17 , skill_dim = 10 , hidden_dim = 256 , action_dim = 6):
        super().__init__()

        self.input_dim = input_dim
        self.skill_dim = skill_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim + skill_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self,observation,skill):
        x = torch.cat([observation,skill],dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        mean = self.mean(x)
        log_std = self.std(x)

        return mean,log_std

    def sample(self,observation,skill):
        

        mean , log_std = self.forward(observation,skill)

        log_std = torch.clamp(log_std,min=-20,max=2)
        std = log_std.exp()
        sample = torch.distributions.Normal(mean,std)

        x = sample.rsample()
        action = torch.tanh(x)

        log_prob = sample.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean)

        return action, log_prob, mean





class Discriminator(nn.Module):


    def __init__(self, observation_dim = 17 , hidden_dim = 256 , skill_dim = 10):
        super().__init__()

        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.skill_dim = skill_dim


        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim,skill_dim)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self,observation):
        
        x = self.relu(self.fc1(observation))
        x = self.relu(self.fc2(x))
        x = self.output(x)
    
        return x


# gives q values of state-action-skill tuples
class Critic(nn.Module):

    def __init__(self,observation_dim = 17 , hidden_dim = 256 , action_dim = 6 , skill_dim = 10):
        super().__init__()

        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim

        self.fc1 = nn.Linear(observation_dim + action_dim + skill_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self,observation,action,skill):
        x = torch.cat([observation,action,skill],dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)
        return x    



    
