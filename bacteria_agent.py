import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class BacteriaNet(nn.Module):
    """Simple feedforward network for bacteria control"""
    
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Actions are in [-1, 1]
        )
    
    def forward(self, x):
        return self.network(x)


class PPOAgent:
    """Simplified PPO agent for bacteria training"""
    
    def __init__(self, obs_dim, action_dim, lr=1e-4, gamma=0.99, 
                 clip_epsilon=0.1, value_coef=0.8, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Actor network
        self.actor = BacteriaNet(obs_dim, action_dim)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Separate optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # For logging action std (exploration)
        self.action_std = 0.8
        self.min_action_std = 0.05
        self.action_std_decay = 0.9995
    
    def select_action(self, obs, training=True):
        """Select action with exploration noise during training"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action_mean = self.actor(obs_tensor).squeeze(0).numpy()
        
        if training:
            # Add Gaussian noise for exploration
            action = action_mean + np.random.normal(0, self.action_std, size=action_mean.shape)
            action = np.clip(action, -1, 1)
        else:
            action = action_mean
        
        return action
    
    def train_step(self, trajectories):
        """
        Train on a batch of trajectories.
        trajectories: list of (obs, action, reward, next_obs, done)
        """
        if len(trajectories) == 0:
            return {}
        
        # Convert to tensors (using numpy arrays for efficiency)
        obs = torch.FloatTensor(np.array([t[0] for t in trajectories]))
        actions = torch.FloatTensor(np.array([t[1] for t in trajectories]))
        rewards = torch.FloatTensor(np.array([t[2] for t in trajectories]))
        next_obs = torch.FloatTensor(np.array([t[3] for t in trajectories]))
        dones = torch.FloatTensor(np.array([t[4] for t in trajectories]))
        
        # Compute returns and advantages
        with torch.no_grad():
            values = self.critic(obs).squeeze()
            next_values = self.critic(next_obs).squeeze()
            
            # TD target
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Old action log probs (for PPO clipping)
            old_action_logprobs = -0.5 * ((actions - self.actor(obs))**2).sum(dim=1)
        
        # Initialize losses for return
        actor_loss = None
        critic_loss = None
        
        # PPO update for multiple epochs
        for _ in range(8):  # Increase epochs from 4 to 8 for better training
            # Current policy
            action_means = self.actor(obs)
            action_logprobs = -0.5 * ((actions - action_means)**2).sum(dim=1)
            
            # Ratio for PPO
            ratios = torch.exp(action_logprobs - old_action_logprobs)
            
            # Surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            
            # Actor loss (PPO clipped objective)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus (encourages exploration)
            entropy = 0.5 * torch.log(torch.tensor(2 * np.pi * np.e * self.action_std**2))
            actor_loss -= self.entropy_coef * entropy
            
            # Critic loss (value function)
            current_values = self.critic(obs).squeeze()
            critic_loss = nn.MSELoss()(current_values, td_targets)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Decay exploration noise
        self.action_std = max(self.min_action_std, self.action_std * self.action_std_decay)
        
        return {
            'actor_loss': actor_loss.item() if actor_loss is not None else 0,
            'critic_loss': critic_loss.item() if critic_loss is not None else 0,
            'mean_advantage': advantages.mean().item(),
            'action_std': self.action_std
        }
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'action_std': self.action_std
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.action_std = checkpoint['action_std']
