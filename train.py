import numpy as np
from bacteria_env import BacteriaEnv
from bacteria_agent import PPOAgent
import json
import os


def train_bacteria(num_episodes=1000, batch_size=128, save_interval=100, save_dir='./models'):
    """Train the bacteria agent"""
    
    # Create save directory if it doesn't exist
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = BacteriaEnv(arena_size=500, max_steps=5000, num_food=10, num_danger=3)
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim, lr=1e-4)
    
    # Training storage
    trajectories = []
    episode_rewards = []
    episode_lengths = []
    
    # Statistics
    best_reward = -float('inf')
    
    print("Starting training...")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Episodes: {num_episodes}, Batch size: {batch_size}")
    print("-" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(obs, training=True)
            
            # Environment step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store trajectory
            trajectories.append((obs, action, reward, next_obs, float(done)))
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Train when batch is full
        if len(trajectories) >= batch_size:
            train_info = agent.train_step(trajectories)
            trajectories = []
        
# Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Length (last 10): {avg_length:.1f}")
            print(f"  Last Episode Reward: {episode_reward:.2f}")
            print(f"  Exploration Std: {agent.action_std:.3f}")
            print("-" * 60)
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(os.path.join(save_dir, 'best_bacteria_model.pt'))
                print(f"  ✓ New best model saved! Reward: {best_reward:.2f}")
                print("-" * 60)
        
        # Regular checkpoints
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(save_dir, f'bacteria_model_ep{episode+1}.pt'))
            
            # Save training stats
            stats = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'best_reward': best_reward
            }
            with open(os.path.join(save_dir, f'training_stats_ep{episode+1}.json'), 'w') as f:
                json.dump(stats, f)
    
    # Final save
    agent.save(os.path.join(save_dir, 'final_bacteria_model.pt'))
    
    print("\nTraining complete!")
    print(f"Best average reward: {best_reward:.2f}")
    print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Models saved to: {save_dir}/")
    
    return agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    agent, rewards, lengths = train_bacteria(num_episodes=1000, batch_size=128)
