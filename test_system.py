#!/usr/bin/env python3
"""
Quick test script to verify the bacteria simulation works correctly.
Run this before starting training to catch any issues early.
"""

import numpy as np
import sys

def test_environment():
    """Test that the environment works correctly"""
    print("Testing BacteriaEnv...")
    
    try:
        from bacteria_env import BacteriaEnv
        
        env = BacteriaEnv(arena_size=500, max_steps=100, num_food=3, num_danger=2)
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (20,), f"Expected obs shape (20,), got {obs.shape}"
        print(f"  ✓ Reset successful, obs shape: {obs.shape}")
        
        # Test random actions
        total_reward = 0
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                print(f"  ✓ Episode ended at step {step}, total reward: {total_reward:.2f}")
                break
        
        # Test render
        state = env.render()
        assert 'bacteria_pos' in state
        assert 'food_positions' in state
        print(f"  ✓ Render successful")
        
        print("  ✓ Environment test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}\n")
        return False


def test_agent():
    """Test that the agent works correctly"""
    print("Testing BacteriaAgent...")
    
    try:
        from bacteria_agent import PPOAgent
        import torch
        
        obs_dim = 20
        action_dim = 2
        
        agent = PPOAgent(obs_dim, action_dim)
        print(f"  ✓ Agent created")
        
        # Test action selection
        dummy_obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.select_action(dummy_obs, training=True)
        assert action.shape == (action_dim,), f"Expected action shape (2,), got {action.shape}"
        assert np.all(action >= -1) and np.all(action <= 1), "Actions should be in [-1, 1]"
        print(f"  ✓ Action selection works, action: {action}")
        
        # Test training step
        trajectories = []
        for _ in range(32):
            obs = np.random.randn(obs_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_obs = np.random.randn(obs_dim).astype(np.float32)
            done = np.random.rand() > 0.9
            trajectories.append((obs, action, reward, next_obs, done))
        
        train_info = agent.train_step(trajectories)
        assert 'actor_loss' in train_info
        assert 'critic_loss' in train_info
        print(f"  ✓ Training step works")
        print(f"    Actor loss: {train_info['actor_loss']:.4f}")
        print(f"    Critic loss: {train_info['critic_loss']:.4f}")
        
        # Test save/load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        agent.save(temp_path)
        print(f"  ✓ Model saved")
        
        agent2 = PPOAgent(obs_dim, action_dim)
        agent2.load(temp_path)
        print(f"  ✓ Model loaded")
        
        import os
        os.remove(temp_path)
        
        print("  ✓ Agent test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Agent test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test environment + agent together"""
    print("Testing Integration (Env + Agent)...")
    
    try:
        from bacteria_env import BacteriaEnv
        from bacteria_agent import PPOAgent
        
        env = BacteriaEnv(arena_size=500, max_steps=100)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPOAgent(obs_dim, action_dim)
        
        # Run one episode
        obs, _ = env.reset()
        trajectories = []
        
        for step in range(50):
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            trajectories.append((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            
            if done or truncated:
                break
        
        print(f"  ✓ Episode completed: {len(trajectories)} steps")
        
        # Train on trajectories
        if len(trajectories) >= 16:
            train_info = agent.train_step(trajectories)
            print(f"  ✓ Training successful")
        
        print("  ✓ Integration test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check that all required packages are installed"""
    print("Checking dependencies...")
    
    required = {
        'numpy': 'numpy',
        'torch': 'torch',
        'gymnasium': 'gymnasium',
        'pygame': 'pygame'
    }
    
    all_good = True
    for package_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            all_good = False
    
    print()
    
    if not all_good:
        print("Missing dependencies! Install with:")
        print("  pip install -r requirements.txt\n")
    
    return all_good


def main():
    print("=" * 60)
    print("Bacteria Simulation - System Test")
    print("=" * 60)
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies before continuing.")
        return False
    
    # Run tests
    tests = [
        ("Environment", test_environment),
        ("Agent", test_agent),
        ("Integration", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20s} {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to train.\n")
        print("Next steps:")
        print("  1. Train the agent:    python train.py")
        print("  2. Visualize results:  python visualize.py")
        print("  3. Manual control:     python visualize.py manual")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
