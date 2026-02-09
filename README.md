# Bacteria Life Simulation - RL Environment

A 2D artificial life simulation where a bacteria-like agent learns to find food and avoid danger using reinforcement learning.

## Features

- **2D Physics Environment**: Bacteria moves through a water-like medium with realistic damping
- **Chemical Sensors**: 8 directional sensors detect food and danger gradients (like chemotaxis)
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) agent learns survival behavior
- **Real-time Visualization**: Beautiful pygame rendering with chemical gradient visualization
- **Simple Neural Network**: Lightweight MLP (~50K-200K parameters) for fast training

## Installation

```bash
# CPU-only PyTorch (recommended for your system)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Other dependencies
pip install numpy gymnasium pygame
```

Or install everything from requirements:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy - Fast numerical operations
- torch - Neural networks 
- gymnasium - RL environment interface
- pygame - Real-time visualization

## Quick Start

**You don't need to train first!** Start by watching the random agent or trying manual control to see what the bacteria needs to learn.

### 1. Watch Random Agent (No Training Needed!)

```bash
python visualize.py random
```

See how poorly a random agent performs - pure chaos!

### 2. Manual Control Mode (No Training Needed!)

```bash
python visualize.py manual
```

Control the bacteria yourself with arrow keys:
- **UP**: Move forward
- **LEFT/RIGHT**: Rotate
- **SPACE**: Pause
- **R**: Reset episode
- **Q**: Quit

Try to survive and find food - it's harder than it looks!

### 3. Train the Agent

```bash
python train.py
```

This will:
- Train for 1000 episodes (takes ~10-20 minutes on CPU)
- Save checkpoints every 100 episodes
- Save the best model to `best_bacteria_model.pt`
- Print training progress every 10 episodes

Training parameters in `train.py`:
- `num_episodes`: Number of training episodes (default: 1000)
- `batch_size`: Batch size for PPO updates (default: 64)
- `save_interval`: How often to save checkpoints (default: 100)

### 4. Visualize Trained Agent

```bash
python visualize.py
```

This loads the best model and runs continuous episodes with visualization.

### 5. Watch Random Agent Again (After Training)

```bash
python visualize.py random
Compare the trained agent vs random to see how much it learned!

## Environment Details

### Observation Space (20 dimensions)
- 8 food chemical sensors (concentration at each sensor)
- 8 danger chemical sensors (concentration at each sensor)
- 2 velocity components (vx, vy)
- 1 angular velocity
- 1 energy level (normalized 0-1)

### Action Space (2 dimensions)
- Thrust: [-1, 1] - forward/backward movement
- Rotation: [-1, 1] - turning left/right

### Rewards
- **+10**: Eating food (also restores 20 energy)
- **-10**: Touching danger (episode terminates)
- **-5**: Running out of energy (episode terminates)
- **-0.01**: Per timestep (encourages efficiency)

### Physics
- Velocity damping: 0.95 (water resistance)
- Max velocity: 5.0 units/step
- Max thrust: 2.0 units/step²
- Max rotation: 0.3 radians/step
- Energy decay: 0.05 per step

### Chemical Sensing
- Sensors use Gaussian falloff: `concentration = exp(-(distance²) / (2σ²))`
- Diffusion range: 150 units
- 8 sensors evenly spaced around the bacteria body

## File Structure

```
bacteria_env.py      - Gymnasium environment implementation
bacteria_agent.py    - PPO agent and neural network
train.py             - Training script
visualize.py         - Visualization and rendering
requirements.txt     - Python dependencies
```

## Neural Network Architecture

### Actor (Policy Network)
```
Input (20) → Linear(128) → ReLU → 
Linear(128) → ReLU → 
Linear(128) → ReLU → 
Linear(2) → Tanh → Output (2)
```

### Critic (Value Network)
```
Input (20) → Linear(128) → ReLU → 
Linear(128) → ReLU → 
Linear(1) → Output (value)
```

Total parameters: ~50K-200K depending on hidden size

## Training Tips

1. **Monitor exploration**: The action std should decay from 0.5 to 0.1 during training
2. **Check rewards**: Average reward should increase from negative to positive values
3. **Episode length**: Successful agents survive longer (closer to max_steps=500)
4. **Best model**: Saved when average reward (over last 10 episodes) improves

## Customization

### Change Environment Parameters

Edit `bacteria_env.py`:
```python
env = BacteriaEnv(
    arena_size=500,      # Arena dimensions
    max_steps=500,       # Episode length
    num_food=5,          # Number of food sources
    num_danger=3         # Number of danger sources
)
```

### Adjust Agent Hyperparameters

Edit `bacteria_agent.py`:
```python
agent = PPOAgent(
    obs_dim=20,
    action_dim=2,
    lr=3e-4,             # Learning rate
    gamma=0.99,          # Discount factor
    clip_epsilon=0.2,    # PPO clipping
    hidden_size=128      # Network size
)
```

### Visualization Settings

Edit `visualize.py`:
```python
visualizer = BacteriaVisualizer(
    env, 
    agent, 
    fps=60  # Frames per second
)
```

## Next Steps

Once you have a working agent, you can:

1. **Scale up to transformer**: Replace MLP with a small transformer architecture
2. **Add 3D swimming worm**: Implement segmented body with coordinated movement
3. **Multi-agent environment**: Add multiple bacteria (competition/cooperation)
4. **Evolution**: Use genetic algorithms to evolve both network and morphology
5. **More complex sensors**: Add vision (raycasting), hearing, or temperature
6. **Richer behaviors**: Reproduction, metabolism, predator-prey dynamics

## Troubleshooting

**Training is slow**: Reduce batch_size or num_episodes
**Agent doesn't learn**: Try increasing learning rate or training longer
**Pygame window doesn't open**: Make sure you have display/X11 available
**Import errors**: Run `pip install -r requirements.txt`

## Performance

On AMD AI Max 395 (128GB unified memory):
- Training: ~5-10 minutes for 1000 episodes (excellent CPU performance)
- Inference: 60 FPS visualization with trained agent
- Memory: <500MB (plenty of room to scale up!)


Enjoy your artificial life! 🦠
