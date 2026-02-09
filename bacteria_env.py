import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BacteriaEnv(gym.Env):
    """
    A 2D environment where a bacteria-like agent learns to find food and avoid danger.
    """
    
    def __init__(self, arena_size=500, max_steps=5000, num_food=5, num_danger=3):
        super().__init__()
        
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.num_food = num_food
        self.num_danger = num_danger
        
        # Bacteria properties
        self.bacteria_radius = 10
        self.max_velocity = 5.0
        self.max_thrust = 2.0
        self.max_rotation = 0.3
        
        # Object properties
        self.food_radius = 8
        self.danger_radius = 12
        self.diffusion_range = 150  # How far chemicals diffuse
        
        # Sensor configuration (8 sensors around the bacteria)
        self.num_sensors = 8
        self.sensor_angles = np.linspace(0, 2*np.pi, self.num_sensors, endpoint=False)
        
        # Action space: [thrust, rotation]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: 
        # - 8 food chemical sensors
        # - 8 danger chemical sensors  
        # - velocity_x, velocity_y
        # - angular_velocity
        # - energy_level
        obs_dim = self.num_sensors * 2 + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize bacteria at random position
        self.bacteria_pos = np.random.uniform(50, self.arena_size - 50, 2)
        self.bacteria_angle = np.random.uniform(0, 2*np.pi)
        self.bacteria_vel = np.zeros(2)
        self.bacteria_angular_vel = 0.0
        self.energy = 100.0
        
        # Spawn food
        self.food_positions = []
        for _ in range(self.num_food):
            pos = np.random.uniform(20, self.arena_size - 20, 2)
            self.food_positions.append(pos)
        
        # Spawn danger
        self.danger_positions = []
        for _ in range(self.num_danger):
            pos = np.random.uniform(20, self.arena_size - 20, 2)
            self.danger_positions.append(pos)
        
        self.steps = 0
        self.total_reward = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        obs = []
        
        # Food chemical sensors (8 directional sensors)
        food_sensors = self._get_chemical_sensors(self.food_positions)
        obs.extend(food_sensors)
        
        # Danger chemical sensors (8 directional sensors)
        danger_sensors = self._get_chemical_sensors(self.danger_positions)
        obs.extend(danger_sensors)
        
        # Velocity
        obs.extend(self.bacteria_vel.tolist())
        
        # Angular velocity
        obs.append(self.bacteria_angular_vel)
        
        # Energy level (normalized)
        obs.append(self.energy / 100.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_chemical_sensors(self, source_positions):
        """
        Calculate chemical concentration at each sensor position.
        Uses Gaussian falloff from sources.
        """
        sensors = []
        
        for sensor_angle in self.sensor_angles:
            # Sensor position (at bacteria surface)
            absolute_angle = self.bacteria_angle + sensor_angle
            sensor_offset = np.array([
                np.cos(absolute_angle),
                np.sin(absolute_angle)
            ]) * self.bacteria_radius
            sensor_pos = self.bacteria_pos + sensor_offset
            
            # Sum concentration from all sources
            concentration = 0.0
            for source_pos in source_positions:
                distance = np.linalg.norm(sensor_pos - source_pos)
                # Gaussian falloff
                concentration += np.exp(-(distance**2) / (2 * (self.diffusion_range/3)**2))
            
            sensors.append(concentration)
        
        return sensors
    
    def step(self, action):
        # Parse action
        thrust = np.clip(action[0], -1, 1) * self.max_thrust
        rotation = np.clip(action[1], -1, 1) * self.max_rotation
        
        # Update angular velocity and angle
        self.bacteria_angular_vel = rotation
        self.bacteria_angle += self.bacteria_angular_vel
        self.bacteria_angle = self.bacteria_angle % (2 * np.pi)
        
        # Apply thrust in current direction
        thrust_vec = np.array([
            np.cos(self.bacteria_angle),
            np.sin(self.bacteria_angle)
        ]) * thrust
        
        # Update velocity with damping
        self.bacteria_vel += thrust_vec
        self.bacteria_vel *= 0.95  # Damping (water resistance)
        
        # Limit velocity
        speed = np.linalg.norm(self.bacteria_vel)
        if speed > self.max_velocity:
            self.bacteria_vel = self.bacteria_vel / speed * self.max_velocity
        
        # Update position
        self.bacteria_pos += self.bacteria_vel
        
        # Boundary handling (bounce off walls)
        if self.bacteria_pos[0] < 0 or self.bacteria_pos[0] > self.arena_size:
            self.bacteria_vel[0] *= -0.5
            self.bacteria_pos[0] = np.clip(self.bacteria_pos[0], 0, self.arena_size)
        
        if self.bacteria_pos[1] < 0 or self.bacteria_pos[1] > self.arena_size:
            self.bacteria_vel[1] *= -0.5
            self.bacteria_pos[1] = np.clip(self.bacteria_pos[1], 0, self.arena_size)
        
        # Energy decay
        self.energy -= 0.05
        
        # Check collisions and calculate reward
        # Base time penalty (more significant now to encourage efficiency)
        reward = -0.1  # Increased from -0.01 for better efficiency incentives
        
        # Track distances for proximity rewards
        nearest_food_dist = float('inf')
        nearest_danger_dist = float('inf')
        
        terminated = False
        
        # Check food collision
        for i, food_pos in enumerate(self.food_positions):
            distance = np.linalg.norm(self.bacteria_pos - food_pos)
            if distance < (self.bacteria_radius + self.food_radius):                
                reward += 100.0
                self.energy = min(100.0, self.energy + 20.0)
                # Respawn food
                self.food_positions[i] = np.random.uniform(20, self.arena_size - 20, 2)
            else:
                nearest_food_dist = min(nearest_food_dist, distance)
        
        # Check danger collision (with stronger penalty as proposed) 
        for danger_pos in self.danger_positions:
            distance = np.linalg.norm(self.bacteria_pos - danger_pos)
            if distance < (self.bacteria_radius + self.danger_radius):
                reward = -50.0  # Stronger penalty as suggested
                terminated = True
            else:
                nearest_danger_dist = min(nearest_danger_dist, distance)
        
        # Add proximity-based rewards for better guidance during learning  
        if nearest_food_dist < float('inf') and nearest_food_dist > 0:
            # Reward based on how close to food (inverse of distance)
            reward += max(0.1, 5.0 / (nearest_food_dist + 1.0))
            
        if nearest_danger_dist < float('inf') and nearest_danger_dist > 0:
            # Penalty based on danger proximity (inverse of distance)  
            reward -= max(0.1, 3.0 / (nearest_danger_dist + 1.0))
        
        # Check energy death
        if self.energy <= 0:
            reward -= 5.0
            terminated = True
        
        self.steps += 1
        self.total_reward += reward
        
        # Episode timeout
        truncated = self.steps >= self.max_steps
        
        obs = self._get_observation()
        info = {
            'total_reward': self.total_reward,
            'energy': self.energy,
            'position': self.bacteria_pos.copy()
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Returns current state for visualization"""
        return {
            'bacteria_pos': self.bacteria_pos.copy(),
            'bacteria_angle': self.bacteria_angle,
            'bacteria_radius': self.bacteria_radius,
            'food_positions': [pos.copy() for pos in self.food_positions],
            'danger_positions': [pos.copy() for pos in self.danger_positions],
            'food_radius': self.food_radius,
            'danger_radius': self.danger_radius,
            'energy': self.energy,
            'arena_size': self.arena_size
        }
