import pygame
import numpy as np
import sys
from bacteria_env import BacteriaEnv
from bacteria_agent import PPOAgent
import torch


class BacteriaVisualizer:
    """Real-time visualization of bacteria behavior"""
    
    def __init__(self, env, agent=None, fps=60):
        self.env = env
        self.agent = agent
        self.fps = fps
        
        # Initialize pygame
        pygame.init()
        self.screen_size = 800
        self.scale = self.screen_size / env.arena_size
        
        self.screen = pygame.display.set_mode((self.screen_size + 200, self.screen_size))
        pygame.display.set_caption("Bacteria Life Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Colors
        self.BG_COLOR = (240, 248, 255)  # Light blue background
        self.BACTERIA_COLOR = (50, 150, 255)  # Blue bacteria
        self.FOOD_COLOR = (50, 255, 100)  # Green food
        self.DANGER_COLOR = (255, 50, 50)  # Red danger
        self.CHEMICAL_FOOD_COLOR = (150, 255, 180, 30)  # Light green with alpha
        self.CHEMICAL_DANGER_COLOR = (255, 150, 150, 30)  # Light red with alpha
        self.WALL_COLOR = (100, 100, 100)
        
    def to_screen_coords(self, pos):
        """Convert environment coordinates to screen coordinates"""
        x = int(pos[0] * self.scale)
        y = int(pos[1] * self.scale)
        return (x, y)
    
    def draw_chemical_gradient(self, surface, positions, color, radius):
        """Draw chemical gradient visualization"""
        for pos in positions:
            screen_pos = self.to_screen_coords(pos)
            screen_radius = int(radius * self.scale)
            
            # Draw gradient circles
            for r in range(screen_radius, 0, -max(1, screen_radius // 5)):
                alpha = int(30 * (r / screen_radius))
                temp_surface = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
                pygame.draw.circle(temp_surface, (*color[:3], alpha), screen_pos, r)
                surface.blit(temp_surface, (0, 0))
    
    def draw_state(self, state, episode_info=None):
        """Draw current environment state"""
        # Fill background
        self.screen.fill(self.BG_COLOR)
        
        # Create arena surface
        arena_surface = pygame.Surface((self.screen_size, self.screen_size))
        arena_surface.fill(self.BG_COLOR)
        
        # Draw chemical gradients
        self.draw_chemical_gradient(
            arena_surface,
            state['food_positions'],
            self.CHEMICAL_FOOD_COLOR,
            150 * self.scale
        )
        self.draw_chemical_gradient(
            arena_surface,
            state['danger_positions'],
            self.CHEMICAL_DANGER_COLOR,
            150 * self.scale
        )
        
        # Draw food
        for food_pos in state['food_positions']:
            screen_pos = self.to_screen_coords(food_pos)
            radius = int(state['food_radius'] * self.scale)
            pygame.draw.circle(arena_surface, self.FOOD_COLOR, screen_pos, radius)
            pygame.draw.circle(arena_surface, (30, 180, 70), screen_pos, radius, 2)
        
        # Draw danger
        for danger_pos in state['danger_positions']:
            screen_pos = self.to_screen_coords(danger_pos)
            radius = int(state['danger_radius'] * self.scale)
            pygame.draw.circle(arena_surface, self.DANGER_COLOR, screen_pos, radius)
            pygame.draw.circle(arena_surface, (200, 30, 30), screen_pos, radius, 2)
        
        # Draw bacteria
        bacteria_screen_pos = self.to_screen_coords(state['bacteria_pos'])
        bacteria_radius = int(state['bacteria_radius'] * self.scale)
        
        # Bacteria body
        pygame.draw.circle(arena_surface, self.BACTERIA_COLOR, bacteria_screen_pos, bacteria_radius)
        pygame.draw.circle(arena_surface, (30, 100, 200), bacteria_screen_pos, bacteria_radius, 2)
        
        # Direction indicator
        angle = state['bacteria_angle']
        direction_end = (
            bacteria_screen_pos[0] + int(np.cos(angle) * bacteria_radius * 1.5),
            bacteria_screen_pos[1] + int(np.sin(angle) * bacteria_radius * 1.5)
        )
        pygame.draw.line(arena_surface, (255, 255, 255), bacteria_screen_pos, direction_end, 3)
        
        # Draw arena border
        pygame.draw.rect(arena_surface, self.WALL_COLOR, 
                        (0, 0, self.screen_size, self.screen_size), 3)
        
        # Blit arena to screen
        self.screen.blit(arena_surface, (0, 0))
        
        # Draw info panel
        self.draw_info_panel(state, episode_info)
        
        pygame.display.flip()
    
    def draw_info_panel(self, state, episode_info):
        """Draw information panel on the right side"""
        panel_x = self.screen_size + 10
        y_offset = 20
        
        # Title
        title = self.font.render("Bacteria Stats", True, (0, 0, 0))
        self.screen.blit(title, (panel_x, y_offset))
        y_offset += 40
        
        # Energy bar
        energy_text = self.small_font.render("Energy:", True, (0, 0, 0))
        self.screen.blit(energy_text, (panel_x, y_offset))
        y_offset += 25
        
        # Energy bar visualization
        bar_width = 160
        bar_height = 20
        energy_pct = state['energy'] / 100.0
        
        # Background
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (panel_x, y_offset, bar_width, bar_height))
        
        # Energy fill (gradient from green to red)
        if energy_pct > 0:
            fill_width = int(bar_width * energy_pct)
            color = (
                int(255 * (1 - energy_pct)),
                int(255 * energy_pct),
                50
            )
            pygame.draw.rect(self.screen, color, 
                           (panel_x, y_offset, fill_width, bar_height))
        
        # Border
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (panel_x, y_offset, bar_width, bar_height), 2)
        
        # Energy value
        energy_val = self.small_font.render(f"{state['energy']:.1f}%", True, (0, 0, 0))
        self.screen.blit(energy_val, (panel_x + bar_width + 5, y_offset))
        y_offset += 40
        
        # Episode info
        if episode_info:
            info_texts = [
                f"Episode: {episode_info.get('episode', 0)}",
                f"Steps: {episode_info.get('steps', 0)}",
                f"Reward: {episode_info.get('total_reward', 0):.1f}",
            ]
            
            for text in info_texts:
                rendered = self.small_font.render(text, True, (0, 0, 0))
                self.screen.blit(rendered, (panel_x, y_offset))
                y_offset += 25
        
        y_offset += 20
        
        # Legend
        legend_title = self.font.render("Legend:", True, (0, 0, 0))
        self.screen.blit(legend_title, (panel_x, y_offset))
        y_offset += 30
        
        legend_items = [
            (self.BACTERIA_COLOR, "Bacteria"),
            (self.FOOD_COLOR, "Food"),
            (self.DANGER_COLOR, "Danger"),
        ]
        
        for color, label in legend_items:
            pygame.draw.circle(self.screen, color, (panel_x + 10, y_offset + 8), 8)
            text = self.small_font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (panel_x + 25, y_offset))
            y_offset += 25
        
        # Controls
        y_offset += 20
        controls_title = self.font.render("Controls:", True, (0, 0, 0))
        self.screen.blit(controls_title, (panel_x, y_offset))
        y_offset += 30
        
        controls = [
            "SPACE - Pause",
            "R - Reset",
            "Q - Quit",
        ]
        
        for control in controls:
            text = self.small_font.render(control, True, (0, 0, 0))
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 22
    
    def run_episode(self, max_steps=5000, manual_control=False):
        """Run one episode with visualization"""
        obs, _ = self.env.reset()
        total_reward = 0
        steps = 0
        paused = False
        
        running = True
        while running and steps < max_steps:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return False
                    elif event.key == pygame.K_r:
                        return True
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
            
            if paused:
                self.clock.tick(self.fps)
                continue
            
            # Select action
            if manual_control:
                # Manual control with arrow keys
                keys = pygame.key.get_pressed()
                thrust = 0.0
                rotation = 0.0
                if keys[pygame.K_UP]:
                    thrust = 1.0
                if keys[pygame.K_DOWN]:
                    thrust = -1.0
                if keys[pygame.K_LEFT]:
                    rotation = -1.0
                if keys[pygame.K_RIGHT]:
                    rotation = 1.0
                action = np.array([thrust, rotation])
            else:
                # Agent control
                if self.agent is not None:
                    action = self.agent.select_action(obs, training=False)
                else:
                    action = self.env.action_space.sample()
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            # Render
            state = self.env.render()
            episode_info = {
                'episode': 0,
                'steps': steps,
                'total_reward': total_reward
            }
            self.draw_state(state, episode_info)
            
            # Check if done
            if done or truncated:
                # Show final state for a moment
                pygame.time.wait(1000)
                running = False
            
            self.clock.tick(self.fps)
        
        return True
    
    def run_continuous(self, manual_control=False):
        """Run continuous episodes"""
        episode = 0
        
        while True:
            episode += 1
            print(f"\nStarting episode {episode}")
            
            should_continue = self.run_episode(manual_control=manual_control)
            
            if not should_continue:
                break
        
        pygame.quit()


def visualize_trained_agent(model_path='./models/best_bacteria_model.pt'):
    """Load and visualize a trained agent"""
    env = BacteriaEnv()
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim)
    
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}, using random agent")
        agent = None
    
    visualizer = BacteriaVisualizer(env, agent, fps=60)
    visualizer.run_continuous(manual_control=False)


def visualize_random():
    """Visualize random agent"""
    env = BacteriaEnv()
    visualizer = BacteriaVisualizer(env, agent=None, fps=60)
    visualizer.run_continuous(manual_control=False)


def manual_control():
    """Play manually with arrow keys"""
    env = BacteriaEnv()
    visualizer = BacteriaVisualizer(env, agent=None, fps=60)
    print("\nManual Control Mode")
    print("Use arrow keys to control the bacteria:")
    print("  UP - Move forward")
    print("  LEFT/RIGHT - Rotate")
    print("  SPACE - Pause")
    print("  R - Reset")
    print("  Q - Quit")
    visualizer.run_continuous(manual_control=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "manual":
            manual_control()
        elif sys.argv[1] == "random":
            visualize_random()
        else:
            visualize_trained_agent(sys.argv[1])
    else:
        # Try to load best model from default location
        visualize_trained_agent('./models/best_bacteria_model.pt')
