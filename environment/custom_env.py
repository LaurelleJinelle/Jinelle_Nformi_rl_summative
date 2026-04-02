import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import os

class EnergyGridEnv(gym.Env):
    """
    Smart Energy Grid Environment
    5 regions visualized as a mini-city map with buildings, trees, roads
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=50):
        super().__init__()
        self.num_regions = 5
        self.max_steps = max_steps
        self.total_supply = 250

        # Priority weights for regions
        self.priority = np.array([2.0, 1.0, 1.5, 1.2, 2.5], dtype=np.float32)

        # Actions: increase allocation for region 0..4, or rebalance evenly (5)
        self.action_space = spaces.Discrete(self.num_regions + 1)

        # Observation: demand + allocation + priority + total_supply + timestep
        obs_size = self.num_regions * 3 + 2
        low = np.zeros(obs_size, dtype=np.float32)
        high = np.array([200]*(self.num_regions*2) + list(self.priority) + [self.total_supply, self.max_steps], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Pygame visualization setup
        self.screen_width = 800
        self.screen_height = 500
        self.region_width = 140
        self.region_height = 140
        self.margin = 40

        self.render_initialized = False
        self.reset()

    # ----------------------------------
    # Reset
    # ----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.demand = np.random.randint(20, 80, size=self.num_regions).astype(np.float32)
        self.allocation = np.ones(self.num_regions) * (self.total_supply / self.num_regions)
        return self._get_obs(), {}

    # ----------------------------------
    # Step
    # ----------------------------------
    def step(self, action):
        self.current_step += 1
        self._apply_action(action)
        self._update_demand()
        reward = self._calculate_reward()
        terminated = False
        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    # ----------------------------------
    # Apply action
    # ----------------------------------
    def _apply_action(self, action):
        adjustment = self.total_supply * 0.1
        if action < self.num_regions:
            self.allocation[action] += adjustment
            for i in range(self.num_regions):
                if i != action:
                    self.allocation[i] -= adjustment / (self.num_regions-1)
        else:
            self.allocation = np.ones(self.num_regions) * (self.total_supply / self.num_regions)

        self.allocation = np.clip(self.allocation, 0, self.total_supply)
        total_alloc = np.sum(self.allocation)
        if total_alloc > self.total_supply:
            self.allocation = (self.allocation / total_alloc) * self.total_supply

    # ----------------------------------
    # Update demand
    # ----------------------------------
    def _update_demand(self):
        fluctuation = np.random.randint(-5,6, size=self.num_regions)
        self.demand += fluctuation
        self.demand = np.clip(self.demand, 10, 150)

        # Random spikes
        if np.random.rand() < 0.1:
            spike_region = np.random.randint(0,self.num_regions)
            self.demand[spike_region] += 30

    # ----------------------------------
    # Reward
    # ----------------------------------
    def _calculate_reward(self):
        reward = 0
        served = np.minimum(self.allocation, self.demand)
        reward += np.sum(served)
        shortage = np.maximum(0, self.demand - self.allocation)
        reward -= np.sum(shortage * 2)
        waste = np.maximum(0, self.allocation - self.demand)
        reward -= np.sum(waste)
        reward += np.sum(served * self.priority * 0.5)
        fluctuation_penalty = np.sum(np.abs(self.allocation - self.demand)) * 0.05
        reward -= fluctuation_penalty
        return float(reward)

    # ----------------------------------
    # Observation
    # ----------------------------------
    def _get_obs(self):
        return np.concatenate([
            self.demand,
            self.allocation,
            self.priority,
            [self.total_supply, self.current_step]
        ]).astype(np.float32)

    # ----------------------------------
    # Render with mini-city map
    # ----------------------------------
    def render(self):
        if not self.render_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Smart Energy Grid Mini-City")
            self.font = pygame.font.SysFont(None, 24)
            self.render_initialized = True

            # Load sprites
            asset_path = os.path.join(os.path.dirname(__file__), "assets")
            self.building_img = pygame.image.load(os.path.join(asset_path, "building.png")).convert_alpha()
            self.house_img = pygame.image.load(os.path.join(asset_path, "house.png")).convert_alpha()
            self.tree_img = pygame.image.load(os.path.join(asset_path, "tree.png")).convert_alpha()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((100, 200, 100))  # grass

        # Roads
        pygame.draw.rect(self.screen, (50,50,50), (0, self.screen_height//2 - 20, self.screen_width, 40))
        pygame.draw.rect(self.screen, (50,50,50), (self.screen_width//2 - 20, 0, 40, self.screen_height))

        # Region positions
        region_positions = [
            (self.margin, self.margin),
            (self.screen_width - self.region_width - self.margin, self.margin),
            (self.margin, self.screen_height - self.region_height - self.margin),
            (self.screen_width - self.region_width - self.margin, self.screen_height - self.region_height - self.margin),
            (self.screen_width//2 - self.region_width//2, self.screen_height//2 - self.region_height//2)
        ]

        for i, (x, y) in enumerate(region_positions):
            # Base rectangle
            pygame.draw.rect(self.screen, (200, 200, 200), (x, y, self.region_width, self.region_height))

            # Allocation overlay
            alloc_height = int((self.allocation[i]/150)*self.region_height)
            pygame.draw.rect(self.screen, (50, 200, 50, 128), (x, y + self.region_height - alloc_height, self.region_width, alloc_height))

            # Unmet demand overlay
            shortage = max(self.demand[i] - self.allocation[i], 0)
            shortage_height = int((shortage/150)*self.region_height)
            pygame.draw.rect(self.screen, (200, 50, 50, 128), (x, y + self.region_height - shortage_height, self.region_width, shortage_height))

            # Draw sprites
            self.screen.blit(pygame.transform.scale(self.building_img, (50,50)), (x + 10, y + 20))
            self.screen.blit(pygame.transform.scale(self.house_img, (30,30)), (x + 80, y + 20))
            self.screen.blit(pygame.transform.scale(self.tree_img, (20,20)), (x + 30, y + 100))
            self.screen.blit(pygame.transform.scale(self.tree_img, (20,20)), (x + 100, y + 100))

            # Priority text
            text = self.font.render(f"P:{self.priority[i]}", True, (0,0,0))
            self.screen.blit(text, (x + 5, y + self.region_height - 20))

            # Region label
            label = self.font.render(f"R{i}", True, (0,0,0))
            self.screen.blit(label, (x + self.region_width//2 - 10, y + 5))

        # Step info
        step_text = self.font.render(f"Step: {self.current_step}", True, (0,0,0))
        self.screen.blit(step_text, (10, 10))

        pygame.display.flip()

    # ----------------------------------
    # Close
    # ----------------------------------
    def close(self):
        if self.render_initialized:
            pygame.quit()