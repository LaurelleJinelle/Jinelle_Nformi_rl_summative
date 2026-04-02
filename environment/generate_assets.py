import pygame
import os

# Initialize Pygame
pygame.init()

# Folder to save assets
asset_folder = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(asset_folder, exist_ok=True)

# ------------------------
# Building - factory style
# ------------------------
building_size = (50, 50)
building_surface = pygame.Surface(building_size, pygame.SRCALPHA)
building_surface.fill((0,0,0,0))  # Transparent

# Draw building body
pygame.draw.rect(building_surface, (100, 100, 200), (5, 10, 40, 35))  
# Draw windows
for i in range(2):
    for j in range(3):
        pygame.draw.rect(building_surface, (200, 200, 255), (8 + i*20, 13 + j*10, 8, 8))

pygame.image.save(building_surface, os.path.join(asset_folder, "building.png"))

# ------------------------
# House - small house
# ------------------------
house_size = (30, 30)
house_surface = pygame.Surface(house_size, pygame.SRCALPHA)
house_surface.fill((0,0,0,0))

# Draw house body
pygame.draw.rect(house_surface, (200, 50, 50), (5, 10, 20, 15))
# Draw roof
pygame.draw.polygon(house_surface, (150, 0, 0), [(0,10),(15,0),(30,10)])

pygame.image.save(house_surface, os.path.join(asset_folder, "house.png"))

# ------------------------
# Tree
# ------------------------
tree_size = (20, 20)
tree_surface = pygame.Surface(tree_size, pygame.SRCALPHA)
tree_surface.fill((0,0,0,0))

# Draw trunk
pygame.draw.rect(tree_surface, (101, 67, 33), (8, 12, 4, 8))
# Draw foliage
pygame.draw.circle(tree_surface, (0, 150, 0), (10, 8), 8)

pygame.image.save(tree_surface, os.path.join(asset_folder, "tree.png"))

print("Sprites generated successfully in the 'assets/' folder.")
pygame.quit()