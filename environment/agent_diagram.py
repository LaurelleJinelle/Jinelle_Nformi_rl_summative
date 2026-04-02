import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Example data (you can replace with actual env state)
regions = {
    0: {"demand": 70, "allocation": 60, "priority": 3},
    1: {"demand": 50, "allocation": 55, "priority": 1},
    2: {"demand": 65, "allocation": 65, "priority": 2},
    3: {"demand": 80, "allocation": 70, "priority": 3},
    4: {"demand": 60, "allocation": 50, "priority": 2},
}

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Agent position
agent_x, agent_y = 5, 9
ax.plot(agent_x, agent_y, 'o', markersize=20, color='blue')
ax.text(agent_x, agent_y+0.3, 'Agent', ha='center', fontsize=12)

# Region positions
positions = {
    0: (2, 7),
    1: (7, 7),
    2: (2, 3),
    3: (7, 3),
    4: (5, 5)
}

for idx, pos in positions.items():
    x, y = pos
    r = regions[idx]
    
    # Draw region rectangle
    rect = patches.Rectangle((x-1, y-1), 2, 2, linewidth=1, edgecolor='black', facecolor='lightgray')
    ax.add_patch(rect)
    
    # Draw allocation arrow from agent
    ax.annotate(
        '', 
        xy=(x, y), 
        xytext=(agent_x, agent_y), 
        arrowprops=dict(facecolor='green' if r['allocation']>=r['demand'] else 'red', shrink=0.05)
    )
    
    # Labels
    ax.text(x, y+0.5, f"R{idx}", ha='center', fontsize=10, fontweight='bold')
    ax.text(x, y, f"D:{r['demand']}", ha='center', fontsize=9)
    ax.text(x, y-0.3, f"A:{r['allocation']}", ha='center', fontsize=9)
    ax.text(x, y-0.6, f"P:{r['priority']}", ha='center', fontsize=9)

plt.title("Agent Allocating Energy to 5 Regions")
plt.show()