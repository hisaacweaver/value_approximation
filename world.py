import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility

class World:
    def __init__(self, size, pct_walls=0.15, pct_holes=0.01):
        self.size = size
        self.grid = np.ones((size, size))
        self.pct_walls = pct_walls
        self.pct_holes = pct_holes
        self.generate_walls()
        self.generate_sinkholes()
        self.goal = np.unravel_index(np.random.choice(self.size * self.size, 1), self.grid.shape)
        self.grid[self.goal] = 1
        self.agent = np.unravel_index(np.random.choice(self.size * self.size, 1), self.grid.shape)

    def generate_walls(self):
        num_walls = int(self.size * self.size * self.pct_walls)
        wall_positions = np.random.choice(self.size * self.size, num_walls, replace=False)
        for pos in wall_positions:
            x, y = divmod(pos, self.size)
            self.grid[x, y] = 0

    def generate_sinkholes(self):
        num_holes = int(self.size * self.size * self.pct_holes)
        hole_positions = np.random.choice(self.size * self.size, num_holes, replace=False)
        for pos in hole_positions:
            x, y = divmod(pos, self.size)
            if self.grid[x, y] != 0:  # Don't place sinkholes on walls
                self.grid[x, y] = -1

    def render(self, agent=False):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot walls as black squares
        wall_positions = np.where(self.grid == 0)
        for i in range(len(wall_positions[0])):
            x, y = wall_positions[0][i], wall_positions[1][i]
            rect = plt.Rectangle((y-0.5, x-0.5), 1, 1, facecolor='black', edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
        
        # Plot sinkholes as red squares
        sinkhole_positions = np.where(self.grid == -1)
        for i in range(len(sinkhole_positions[0])):
            x, y = sinkhole_positions[0][i], sinkhole_positions[1][i]
            rect = plt.Rectangle((y-0.5, x-0.5), 1, 1, facecolor='red', edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
        
        # Plot goal as green square
        rect = plt.Rectangle((self.goal[1]-0.5, self.goal[0]-0.5), 1, 1, facecolor='green', edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect)
        agent_handle = None
        if agent:
            # Plot agent as blue dot
            plt.scatter(self.agent[1], self.agent[0], c='blue', s=200, marker='o', zorder=10, label='Agent')
            agent_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Agent')
        
        # Set up the plot
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis to match array indexing
        
        # Add grid lines
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True, alpha=0.3)
        
        # Create legend
        wall_patch = Patch(facecolor='black', edgecolor='gray', label='Wall')
        goal_patch = Patch(facecolor='green', edgecolor='gray', label='Goal')
        
        handles = [wall_patch, goal_patch]

        if agent:
            handles.append(agent_handle)

        
        if len(sinkhole_positions[0]) > 0:
            sinkhole_patch = Patch(facecolor='red', edgecolor='gray', label='Sinkholes')
            handles.insert(2, sinkhole_patch)  # Insert before agent
        
        plt.legend(handles=handles, loc='upper right')
        plt.title('World Environment')
        plt.show()

    def transition(self, action, state):
        action_space = {
            0: (0, 1),   # Up
            1: (1, 0),   # Right
            2: (0, -1),  # Down
            3: (-1, 0)   # Left
        }
        dx, dy = action_space[action]
        new_position = (state[0] + dy, state[1] + dx)
        if not self.is_wall(new_position):
            return new_position
        return state
    
    def is_wall(self, pos):
        return self.grid[pos] == 0

    def get_goal(self):
        return self.goal

    def get_agent(self):
        return self.agent
    
    def goal_distance(self, state):
        return np.linalg.norm(np.array(state) - np.array(self.goal))

w = World(size=10, pct_walls=0.1, pct_holes=0.05)
w.render()