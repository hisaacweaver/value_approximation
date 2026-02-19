import numpy as np
from world import World
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
import time


class ApproximateValueIteration:
    def __init__(self, world, pct_anchors=0.1,gamma=0.95, tolerance=1e-4, max_iterations=1000):
        self.world = world
        traversable_count = int(np.sum(world.grid != 0))
        target_anchor_count = max(1, int(round(traversable_count * pct_anchors)))
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.size = world.size
        
        # 1. Setup Rewards
        self.rewards = np.ones((self.size, self.size)) * (-1)
        self.rewards[self.world.get_goal()] = 100
        # Handle sinkholes/walls
        for x, y in zip(*np.where(self.world.grid == -1)): self.rewards[x, y] = -50
        for x, y in zip(*np.where(self.world.grid == 0)): self.rewards[x, y] = 0

        self.anchors = self._generate_anchors(target_anchor_count)
        self.theta = np.zeros(len(self.anchors))

    def _generate_anchors(self, target_anchor_count):
            traversable = [tuple(pos) for pos in np.argwhere(self.world.grid != 0)]
            target_anchor_count = min(len(traversable), max(1, int(target_anchor_count)))

            goal = self.world.get_goal()
            sinkholes = [tuple(pos) for pos in np.argwhere(self.world.grid == -1)]

            required_anchors = []
            seen = set()
            for anchor in [goal] + sinkholes:
                if anchor not in seen:
                    required_anchors.append(anchor)
                    seen.add(anchor)

            if len(required_anchors) >= target_anchor_count:
                return required_anchors

            anchors = required_anchors.copy()
            optional_pool = [candidate for candidate in traversable if candidate not in seen]
            additional_needed = target_anchor_count - len(anchors)

            if additional_needed >= len(optional_pool):
                return anchors + optional_pool

            sampled_indices = np.random.choice(len(optional_pool), size=additional_needed, replace=False)
            for idx in sampled_indices:
                anchors.append(optional_pool[idx])

            return anchors

    def get_approx_value(self, state):
        # We still need this to CALCULATE the updates, even if we return a sparse matrix later
        dists = np.sum((np.array(self.anchors) - np.array(state))**2, axis=1)
        return self.theta[np.argmin(dists)]

    def run(self):
        # Algorithm 8.1 Loop
        start_time = time.time()
        for iteration in range(self.max_iterations):
            new_theta = np.copy(self.theta)
            max_change = 0
            
            for i, (x, y) in enumerate(self.anchors):
                if self.world.grid[x, y] == -1 or (x, y) == self.world.get_goal():
                    new_theta[i] = self.rewards[x, y]
                    continue
                
                # Bellman Backup
                q_values = []
                for action in range(4):
                    next_s = self.world.transition(action, (x, y))
                    v_next = self.get_approx_value(next_s)
                    q_values.append(self.rewards[x, y] + self.gamma * v_next)
                
                new_theta[i] = max(q_values)
                max_change = max(max_change, abs(new_theta[i] - self.theta[i]))
            
            self.theta = new_theta
            if max_change < self.tolerance:
                self.iterations_to_converge = iteration + 1
                self.computation_time = time.time() - start_time
                break
        else:
            self.iterations_to_converge = self.max_iterations
            self.computation_time = time.time() - start_time
        
        sparse_U = np.zeros((self.size, self.size))
        for idx, (x, y) in enumerate(self.anchors):
            sparse_U[x, y] = self.theta[idx]
            
        return sparse_U

    def get_fig(self, grid_values):
        """Plot the value function"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a copy of values for visualization
        plot_values = np.copy(grid_values)
        
        # Set wall values to NaN for better visualization
        wall_positions = np.where(self.world.grid == 0)
        for i in range(len(wall_positions[0])):
            x, y = wall_positions[0][i], wall_positions[1][i]
            plot_values[x, y] = np.nan
        
        # Plot values as heatmap
        im = ax.imshow(plot_values, cmap='RdYlBu', interpolation='nearest')
        
        # Add text annotations
        for x in range(self.size):
            for y in range(self.size):
                if not np.isnan(plot_values[x, y]):
                    text = ax.text(y, x, f'{grid_values[x, y]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        # Plot walls as black squares
        wall_positions = np.where(self.world.grid == 0)
        for i in range(len(wall_positions[0])):
            x, y = wall_positions[0][i], wall_positions[1][i]
            rect = Rectangle((y-0.5, x-0.5), 1, 1, facecolor='black', edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
        
        # Highlight goal and sinkholes
        goal_pos = self.world.get_goal()
        rect = Rectangle((goal_pos[1]-0.5, goal_pos[0]-0.5), 1, 1, 
                        facecolor='none', edgecolor='green', linewidth=3)
        ax.add_patch(rect)
        
        sinkhole_positions = np.where(self.world.grid == -1)
        for i in range(len(sinkhole_positions[0])):
            x, y = sinkhole_positions[0][i], sinkhole_positions[1][i]
            rect = Rectangle((y-0.5, x-0.5), 1, 1, 
                           facecolor='none', edgecolor='red', linewidth=3)
            ax.add_patch(rect)
        
        # Set up the plot
        ax.set_xlim(-0.5, self.size-0.5)
        ax.set_ylim(-0.5, self.size-0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value', rotation=270, labelpad=15)

        plt.tight_layout()

        return fig

    def plot_values(self, grid_values):
        fig = self.get_fig(grid_values)
        plt.show()


def plot_reconstructed_values(world, grid_values, title):
    # This function is kept for backwards compatibility but now uses the class method approach
    fig, ax = plt.subplots(figsize=(10, 10))
    
    plot_values = np.copy(grid_values)
    
    wall_positions = np.where(world.grid == 0)
    for i in range(len(wall_positions[0])):
        x, y = wall_positions[0][i], wall_positions[1][i]
        plot_values[x, y] = np.nan
    
    im = ax.imshow(plot_values, cmap='RdYlBu', interpolation='nearest')
    
    for x in range(world.size):
        for y in range(world.size):
            if not np.isnan(plot_values[x, y]):
                text = ax.text(y, x, f'{grid_values[x, y]:.1f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    for i in range(len(wall_positions[0])):
        x, y = wall_positions[0][i], wall_positions[1][i]
        rect = Rectangle((y-0.5, x-0.5), 1, 1, facecolor='black', edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect)
    
    goal_pos = world.get_goal()
    rect = Rectangle((goal_pos[1]-0.5, goal_pos[0]-0.5), 1, 1, 
                     facecolor='none', edgecolor='green', linewidth=3)
    ax.add_patch(rect)
    
    sinkhole_positions = np.where(world.grid == -1)
    for i in range(len(sinkhole_positions[0])):
        x, y = sinkhole_positions[0][i], sinkhole_positions[1][i]
        rect = Rectangle((y-0.5, x-0.5), 1, 1, 
                         facecolor='none', edgecolor='red', linewidth=3)
        ax.add_patch(rect)
    
    ax.set_xlim(-0.5, world.size-0.5)
    ax.set_ylim(-0.5, world.size-0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    ax.set_xticks(np.arange(-0.5, world.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()

def reconstruct_grid(sparse_U, world_size, method='nearest'):
    anchor_rows, anchor_cols = np.nonzero(sparse_U)
    anchor_points = np.column_stack((anchor_rows, anchor_cols))
    anchor_values = sparse_U[anchor_rows, anchor_cols]
    
    grid_x, grid_y = np.mgrid[0:world_size, 0:world_size]
    
    full_grid = griddata(
        anchor_points, 
        anchor_values, 
        (grid_x, grid_y), 
        method=method,
        fill_value=np.min(anchor_values) 
    )
    
    return full_grid

if __name__ == "__main__":
    print("Creating world...")
    world = World(size=10, pct_walls=0.15, pct_holes=0.08) 
    
    avi = ApproximateValueIteration(world, pct_anchors=0.50, gamma=0.8, tolerance=1e-6)
    sparse_U = avi.run()

    nearest_grid = reconstruct_grid(sparse_U, world.size, method='nearest')
    linear_grid  = reconstruct_grid(sparse_U, world.size, method='linear')

    print("\nPlotting results...")
    # Use the new class method format to match exact_value.py
    avi.plot_values(sparse_U)
    avi.plot_values(nearest_grid)
    avi.plot_values(linear_grid)