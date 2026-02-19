import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from world import World

np.random.seed(42)  # For reproducibility

class ValueIteration:
    def __init__(self, world, gamma=0.95, tolerance=1e-6, max_iterations=1000, reward_values=[100,-50,-1]):
        """
        Initialize Value Iteration algorithm
        
        Args:
            world: World object containing the grid environment
            gamma: Discount factor
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
        """
        self.world = world
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.size = world.size
        self.reward_values = reward_values
        
        # Initialize value function
        self.values = np.zeros((self.size, self.size))
        
        # Set rewards
        self.set_rewards()
        
        # Action space: 0=Up, 1=Right, 2=Down, 3=Left
        self.actions = [0, 1, 2, 3]
        
    def set_rewards(self):
        """Set reward values for different cell types"""
        self.rewards = np.ones((self.size, self.size)) * (self.reward_values[2])  # Step cost of -1
        
        # Set goal reward
        goal_pos = self.world.get_goal()
        self.rewards[goal_pos] = self.reward_values[0] 
        
        # Set sinkhole rewards
        sinkhole_positions = np.where(self.world.grid == -1)
        for i in range(len(sinkhole_positions[0])):
            x, y = sinkhole_positions[0][i], sinkhole_positions[1][i]
            self.rewards[x, y] = self.reward_values[1]
        
        # Walls have no reward (inaccessible)
        wall_positions = np.where(self.world.grid == 0)
        for i in range(len(wall_positions[0])):
            x, y = wall_positions[0][i], wall_positions[1][i]
            self.rewards[x, y] = 0
    
    def is_valid_state(self, state):
        """Check if state is within bounds and not a wall"""
        x, y = state
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self.world.grid[x, y] != 0  # Not a wall
    
    def get_next_states(self, state, action):
        """Get possible next states given current state and action"""
        next_state = self.world.transition(action, state)
        return [(next_state, 1.0)]  # Deterministic transitions
    
    def value_iteration(self):
        """Run value iteration algorithm"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            new_values = np.copy(self.values)
            max_change = 0
            
            for x in range(self.size):
                for y in range(self.size):
                    state = (x, y)
                    
                    # Skip walls
                    if not self.is_valid_state(state):
                        continue
                    
                    # For terminal states (goal or sinkholes), value is just the reward
                    if self.world.grid[x, y] == -1 or state == self.world.get_goal():
                        new_values[x, y] = self.rewards[x, y]
                        continue
                    
                    # Calculate Q-values for all actions
                    q_values = []
                    for action in self.actions:
                        q_value = 0
                        next_states = self.get_next_states(state, action)
                        
                        for next_state, prob in next_states:
                            reward = self.rewards[x, y]  # Immediate reward for current state
                            future_value = self.values[next_state[0], next_state[1]]
                            q_value += prob * (reward + self.gamma * future_value)
                        
                        q_values.append(q_value)
                    
                    # Take the maximum Q-value
                    new_values[x, y] = max(q_values)
                    
                    # Track maximum change for convergence
                    change = abs(new_values[x, y] - self.values[x, y])
                    max_change = max(max_change, change)
            
            self.values = new_values
            
            # Check for convergence
            if max_change < self.tolerance:
                end_time = time.time()
                computation_time = end_time - start_time
                iterations_to_converge = iteration + 1
                
                # print(f"Converged after {iterations_to_converge} iterations")
                # print(f"Total computation time: {computation_time:.4f} seconds")
                # print(f"Time per iteration: {computation_time/iterations_to_converge:.4f} seconds")
                
                # Store metrics for later access
                self.iterations_to_converge = iterations_to_converge
                self.computation_time = computation_time
                
                return self.values
                
            if (iteration + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                # print(f"Iteration {iteration + 1}, max change: {max_change:.6f}, elapsed time: {elapsed_time:.4f}s")
        
        # If we reach here, we didn't converge
        end_time = time.time()
        computation_time = end_time - start_time
        # print(f"Did not converge after {self.max_iterations} iterations")
        # print(f"Total computation time: {computation_time:.4f} seconds")
        
        self.iterations_to_converge = self.max_iterations
        self.computation_time = computation_time
        
        return self.values
    
    def extract_policy(self):
        """Extract optimal policy from value function"""
        policy = np.zeros((self.size, self.size), dtype=int)
        
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                
                # Skip walls and terminal states
                if not self.is_valid_state(state) or self.world.grid[x, y] == -1 or state == self.world.get_goal():
                    continue
                
                # Find best action
                best_action = 0
                best_value = float('-inf')
                
                for action in self.actions:
                    q_value = 0
                    next_states = self.get_next_states(state, action)
                    
                    for next_state, prob in next_states:
                        reward = self.rewards[x, y]
                        future_value = self.values[next_state[0], next_state[1]]
                        q_value += prob * (reward + self.gamma * future_value)
                    
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                policy[x, y] = best_action
        
        return policy
    
    def get_fig(self):
        """Plot the value function"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a copy of values for visualization
        plot_values = np.copy(self.values)
        
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
                    text = ax.text(y, x, f'{self.values[x, y]:.1f}',
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

    def plot_values(self):
        fig = self.get_fig()
        plt.show()
    
def run_value_iteration_example():
    """Run value iteration on a sample world"""
    print("Creating world...")
    world = World(size=10, pct_walls=0.15, pct_holes=0.08)
    
    print("Running value iteration...")
    vi = ValueIteration(world, gamma=0.8, tolerance=1e-6)
    values = vi.value_iteration()
    
    # Report performance metrics
    print("\n" + "="*50)
    print("PERFORMANCE METRICS:")
    print("="*50)
    print(f"Grid size: {world.size}x{world.size} = {world.size**2} states")
    print(f"Convergence tolerance: {vi.tolerance}")
    print(f"Discount factor (gamma): {vi.gamma}")
    print(f"Iterations to convergence: {vi.iterations_to_converge}")
    print(f"Total computation time: {vi.computation_time:.4f} seconds")
    print(f"Average time per iteration: {vi.computation_time/vi.iterations_to_converge:.4f} seconds")
    print(f"States processed per second: {(world.size**2 * vi.iterations_to_converge) / vi.computation_time:.0f}")
    print("="*50)
    
    print("\nPlotting results...")
    vi.plot_values()
    
    return vi, values


if __name__ == "__main__":
    vi, values = run_value_iteration_example()