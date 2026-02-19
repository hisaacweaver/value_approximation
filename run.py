import time
import numpy as np
import matplotlib.pyplot as plt

from world import World
from exact_value import ValueIteration
from approximate_value import ApproximateValueIteration, reconstruct_grid


def main():
	np.random.seed(42)

	grid_size = 10
	pct_walls = 0.15
	pct_sinkholes = 0.05

	goal_reward = 100
	pit_reward = -50
	step_cost = -1

	gamma = 0.95
	tolerance = 1e-6
	max_iterations = 1000

	pct_anchors = 0.30

	print("=" * 60)
	print("VALUE APPROXIMATION RUN")
	print("=" * 60)

	world = World(size=grid_size, pct_walls=pct_walls, pct_holes=pct_sinkholes)

	value_iteration = ValueIteration(
		world,
		gamma=gamma,
		tolerance=tolerance,
		max_iterations=max_iterations,
		reward_values=(goal_reward, pit_reward, step_cost),
	)
	value_iteration.value_iteration()

	approximate_value = ApproximateValueIteration(
		world,
		pct_anchors=pct_anchors,
		gamma=gamma,
		tolerance=tolerance,
		max_iterations=max_iterations,
	)
	sparse_u = approximate_value.run()

	nearest_start = time.time()
	nearest_grid = reconstruct_grid(sparse_u, world.size, method="nearest")
	nearest_time = time.time() - nearest_start

	linear_start = time.time()
	linear_grid = reconstruct_grid(sparse_u, world.size, method="linear")
	linear_time = time.time() - linear_start

	print("\nPERFORMANCE METRICS")
	print("-" * 60)
	print(f"Grid size: {world.size}x{world.size} = {world.size**2} states")

	print("\n[Exact Value Iteration]")
	print(f"Iterations to convergence: {value_iteration.iterations_to_converge}")
	print(f"Total computation time: {value_iteration.computation_time:.4f} seconds")

	print("\n[Approximation Anchor Points]")
	print(f"Anchor points used: {len(approximate_value.anchors)}")
	print(f"Iterations to convergence: {approximate_value.iterations_to_converge}")
	print(f"Total computation time: {approximate_value.computation_time:.4f} seconds")

	print("\n[Nearest Neighbor Interpolation]")
	print("Iterations required: 1 (single interpolation pass)")
	print(f"Total computation time: {nearest_time:.4f} seconds")

	print("\n[Linear Interpolation]")
	print("Iterations required: 1 (single interpolation pass)")
	print(f"Total computation time: {linear_time:.4f} seconds")
	print("=" * 60)

	world_fig = world.get_fig()
	world_fig.suptitle("Grid World Environment")

	exact_fig = value_iteration.get_fig()
	exact_fig.suptitle("Optimal Value Function")

	sparse_fig = approximate_value.get_fig(sparse_u)
	sparse_fig.suptitle("Approximation Anchor Points")

	nearest_fig = approximate_value.get_fig(nearest_grid)
	nearest_fig.suptitle("Nearest Neighbor Interpolation")

	linear_fig = approximate_value.get_fig(linear_grid)
	linear_fig.suptitle("Linear Interpolation")

	plt.show()


if __name__ == "__main__":
	main()
