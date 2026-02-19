import streamlit as st
from world import World
from exact_value import ValueIteration
import numpy as np
from approximate_value import ApproximateValueIteration, reconstruct_grid
import time

np.random.seed(42)  # For reproducibility

st.set_page_config(page_title="Value Approximation Demo", layout="wide")

st.title("Value Approximation Demo")
st.header("By Benjamin Weaver and Isaac Weaver")


world_cols = st.columns([1, 2, 1])

with world_cols[1]:
    st.header("Grid World Environment")
    st.text("Define the grid world parameters")
    grid_size = st.number_input("Grid Size", min_value=8, max_value=100, value=10)
    pct_walls = st.slider("Percentage of Walls", min_value=0.0, max_value=0.3, value=0.15, step=0.01, format="%.2f")
    pct_sinkholes = st.slider("Percentage of Sinkholes", min_value=0.0, max_value=0.2, value=0.05, step=0.01, format="%.2f")
    world = World(size=grid_size, pct_walls=pct_walls, pct_holes=pct_sinkholes)
    fig = world.get_fig()
    st.pyplot(fig)

    st.header("Reward parameters")
    goal_reward = st.number_input("Goal reward", value=100, step=1)
    pit_reward = st.number_input("Pit reward", value=-50, step=1)
    step_cost = st.number_input("Step cost (per move)", value=-1, step=1)

    st.header("Value iteration parameters")
    gamma = st.number_input("Discount factor (gamma)", value=0.95, step=0.01, format="%.2f")
    tolerance = st.number_input("Convergence tolerance", value=1e-6, step=1e-6, format="%.6f")
    max_iterations = st.number_input("Maximum iterations", value=1000, step=100)

    st.write(f"Using goal reward={goal_reward}, pit reward={pit_reward}, step cost={step_cost}")
    value_iteration = ValueIteration(world, gamma=gamma, tolerance=tolerance, max_iterations=max_iterations, reward_values=(goal_reward, pit_reward, step_cost))
    optimal_values = value_iteration.value_iteration()

    pct_anchors = st.slider("Percentage of Anchor Points", min_value=0.01, max_value=0.9, value=0.30, step=0.01, format="%.2f") 

value_cols = st.columns(4)
with value_cols[0]:
    st.header("Optimal Value Function")
    value_fig = value_iteration.get_fig()
    st.pyplot(value_fig)

    # Report performance metrics
    st.header("Performance Metrics")

    st.write(f"Grid size: {world.size}x{world.size} = {world.size**2} states")
    st.write(f"Iterations to convergence: {value_iteration.iterations_to_converge}")
    st.write(f"Total computation time: {value_iteration.computation_time:.4f} seconds")




with value_cols[1]:
    st.header("Approximation Anchor Points")
    
    approximate_value = ApproximateValueIteration(world, pct_anchors=pct_anchors, gamma=gamma, tolerance=tolerance, max_iterations=max_iterations)
    sparse_U = approximate_value.run()
    fig = approximate_value.get_fig(sparse_U)
    st.pyplot(fig)

    st.header("Performance Metrics")
    st.write(f"Anchor points used: {len(approximate_value.anchors)}")
    st.write(f"Iterations to convergence: {approximate_value.iterations_to_converge}")
    st.write(f"Total computation time: {approximate_value.computation_time:.4f} seconds")

with value_cols[2]:
    st.header("Nearest Neighbor Interpolation")
    nearest_start = time.time()
    nearest_grid = reconstruct_grid(sparse_U, world.size, method='nearest')
    nearest_time = time.time() - nearest_start
    fig = approximate_value.get_fig(nearest_grid)
    st.pyplot(fig)

    st.header("Performance Metrics")
    st.write("Iterations required: 1 (single interpolation pass)")
    st.write(f"Total computation time: {nearest_time:.4f} seconds")

with value_cols[3]:
    st.header("Linear Interpolation")
    linear_start = time.time()
    linear_grid = reconstruct_grid(sparse_U, world.size, method='linear')
    linear_time = time.time() - linear_start
    fig = approximate_value.get_fig(linear_grid)
    st.pyplot(fig)

    st.header("Performance Metrics")
    st.write("Iterations required: 1 (single interpolation pass)")
    st.write(f"Total computation time: {linear_time:.4f} seconds")