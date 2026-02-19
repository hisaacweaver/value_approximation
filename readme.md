# Value Approximation Project

by Benjamin Weaver and Isaac Weaver

## Problem Description

This project models robotic navigation in a grid world (e.g., a warehouse or disaster zone).

Each cell can be:

- **Goal** (positive terminal reward)
- **Sinkhole/Pit** (negative terminal reward)
- **Wall** (blocked, cannot be entered)
- **Free space** (step cost for moving)

The goal is to compare:

- **Exact value iteration** (high-accuracy planning)
- **Approximate value iteration** with anchors and reconstruction (faster, scalable approximation)

## Project Files

- `world.py`: Environment generation, transitions, and world visualization.
- `exact_value.py`: Exact Value Iteration implementation and plotting.
- `approximate_value.py`: Approximate Value Iteration + grid reconstruction (`nearest` and `linear`).
- `demo.py`: Streamlit UI for interactive world + exact method parameters.

## Setup

### 1 - Create and activate a virtual environment (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2 - Install dependencies

```powershell
pip install -r requirements.txt
```

## Run python scrip

```powershell
python run.py
```

## Run the Streamlit App

```powershell
streamlit run demo.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Using the App (Interactive Exact Method)

In the Streamlit app you can control:

- **Grid Size**
- **Percentage of Walls**
- **Percentage of Sinkholes**
- **Goal reward, Pit reward, Step cost**
- **Discount factor (`gamma`)**
- **Convergence tolerance**
- **Maximum iterations**

The app displays:

- The generated world map
- The exact optimal value heatmap
- Performance metrics (iterations and computation time)

## Run Script-Based Comparisons

### Exact value iteration only

```powershell
python exact_value.py
```

### Approximate value iteration + reconstruction

```powershell
python approximate_value.py
```

`approximate_value.py` reconstructs a full grid from anchor values using:

- `nearest` interpolation
- `linear` interpolation

## How to Interpret Results

### 1 - Value Heatmaps

- **Higher values** indicate states with better expected long-term return.
- Values generally increase as you move toward the goal (unless blocked by walls/sinkholes).
- Sinkholes should show strongly negative values and influence nearby states.

### 2 - Parameter Effects

- **Higher gamma**: more future-focused planning, broader value propagation.
- **Lower gamma**: more short-term behavior.
- **More negative step cost**: encourages shorter paths.
- **More negative pit reward**: stronger avoidance of sinkholes.

### 3 - Convergence and Speed (Exact)

- Smaller tolerance usually needs more iterations.
- Larger grids increase runtime substantially.

### 4 - Exact vs Approximate

- **Exact** is more accurate but can become expensive on larger grids.
- **Approximate** is faster with fewer anchors, but reconstruction may smooth details.
- Compare nearest vs linear reconstruction to see trade-offs in smoothness vs fidelity.

## Notes

- Random seed is fixed in the code (`np.random.seed(42)`), improving reproducibility.
- The current Streamlit app visualizes the exact method interactively.
- Approximate-method visualizations are currently run through `approximate_value.py`.
