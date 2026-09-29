"""
feature_engineering.py
-----------------------
Turns raw placement data (cell_positions.csv, pin_positions.csv) into a
grid-based "image" that a neural network can read.

Output: a (3, GRID_SIZE, GRID_SIZE) NumPy array saved as .npy, where the
3 channels are:
    0: cell density
    1: pin density
    2: RUDY (Rectangular Uniform wire DensitY estimate)
"""

import numpy as np
import pandas as pd
import os

GRID_SIZE = 256  # 256 x 256 grid, like graph paper over the chip


def load_die_area(cell_csv):
    """Estimate chip boundary from the extent of all cell positions."""
    df = pd.read_csv(cell_csv)
    x_min, x_max = df["x"].min(), (df["x"] + df["width"]).max()
    y_min, y_max = df["y"].min(), (df["y"] + df["height"]).max()
    return x_min, x_max, y_min, y_max


def to_grid_index(x, y, bounds, grid_size=GRID_SIZE):
    x_min, x_max, y_min, y_max = bounds
    col = int((x - x_min) / (x_max - x_min) * (grid_size - 1))
    row = int((y - y_min) / (y_max - y_min) * (grid_size - 1))
    col = np.clip(col, 0, grid_size - 1)
    row = np.clip(row, 0, grid_size - 1)
    return row, col


def compute_cell_density(cell_csv, bounds, grid_size=GRID_SIZE):
    df = pd.read_csv(cell_csv)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for _, row_data in df.iterrows():
        cx = row_data["x"] + row_data["width"] / 2
        cy = row_data["y"] + row_data["height"] / 2
        r, c = to_grid_index(cx, cy, bounds, grid_size)
        grid[r, c] += 1.0
    # Normalize 0-1
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


def compute_pin_density(pin_csv, bounds, grid_size=GRID_SIZE):
    df = pd.read_csv(pin_csv)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for _, row_data in df.iterrows():
        r, c = to_grid_index(row_data["x"], row_data["y"], bounds, grid_size)
        grid[r, c] += 1.0
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


def compute_rudy(cell_csv, bounds, grid_size=GRID_SIZE):
    """
    Simplified RUDY: for every cell, spread an estimated wiring demand
    (proportional to its bounding box) uniformly over the grid cells its
    bounding box overlaps. This is a common simplification used in
    congestion-prediction papers.
    """
    df = pd.read_csv(cell_csv)
    x_min, x_max, y_min, y_max = bounds
    cell_w = (x_max - x_min) / grid_size
    cell_h = (y_max - y_min) / grid_size
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for _, row_data in df.iterrows():
        w, h = row_data["width"], row_data["height"]
        demand = (w + h)  # rough proxy for wirelength demand (half-perimeter)
        r, c = to_grid_index(
            row_data["x"] + w / 2, row_data["y"] + h / 2, bounds, grid_size
        )
        # spread demand over a small neighborhood proportional to cell size
        span_r = max(1, int(h / cell_h))
        span_c = max(1, int(w / cell_w))
        r0, r1 = max(0, r - span_r // 2), min(grid_size, r + span_r // 2 + 1)
        c0, c1 = max(0, c - span_c // 2), min(grid_size, c + span_c // 2 + 1)
        grid[r0:r1, c0:c1] += demand / ((r1 - r0) * (c1 - c0))

    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


def build_feature_tensor(cell_csv, pin_csv, out_path, grid_size=GRID_SIZE):
    bounds = load_die_area(cell_csv)
    cell_density = compute_cell_density(cell_csv, bounds, grid_size)
    pin_density = compute_pin_density(pin_csv, bounds, grid_size)
    rudy = compute_rudy(cell_csv, bounds, grid_size)

    feature_tensor = np.stack([cell_density, pin_density, rudy], axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, feature_tensor)
    print(f"Saved input features {feature_tensor.shape} -> {out_path}")
    return feature_tensor


if __name__ == "__main__":
    build_feature_tensor(
        cell_csv="./data/raw/cell_positions.csv",
        pin_csv="./data/raw/pin_positions.csv",
        out_path="./data/processed/design1_input.npy",
    )
