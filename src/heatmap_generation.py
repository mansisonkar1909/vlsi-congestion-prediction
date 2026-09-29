"""
heatmap_generation.py
-----------------------
Turns OpenROAD's congestion_report.csv (from global_route) into a
(GRID_SIZE, GRID_SIZE) target heatmap - the "ground truth" (Y) your
model will be trained to predict.

OpenROAD's report is per-GCell, with overflow = usage - capacity
(positive overflow = congested / not enough room for the wires).
"""

import numpy as np
import pandas as pd
import os

GRID_SIZE = 256


def build_congestion_heatmap(congestion_csv, out_path, grid_size=GRID_SIZE):
    df = pd.read_csv(congestion_csv)

    # Expected columns from OpenROAD's congestion report (verify against
    # your OpenROAD version's actual output header and adjust names below)
    x_min, x_max = df["x1"].min(), df["x2"].max()
    y_min, y_max = df["y1"].min(), df["y2"].max()

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)

    for _, row in df.iterrows():
        cx = (row["x1"] + row["x2"]) / 2
        cy = (row["y1"] + row["y2"]) / 2
        col = int((cx - x_min) / (x_max - x_min) * (grid_size - 1))
        r = int((cy - y_min) / (y_max - y_min) * (grid_size - 1))
        col = np.clip(col, 0, grid_size - 1)
        r = np.clip(r, 0, grid_size - 1)

        overflow = max(0.0, row["usage"] - row["capacity"])
        heatmap[r, col] += overflow
        counts[r, col] += 1

    # average where multiple GCells land in same bin
    nonzero = counts > 0
    heatmap[nonzero] /= counts[nonzero]

    # normalize 0-1 so it's comparable across different chip sizes
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, heatmap)
    print(f"Saved congestion heatmap {heatmap.shape} -> {out_path}")
    return heatmap


if __name__ == "__main__":
    build_congestion_heatmap(
        congestion_csv="./data/raw/congestion_report.csv",
        out_path="./data/processed/design1_target.npy",
    )
