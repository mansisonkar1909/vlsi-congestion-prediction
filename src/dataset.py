"""
dataset.py
-----------
PyTorch Dataset that loads (input_features.npy, target_heatmap.npy) pairs
for many different chip designs.

Expected folder layout in data/processed/:
    design1_input.npy   design1_target.npy
    design2_input.npy   design2_target.npy
    ...
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CongestionDataset(Dataset):
    def __init__(self, processed_dir="./data/processed"):
        self.pairs = []
        input_files = sorted(glob.glob(os.path.join(processed_dir, "*_input.npy")))
        for inp_path in input_files:
            design_name = os.path.basename(inp_path).replace("_input.npy", "")
            target_path = os.path.join(processed_dir, f"{design_name}_target.npy")
            if os.path.exists(target_path):
                self.pairs.append((inp_path, target_path))
            else:
                print(f"Warning: no target found for {design_name}, skipping")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, target_path = self.pairs[idx]
        x = np.load(inp_path).astype(np.float32)          # (3, H, W)
        y = np.load(target_path).astype(np.float32)        # (H, W)
        y = np.expand_dims(y, axis=0)                       # (1, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)
