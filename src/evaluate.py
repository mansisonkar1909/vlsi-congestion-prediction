"""
evaluate.py
------------
Loads the trained U-Net, runs it on the held-out 20% test designs
(input features ONLY - it never sees the real routing/congestion data),
and scores predictions against the real heatmaps using MSE and SSIM.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset

from dataset import CongestionDataset
from model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    dataset = CongestionDataset("./data/processed")

    with open("./checkpoints/test_indices.txt") as f:
        test_indices = [int(i) for i in f.read().split(",") if i]
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("./checkpoints/unet_congestion.pth", map_location=DEVICE))
    model.eval()

    mse_scores, ssim_scores = [], []

    with torch.no_grad():
        for i, (x, y_true) in enumerate(test_loader):
            x = x.to(DEVICE)
            y_pred = model(x).cpu().numpy()[0, 0]   # (H, W)
            y_true = y_true.numpy()[0, 0]            # (H, W)

            mse = np.mean((y_pred - y_true) ** 2)
            score = ssim(y_true, y_pred, data_range=1.0)

            mse_scores.append(mse)
            ssim_scores.append(score)

            # save a side-by-side comparison image for the report
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(y_true, cmap="hot")
            axes[0].set_title("Real congestion")
            axes[1].imshow(y_pred, cmap="hot")
            axes[1].set_title("Predicted congestion")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"./checkpoints/test_sample_{i}.png")
            plt.close()

    print(f"Average MSE:  {np.mean(mse_scores):.5f}")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}  (closer to 1.0 = better)")


if __name__ == "__main__":
    main()
