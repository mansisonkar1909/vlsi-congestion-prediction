"""
train.py
---------
Trains the U-Net on (input_features, congestion_heatmap) pairs.
80% of designs used for training, 20% held out for testing (see evaluate.py).
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm

from dataset import CongestionDataset
from model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4


def main():
    dataset = CongestionDataset("./data/processed")
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # simple pixel-wise error between predicted & real heatmap

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_set)
        print(f"Epoch {epoch}: avg train MSE loss = {avg_loss:.5f}")

    torch.save(model.state_dict(), "./checkpoints/unet_congestion.pth")
    print("Model saved to ./checkpoints/unet_congestion.pth")

    # save the test split indices so evaluate.py uses the SAME held-out set
    test_indices = test_set.indices
    with open("./checkpoints/test_indices.txt", "w") as f:
        f.write(",".join(map(str, test_indices)))


if __name__ == "__main__":
    main()
