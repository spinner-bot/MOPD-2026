# Handwritten Digit Recognition v2 — GPU-preloaded, full-batch
# Author: Langxi (hhu2524030232 Zhang Ruihan)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.model_selection import train_test_split


def reset():
    """Load CSV, split, and pre-allocate ALL data on GPU once."""
    data = pd.read_csv("data.csv")
    print(f"Data loaded: {data.shape}")

    x, y = data.iloc[1:, 1:], data.iloc[1:, 0]
    print(f"Samples: {data.shape[0] - 1}")

    size = float(input("Train set ratio (default 0.8): ") or "0.8")
    if size > 1 or size < 0:
        size = 0.8
        print("Invalid input, using default: 0.8")

    x = x.astype(np.float32).values
    y = y.astype(np.int64).values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=1 - size, random_state=42
    )

    # -- KEY CHANGE: pre-load everything to GPU once --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.from_numpy(x_train).to(device)
    Y_train = torch.from_numpy(y_train).to(device)
    X_test  = torch.from_numpy(x_test).to(device)
    Y_test  = torch.from_numpy(y_test).to(device)

    return X_train, Y_train, X_test, Y_test, x_train.shape[1], len(np.unique(y))


class NeuralNetwork(nn.Module):
    """3-layer MLP: input -> 256 -> 128 -> output."""

    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.linear1 = nn.Linear(i_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, o_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


def train_model(X, Y, i_dim, o_dim, epochs, lr):
    """Full-batch training: entire dataset in one shot each epoch. No DataLoader overhead."""
    device = X.device
    model = NeuralNetwork(i_dim, o_dim).to(device)
    torch.manual_seed(1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    n_samples = X.shape[0]
    print(f"Training (full-batch) | samples={n_samples} lr={lr:.4f} device={device}")
    for epoch in range(epochs):
        t0 = time.time()
        optimizer.zero_grad()
        logits = model(X)                    # forward entire dataset at once
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()
        elapsed = time.time() - t0
        print(f"Epoch [{epoch + 1}/{epochs}] | loss={loss.item():.4f} | {elapsed:.2f}s")
    return model


def test_model(model, X, Y):
    """Evaluate on pre-loaded GPU test set."""
    device = X.device
    print(f"Testing | samples={X.shape[0]} device={device}")
    model.eval()

    with torch.no_grad():
        logits = model(X)
        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == Y).sum().item()
        total = Y.size(0)

    acc = 100 * correct / total
    print(f"\n===== Test Results =====")
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {acc:.2f}%")
    print(f"Device        : {device}")
    return acc


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, i_dim, class_n = reset()

    epochs = int(input("Epochs: "))
    lr = float(input("Learning rate: "))

    t_total = time.time()
    model = train_model(X_train, Y_train, i_dim, class_n, epochs, lr)
    test_model(model, X_test, Y_test)
    print(f"Total wall time: {time.time() - t_total:.2f}s")
