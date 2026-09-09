# Handwritten Digit Recognition | Author: Langxi (hhu2524030232 Zhang Ruihan)
#
# A simple 3-layer MLP for recognizing 28x28 handwritten digits from CSV.
# Adapted for headless server (no matplotlib, English output, ASCII-safe paths).

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def reset():
    """Load CSV, split into train/test sets."""
    data = pd.read_csv("data.csv")
    print(f"Data loaded: {data.shape}")

    # Row 0 = header/label, columns from index 1 = pixel data
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
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test), torch.from_numpy(y_test)
    )
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


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


def train_model(train_dataset, i_dim, o_dim, epochs, batch_size, lr):
    """Train the MLP, return trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(i_dim, o_dim).to(device)
    torch.manual_seed(1)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    print(f"Training started | batch={batch_size} lr={lr:.4f} device={device}")
    for epoch in range(epochs):
        total_loss = 0.0
        t0 = time.time()
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - t0
        print(f"Epoch [{epoch + 1}/{epochs}] | loss={avg_loss:.4f} | {elapsed:.2f}s")
    return model


def test_model(model, test_dataset, batch_size):
    """Evaluate the model on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing started | batch={batch_size} device={device}")
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            _, predicted = torch.max(logits, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100 * correct / total
    print(f"\n===== Test Results =====")
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {acc:.2f}%")
    return acc


if __name__ == "__main__":
    train_dataset, test_dataset, i_dim, class_n = reset()

    epochs = int(input("Epochs: "))
    batch_size = int(input("Batch size: "))
    lr = float(input("Learning rate: "))

    trained_model = train_model(train_dataset, i_dim, class_n, epochs, batch_size, lr)
    test_model(trained_model, test_dataset, batch_size)
