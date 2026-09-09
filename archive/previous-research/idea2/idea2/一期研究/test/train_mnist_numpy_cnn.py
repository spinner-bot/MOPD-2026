import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny NumPy CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--limit-train", type=int, default=512)
    parser.add_argument("--limit-test", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_mnist(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for url in MNIST_URLS.values():
        target = data_dir / url.rsplit("/", 1)[-1]
        if target.exists():
            continue
        print(f"downloading={target.name}")
        urllib.request.urlretrieve(url, target)


def load_images(path: Path, limit: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"invalid image file: {path}")
        count = min(count, limit) if limit > 0 else count
        data = np.frombuffer(f.read(rows * cols * count), dtype=np.uint8)
    images = data.reshape(count, 1, rows, cols).astype(np.float32) / 255.0
    return (images - 0.1307) / 0.3081


def load_labels(path: Path, limit: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid label file: {path}")
        count = min(count, limit) if limit > 0 else count
        return np.frombuffer(f.read(count), dtype=np.uint8).astype(np.int64)


def conv2d_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, channels, height, width = x.shape
    filters, _, kernel_h, kernel_w = weight.shape
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1
    out = np.zeros((batch, filters, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, i : i + kernel_h, j : j + kernel_w]
            out[:, :, i, j] = np.tensordot(patch, weight, axes=([1, 2, 3], [1, 2, 3]))
    out += bias.reshape(1, -1, 1, 1)
    return out


def conv2d_backward(
    x: np.ndarray,
    weight: np.ndarray,
    grad_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, channels, height, width = x.shape
    filters, _, kernel_h, kernel_w = weight.shape
    out_h, out_w = grad_out.shape[2:]
    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(weight)
    grad_b = grad_out.sum(axis=(0, 2, 3))

    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, i : i + kernel_h, j : j + kernel_w]
            grad_w += np.tensordot(grad_out[:, :, i, j], patch, axes=([0], [0]))
            grad_x[:, :, i : i + kernel_h, j : j + kernel_w] += np.tensordot(
                grad_out[:, :, i, j], weight, axes=([1], [0])
            )

    return grad_x, grad_w / batch, grad_b / batch


def maxpool2_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, channels, height, width = x.shape
    out_h = height // 2
    out_w = width // 2
    out = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)
    mask = np.zeros_like(x, dtype=bool)

    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            max_vals = patch.max(axis=(2, 3), keepdims=True)
            out[:, :, i, j] = max_vals[:, :, 0, 0]
            mask[:, :, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = patch == max_vals

    return out, mask


def maxpool2_backward(grad_out: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grad_x = np.zeros_like(mask, dtype=np.float32)
    for i in range(grad_out.shape[2]):
        for j in range(grad_out.shape[3]):
            patch_mask = mask[:, :, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            denom = patch_mask.sum(axis=(2, 3), keepdims=True)
            grad_x[:, :, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
                patch_mask * grad_out[:, :, i : i + 1, j : j + 1] / denom
            )
    return grad_x


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(labels.size), labels] + 1e-12).mean()
    grad = probs
    grad[np.arange(labels.size), labels] -= 1.0
    return float(loss), grad / labels.size


class TinyCNN:
    def __init__(self, rng: np.random.Generator) -> None:
        self.conv_w = rng.normal(0, 0.1, size=(8, 1, 3, 3)).astype(np.float32)
        self.conv_b = np.zeros(8, dtype=np.float32)
        self.fc_w = rng.normal(0, 0.1, size=(8 * 13 * 13, 10)).astype(np.float32)
        self.fc_b = np.zeros(10, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        conv = conv2d_forward(x, self.conv_w, self.conv_b)
        act = relu(conv)
        pooled, pool_mask = maxpool2_forward(act)
        flat = pooled.reshape(x.shape[0], -1)
        logits = flat @ self.fc_w + self.fc_b
        cache = {"x": x, "conv": conv, "pool_mask": pool_mask, "flat": flat}
        return logits, cache

    def train_batch(self, x: np.ndarray, labels: np.ndarray, lr: float) -> float:
        logits, cache = self.forward(x)
        loss, grad_logits = softmax_cross_entropy(logits, labels)

        grad_fc_w = cache["flat"].T @ grad_logits
        grad_fc_b = grad_logits.sum(axis=0)
        grad_flat = grad_logits @ self.fc_w.T
        grad_pool = grad_flat.reshape(x.shape[0], 8, 13, 13)
        grad_act = maxpool2_backward(grad_pool, cache["pool_mask"])
        grad_conv = grad_act * (cache["conv"] > 0)
        _, grad_conv_w, grad_conv_b = conv2d_backward(cache["x"], self.conv_w, grad_conv)

        self.fc_w -= lr * grad_fc_w
        self.fc_b -= lr * grad_fc_b
        self.conv_w -= lr * grad_conv_w
        self.conv_b -= lr * grad_conv_b
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return logits.argmax(axis=1)


def iterate_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    indices = rng.permutation(len(images))
    for start in range(0, len(images), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield images[batch_idx], labels[batch_idx]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    download_mnist(args.data_dir)

    train_images = load_images(args.data_dir / "train-images-idx3-ubyte.gz", args.limit_train)
    train_labels = load_labels(args.data_dir / "train-labels-idx1-ubyte.gz", args.limit_train)
    test_images = load_images(args.data_dir / "t10k-images-idx3-ubyte.gz", args.limit_test)
    test_labels = load_labels(args.data_dir / "t10k-labels-idx1-ubyte.gz", args.limit_test)

    model = TinyCNN(rng)
    print("backend=numpy")
    print(f"train_samples={len(train_images)} test_samples={len(test_images)}")

    for epoch in range(1, args.epochs + 1):
        losses = []
        for x_batch, y_batch in iterate_batches(train_images, train_labels, args.batch_size, rng):
            losses.append(model.train_batch(x_batch, y_batch, args.lr))
        predictions = model.predict(test_images)
        accuracy = (predictions == test_labels).mean()
        print(f"epoch={epoch:02d} train_loss={np.mean(losses):.4f} test_acc={accuracy:.2%}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "mnist_numpy_cnn.npz"
    np.savez(model_path, conv_w=model.conv_w, conv_b=model.conv_b, fc_w=model.fc_w, fc_b=model.fc_b)
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    main()
