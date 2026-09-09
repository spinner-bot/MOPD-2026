# MNIST CNN Test

This folder contains a minimal CNN training test for MNIST.

## Files

- `train_mnist_numpy_cnn.py`: trains and evaluates a tiny CNN using only NumPy.
- `train_mnist_cnn.py`: a PyTorch version kept for reference. On this machine, PyTorch currently fails while loading `c10.dll`.
- `requirements.txt`: Python dependencies.
- `run_test.ps1`: creates `.venv`, installs dependencies, and runs a quick smoke test.

## Quick Start

From this folder:

```powershell
.\run_test.ps1
```

Or from the repository root:

```powershell
& 'research/main/idea2/一期研究/test/run_test.ps1'
```

The default smoke test trains on 512 MNIST training samples and evaluates on 256 test samples for 1 epoch. MNIST data is downloaded to `data/`, and the trained NumPy model is saved to `outputs/mnist_numpy_cnn.npz`.

For a longer run:

```powershell
.venv\Scripts\python.exe train_mnist_numpy_cnn.py --epochs 5 --limit-train 2048 --limit-test 512
```
