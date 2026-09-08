$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    python -m venv (Join-Path $ProjectDir ".venv")
}

& $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")
& $VenvPython (Join-Path $ProjectDir "train_mnist_numpy_cnn.py") --epochs 1 --limit-train 512 --limit-test 256
