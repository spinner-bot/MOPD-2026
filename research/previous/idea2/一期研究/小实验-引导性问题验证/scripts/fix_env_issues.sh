#!/bin/bash
# Fix environment issues
# Date: 2026-09-04

set -e

LOG_FILE=~/text_opd_fix.log

echo "======================================================================" | tee -a $LOG_FILE
echo "Text-OPD Environment Fix - $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate text-opd

# Step 1: Fix PyTorch CUDA libraries
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 1: Fixing PyTorch CUDA libraries..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# Install nvidia libraries that match PyTorch CUDA 12.9
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 \
    nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 \
    --index-url https://download.pytorch.org/whl/cu129 2>&1 | tee -a $LOG_FILE

# Verify PyTorch
echo "" | tee -a $LOG_FILE
echo "Verifying PyTorch..." | tee -a $LOG_FILE
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1 | tee -a $LOG_FILE

# Step 2: Install vLLM with retry
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 2: Installing vLLM 0.18.0..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

for i in 1 2 3; do
    echo "Attempt $i..." | tee -a $LOG_FILE
    if pip install vllm==0.18.0 2>&1 | tee -a $LOG_FILE; then
        echo "✓ vLLM installed successfully" | tee -a $LOG_FILE
        break
    else
        echo "Attempt $i failed, retrying..." | tee -a $LOG_FILE
        sleep 10
    fi
done

# Step 3: Install verl
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 3: Installing verl 0.7.0..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

for i in 1 2 3; do
    echo "Attempt $i..." | tee -a $LOG_FILE
    if pip install verl==0.7.0 2>&1 | tee -a $LOG_FILE; then
        echo "✓ verl installed successfully" | tee -a $LOG_FILE
        break
    else
        echo "Attempt $i failed, retrying..." | tee -a $LOG_FILE
        sleep 10
    fi
done

# Step 4: Install flash-attn (this may take a while)
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 4: Installing flash-attn 2.8.3..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

for i in 1 2 3; do
    echo "Attempt $i..." | tee -a $LOG_FILE
    if pip install flash-attn==2.8.3 --no-build-isolation 2>&1 | tee -a $LOG_FILE; then
        echo "✓ flash-attn installed successfully" | tee -a $LOG_FILE
        break
    else
        echo "Attempt $i failed, retrying..." | tee -a $LOG_FILE
        sleep 10
    fi
done

# Step 5: Final verification
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 5: Final verification..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

python -c "
import sys
print('Python:', sys.version)
print()

results = []

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    results.append(('PyTorch', True, f'{torch.__version__}, CUDA: {cuda_ok}'))
except Exception as e:
    results.append(('PyTorch', False, str(e)))

try:
    import transformers
    results.append(('Transformers', True, transformers.__version__))
except Exception as e:
    results.append(('Transformers', False, str(e)))

try:
    import vllm
    results.append(('vLLM', True, vllm.__version__))
except Exception as e:
    results.append(('vLLM', False, str(e)))

try:
    import verl
    results.append(('verl', True, verl.__version__))
except Exception as e:
    results.append(('verl', False, str(e)))

try:
    import flash_attn
    results.append(('flash-attn', True, flash_attn.__version__))
except Exception as e:
    results.append(('flash-attn', False, str(e)))

for name, success, msg in results:
    status = '✓' if success else '✗'
    print(f'{status} {name}: {msg}')
" 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Fix completed at $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
