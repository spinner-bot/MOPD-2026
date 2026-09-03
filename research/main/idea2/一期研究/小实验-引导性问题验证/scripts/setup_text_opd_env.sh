#!/bin/bash
# Text-OPD Environment Setup Script
# Date: 2026-09-04
# Purpose: Install all dependencies for Text-OPD experiment

set -e  # Exit on error

LOG_FILE=~/text_opd_setup.log

echo "======================================================================" | tee -a $LOG_FILE
echo "Text-OPD Environment Setup - $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# Activate environment
echo "Activating text-opd environment..." | tee -a $LOG_FILE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate text-opd

echo "Python version: $(python --version)" | tee -a $LOG_FILE
echo "Pip version: $(pip --version)" | tee -a $LOG_FILE

# Upgrade pip
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 1: Upgrading pip..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
pip install --upgrade pip 2>&1 | tee -a $LOG_FILE

# Install PyTorch with CUDA 12.9
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 2: Installing PyTorch 2.10.0 with CUDA 12.9..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129 2>&1 | tee -a $LOG_FILE

# Verify PyTorch installation
echo "" | tee -a $LOG_FILE
echo "Verifying PyTorch installation..." | tee -a $LOG_FILE
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" 2>&1 | tee -a $LOG_FILE

# Install transformers
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 3: Installing transformers 5.5.0..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
pip install transformers==5.5.0 2>&1 | tee -a $LOG_FILE

# Install vLLM
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 4: Installing vLLM 0.18.0..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
pip install vllm==0.18.0 2>&1 | tee -a $LOG_FILE

# Install verl
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 5: Installing verl 0.7.0..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
pip install verl==0.7.0 2>&1 | tee -a $LOG_FILE

# Install flash-attn (this may take a while)
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 6: Installing flash-attn 2.8.3 (compilation may take 30-60 min)..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
pip install flash-attn==2.8.3 --no-build-isolation 2>&1 | tee -a $LOG_FILE

# Install other dependencies
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 7: Installing other dependencies..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
pip install datasets accelerate sentencepiece protobuf 2>&1 | tee -a $LOG_FILE

# Final verification
echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Step 8: Final verification..." | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

python -c "
import sys
print('Python:', sys.version)
print()

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA version: {torch.version.cuda}')
        print(f'  GPU count: {torch.cuda.device_count()}')
except Exception as e:
    print(f'✗ PyTorch: {e}')

print()

try:
    import transformers
    print(f'✓ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'✗ Transformers: {e}')

print()

try:
    import vllm
    print(f'✓ vLLM: {vllm.__version__}')
except Exception as e:
    print(f'✗ vLLM: {e}')

print()

try:
    import verl
    print(f'✓ verl: {verl.__version__}')
except Exception as e:
    print(f'✗ verl: {e}')

print()

try:
    import flash_attn
    print(f'✓ flash-attn: {flash_attn.__version__}')
except Exception as e:
    print(f'✗ flash-attn: {e}')
" 2>&1 | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
echo "Setup completed at $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
