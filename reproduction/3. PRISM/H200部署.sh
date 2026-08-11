#!/bin/bash
# PRISM H200 服务器一键部署脚本
# 使用方法: bash H200部署.sh

set -e

echo "=== PRISM H200 环境部署 ==="

# 1. 创建 conda 环境
conda create -n prism python=3.12 -y
source activate prism

# 2. 安装依赖
cd code
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || echo "requirements.txt not found, installing core deps manually"
pip install torch transformers accelerate vllm flash-attn

# 3. 安装 patched transformers
cd transformers-4.57.0
pip install -e . --no-deps
cd ..

# 4. 验证
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.device_count())"

echo "=== PRISM 环境部署完成 ==="
echo ""
echo "三阶段训练:"
echo "  Stage1 SFT: LLaMA-Factory (LR 1e-5, 1 epoch)"
echo "  Stage2 PRISM: bash scripts/train/experiment/qwen3_vl_prism.sh"
echo "  Stage3 RLVR: bash scripts/train/experiment/qwen3_vl_after_prism.sh"
