#!/bin/bash
# DiffusionOPD H200 服务器一键部署脚本
# 使用方法: bash H200部署.sh

set -e

echo "=== DiffusionOPD H200 环境部署 ==="

# 1. 创建 conda 环境
conda create -n diffusion-opd python=3.12 -y
source activate diffusion-opd

# 2. 安装依赖
cd code
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || echo "Installing core deps manually"
pip install torch transformers diffusers accelerate peft wandb

# 3. 验证
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.device_count())"

echo "=== DiffusionOPD 环境部署完成 ==="
echo ""
echo "基于 SD3.5-Medium，多教师 Flow-Matching OPD"
echo "训练: 约 97 GPU-hours on 8×H200 (~12h)"
