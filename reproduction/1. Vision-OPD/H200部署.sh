#!/bin/bash
# Vision-OPD H200 服务器一键部署脚本
# 使用方法: bash H200部署.sh

set -e

echo "=== Vision-OPD H200 环境部署 ==="

# 1. 创建 conda 环境
conda create -n vision-opd python=3.12 -y
source activate vision-opd

# 2. 安装依赖
pip install --upgrade pip
cd code
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation

# 3. 验证
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.device_count())"
python -c "import vllm; print('vLLM OK')"

echo "=== Vision-OPD 环境部署完成 ==="
echo ""
echo "下一步:"
echo "  推理: vllm serve yuanqianhao/Vision-OPD-9B --gpu-memory-utilization 0.85 --tensor-parallel-size 8"
echo "  训练: bash scripts/run_vision_opd.sh"
