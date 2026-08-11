#!/bin/bash
# V-Zero H200 服务器一键部署脚本
# 使用方法: bash H200部署.sh

set -e

echo "=== V-Zero H200 环境部署 ==="

# 1. 克隆代码（如果还没克隆）
if [ ! -d "code" ]; then
    git clone https://github.com/eVI-group-SCU/V-Zero.git code
fi

# 2. 创建 conda 环境
conda create -n v-zero python=3.12 -y
source activate v-zero

# 3. 安装依赖
cd code
uv pip install -e .

# 4. 验证
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.device_count())"

echo "=== V-Zero 环境部署完成 ==="
echo ""
echo "下一步:"
echo "  推理权重: hao05/v-zero-4b (HuggingFace)"
echo "  训练: bash scripts/run_vzero_qwen35_vl_fsdp.sh"
