#!/bin/bash
# MCDC 部署脚本
# 用法: bash deploy.sh

set -e

echo "=== MCDC 部署开始 ==="

# 1. 创建项目目录
DEPLOY_DIR="$HOME/mcd"
mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

echo "部署目录: $DEPLOY_DIR"

# 2. 复制代码
echo "[1/5] 复制代码..."
cp -r /path/to/sync/code/* ./
cp /path/to/sync/scripts/run_vision_opd.sh ./scripts/

# 3. 激活 conda 环境
echo "[2/5] 激活 conda 环境..."
source activate vision-opd

# 4. 下载 Text 教师模型
echo "[3/5] 下载 Text 教师模型..."
mkdir -p models
cd models
if [ ! -d "Qwen2.5-7B-Instruct" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='Qwen2.5-7B-Instruct')
"
    echo "Text 教师模型下载完成"
else
    echo "Text 教师模型已存在，跳过下载"
fi
cd ..

# 5. 验证环境
echo "[4/5] 验证环境..."
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
"

python -c "from verl.workers.actor.mcd_distillation import mcd_distillation_loss; print('MCDC module OK')"

# 6. 设置环境变量
echo "[5/5] 设置环境变量..."
cat > env.sh << 'EOF'
export WORLD_SIZE=1
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
EOF

echo "=== MCDC 部署完成 ==="
echo ""
echo "下一步："
echo "  cd $DEPLOY_DIR"
echo "  source env.sh"
echo "  bash scripts/run_vision_opd.sh"
