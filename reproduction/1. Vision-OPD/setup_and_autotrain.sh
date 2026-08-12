#!/bin/bash
# Vision-OPD 一键准备 + 自动训练脚本（带下载重试）
# 用法: 在 tmux 里执行  bash setup_and_autotrain.sh
# 功能: 下载数据(可断线重试) → 解压 → 转parquet → 下载基座模型(可重试) → 等GPU空闲 → 自动训练

cd ~/spinner-bot/reproduction/1.\ Vision-OPD/code
export HF_ENDPOINT=https://hf-mirror.com

# 通用重试函数：命令失败自动重试，直到成功或达到上限
retry() {
    local desc="$1"; shift
    local max=15
    for i in $(seq 1 $max); do
        echo "[$desc] 第 $i/$max 次尝试 $(date '+%H:%M:%S') ..."
        if "$@"; then
            echo "[$desc] 成功"
            return 0
        fi
        echo "[$desc] 失败，60 秒后重试 ..."
        sleep 60
    done
    echo "[$desc] 重试 $max 次仍失败，退出。稍后重跑本脚本可续传。"
    return 1
}

echo "====================  Step 1/5: 下载数据集 Vision-OPD-6K ===================="
retry "下载数据集" python -c "from huggingface_hub import snapshot_download; import os; snapshot_download('yuanqianhao/Vision-OPD-6K', repo_type='dataset', local_dir=os.path.abspath('data'))"

echo "====================  Step 2/5: 解压图片 ===================="
cd data/images
cat images.tar.gz* | tar -xf -
rm -f images.tar.gz*
cd ../teacher_images
tar -xf teacher_images.tar.gz
rm -f teacher_images.tar.gz
cd ../..

echo "====================  Step 3/5: 转换 train.jsonl -> train.parquet ===================="
python scripts/prepare_data.py --data-dir ./data --skip-download

echo "====================  Step 4/5: 下载基座模型 Qwen3.5-4B ===================="
retry "下载基座模型" python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"

echo "====================  Step 5/5: 等待 GPU 空闲后自动启动训练 ===================="
while true; do
    free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 < 10000 {c++} END {print c+0}')
    echo "$(date '+%Y-%m-%d %H:%M:%S')  空闲 GPU: $free/8"
    if [ "$free" -ge 8 ]; then
        echo ">>> 检测到 8 张 GPU 全部空闲，开始训练！"
        break
    fi
    sleep 120
done

bash scripts/run_vision_opd.sh 2>&1 | tee train.log
echo "训练结束，日志已保存到 train.log"
