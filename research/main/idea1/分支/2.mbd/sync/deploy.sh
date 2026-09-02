#!/bin/bash
# ============================================================
# MBD (Modal Balanced Distillation) 部署脚本
# 用法: 在服务器上 bash deploy.sh
# 前置: 已按 mcd 分支部署好完整 verl 训练框架（~//mcd/code 或 ~/vopd/code）
# ============================================================

set -e

echo "=== MBD 部署开始 ==="

# 1. 确定目标目录（默认用 mcd 基座部署目录）
CODE_DIR="${CODE_DIR:-$HOME/mcd/code}"
echo "目标代码目录: $CODE_DIR"

# 2. 检查目标目录存在
if [ ! -d "$CODE_DIR/verl" ]; then
  echo "错误: 未找到 verl 训练框架 ($CODE_DIR/verl)"
  echo "请先部署 mcd 分支的完整代码，或设置 CODE_DIR 指向完整框架目录"
  exit 1
fi

# 3. 复制 MBD 蒸馏模块到 verl actor 目录
echo "[1/3] 复制 MBD 模块到 verl..."
cp code/mbd_distillation.py "$CODE_DIR/verl/workers/actor/"

# 4. 运行单元测试
echo "[2/3] 运行单元测试..."
cd "$CODE_DIR"
python -c "
import sys; sys.path.insert(0, 'verl/workers/actor')
from mbd_distillation import MBDDistillation
print('MBD 模块导入 OK')
"

# 5. 提示集成
echo "[3/3] MBD 模块已就位"
echo ""
echo "=== MBD 部署完成 ==="
echo ""
echo "下一步："
echo "  1. 参考 mcd 分支集成方式，在 dp_actor.py 中调用 MBDDistillation"
echo "  2. 运行 test_mbd_distillation.py 验证算法正确性："
echo "     python test_mbd_distillation.py"
