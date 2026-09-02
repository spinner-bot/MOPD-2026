# MCDC 部署包

## 目录结构

```
sync/
├── code/                    # MCDC 代码（基于 Vision-OPD）
│   ├── verl/               # 训练框架
│   │   └── workers/actor/
│   │       ├── dp_actor.py              # 修改：集成 MCDC
│   │       └── mcd_distillation.py      # 新增：MCDC 核心模块
│   ├── scripts/
│   │   └── run_vision_opd.sh            # 训练脚本
│   └── ...
├── scripts/
│   └── run_vision_opd.sh    # 训练脚本（已配置 MCDC）
├── models/                  # 模型目录（部署时下载）
└── deploy.sh                # 部署脚本
```

## 部署步骤

### 1. 上传 sync 目录到服务器

```bash
# 在本地
scp -r sync/ username@server:~/
```

### 2. 在服务器上运行部署脚本

```bash
cd ~/sync
bash deploy.sh
```

部署脚本会：
- 创建项目目录 `~/mcd`
- 复制代码
- 下载 Text 教师模型（Qwen2.5-7B-Instruct）
- 验证环境
- 生成环境变量配置 `env.sh`

### 3. 启动训练

```bash
cd ~/mcd
source env.sh

# 使用 tmux 保护训练
tmux new -s mcd

# 启动训练（单卡 H200）
CUDA_VISIBLE_DEVICES=1 bash scripts/run_vision_opd.sh \
  trainer.n_gpus_per_node=1 \
  data.train_batch_size=24 \
  actor_rollout_ref.actor.ppo_mini_batch_size=24 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  data.dataloader_num_workers=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  2>&1 | tee train.log

# 脱离 tmux
# 按 Ctrl+B，松开，再按 D
```

## 训练配置

### MCDC 超参数

- `mcd_alpha=0.5`：置信度权重
- `mcd_beta=0.5`：优势差权重
- `mcd_tau=1.0`：温度参数
- `mcd_epsilon=0.1`：门控阈值

### 模型配置

- **学生模型**：Qwen3.5-4B（多模态）
- **VL 教师**：Qwen3.5-4B（与 Vision-OPD 相同）
- **Text 教师**：Qwen2.5-7B-Instruct（纯文本）

## 查看训练进度

```bash
# 查看日志
tail -f ~/mcd/train.log

# 查看 step 进度
grep "Training Progress" ~/mcd/train.log | tail -1

# 进入 tmux
tmux attach -t mcd
```

## 注意事项

1. **Text 教师模型**：首次运行会下载 Qwen2.5-7B-Instruct（约 14GB），需要网络连接
2. **显存需求**：单卡 H200（140GB）足够，显存占用约 50%
3. **训练时长**：预计 40-50 小时（260 步）
4. **tmux 保护**：长时间训练必须用 tmux，防止 SSH 断开导致训练中断

## 故障排查

### Text 教师加载失败

检查模型是否下载完整：
```bash
ls -lh ~/mcd/models/Qwen2.5-7B-Instruct/
```

### 显存不足

降低 `gpu_memory_utilization`：
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

### MCDC 模块导入错误

验证 MCDC 模块：
```bash
python -c "from verl.workers.actor.mcd_distillation import mcd_distillation_loss; print('OK')"
```
