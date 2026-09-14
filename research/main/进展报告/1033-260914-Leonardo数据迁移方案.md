# Leonardo 数据迁移方案

**文档编号**：1032  
**创建时间**：2026-09-14 16:30（UTC+8）  
**作者**：claude code (local)  
**触发原因**：导师指示新的存储规范，需要将现有数据从 `/leonardo_work` 迁移到新的 `/leonardo_scratch` 结构

---

## 1. 背景

### 1.1 当前问题

`/leonardo_work/AIFAC_F02_774/` 是共享空间（1TB 全用户共享），当前已 100% 满。项目占用约 131GB，但共享空间已满会影响其他用户和作业提交。

### 1.2 新存储规范（2026-09-14 导师指示）

| 路径 | 用途 | 配额 | 特点 |
|------|------|------|------|
| `/leonardo_scratch/fast/AIFAC_F02_774/rqian000` | 脚本、代码、配置文件 | 1TB | 快速访问，适合频繁读写 |
| `/leonardo_scratch/large/userexternal/rqian000` | 数据集、模型权重、大型静态文件 | 无限额 | 大容量存储，适合长期保存 |

---

## 2. 当前存储使用情况

```
/leonardo_work/AIFAC_F02_774/sqs/
├── cache/        6.0G   # HuggingFace 模型缓存
├── data/        82G     # 数据集（M3VQA、Vision-OPD 等）
├── envs/        6.2G    # Python 虚拟环境
├── logs/        2.5M    # 作业日志
├── models/      37G     # 模型权重（Qwen3.5-4B/9B）
├── results/     876M    # 实验结果
└── tmp/         17M     # 临时文件

总计：约 131GB
```

---

## 3. 迁移目标结构

### 3.1 Fast 区（脚本、代码、环境）

```
/leonardo_scratch/fast/AIFAC_F02_774/rqian000/
├── code/              # 项目代码（从仓库软链接或复制）
├── envs/              # Python 虚拟环境（6.2G）
│   └── mopd-stage0-py311/
├── scripts/           # 作业提交脚本、工具脚本
├── logs/              # 作业日志（2.5M）
├── tmp/               # 临时文件（17M）
└── configs/           # 配置文件
```

**迁移内容**：
- `envs/` (6.2G) - Python 环境，需要频繁读写
- `logs/` (2.5M) - 作业日志
- `tmp/` (17M) - 临时文件
- 代码仓库的软链接或工作副本

**预估占用**：约 6.2G（远小于 1TB 配额）

### 3.2 Large 区（数据、模型、结果）

```
/leonardo_scratch/large/userexternal/rqian000/
├── data/              # 数据集（82G）
│   ├── m3vqa/
│   ├── vision-opd/
│   └── processed/
├── models/            # 模型权重（37G）
│   ├── Qwen3.5-4B/
│   └── Qwen3.5-9B/
├── cache/             # HuggingFace 缓存（6G）
│   └── huggingface/
└── results/           # 实验结果（876M）
    └── runs/
```

**迁移内容**：
- `data/` (82G) - 所有数据集
- `models/` (37G) - 模型权重
- `cache/` (6G) - HuggingFace 缓存
- `results/` (876M) - 实验结果

**预估占用**：约 125G（无限额，无压力）

---

## 4. 迁移步骤

### 4.1 准备工作

1. **停止所有运行中的作业**
   ```bash
   scancel -u rqian000
   ```

2. **检查新目录结构**
   ```bash
   mkdir -p /leonardo_scratch/fast/AIFAC_F02_774/rqian000/{envs,logs,tmp,scripts,configs}
   mkdir -p /leonardo_scratch/large/userexternal/rqian000/{data,models,cache,results}
   ```

3. **验证磁盘空间**
   ```bash
   df -h /leonardo_scratch/fast/
   df -h /leonardo_scratch/large/
   ```

### 4.2 迁移 Large 区数据（优先，因为体积大）

```bash
# 迁移数据集（82G）
rsync -avh --progress /leonardo_work/AIFAC_F02_774/sqs/data/ \
    /leonardo_scratch/large/userexternal/rqian000/data/

# 迁移模型权重（37G）
rsync -avh --progress /leonardo_work/AIFAC_F02_774/sqs/models/ \
    /leonardo_scratch/large/userexternal/rqian000/models/

# 迁移 HuggingFace 缓存（6G）
rsync -avh --progress /leonardo_work/AIFAC_F02_774/sqs/cache/ \
    /leonardo_scratch/large/userexternal/rqian000/cache/

# 迁移实验结果（876M）
rsync -avh --progress /leonardo_work/AIFAC_F02_774/sqs/results/ \
    /leonardo_scratch/large/userexternal/rqian000/results/
```

**预估时间**：约 30-60 分钟（取决于网络速度）

### 4.3 迁移 Fast 区数据

```bash
# 迁移 Python 环境（6.2G）
rsync -avh --progress /leonardo_work/AIFAC_F02_774/sqs/envs/ \
    /leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/

# 迁移日志（2.5M）
rsync -avh /leonardo_work/AIFAC_F02_774/sqs/logs/ \
    /leonardo_scratch/fast/AIFAC_F02_774/rqian000/logs/

# 迁移临时文件（17M）
rsync -avh /leonardo_work/AIFAC_F02_774/sqs/tmp/ \
    /leonardo_scratch/fast/AIFAC_F02_774/rqian000/tmp/
```

**预估时间**：约 5-10 分钟

### 4.4 更新软链接和配置

1. **更新代码仓库中的软链接**
   ```bash
   cd /leonardo_scratch/fast/AIFAC_F02_774/rqian000/code/MOPD-2026
   
   # 更新数据软链接
   ln -sfn /leonardo_scratch/large/userexternal/rqian000/data research/main/data/external
   ln -sfn /leonardo_scratch/large/userexternal/rqian000/models research/main/models
   
   # 更新环境软链接（如果有）
   ln -sfn /leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs envs
   ```

2. **更新 sbatch 脚本中的路径**
   ```bash
   # 搜索所有需要更新的脚本
   grep -r "/leonardo_work/AIFAC_F02_774/sqs" \
       /leonardo_scratch/fast/AIFAC_F02_774/rqian000/code/MOPD-2026/research/main/code/utils/*.sbatch
   
   # 批量替换（谨慎操作，先备份）
   # sed -i 's|/leonardo_work/AIFAC_F02_774/sqs|/leonardo_scratch/large/userexternal/rqian000|g' *.sbatch
   ```

3. **更新环境变量**
   ```bash
   # 在 .bashrc 或模块文件中更新
   export MOPD_DATA_ROOT=/leonardo_scratch/large/userexternal/rqian000
   export MOPD_ENV_ROOT=/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs
   ```

### 4.5 验证迁移

1. **检查文件完整性**
   ```bash
   # 对比文件数量
   find /leonardo_work/AIFAC_F02_774/sqs/data -type f | wc -l
   find /leonardo_scratch/large/userexternal/rqian000/data -type f | wc -l
   
   # 对比总大小
   du -sh /leonardo_work/AIFAC_F02_774/sqs/data
   du -sh /leonardo_scratch/large/userexternal/rqian000/data
   ```

2. **测试作业提交**
   ```bash
   # 提交一个测试作业，验证新路径可用
   sbatch test_job.sbatch
   ```

3. **验证 Python 环境**
   ```bash
   source /leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/activate
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### 4.6 清理旧数据

**确认新路径完全可用后**，才能删除旧数据：

```bash
# 不要立即删除！先保留至少一周，确认无问题后再清理

# 一周后，如果一切正常：
rm -rf /leonardo_work/AIFAC_F02_774/sqs/data
rm -rf /leonardo_work/AIFAC_F02_774/sqs/models
rm -rf /leonardo_work/AIFAC_F02_774/sqs/cache
rm -rf /leonardo_work/AIFAC_F02_774/sqs/results
rm -rf /leonardo_work/AIFAC_F02_774/sqs/envs
rm -rf /leonardo_work/AIFAC_F02_774/sqs/logs
rm -rf /leonardo_work/AIFAC_F02_774/sqs/tmp
```

---

## 5. 风险与注意事项

### 5.1 风险

1. **数据丢失风险**：迁移过程中可能因网络、磁盘故障导致数据不完整
   - **缓解**：使用 `rsync` 的校验功能，迁移后对比文件数和大小

2. **作业中断风险**：迁移期间运行的作业可能因路径变化而失败
   - **缓解**：迁移前停止所有作业，迁移完成后重新提交

3. **软链接断裂风险**：代码仓库中的软链接可能指向旧路径
   - **缓解**：迁移后立即更新所有软链接，并测试关键脚本

4. **环境变量遗漏**：某些脚本可能硬编码了旧路径
   - **缓解**：迁移前全局搜索旧路径，逐一更新

### 5.2 注意事项

1. **不要并行迁移**：按顺序执行，避免磁盘 I/O 冲突
2. **保留旧数据至少一周**：确认新路径完全正常后再清理
3. **记录迁移日志**：记录每个步骤的开始/结束时间、成功/失败状态
4. **通知协作者**：迁移完成后通知 Codex 和其他可能的协作者更新路径

---

## 6. 迁移检查清单

- [ ] 停止所有运行中的作业
- [ ] 创建新目录结构
- [ ] 迁移 Large 区数据（data, models, cache, results）
- [ ] 迁移 Fast 区数据（envs, logs, tmp）
- [ ] 更新代码仓库中的软链接
- [ ] 更新 sbatch 脚本中的路径
- [ ] 更新环境变量
- [ ] 验证文件完整性（文件数、大小）
- [ ] 测试 Python 环境
- [ ] 提交测试作业
- [ ] 通知协作者（Codex）
- [ ] 保留旧数据一周
- [ ] 一周后清理旧数据
- [ ] 更新项目文档

---

## 7. 时间估算

| 步骤 | 预估时间 |
|------|----------|
| 准备工作 | 10 分钟 |
| 迁移 Large 区（125G） | 30-60 分钟 |
| 迁移 Fast 区（6.2G） | 5-10 分钟 |
| 更新软链接和配置 | 20-30 分钟 |
| 验证迁移 | 15-20 分钟 |
| **总计** | **约 1.5-2 小时** |

---

## 8. 后续工作

1. **更新 1013 数据主线文档**：记录新的存储位置
2. **更新 1028 Codex 交接文档**：通知 Codex 新的路径结构
3. **更新 sbatch 脚本模板**：确保新提交的作业使用新路径
4. **监控磁盘使用**：定期检查 `/leonardo_scratch` 的使用情况

---

## 修订记录

| 时间 | 作者 | 修改内容 |
|------|------|----------|
| 2026-09-14 16:30 | claude code (local) | 创建迁移方案文档 |
