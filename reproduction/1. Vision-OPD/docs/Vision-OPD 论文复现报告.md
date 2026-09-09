# Vision-OPD 复现进展报告

> 日期：2026-08-26  
> 项目：Vision-OPD 论文复现（arXiv: 2605.18740）

---

## 一、项目目标

复现 Vision-OPD 论文，在 USTC 8×H200 服务器上完成训练与评估，验证其多模态细粒度视觉理解能力。

---

## 二、已完成工作

### 2.1 环境搭建
- **硬件**：8×NVIDIA H200（每张 140GB 显存，共 1120GB）
- **软件栈**：
  - Python 3.12 + conda 环境 `vision-opd`
  - PyTorch 2.10.0 + CUDA 12.9
  - vLLM 0.18.0（推理引擎）
  - verl 0.7.0.dev0（分布式 RL 训练框架）
  - transformers 5.5.0
  - flash-attn 2.8.3（注意力加速）

### 2.2 数据准备
- 下载 Vision-OPD-6K 数据集（6000 条多模态指令）
- 解压图片（images.tar.gz + teacher_images.tar.gz）
- 转换为训练格式（train.parquet）

### 2.3 模型准备
- 下载 Qwen3.5-4B 基座模型到本地目录（避免网络解析失败）
- 开启 `trust_remote_code=True`（Qwen3.5 tokenizer 需远程代码）

### 2.4 训练脚本调试
- 解决词表加载失败问题
- 解决 FlashInfer JIT 编译失败问题（CUDA 版本不匹配）
- 解决 libcuda.so 链接问题
- 训练脚本能成功启动，rollout 阶段通过

---

## 三、遇到的关键问题与解决方案

| 问题 | 根因 | 解决方案 | 耗时 |
|------|------|---------|------|
| `OSError: Unable to load vocabulary` | tokenizer 需远程代码 + 模型走网络解析断网 | 开 trust_remote_code + 模型下到本地 | 2h |
| `nvcc fatal: Value 'c++20' is not defined` | 系统只有 CUDA 11.8，FlashInfer JIT 需 CUDA ≥12 | conda 装 cuda-toolkit=12.9 | 1h |
| `ld: cannot find -lcuda` | conda CUDA 不含驱动库 | 指向系统 /usr/local/cuda-11.8/.../stubs | 30min |
| `ModuleNotFoundError: No module named 'flash_attn'` | PyPI 无预编译 wheel（torch 2.10 + CUDA 12.9 太新） | 源码编译 flash-attn | 30min |

**关键经验**：
1. 环境依赖要一次到位，不该留到训练时才发现
2. 国内服务器连 GitHub 不稳，预编译 wheel 下载需代理
3. conda 装的 CUDA toolkit 不含驱动库，需指向系统 CUDA

---

## 四、当前状态

### 4.1 训练启动
- ✅ 环境就绪（conda + CUDA 12.9 + flash-attn）
- ✅ 模型、数据准备完毕
- ✅ 训练脚本能成功启动
- ✅ rollout 阶段通过（生成 8 条样本/step）
- ✅ **成功进入稳态训练**（260 步，训练持续运行中）

### 4.2 资源配置
- **GPU**：卡 5、7（2×H200，140GB×2 = 280GB）
- **batch size**：24（prompt 数）× 8（rollout 数）= 192 样本/step
- **显存占用**：约 50%（FSDP + offload 优化）
- **预计训练时长**：3-5 小时

---

## 五、下一步计划

### 5.1 训练监控（持续进行）
- 监控训练进度，确认 loss 持续下降
- 等待训练完成（260 步，预计 3-5 小时）
- 保存 checkpoint（`~/vopd/checkpoints/Vision-OPD-Qwen3.5-4B/`）

### 5.2 模型评估（后续工作）
- 合并 checkpoint：`bash scripts/merge_checkpoint.sh`
- 推理测试：用验证集评估细粒度视觉理解能力
- 与论文 Table 1/2 对照，计算准确率/相似度指标

### 5.3 结果分析（后续工作）
- 对比基线（Qwen3.5-4B 原始模型 vs Vision-OPD 训练后）
- 可视化注意力图，验证细粒度特征学习
- 撰写完整复现报告，记录偏差与改进方向

---

## 六、耗时统计

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 环境搭建 | 2 天 | 含依赖调试 |
| 数据准备 | 0.5 天 | 下载 + 预处理 |
| 训练调试 | 1 天 | 解决词表/CUDA/链接问题 |
| flash-attn 编译 | 0.5 小时 | 源码编译 |
| 训练（预计） | 3-5 小时 | 260 步 |
| 评估（预计） | 1 天 | 推理 + 指标计算 |
| **总计** | **约 5 天** | 含等待时间 |

---

## 七、总结

Vision-OPD 复现项目已完成环境搭建、数据准备、训练调试，**成功实现稳定训练**。过程中解决了多个环境依赖问题（词表加载、CUDA 版本、驱动库链接、flash-attn 编译），积累了 H200 服务器部署经验。训练持续运行中，后续将完成评估与结果分析。

---
