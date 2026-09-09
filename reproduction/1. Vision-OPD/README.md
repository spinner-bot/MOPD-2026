# 多模态 OPD 论文复现

> 目录：F:\资源夹\科研\SQS Team\multimodal-OPD-research\reproduction
> 目标硬件：8×H200 (141GB × 8)
> 本地开发：Windows 11 + RTX 5060 Laptop (8GB, 仅代码开发用)

---

## 论文列表

| # | 论文 | arXiv | 类型 | 代码 | 权重 | 复现难度 |
|---|------|-------|------|------|------|---------|
| 1 | Vision-OPD | 2605.18740 | OPSD, MLLM 细粒度视觉 | ✅ | ✅ HF | 🟢 易 |
| 2 | V-Zero | 2606.25319 | OPD, MLLM 无标签 | ✅ | ✅ HF | 🟢 易 |
| 3 | PRISM | 2604.28123 | Black-Box OPD, Multimodal RL | ✅ | ✅ HF | 🟡 中 |
| 4 | DiffusionOPD | 2605.15055 | Flow Matching, 多教师 | ✅ | TBD | 🟡 中 |

---

## 目录结构

```
reproduction/
├── README.md
├── 1. Vision-OPD/
│   ├── URL.txt          # 论文/代码/权重/数据链接
│   ├── 论文解读.md       # 论文内容详解
│   ├── 复现计划.md       # 5阶段复现步骤
│   ├── 学习指南.md       # 新手知识讲解
│   ├── 环境搭建详解.md   # 环境搭建 why&how
│   ├── H200部署.sh       # H200 一键部署脚本
│   ├── venv/            # 本地 Python 虚拟环境
│   └── code/            # 官方代码
├── 2. V-Zero/
│   ├── H200部署.sh
│   ├── venv/
│   └── code/
├── 3. PRISM/
│   ├── H200部署.sh
│   ├── venv/
│   └── code/
└── 4. DiffusionOPD/
    ├── H200部署.sh
    ├── venv/
    └── code/
```

---

## 当前状态

### 本地环境 (Windows)
- [x] Python 3.13 + pip + git
- [x] 4 篇代码已克隆
- [x] 4 篇 venv 已创建
- [x] H200 部署脚本已就绪
- [ ] V-Zero 论文解读
- [ ] PRISM 论文解读
- [ ] DiffusionOPD 论文解读

### H200 服务器
- [ ] SSH 连接修复（密钥需管理员添加）
- [ ] 4 篇环境按序部署

---

## H200 部署顺序

1. Vision-OPD（最优先——最易复现、有预训练权重）
2. V-Zero（4B 模型、无标签、快）
3. PRISM（三阶段流水线，已深度分析）
4. DiffusionOPD（需要~97 GPU-hours）

---

## Git 规范

- 每完成1个最小单元 commit 1次
- commit message 格式：`{N}-{Paper}: {action} {中文说明}`
