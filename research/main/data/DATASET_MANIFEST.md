# 数据来源清单（Dataset Manifest）

> 本清单用于追溯所有实验依赖的数据来源。下载前请确认每一项已获得官方授权。
>
> 创建时间：2026-09-09
> 状态：待用户确认

---

## 1. 核心训练数据

### 1.1 Vision-OPD 6.2K（Track A 复现）

| 字段 | 值 |
|---|---|
| **HuggingFace** | `yuanqianhao/Vision-OPD-6K` |
| **GitHub** | `VisionOPD/Vision-OPD` |
| **arXiv** | [2605.18740](https://arxiv.org/abs/2605.18740) |
| **会议** | ICCV 2025 |
| **许可** | Apache-2.0 |
| **规模** | 6,241 样本，~37.5 GB |
| **下载** | `huggingface-cli download yuanqianhao/Vision-OPD-6K --repo-type dataset` |
| **用途** | Track A 严格复现 |

⚠️ 存在镜像（`zwyang6/`, `williamium/`），应使用 `yuanqianhao` 作为官方源。

---

### 1.2 自建 P/R/PR 数据（主实验 + Stage 0）

| 字段 | 值 |
|---|---|
| **来源** | 自行生成 |
| **生成模型** | 待定（GPT-5.5 / GPT-5 等） |
| **目标规模** | Train 8K + Val 1K + Test 2K + Stage0 400-600 |
| **许可** | 自持（需记录生成模型版本） |
| **状态** | 未开始（依赖数据生成 pipeline） |

---

## 2. 外部评测集 —— 视觉感知

### 2.1 ZoomBench

| 字段 | 值 |
|---|---|
| **HuggingFace** | `inclusionAI/ZoomBench` |
| **GitHub** | `inclusionAI/Zooming-without-Zooming` |
| **arXiv** | [2602.11858](https://arxiv.org/abs/2602.11858) |
| **会议** | ICML 2026 |
| **许可** | Apache-2.0 |
| **规模** | 845 VQA 样本，~4.03 GB |
| **下载** | `load_dataset("inclusionAI/ZoomBench")` |
| **备注** | 训练数据另在 `inclusionAI/ZwZ-RL-VQA`（~74K 对）|

### 2.2 HR-Bench 4K / 8K

| 字段 | 值 |
|---|---|
| **HuggingFace** | `DreamMr/HR-Bench`（含两个子集 `hrbench_4k` / `hrbench_8k`）|
| **GitHub** | `DreamMr/HR-Bench` |
| **arXiv** | [2408.15556](https://arxiv.org/abs/2408.15556) |
| **会议** | AAAI 2025 |
| **许可** | ⚠️ **未在卡片中显式声明**，需查看仓库 LICENSE 文件 |
| **规模** | 共 200 样本（100 FSP + 100 FCP），~9.82 GB |
| **下载** | `load_dataset("DreamMr/HR-Bench", "hrbench_4k")` / `load_dataset("DreamMr/HR-Bench", "hrbench_8k")` |

⚠️ 注意区分另一篇同名论文 "HRBench"（arXiv:2605.28398，LLM 推理模式），**不是同一数据集**。

### 2.3 V* Bench（VStar Bench）

| 字段 | 值 |
|---|---|
| **HuggingFace** | `craigwu/vstar_bench` |
| **GitHub** | `penghao-wu/vstar` |
| **arXiv** | [2312.14135](https://arxiv.org/abs/2312.14135) |
| **会议** | CVPR 2024 |
| **许可** | 代码 MIT；数据集 SA-1B Research License（需确认） |
| **规模** | 191 MCQ 样本，~380 MB |
| **下载** | `load_dataset("craigwu/vstar_bench")` |

⚠️ 注意区分 `V-STaR-Bench/V-STaR`（视频-LLM 时空推理），**不是同一数据集**。

---

## 3. 外部评测集 —— 通用能力

### 3.1 MMStar

| 字段 | 值 |
|---|---|
| **HuggingFace** | `Lin-Chen/MMStar` |
| **GitHub** | `MMStar-Benchmark/MMStar` |
| **arXiv** | [2403.20330](https://arxiv.org/abs/2403.20330) |
| **会议** | NeurIPS 2024 |
| **许可** | Apache-2.0 |
| **规模** | 1,500 样本，~101 MB（TSV）|
| **下载** | `load_dataset("Lin-Chen/MMStar", split="val")` |

### 3.2 CV-Bench

| 字段 | 值 |
|---|---|
| **HuggingFace** | `nyu-visionx/CV-Bench` |
| **GitHub** | `cambrian-mllm/cambrian`（Cambrian 项目的一部分）|
| **arXiv** | [2406.16860](https://arxiv.org/abs/2406.16860) |
| **会议** | NeurIPS 2024 |
| **许可** | Apache-2.0 |
| **规模** | 2,638 样本，~407 MB |
| **下载** | `load_dataset("nyu-visionx/CV-Bench")` |

### 3.3 MMVP

| 字段 | 值 |
|---|---|
| **HuggingFace** | `MMVP/MMVP`（主集，300 图）|
| **GitHub** | `tsb0601/MMVP` |
| **arXiv** | [2401.06209](https://arxiv.org/abs/2401.06209) |
| **会议** | CVPR 2024 Oral |
| **许可** | MIT |
| **规模** | 300 图像（150 CLIP-blind pairs），~3.16 MB |
| **下载** | `load_dataset("MMVP/MMVP")` |

⚠️ **已知问题**：索引 99, 100, 279, 280 有 ground-truth 错误。评测应使用修正版 `lmms-lab-eval/MMVP`。

### 3.4 POPE

| 字段 | 值 |
|---|---|
| **HuggingFace** | `lmms-lab-encoder/POPE`（格式化版，~510 MB）|
| **GitHub** | `RUCAIBox/POPE`（官方）|
| **arXiv** | [2305.10355](https://arxiv.org/abs/2305.10355) |
| **会议** | EMNLP 2023 |
| **许可** | MIT |
| **规模** | 9,000 样本（500 COCO val × 6 问 × 3 变体）|
| **下载** | `git clone https://github.com/RUCAIBox/POPE.git` 或 `load_dataset("lmms-lab-encoder/POPE")` |

### 3.5 VisualProbe

| 字段 | 值 |
|---|---|
| **HuggingFace** | `Mini-o3/VisualProbe_Easy` / `_Medium` / `_Hard` / `_train` |
| **GitHub** | `Mini-o3/Mini-o3` |
| **arXiv** | [2509.07969](https://arxiv.org/abs/2509.07969) |
| **会议** | ICLR 2026 |
| **许可** | **CC BY-NC 4.0**（仅非商业用途）|
| **规模** | Train ~4K QA / Test ~500（Easy/Medium/Hard）|
| **下载** | `load_dataset("Mini-o3/VisualProbe_Easy")` 等 |

⚠️ 非商业许可，若未来有商业化计划需另行授权。

---

## 4. 外部评测集 —— 长推理

### 4.1 M³-VQA

| 字段 | 值 |
|---|---|
| **HuggingFace** | `CASIA-IVA-Lab/M3VQA`（注意 ASCII 写法，非 M³）|
| **GitHub** | `CASIA-IVA-Lab/M3VQA` |
| **arXiv** | [2604.25122](https://arxiv.org/abs/2604.25122) |
| **会议** | ACL 2026 Main |
| **许可** | MIT（数据集），CC BY 4.0（论文）|
| **规模** | 13,125 triples / 10,565 图 / 7,611 问 |
| **下载** | `load_dataset("CASIA-IVA-Lab/M3VQA")` |

### 4.2 VistaHop

| 字段 | 值 |
|---|---|
| **HuggingFace** | ❌ **无 HF 镜像**，仅 GitHub |
| **GitHub** | `ahang0712/vistahop` |
| **arXiv** | [2606.03273](https://arxiv.org/abs/2606.03273) |
| **会议** | arXiv 预印本（2026-06）|
| **许可** | 论文 CC BY 4.0；**代码许可需查 LICENSE 文件** |
| **规模** | v2（最新）600 图像 / 600 任务 |
| **下载** | `git clone https://github.com/ahang0712/vistahop.git` |

---

## 5. 外部评测集 —— 真实世界

### 5.1 MME-RealWorld EN / CN

| 字段 | 值 |
|---|---|
| **HuggingFace（完整版）** | `yifanzhang114/MME-RealWorld`（~93.5 GB，含 EN+CN）|
| **HuggingFace（精简版）** | `yifanzhang114/MME-RealWorld-Lite`（~8.78 GB）|
| **GitHub** | `MME-Benchmarks/MME-RealWorld` |
| **arXiv** | [2408.13257](https://arxiv.org/abs/2408.13257) |
| **会议** | ICLR 2025 |
| **许可** | ⚠️ **仅限学术研究，禁止商业用途**（GitHub README 声明）|
| **规模** | EN：13,366 图 / 29,429 QA；CN：1,889 图 / 5,917 QA |
| **下载** | `huggingface-cli download yifanzhang114/MME-RealWorld --repo-type dataset` |

⚠️ **重要**：EN 与 CN **不是翻译关系**，是独立收集的不同图像来源。

---

## 6. 许可风险汇总

| 数据集 | 许可 | 风险等级 |
|---|---|---|
| Vision-OPD 6K | Apache-2.0 | ✅ 安全 |
| ZoomBench | Apache-2.0 | ✅ 安全 |
| HR-Bench | **未明确声明** | ⚠️ 需确认 |
| V* Bench | MIT / SA-1B Research | ⚠️ 需确认 SA-1B 条款 |
| MMStar | Apache-2.0 | ✅ 安全 |
| CV-Bench | Apache-2.0 | ✅ 安全 |
| MMVP | MIT | ✅ 安全 |
| POPE | MIT | ✅ 安全 |
| VisualProbe | **CC BY-NC 4.0** | ⚠️ 非商用 |
| M³-VQA | MIT | ✅ 安全 |
| VistaHop | **CC BY 4.0 + 代码许可待查** | ⚠️ 需确认 |
| MME-RealWorld | **仅限学术研究** | ⚠️ 非商用 |

---

## 7. 下载优先级建议

### Tier 1：立即历史下载（许可明确、Stage 0 必需）
- Vision-OPD 6K（Track A 复现必需）
- MMStar、CV-Bench、MMVP、POPE（通用能力评测）

### Tier 2：尽快下载（许可明确、外部评测必需）
- ZoomBench、V* Bench、M³-VQA
- MME-RealWorld-Lite（先下载精简版）

### Tier 3：需确认后下载（许可待查）
- HR-Bench（确认 LICENSE）
- VisualProbe（确认非商用可接受）
- VistaHop（确认代码许可）
- MME-RealWorld 完整版（93 GB，按需决定）

---

## 8. 下一步动作

1. **用户确认**：审阅本清单，确认各项许可可接受
2. **确认待查项**：
   - HR-Bench LICENSE 文件
   - V* Bench SA-1B Research License 条款
   - VistaHop 代码 LICENSE
3. **服务器环境准备**：在 Leonardo 上准备 Python 环境 + `huggingface-cli`
4. **分批下载**：按 Tier 优先级在服务器上下载
5. **记录下载日志**：每次下载记录时间、版本、hash
