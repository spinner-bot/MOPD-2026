# 数据来源清单（Dataset Manifest）

> 本清单用于追溯所有实验依赖的数据来源。下载前请确认每一项已获得官方授权。
>
> 创建时间：2026-09-09
> 创建作者：claude code (local)
>
> 修改时间：2026-09-14 04:14（UTC+8）
>
> 修改作者：Codex (CINECA Leonardo)
>
> 修改原因：Leonardo payload 已迁移到 large scratch 并发生一次项目化顶层重排；修正已失效的 Work 路径，登记真实根与兼容入口，避免后续作业继续引用错误位置。
>
> 状态：既有下载已完成；大文件 payload 使用 Leonardo large scratch，main 保存清单、映射和运行记录；1035 裁决 B2a 人工校准前暂停新增大型下载。

---

## 1. 核心训练数据

### 1.1 Vision-OPD 6.2K（Track A 复现）

| 字段 | 值 |
|---|---|
| **HuggingFace** | `yuanqianhao/Vision-OPD-6K` |
| **GitHub** | `VisionOPD/Vision-OPD` |
| **arXiv** | [2605.18740](https://arxiv.org/abs/2605.18740) |
| **发表状态** | arXiv 预印本（2026-05）；不得标注为 ICCV 2025 |
| **许可** | Apache-2.0 |
| **规模** | 6,241 样本，~37.5 GB |
| **下载** | 使用 §9 的固定 revision `hf download` 流程 |
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
| **下载** | 使用 §9 的固定 revision `hf download` 流程 |
| **备注** | 训练数据另在 `inclusionAI/ZwZ-RL-VQA`（~74K 对）|

### 2.2 HR-Bench 4K / 8K

| 字段 | 值 |
|---|---|
| **HuggingFace** | `DreamMr/HR-Bench`（含两个子集 `hrbench_4k` / `hrbench_8k`）|
| **GitHub** | `DreamMr/HR-Bench` |
| **arXiv** | [2408.15556](https://arxiv.org/abs/2408.15556) |
| **会议** | AAAI 2025 |
| **许可** | ⚠️ **未在卡片中显式声明**，需查看仓库 LICENSE 文件 |
| **规模** | 共 200 样本（100 FSP + 100 FCP）；Hub 仓库约 9.82 GB |
| **下载** | 使用 §9 的固定 revision `hf download` 流程；下载后记录实际采用的 4K/8K 文件 |

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
| **下载** | 使用 §9 的固定 revision `hf download` 流程 |

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
| **下载** | 使用 §9 的固定 revision `hf download` 流程 |

### 3.2 CV-Bench

| 字段 | 值 |
|---|---|
| **HuggingFace** | `nyu-visionx/CV-Bench` |
| **GitHub** | `cambrian-mllm/cambrian`（Cambrian 项目的一部分）|
| **arXiv** | [2406.16860](https://arxiv.org/abs/2406.16860) |
| **会议** | NeurIPS 2024 |
| **许可** | Apache-2.0 |
| **规模** | 2,638 样本，~407 MB |
| **下载** | 使用 §9 的固定 revision `hf download` 流程 |

### 3.3 MMVP

| 字段 | 值 |
|---|---|
| **HuggingFace** | `MMVP/MMVP`（主集，300 图）|
| **GitHub** | `tsb0601/MMVP` |
| **arXiv** | [2401.06209](https://arxiv.org/abs/2401.06209) |
| **会议** | CVPR 2024 Oral |
| **许可** | MIT |
| **规模** | 300 图像（150 CLIP-blind pairs），~3.16 MB |
| **下载** | 使用 §9 的固定 revision `hf download` 流程，并记录修正标签来源 |

⚠️ **已知问题**：索引 99, 100, 279, 280 有 ground-truth 错误。评测应使用修正版 `lmms-lab-eval/MMVP`。

### 3.4 POPE

| 字段 | 值 |
|---|---|
| **HuggingFace** | `lmms-lab-encoder/POPE`（格式化版，~510 MB）|
| **GitHub** | `RUCAIBox/POPE`（官方）|
| **arXiv** | [2305.10355](https://arxiv.org/abs/2305.10355) |
| **会议** | EMNLP 2023 |
| **许可** | MIT |
| **规模** | 官方设计常表述为 9,000 个问题；所列 `lmms-lab-encoder/POPE` 镜像当前为 18,000 行，必须记录选定 split/变体及实际行数 |
| **下载** | 使用 §9 的镜像下载流程；若采用官方仓库格式，单独留档转换过程 |

### 3.5 VisualProbe

| 字段 | 值 |
|---|---|
| **HuggingFace** | `Mini-o3/VisualProbe_Easy` / `_Medium` / `_Hard` / `_train` |
| **GitHub** | `Mini-o3/Mini-o3` |
| **arXiv** | [2509.07969](https://arxiv.org/abs/2509.07969) |
| **会议** | ICLR 2026 |
| **许可** | **CC BY-NC 4.0**（仅非商业用途）|
| **规模** | 各 Hub 子集条数独立变化；以 pinned revision 的实际清单为准 |
| **下载** | 使用 §9 分别下载 Easy、Medium、Hard 与 train，并分别记录 revision |

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
| **下载** | 使用 §9 的固定 revision `hf download` 流程 |

### 4.2 VistaHop

| 字段 | 值 |
|---|---|
| **HuggingFace** | ❌ **无 HF 镜像**，仅 GitHub |
| **GitHub** | `ahang0712/vistahop` |
| **arXiv** | [2606.03273](https://arxiv.org/abs/2606.03273) |
| **会议** | arXiv 预印本（2026-06）|
| **许可** | 论文 CC BY 4.0；**代码许可需查 LICENSE 文件** |
| **规模** | v2（最新）600 图像 / 600 任务 |
| **下载** | 先核实仓库和 LICENSE，再依 §9 的 Git 固定 commit 流程下载 |

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
| **许可** | Hub 镜像标注 Apache-2.0；官方 GitHub README 的使用限制须单独保存并优先遵守，当前按受限数据处理 |
| **规模** | EN：13,366 图 / 29,429 QA；CN：1,889 图 / 5,917 QA |
| **下载** | 使用 §9 的固定 revision `hf download` 流程 |

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

## 7. 基础模型

| 项目 | 官方来源 | 用途 | 下载目标 |
|---|---|---|---|
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | Stage 0、Stage 1、Track A 与主实验共同基线 | `models/Qwen3.5-4B/` |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 正式扩展比较 | `models/Qwen3.5-9B/` |

模型的 processor、tokenizer 和视觉配置须与权重一起下载并固定 revision。不得用 Vision-OPD 的训练后 checkpoint 替代共同基线。

## 8. 下载优先级建议

### Tier 1：项目核心输入
- Qwen3.5-4B、Qwen3.5-9B、Vision-OPD 6K
- MMStar、CV-Bench、MMVP、POPE、ZoomBench、HR-Bench、V* Bench

### Tier 2：完整外部评测
- VisualProbe、M³-VQA、VistaHop、MME-RealWorld EN/CN 完整版

### 下载前置条件（许可待查）
- HR-Bench（确认 LICENSE）
- VisualProbe（确认非商用可接受）
- VistaHop（确认代码许可）
- MME-RealWorld 完整版（93 GB；不以 Lite 版替代全量计划）

---

## 9. 可执行下载与验证流程

项目逻辑路径均登记在 `research/main/`。截至 2026-09-14，数据与模型的真实目录分别是：

```text
/leonardo_scratch/large/userexternal/rqian000/ICLR-2027-czy/data
/leonardo_scratch/large/userexternal/rqian000/ICLR-2027-czy/models
```

为兼容仓库软链接、历史清单绝对路径和已归档命令，large scratch 顶层保留 `data`、`models`、`sqs` 三个相对软链接；仓库通过 `research/main/data/external` 与 `research/main/models` 访问。Python 环境位于 `/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/`。1036 记录了兼容入口恢复和 C1/B0/B2a 共 416 条图片路径与 SHA-256 复核。

不要把全量 payload 写入 `/leonardo/home`。任何后续下载必须先经过 1035 的来源裁决并建立独立 run archive；下载前以 `hf models info` 或 `hf datasets info` 记录具体 revision SHA，再把 SHA 替换进命令中的 `<REVISION>`，不得使用浮动的 `main`。使用 `hf`，不使用已弃用的 `huggingface-cli`。

```bash
hf download Qwen/Qwen3.5-4B --revision <REVISION> --local-dir models/Qwen3.5-4B
hf download Qwen/Qwen3.5-9B --revision <REVISION> --local-dir models/Qwen3.5-9B
hf download yuanqianhao/Vision-OPD-6K --repo-type dataset --revision <REVISION> --local-dir data/raw/vision-opd-6k
hf download inclusionAI/ZoomBench --repo-type dataset --revision <REVISION> --local-dir data/external/zoombench
hf download DreamMr/HR-Bench --repo-type dataset --revision <REVISION> --local-dir data/external/hr-bench
hf download craigwu/vstar_bench --repo-type dataset --revision <REVISION> --local-dir data/external/vstar-bench
hf download Lin-Chen/MMStar --repo-type dataset --revision <REVISION> --local-dir data/external/mmstar
hf download nyu-visionx/CV-Bench --repo-type dataset --revision <REVISION> --local-dir data/external/cv-bench
hf download MMVP/MMVP --repo-type dataset --revision <REVISION> --local-dir data/external/mmvp
hf download lmms-lab-encoder/POPE --repo-type dataset --revision <REVISION> --local-dir data/external/pope
hf download Mini-o3/VisualProbe_Easy --repo-type dataset --revision <REVISION> --local-dir data/external/visualprobe/easy
hf download Mini-o3/VisualProbe_Medium --repo-type dataset --revision <REVISION> --local-dir data/external/visualprobe/medium
hf download Mini-o3/VisualProbe_Hard --repo-type dataset --revision <REVISION> --local-dir data/external/visualprobe/hard
hf download Mini-o3/VisualProbe_train --repo-type dataset --revision <REVISION> --local-dir data/external/visualprobe/train
hf download CASIA-IVA-Lab/M3VQA --repo-type dataset --revision <REVISION> --local-dir data/external/m3-vqa
hf download yifanzhang114/MME-RealWorld --repo-type dataset --revision <REVISION> --local-dir data/external/mme-realworld
git clone https://github.com/ahang0712/vistahop.git data/external/vistahop && git -C data/external/vistahop checkout <COMMIT>
```

每项完成后保存 revision、许可证快照、文件数、总字节数、sha256sum 清单、实际命令和验证输出。Vision-OPD 按数据卡解压 images、teacher_images、original_images，并保存解压前后文件清单。

## 10. 下一步动作

当前先完成 B2a 32 图人工盲校准。校准证明 M3VQA 图片可供性不足、且 1035 决定比较新来源后，才执行以下新增下载动作；不能因现有目录中缺少某项就自动补下。

1. **许可确认**：审阅并留档受限项目的可用性；未确认者不得启动下载
2. **确认待查项**：
   - HR-Bench LICENSE 文件
   - V* Bench SA-1B Research License 条款
   - VistaHop 代码 LICENSE
3. **服务器环境准备**：在 Leonardo 上准备 Python 环境 + `hf` CLI
4. **分批下载**：按 §8 顺序下载全部资源，固定 revision 并存入 §9 的目标目录
5. **下载验收**：每项保存时间、revision、许可快照、文件哈希和 run archive；全部验收后才进入正式实验

## 11. 修订记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-14 04:14 | Codex (CINECA Leonardo) | 修正迁移后的真实数据/模型/环境根，登记兼容入口与 1036 校验，并按 1035 暂停 B2a 前的新增大型下载。 |
