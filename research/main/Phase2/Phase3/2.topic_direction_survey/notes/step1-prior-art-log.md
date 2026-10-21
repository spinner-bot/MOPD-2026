# 工作2 第1步：双教师/多教师 OPD 先验核查——过程日志

**日期**：2026-10-22（UTC+8）
**执行**：Claude Code (local)
**目的**：核查导师"一个学生、两个教师（视觉 + 简洁）"设想是否已被做过，为选题方向调研提供依据。

---

## 1. 检索范围与查询

| 时间 | 来源 | 查询/动作 |
|---|---|---|
| 10-21 | WebSearch | `multi-teacher on-policy distillation LLM privileged information collaboration` |
| 10-21 | WebSearch | `dual teacher distillation concise reasoning vision language model system prompt compression` |
| 10-21 | WebSearch | `DualOPSD adaptive privileged teachers on-policy self-distillation arXiv` |
| 10-21 | WebSearch | `DOPD dual on-policy distillation arXiv 2026 privileged teacher advantage` |
| 10-22 | WebSearch | `"on-policy self-distillation" conciseness teacher brevity length compression visual teacher` |
| 10-22 | GitHub raw | `awesome-on-policy-distillation/README.md`（1008 行，2026 年 OPD 论文清单） |
| 10-22 | arXiv API | 批量核验 12 个编号的标题、日期、comment |
| 10-22 | 原文 PDF | H-OPD（2607.02592v3，18 页）全文抽取，读摘要、3.1–3.4 节、仲裁公式 |

## 2. 核验记录

| arXiv | 标题 | 状态/备注 |
|---|---|---|
| 2607.02592 | H-OPD: Confidence Aware Heterogeneous Multi-Teacher Multimodal On-policy Distillation | **EMNLP 2026**（arXiv comment 明确） |
| 2608.26019 | DualOPSD: Closing the Feedback Loop in On-Policy Self-Distillation | 预印本（v2） |
| 2606.30626 | DOPD: Dual On-policy Distillation | 预印本 |
| 2607.19046 | Contrastive On-Policy Distillation（COPD/COPSD） | 预印本，work in progress |
| 2603.05433 | CRISP: Compressed Reasoning via Iterative Self-Policy Distillation | 预印本（v7），即会上讨论的"260304333" |
| 2609.02548 | Learn from Whoever Is Right（MT-SDPO） | 预印本 |
| 2609.15404 | Who Teaches Which Token?（VG-OPD） | 预印本 |
| 2605.01347 | MAD-OPD | 预印本，有代码 |
| 2607.18293 | One Student, Many Teachers（PROMPTSD） | 预印本 |
| 2609.39687 | Neighborhood On-Policy Self-Distillation | 预印本 |
| 2605.27115 | Counteraction-Aware Multi-Teacher On-Policy Distillation | 预印本 |
| 2607.27770 | Beyond the Best Teacher: Expanding and Compressing the Reasoning Solution Manifold | 预印本 |
| 2609.37044 | Learning from Think-Mode Advantage via On-Policy Distillation（ThinkOPD） | 预印本 |
| 2607.22629 | Masked Self-Distillation | 预印本 |
| 2608.24987 | D³-MOPD: Dynamic Domain Scheduling | 预印本 |
| 2610.02179 | From Gradients to Capabilities: Understanding Multi-Teacher OPD | 预印本 |

> 说明：会上口述的 `2603.04333` 经核验是 *What Does Flow Matching Bring To TD Learning?*，与本次讨论无关；实际论文是 `2603.05433`（CRISP）。

## 3. 关键发现

1. **"一个学生两个教师"已有正式发表工作**：H-OPD（EMNLP 2026）用 VL 教师 + 文本教师，在同一 student prefix 上做 **token 级置信度仲裁**。架构与导师设想几乎同构。
2. **H-OPD 的文本教师不是"简洁教师"**：它通过 vision-to-language description transfer，把图像内容转成文本描述喂给文本教师，让文本教师专注逻辑推理；目标是多模态推理提升，不压缩输出长度。
3. **多教师冲突已被研究**：Counteraction-Aware MOPD（2605.27115）明确处理多教师梯度冲突；DOPD（2606.30626）研究"特权幻觉"与教师-学生信息不对称；From Gradients to Capabilities（2610.02179）分析多教师信号的梯度/Adam 效应。导师故事里的"两种特权放在一个教师上会冲突"，已有相邻证据，不能当作全新发现。
4. **OPSD 压缩也已被占**：CRISP/OPSDC（单教师简洁）、COPD/COPSD（对比提示 token advantage）、BIRD（warm start）、MPD（教师改写短轨迹）、Masked Self-Distillation、ThinkOPD 等。
5. **未直接检索到"视觉特权教师 + 简洁教师 + 输出压缩"这个具体组合**；它目前是缝隙，但属于已有组件的组合，novelty 取决于能否给出新的机制或新的受控证据。

## 4. 未解决问题

- "两种特权放进同一教师会冲突"目前只是导师的判断，尚未见到直接实验；需要设计对照才能作为文章第一幕。
- H-OPD 的 token 级置信度仲裁是否可直接复用？它衡量"教师不确定性"，不衡量"该 token 是否值得压缩"。
- 视觉信息增益（VIG 式）计算需要两次前向，放进 OPSD 训练循环的成本需要评估。
- 多教师 OPD 文献更新极快（2026 年下半年密集出现），需要在最终报告前再做一次增量检索。

## 5. 过程材料

- H-OPD 原文 PDF：`materials/h-opd-2607.02592.pdf`
- H-OPD 抽取文本：`materials/h-opd-2607.02592.txt`
- 对比矩阵：`outputs/prior-art-matrix.md`
- 工作1的 63 篇索引：`../1.compression_literature_survey/outputs/paper-index.csv`
