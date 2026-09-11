# Idea2 概述：多模态 OPD 独有现象研究

> 创建时间：2026-09-02
> 方向：多模态在线策略蒸馏（MOPD）独有现象研究
> 状态：前期准备中

---

## 一、研究动机

### 1.1 从 idea1 到 idea2 的转向

idea1（多模态 OPD 方法改进）在 2026-09-02 被废弃，原因详见 `idea1/idea1收尾报告-20260902.md`。核心问题：OPD 方向在 2024-2026 进入爆发期，单模态 OPD 的改进空间已被大量工作占满（教师分歧处理、边际递减、能力整合、训练稳定性等细分问题都已有专文研究）。idea1 的 5 个分支本质上都是已有 trick 在多模态 OPD 中的组合应用，方法贡献不足以构成独立论文。

### 1.2 新定位：现象研究而非方法改进

**核心观察**：现有 sophisticated 的 OPD 改进方法（SA-OPD、Open-MOPD、ReNIO、Matching Supervision 等）**几乎全是单模态（纯文本 LLM）的**。真正做多模态 OPD 的工作只有 H-OPD 和 Vision-OPD 两篇。

**推论**：多模态 OPD 是相对蓝海。单模态 OPD 里已被深入理解的现象（教师分歧、边际递减、能力整合鸿沟等），在多模态场景下**是否成立、是否以新的形式存在、是否会出现单模态没有的新现象**——这些问题目前几乎没人回答过。

**研究 thesis**：

> 多模态 OPD 不是单模态 OPD 的简单扩展。由于跨模态语义对齐的介入，多模态 OPD 会产生单模态 OPD 中不存在的新现象、新失败模式、新机会。系统性地揭示这些现象，是一类未被占据的研究贡献。

---

## 二、核心研究问题

### 2.1 方向 A（已选定）：跨模态教师对齐动力学

**核心问题**：在多模态 OPD 训练过程中，VL 教师与 Text 教师之间的"语义对齐度"如何演化？

**具体子问题**：
1. **整体趋势**：对齐度随训练步数单调上升/下降？是否存在 plateau？是否存在**相变点**（某阶段对齐度突然跳变）？
2. **任务依赖性**：不同任务类型（视觉 grounding / 图表理解 / OCR / 常识问答）的对齐演化模式是否不同？是否存在"某类 token 一直低对齐，某类 token 对齐快速上升"的模式？
3. **性能预测**：对齐动力学曲线是否能预测最终学生性能？如果能，可作为训练监控信号。
4. **机制分析**：Attention 模式是否随训练演化？Text 教师是否开始"学"VL 教师的注意力模式？

### 2.2 备选方向（暂不推进，留作 extension）

- **方向 B**：视觉信息传递的机制分析——VL 教师的"视觉能力"是如何传递到学生模型的？是否存在"视觉捷径"？
- **方向 C**：模态不对称下的反直觉监督——是否存在某些 token，Text 教师（看不见图）比 VL 教师（看得见图）给出更可靠的监督？

---

## 三、方法设计（初步）

### 3.1 对齐度量化指标

建议同时使用多个 metric，互相印证：

| Metric | 公式 | 含义 | 优点 |
|--------|------|------|------|
| **JS 散度**（主 metric） | $D_{JS}(p_{VL} \parallel p_{Text})$ | 分布整体差异 | 对称、bounded、数值稳定 |
| **Top-K Jaccard** | $\|J_{VL} \cap J_{Text}\| / \|J_{VL} \cup J_{Text}\|$ | top-k token 集合重叠度 | 直观、robust |
| **Top-1 一致率** | $\mathbb{1}[\arg\max p_{VL} = \arg\max p_{Text}]$ | 两教师最高概率 token 是否相同 | 最简单、易解释 |
| **Attention 相似度**（supplementary） | $\cos(\text{attn}_{VL}, \text{attn}_{Text})$ | 注意力模式相似性 | 机制层面洞察 |

### 3.2 实验设计

#### 3.2.1 训练配置

- 沿用 Vision-OPD 复现的 verl 框架（`repro` 分支的 `code/verl/`）
- 在训练循环里**额外加载 Text 教师**（参考 idea1 `mcd` 分支的 `dp_actor.py`，已实现 text_teacher 加载逻辑）
- 每 N 步（如每 10 步）对**固定的 probe batch** 跑两个教师的 forward，记录输出分布

#### 3.2.2 Probe batch 设计

- 从训练集里**固定抽 100-200 条样本**（整个训练过程用同一个 probe batch，保证可比性）
- 样本要**覆盖不同任务类型**：视觉 grounding / 图表理解 / OCR / 常识问答
- 这样既能看整体趋势，也能按任务类型分组看

#### 3.2.3 数据记录

每一步每个 probe token，存：
```python
{
    'step': int,
    'token_id': int,
    'token_text': str,
    'task_type': str,           # 人工或自动分类
    'p_vl_topk': List[float],   # VL 教师 top-k 概率
    'p_text_topk': List[float], # Text 教师 top-k 概率
    'js_divergence': float,
    'top1_agree': bool,
    'topk_jaccard': float,
}
```

**数据量估算**：100 probe 样本 × 平均 200 token × 260 步 / 10（采样间隔）= ~520k 条记录，很小。

#### 3.2.4 分析维度

1. **整体对齐度 vs 训练步数曲线**
2. **按任务类型分组**的对齐演化
3. **对齐度与最终性能的相关性**（多组超参配置）
4. **相变点检测**（CUSUM / Bayesian change point detection）
5. **Attention 相似度分析**（supplementary）

---

## 四、预期结果与可发表性

| 发现类型 | 可发表性 | 怎么包装 |
|---------|---------|---------|
| 找到相变点：对齐度在某训练阶段突然跳变 | ⭐⭐⭐⭐⭐ 顶会主会 | "We discover a phase transition in cross-modal teacher alignment during OPD training" |
| 不同任务类型的对齐演化不同 | ⭐⭐⭐⭐ 顶会/二档 | "Task-dependent alignment dynamics reveal modality-specific learning bottlenecks" |
| 对齐动力学能预测最终性能 | ⭐⭐⭐⭐ 顶会/二档 | "Alignment dynamics as a training monitor" |
| 对齐度整体平稳，但某类 token 异常 | ⭐⭐⭐ 二档/workshop | "Localized alignment anomalies in multimodal OPD" |
| 完全找不到任何 pattern | ⭐⭐ workshop | "Characterizing cross-modal alignment in OPD: a null study" |

**风险评估**：这个方向的风险**很低**——只要数据不是完全乱，总能讲出 story。最差情况（null result）也能发 workshop。

---

## 五、时间规划

| 阶段 | 时长 | 任务 |
|------|------|------|
| 1. 精读 + 问题打磨 | 2 周 | 精读 Open-MOPD, SA-OPD, H-OPD 三篇，把方向 A 收敛到具体问题 |
| 2. 实验设计 + 工具准备 | 2 周 | 在 verl 框架上加 probe 日志；准备 probe batch |
| 3. 实验 + 数据收集 | 4-6 周 | 跑训练、做分析、收集现象证据 |
| 4. 写作 | 2-3 周 | 成文 |

**总时长**：10-13 周 ≈ 一个学期

**投稿目标**（按优先级）：
1. ECCV 2026（多模态顶会，与方向天然契合）
2. CVPR 2027 / ICLR 2027 主会（如果现象发现够惊人）
3. NeurIPS 2026 Workshop on Multimodal LLMs / CVPR 2027 Workshop（保底）

---

## 六、资源复用

Idea2 可直接复用以下资源：

- **verl 训练框架**：从 idea1 `mcd` 分支的 `code/mcd/` 复制
- **双教师加载逻辑**：`dp_actor.py` 中 `self.text_teacher` 段
- **Vision-OPD 数据集**：`repro` 分支的 `reproduction/1. Vision-OPD/`
- **评估流水线**：`repro` 分支的评估脚本

---

## 七、下一步

1. 精读 Open-MOPD, SA-OPD, H-OPD 三篇论文
2. 设计 probe batch 挑选策略
3. 在 verl 框架里加 probe 日志（参考 mcd 的 text_teacher 集成）
4. 跑 dry run（10 步训练 + 1 次 probe），验证 pipeline 通

---

## 八、参考文献

核心必读：
- Open-MOPD: arXiv 2608.19098
- SA-OPD: arXiv 2608.03632
- H-OPD: arXiv 2607.02592
- Vision-OPD: arXiv 2605.18740

相关扩展：
- Lightning OPD 2.0: arXiv 2607.28449
- ReNIO: arXiv 2606.23104
- Matching Supervision to Student Capacity: arXiv 2608.08176
- Trust Region Policy Distillation: arXiv 2607.04751
- Revisiting OPD: Empirical Failure Modes: arXiv 2603.25562
- Teacher-Informed Multi-Teacher KD: arXiv 2605.27967
