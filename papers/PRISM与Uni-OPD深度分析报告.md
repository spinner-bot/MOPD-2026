# PRISM 与 Uni-OPD 深度对比分析报告

> **组会准备用 · 2026/07/16 · 预计阅读时间：45分钟**

---

## 目录
1. [两篇论文的定位与关系](#1-两篇论文的定位与关系)
2. [PRISM 深度解析](#2-prism-深度解析)
3. [Uni-OPD 深度解析](#3-uni-opd-深度解析)
4. [关键对比：相同与不同](#4-关键对比相同与不同)
5. [导师可能追问的20个问题](#5-导师可能追问的20个问题)
6. [报告PPT建议结构](#6-报告ppt建议结构)

---

## 1. 两篇论文的定位与关系

### 1.1 共同的技术背景：On-Policy Distillation (OPD)

两篇论文都围绕 **On-Policy Distillation（在策略蒸馏）** 这一核心技术展开。

**为什么需要OPD？** 传统的大模型后训练流程是 SFT → RLVR 两阶段：
- **SFT（监督微调）**：在静态数据集上做token-level模仿学习，因为训练和推理时的数据分布不一致，存在 **exposure bias（曝光偏差）**——模型在推理时可能进入训练时没见过的错误状态，导致错误级联放大。
- **RLVR（可验证奖励的强化学习）**：只能给出序列级别的奖励信号（答案对/错），无法提供细粒度的token级别归因。

**OPD的解决方案**：让学生模型在自己的on-policy生成轨迹上接受教师模型的token-level监督。这结合了RL的on-policy采样优势和SFT的细粒度信号优势。其核心目标是：

$$J_{OPD}(\theta) = \min_\theta \mathbb{E}_{q \sim \mathcal{D}, \tau \sim \pi_\theta(\cdot|q)} D_{KL}\left(\pi_\theta(\tau|q) \| \pi_T(\tau|q)\right)$$

即：让学生在自己生成的轨迹上，最小化与教师分布的reverse KL散度。

**OPD的梯度可以写成token-level奖励的形式**：
$$r_t^{OPD} = \log \frac{\pi_T(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})}$$

当教师比学生更偏好某个token时（$r_t^{OPD} > 0$），学生向该方向更新；反之则远离。

### 1.2 两篇论文在OPD研究版图中的位置

| 维度 | PRISM | Uni-OPD |
|------|-------|---------|
| **核心关注** | OPD作为SFT和RLVR之间的**中间对齐阶段** | OPD本身作为**独立的后训练范式** |
| **解决问题** | SFT引入的**分布漂移**（尤其多模态场景） | OPD的两个**基础瓶颈**（探索不足+监督不可靠） |
| **技术贡献** | MoE判别器的对抗式OPD | 双视角优化策略（数据平衡+边距校准） |
| **模态范围** | 聚焦多模态（MLLM） | LLM和MLLM统一框架 |
| **OPD角色** | 过渡阶段（pre-alignment） | 独立训练范式 |
| **教师形式** | 黑盒监督池（仅需demonstration） | 白盒教师（需logits访问） |

**关键理解**：PRISM 把 OPD 当作 **管道中的一环**（修复SFT→RLVR之间的gap），而 Uni-OPD 把 OPD 当作 **独立的优化范式** 来系统性研究。

---

## 2. PRISM 深度解析

### 2.1 论文信息

- **标题**：PRISM: Pre-alignment via Black-box On-policy Distillation for Multimodal Reinforcement Learning
- **作者**：Sudong Wang, Weiquan Huang, Xiaomin Yu 等（HKUST(GZ)、清华、NTU等）
- **发表**：arXiv:2604.28123v1, 2026年4月30日
- **代码**：https://github.com/XIAO4579/PRISM

### 2.2 核心动机

**SFT + RLVR 两阶段范式的隐藏问题**：

SFT 虽然给模型注入了基础推理能力，但也将模型推离了其原始分布。这种 **distributional drift（分布漂移）** 在多模态场景下尤为严重：

1. **感知漂移（Perception Drift）**：视觉描述的准确性下降
2. **推理漂移（Reasoning Drift）**：推理路径偏离正确的逻辑链

这两种漂移是 **异质的（heterogeneous）**——它们产生的原因不同，需要不同的矫正信号。但传统的单一判别器或单一矫正目标无法分别处理这两种漂移。

**关键发现**：模型越强，SFT漂移的代价越大。因为强模型本来就有好的推理分布，SFT反而可能破坏其原生优势。

**PRISM的核心思想**：在SFT和RLVR之间插入一个 **distribution alignment（分布对齐）** 阶段，使用对抗式OPD将post-SFT策略拉回监督分布，为RLVR提供更好的初始化。

### 2.3 三阶段管道

```
SFT (冷启动) → Alignment (分布对齐) → RLVR (最终优化)
```

**Stage 1：冷启动SFT**
- 数据：113K Gemini 3 Flash蒸馏数据 + 1.26M公开数据 = 1.37M样本
- 蒸馏要求：每个样本必须包含完整的 `<caption>`（细粒度视觉描述）+ `<think>`（逐步推理）+ `<answer>`
- 目的：建立初始推理能力，缩小与监督分布的差距，使后续对抗训练可行

**Stage 2：对抗式OPD对齐（核心贡献）**
- 持续500步
- 使用 **MoE判别器**（感知专家 + 推理专家）与策略进行 **minimax博弈**
- 判别器：学习区分策略rollout和监督数据
- 策略：最大化判别器给出的奖励（即让rollout更接近监督数据）
- **Black-box设计**：不需要教师logits，只需要监督pool中的demonstration样本
- **去掉KL正则化**：因为目标就是让分布发生偏移（向监督分布靠拢）

**Stage 3：RLVR**
- 在2K难度筛选的样本上做outcome-based RL（GRPO/DAPO/GSPO）
- 奖励 = 答案准确性 + 格式合规性

### 2.4 MoE判别器设计（核心技术）

判别器包含两个专家：
- **感知专家 $D_v$**：评估视觉描述 $c$ 的准确性
- **推理专家 $D_r$**：评估推理链 $t$ 的一致性

MoE奖励：
$$r(x, y) = \alpha \cdot D_v(x, c) + (1-\alpha) \cdot D_r(x, t)$$

判别器训练（Bradley-Terry Loss）：
$$\mathcal{L}_{D_k} = -\mathbb{E}_{(x, y^+, y^-) \sim \mathcal{T}} \log \sigma\left(D_k(x, y_k^+) - D_k(x, y_k^-)\right)$$

策略更新（GRPO-style，组内归一化优势）：
$$A_i = \frac{r(x, y_i^-) - \text{mean}\{r(x, y_j^-)\}_{j=1}^N}{\text{std}\{r(x, y_j^-)\}_{j=1}^N}$$

### 2.5 关键实验结果

| 配置 | 4B Avg | 8B Avg |
|------|--------|--------|
| Instruct（基座） | 59.7 | 63.3 |
| SFT only | 56.8 | 58.1 |
| SFT+GRPO（基线） | 61.8 | 63.3 |
| **PRISM+GRPO** | **66.2** | **69.3** |
| **改进** | **+4.4** | **+6.0** |

- PRISM对齐后的checkpoint单独看准确率并不比SFT高（因为对齐的目标是分布修正而非答案正确率）
- 对齐的收益在RLVR阶段才体现出来
- 基于PRISM的RLVR比SFT+GRPO使用更少的token（更简洁有效的推理）
- 8B模型受SFT漂移影响更大（因为8B的原生推理能力更强）

### 2.6 消融实验关键结论

1. **MoE > 密集判别器**（-3.4 avg）：单一判别器将感知和推理反馈折叠为单一标量，梯度信号含噪；MoE让每个专家专精一个维度
2. **三阶段缺一不可**：去掉对齐→-4.4；去掉SFT→-16.8（差距太大，对抗训练失败）
3. **视觉-语言判别器 > 纯文本判别器**：纯文本判别器只能捕捉文本表层模式（"鹦鹉对齐"），无法验证视觉描述是否忠实于图像
4. **更多SFT数据帮助对齐**：仅107K→-3.7 avg，SFT与对齐是互补的

### 2.7 训练动力学分析

- 感知专家收敛早且快（视觉对齐相对简单）
- 推理专家上升更缓慢且有震荡（推理对齐涉及更复杂的分布修正）
- 两者最终收敛到相似的平衡水平
- 这种异步收敛进一步验证了MoE设计的必要性

---

## 3. Uni-OPD 深度解析

### 3.1 论文信息

- **标题**：Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe
- **作者**：Wenjin Hou, Shangpin Peng, Weinong Wang 等（浙大、Tencent）
- **发表**：arXiv:2605.03677v2, 2026年7月6日
- **代码**：https://github.com/WenjinHou/Uni-OPD

### 3.2 核心动机

**OPD为什么有时有效，有时失效？**

Uni-OPD从第一性原理出发，将OPD的成功条件分解为两个角色视角：

1. **学生视角（Student Perspective）**：学生必须充分探索 **informative states（有信息量的状态）**——即多样且难度适中的自生成轨迹
2. **教师视角（Teacher Perspective）**：教师的token级监督在学生rollout上必须保持 **可靠（reliable）**

当这两个条件不满足时，OPD就会失效。

**教师监督不可靠的三个典型失败模式**：
1. **OOD退化**：学生rollout进入教师训练分布之外的稀疏区域，教师logits变得噪声化
2. **高估错误轨迹**：错误轨迹因局部token模式碰巧匹配教师高置信区域而获得异常高分
3. **低估正确轨迹**：正确轨迹因生成路径与教师主流路径不同而被异常低分

### 3.3 双视角优化策略

#### 学生视角：联合离线和在线数据平衡

**离线难度感知数据平衡（Offline Difficulty-aware Data Balancing）**：

RL中的常见做法是筛掉太简单（全对）或太难（全错）的样本。但Uni-OPD发现：
- 小模型的训练数据常呈现 **镜像J形或U形分布**（大多数样本集中在极难或极易两端）
- 直接过滤会**大幅减少数据多样性**，性能显著下降
- **方案**：对中等难度样本（多次rollout中部分正确）进行**上采样**，reshape成一个更均匀的分布
- 效果：保留数据多样性和难度谱，让学生探索更广阔的解空间

**在线正确率感知数据平衡（Online Correctness-aware Data Balancing）**：

- 训练时每个rollout组内强制保持**正确和错误轨迹的平衡组合**
- 防止退化情况（如所有样本相同结果→无信息量梯度）
- 最佳比例：correct/incorrect ≈ 0.75–1.0

#### 教师视角：结果引导的边距校准

**核心概念：Order Consistency（顺序一致性）**

定义轨迹级蒸馏回报：
$$G_{OPD}(q, \tau) = \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} \log \frac{\pi_T(o_t|q, o_{<t})}{\pi_\theta(o_t|q, o_{<t})}$$

关键判据：对于任意prompt $q$，期望满足：
$$G_{OPD}(q, \tau^+) \geq G_{OPD}(q, \tau^-), \quad \forall \tau^+ \in S^+(q), \tau^- \in S^-(q)$$

即：**所有正确轨迹的蒸馏回报应该高于所有错误轨迹**。

定义prompt-level margin：
$$m(q) = \min_{\tau \in S^+(q)} G_{OPD}(q, \tau) - \max_{\tau \in S^-(q)} G_{OPD}(q, \tau)$$

目标：$m(q) \geq \delta$（$\delta > 0$ 是安全边距）

**两种校准策略**：
1. **Margin Mask（边距掩码）**：只保留满足 $m(q) \geq \delta$ 的prompt组，丢弃违反的组
2. **Margin Shift（边距平移）**：对违反的组，给所有正确轨迹的回报加上最小修正量 $\Delta(q) = \delta - m(q)$，恢复顺序一致性

实验表明Margin Shift略优于Margin Mask，被采用为主方法。

### 3.4 实验设计

Uni-OPD的实验设计极其全面，涵盖：

| 蒸馏场景 | 描述 |
|---------|------|
| 单教师蒸馏 | LLM(MATH/CODE) + MLLM(MATH/LOGIC/DOC) |
| 多教师蒸馏 | 合并多个领域专家的能力到一个学生 |
| 强到弱蒸馏 | 30B teacher → 4B/1.7B student |
| 跨模态蒸馏 | 文本+多模态教师联合训练一个MLLM学生 |

模型：Qwen3-4B/1.7B（文本）、Qwen3-VL-4B/2B（多模态）
基准：5个能力域、16个benchmark

### 3.5 关键实验结果

**单教师蒸馏（LLM）**：
- Math Avg: OPD 44.6 → Uni-OPD 48.7 (+4.1)
- Code Avg: OPD 60.2 → Uni-OPD 63.2 (+3.0)

**多教师蒸馏（LLM）**：
- Math Avg: OPD 46.6 → Uni-OPD 48.5 (+1.9)
- Code Avg: OPD 60.5 → Uni-OPD 63.6 (+3.1)

**强到弱蒸馏（30B → 4B）**：
- Math Avg: OPD 41.2 → Uni-OPD 42.9 (+1.7)
- Code Avg: OPD 59.0 → Uni-OPD 60.8 (+1.8)

**跨模态蒸馏（MLLM学生）**：
- Code Avg: OPD 64.1 → Uni-OPD 65.6 (+1.5)
- Math Avg: OPD 62.7 → Uni-OPD 63.9 (+1.2)

**收敛速度**：Uni-OPD比RL快得多（在更少的优化步数内达到更好性能）

### 3.6 消融实验关键结论

1. **离线数据平衡**：去掉后性能下降，验证了保留难度多样性的必要性
2. **在线数据平衡**：去掉后性能下降，验证了组内正负样本平衡的重要性
3. **边距校准**：去掉后下降最明显，验证了教师监督可靠性是OPD的关键瓶颈
4. **Margin Shift > Margin Mask**：两者都优于基线，Shift略好于Mask

### 3.7 核心Takeaways

1. **平衡推理能力与效率**：Uni-OPD用更少的优化步数达到更好的性能
2. **教师价值来自能力差距，而非绝对强度**：即使相同backbone的RL教师，也能通过能力差距驱动学生提升甚至超越教师
3. **OPD蒸馏推理是模态无关的能力**：文本和多模态数据联合训练下，per-token信号携带的推理模式独立于模态
4. **OPD干净地合并专门化能力，相关能力互相强化**：可以扩展到三个教师，联合提升所有能力

---

## 4. 关键对比：相同与不同

### 4.1 共同点

| 方面 | 共性 |
|------|------|
| **技术范式** | 都基于 On-Policy Distillation |
| **核心洞察** | 都认识到标准post-training pipeline（SFT→RLVR）存在问题 |
| **优化目标** | 都使用 reverse KL divergence 作为OPD的核心目标 |
| **算法基础** | 都使用 GRPO-style 的策略优化 |
| **模型基础** | 都使用 Qwen3 系列模型进行实验验证 |
| **问题意识** | 都关注分布匹配的质量，而不仅仅是最终答案正确率 |

### 4.2 核心差异

| 维度 | PRISM | Uni-OPD |
|------|-------|---------|
| **OPD定位** | 中间过渡阶段（pre-alignment） | 独立后训练范式 |
| **教师需求** | 黑盒（仅需demonstration样本） | 白盒（需logits访问） |
| **核心技术** | 对抗式MoE判别器 | 数据平衡 + 边距校准 |
| **模态范围** | 聚焦多模态 | LLM + MLLM统一 |
| **问题诊断** | SFT引入分布漂移（多模态下异质化） | OPD的两个基础瓶颈（探索+监督） |
| **解决路径** | 插入新阶段（三阶段管道） | 优化现有OPD训练过程 |
| **判断标准** | 下游RLVR性能提升 | 直接蒸馏性能提升 |
| **对抗训练** | 是（minimax game） | 否（直接校准） |
| **多样性** | 1.37M多模态数据 | 5域16基准 |
| **规模** | 4B/8B | 1.7B/4B/30B teacher |

### 4.3 互补关系

**PRISM可以做但Uni-OPD不做的**：
- 利用对抗训练实现黑盒蒸馏（不需要教师logits）
- 分离感知与推理的矫正信号（MoE判别器）
- 将OPD作为管道中间阶段而非终端目标

**Uni-OPD可以做但PRISM不做的**：
- 多教师能力合并
- 强到弱蒸馏
- 跨模态蒸馏
- 系统性地分析和解决OPD的底层瓶颈

**两者结合的可能性**：
- Uni-OPD的数据平衡和边距校准策略可以直接增强PRISM的对齐阶段
- PRISM的MoE判别器设计可以引入Uni-OPD作为教师监督的补充
- Uni-OPD的order consistency概念可以用来分析PRISM的判别器质量

---

## 5. 导师可能追问的20个问题

### 基础理解类（5题）

**Q1：什么是On-Policy Distillation？和普通蒸馏有什么区别？**
> OPD让学生在**自己的生成轨迹**上学习教师分布（reverse KL），而非在教师预生成的静态数据上学习（forward KL）。这消除了exposure bias，因为训练时的数据分布来自学生自身的rollout。普通蒸馏（如标准SFT）是off-policy的——教师先生成数据，学生再在这些数据上训练，训练和推理分布不一致。

**Q2：为什么SFT会导致distributional drift？**
> SFT用uniform token-level loss优化模型去模仿demonstration policy的轨迹，不区分过程和结果。模型可能学到表层模式而非忠实推理能力，同时偏离其原始分布。在多模态场景下尤为严重——视觉感知和逻辑推理的漂移模式不同（heterogeneous drift）。

**Q3：PRISM为什么要用MoE判别器而不是单一判别器？**
> 多模态推理中的错误有两个来源：感知错误（看错）和推理错误（想错）。单一判别器将两者折叠为一个标量，当策略在一个维度提升但另一个维度退化时，梯度信号含噪。MoE让感知专家和推理专家各自专精，提供解耦的矫正信号。实验证明：密集判别器在WeMath上比MoE差6个点。

**Q4：Uni-OPD的"order consistency"是什么意思？**
> 正确的推理轨迹应该比错误的轨迹获得更高的教师评分（蒸馏回报）。即 $G_{OPD}(\tau^+) > G_{OPD}(\tau^-)$。"Order-consistent"意味着这个排序在所有prompt的所有轨迹上都成立。当教师的token级评分在轨迹级聚合后违反了这一排序时（如错误轨迹因局部匹配而得分高于正确轨迹），监督就不可靠了。

**Q5：PRISM和Uni-OPD中OPD的目标函数有何不同？**
> PRISM使用对抗式目标（minimax game with Bradley-Terry loss），判别器输出标量奖励，策略用policy gradient优化。Uni-OPD使用标准的reverse KL散度目标，教师提供token-level log probabilities。PRISM是黑盒的（不需要logits），Uni-OPD是白盒的（需要logits）。

### 深入技术类（5题）

**Q6：PRISM为什么在对齐阶段去掉KL正则化？**
> 标准RLHF/RLVR中KL正则化用于防止策略偏离SFT初始化太远。但PRISM的对齐目标恰恰是**修正SFT引入的分布漂移**——策略需要**向监督分布移动**。如果加KL正则化锚定在SFT checkpoint，就与对齐目标直接矛盾。因此PRISM显式设置KL coefficient=0。

**Q7：PRISM的判别器是如何初始化的？为什么需要预热？**
> 两个专家都从预训练backbone初始化，并在各自的偏好对上预热（感知专家在视觉描述对、推理专家在推理链对上）。需要预热的原因是：如果策略离监督分布太远，判别器可以轻易区分两者，迅速饱和，导致策略梯度消失（这是GAN训练中的经典问题）。预热缩小了差距，使对抗训练可行。额外使用load-balancing loss防止专家坍塌。

**Q8：Uni-OPD的离线数据平衡和数据过滤有什么区别？为什么平衡更好？**
> 数据过滤直接**丢弃**全对或全错的样本，在分布呈J形或U形时会大幅减少多样性。数据平衡通过**上采样**中等难度样本，保留了两端的样本。多样性对于OPD至关重要——学生需要接触到不同难度的状态才能充分探索。实验证明过滤导致性能下降，而平衡带来提升。

**Q9：Uni-OPD的margin mask和margin shift各自的优劣是什么？**
> Margin Mask更保守：只使用满足order consistency的prompt组，丢弃违反的组。这保证了训练数据的质量，但可能浪费大量数据（尤其在训练早期，大多数组可能不满足条件）。Margin Shift更激进：给违反的组施加最小修正来恢复一致性，保留了所有训练数据。实验表明Margin Shift略好，说明保留更多训练信号比保守丢弃更有益。

**Q10：PRISM的token效率从何而来？为什么对齐后模型用更少的token达到更高准确率？**
> 对齐阶段让模型的视觉描述更精确、推理链更符合监督分布的结构。进入RLVR后，模型不需要"试错"那么长的推理链——因为它的推理路径已经接近正确的分布形态。本质上，对齐将搜索空间从"找到任何能得出正确答案的推理"缩小为"在正确的推理分布附近优化"。

### 实验设计类（5题）

**Q11：PRISM的三阶段为什么缺一不可？每一阶段的具体作用是什么？**
> - **SFT**：提供初始能力和缩小分布差距。没有SFT，差距太大，对抗训练失败（-16.8 avg），判别器平凡区分，策略退化。
> - **Alignment**：修正SFT引入的残余分布漂移。没有对齐，RLVR需要从扭曲的分布开始优化，最终性能差-4.4 avg。
> - **RLVR**：将分布正确的推理转化为正确答案。对齐修正了"how to reason"，RLVR优化"whether the reasoning leads to correct answer"。
> 三者是 "能力 → 分布 → 结果" 的递进关系。

**Q12：PRISM为什么用Gemini 3 Flash蒸馏数据，而不是直接用开源数据？**
> 现有开源多模态数据质量不足：很多只有简短答案、不完整推理链、或不精确的视觉描述。PRISM需要每个样本包含完整的视觉grounding + 逐步推理 + 正确答案（三部分结构化输出），以便在SFT和对齐阶段复用。此外，PRISM专门针对当前强模型零通过率的难题进行蒸馏，确保数据的高难度和高信息量。

**Q13：PRISM中为什么RLVR只用2K样本，而SFT用了1.37M？**
> 两个阶段的目标不同：SFT需要广泛覆盖来建立初始能力（广度优先）；RLVR需要精准优化——2K样本是从6K中经过difficulty filter筛选的（pass rate在[0.2, 0.8]），确保RLVR训练在"既不简单到无需优化，也不难到无从学起"的样本上进行（难度适中原则）。

**Q14：Uni-OPD的强到弱蒸馏实验中，1.7B学生从30B教师蒸馏的效果如何？对你有什么启示？**
> 1.7B学生从15.9（自身）提升到25.0（math avg, +9.1），但仍远低于30B教师的58.3。这揭示了OPD的scale limitation——能力差距过大时，学生难以吸收教师的全部能力。但同时，code从8.9→52.7（+43.8!），说明某些领域的能力更容易通过OPD传递。启示：(1) OPD对能力差距敏感；(2) 不同领域的能力可迁移性不同；(3) 对于极小模型，OPD仍有显著提升但无法弥合大规模差距。

**Q15：跨模态蒸馏实验中，文本code能力和多模态math能力在同一个MLLM学生中是如何共存的？**
> Uni-OPD显示两者不仅共存，还**同时提升**（code: 64.1→65.6, math: 62.7→63.9）。这说明OPD的per-token信号携带的推理模式是**模态无关**的——逻辑推理的结构（分步、验证、纠错）在文本和多模态任务间共享。多教师OPD相当于将多种推理模式注入同一参数空间，相关能力相互强化。

### 批判思维类（5题）

**Q16：PRISM的对抗训练是否稳定？论文有没有讨论训练不稳定性的问题？**
> 论文讨论了稳定性问题并采取了多项措施：(1) 判别器和策略都需要初始化预热，否则会出现模式坍塌；(2) 训练动力学图显示推理专家的reward gap先剧烈震荡后收敛，说明训练确实存在不稳定性；(3) 论文仅训练500步就停止，暗示更长训练可能导致退化。但论文没有系统性地讨论GAN训练的经典问题（如mode collapse、non-convergence），也没有尝试Wasserstein GAN等更稳定的对抗训练方法。这是一个潜在的局限性。

**Q17：PRISM的MoE判别器设计是否过度依赖结构化输出格式（<caption> + <think> + <answer>）？**
> 是，这是一个明确的局限。论文在limitations中承认：MoE设计依赖显式的视觉描述和推理链分解，对不自然支持这种分解的任务适用性有限。如果你的任务没有自然的结构化输出（如开放式对话），PRISM的判别器设计需要重新考虑。论文建议未来探索自动或学习式的判别器反馈分解。

**Q18：Uni-OPD的order consistency假设是否总是成立？有没有合理的反例？**
> 这是一个很深的问题。可能存在反例：(1) "幸运的正确"——推理链错误但巧合得出正确答案，其蒸馏回报可能很高（因为局部token匹配教师）而答案正确，打破了排序；(2) "接近正确的错误"——推理链几乎全对但在最后一步出错，其蒸馏回报可能高于一个粗糙但最终正确的推理。Uni-OPD的边距校准无法处理这两种情况——它假设outcome reward是排序的gold standard，但outcome和process quality之间确实存在gap。论文在3.1节提到了这些failure modes但没有给出完美的理论解决方案。

**Q19：为什么8B模型受SFT漂移影响比4B更大？这对实际应用有什么启示？**
> 8B模型的原生推理分布已经很强（Instruct 63.3 avg vs 4B 59.7）。SFT用一个外部demonstration policy强行覆盖这个原生分布时，不仅没有增益，还破坏了原生能力。这是"强模型不需要太多SFT"的证据。实际启示：(1) 模型越强，SFT越要谨慎（可能需要更少的数据或更弱的优化强度）；(2) 对于强模型，可能考虑跳过SFT直接OPD或RL；(3) 评估SFT效果不应只看SFT后的准确率，还应该看下游RLVR的增益。

**Q20：如果让你设计一个结合PRISM和Uni-OPD优势的新方法，你会怎么做？**
> 一个自然的设计：
> 1. 保留PRISM的三阶段管道框架（SFT→Alignment→RLVR）
> 2. 在对齐阶段引入Uni-OPD的优化策略：
>    - 离线/在线数据平衡：确保对抗训练的prompt和rollout具有充分的难度多样性
>    - 边距校准：用outcome reward校准判别器的打分，确保order consistency（而不是完全依赖对抗信号）
> 3. 用Uni-OPD的多教师设置扩展PRISM：多个领域专家的监督pool同时参与对齐
> 4. 将PRISM的MoE判别器迁移到Uni-OPD的框架中（作为教师信号的一种）
>
> 理论上的优势：对抗式OPD提供了分布级别的匹配，边距校准提供了样本级的可靠性保障，两者互补。

---

## 6. 报告PPT建议结构

### 建议结构（约25-30页，45分钟报告）

**Part 0：背景与动机（5页）**
1. 标题页
2. LLM/MLLM后训练的标准范式：SFT → RLVR 及其问题
3. On-Policy Distillation的核心思想与公式
4. 两篇论文的共同背景与不同切入点
5. 本报告的结构

**Part 1：PRISM（8页）**
6. PRISM的核心动机：SFT引入的分布漂移
7. 多模态场景下漂移的异质性（感知 vs 推理）
8. 三阶段管道概览
9. MoE判别器设计（重点页）
10. 对抗式OPD的minimax博弈
11. 主要实验结果
12. 消融实验与训练动力学
13. PRISM小结（贡献与局限）

**Part 2：Uni-OPD（8页）**
14. Uni-OPD的动机：OPD的两个基础瓶颈
15. 双视角优化框架概览
16. 学生视角：离线+在线数据平衡
17. 教师视角：Order Consistency与边距校准
18. 实验设计全景（四种蒸馏场景）
19. 主要实验结果
20. 消融实验与定性分析
21. Uni-OPD小结（贡献与局限）

**Part 3：对比与展望（5页）**
22. 两篇论文的详细对比表
23. 互补关系与结合可能性
24. 对OPD研究的整体启示
25. 未来方向与讨论

**Backup Slides**：
26. 关键公式汇总
27. 实验配置细节
28. 数据构建pipeline

---

## 附录：关键公式速查

### PRISM

**MoE Discriminator Score**：
$$r(x, y) = \alpha \cdot D_v(x, c) + (1-\alpha) \cdot D_r(x, t)$$

**Bradley-Terry Loss (Discriminator)**：
$$\mathcal{L}_{D_k} = -\mathbb{E} \log \sigma(D_k(x, y_k^+) - D_k(x, y_k^-))$$

**Minimax Objective**：
$$\min_\theta \max_\phi \mathbb{E}_{(x,y^+)\sim\mathcal{T}, y^-\sim G_\theta} \left[r_\phi(x, y^+) - r_\phi(x, y^-)\right]$$

**GRPO Advantage (Policy)**：
$$A_i = \frac{r(x, y_i^-) - \text{mean}(\{r(x, y_j^-)\})}{\text{std}(\{r(x, y_j^-)\})}$$

### Uni-OPD

**OPD Token-level Reward**：
$$r_t^{OPD} = \log \frac{\pi_T(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})}$$

**Trajectory-level Distillation Return**：
$$G_{OPD}(q, \tau) = \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} r_t^{OPD}$$

**Prompt-level Margin**：
$$m(q) = \min_{\tau \in S^+(q)} G_{OPD}(q, \tau) - \max_{\tau \in S^-(q)} G_{OPD}(q, \tau)$$

**Margin Shift Correction**：
$$\Delta(q) = \delta - m(q), \quad \tilde{G}_{OPD}(q, \tau) = G_{OPD}(q, \tau) + \Delta(q) \cdot \mathbf{1}\{R(q, \tau) = 1\}$$

---

> **报告撰写：Claude Code · 2026/07/16**
> **论文来源**：arXiv:2604.28123v1 (PRISM) · arXiv:2605.03677v2 (Uni-OPD)
