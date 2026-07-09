# 📘 知识蒸馏（Knowledge Distillation）—— 从入门到精通

> *"How can we compress the knowledge of a large, powerful model into a smaller, faster one?"*
>
> —— Geoffrey Hinton, 2015

---

## 🗺️ 阅读导航

| 章节 | 内容 | 难度 |
|------|------|------|
| [第一章](#第一章-知识蒸馏的起源) | 从模型压缩到 Hinton 2015 奠基 | ⭐ 入门 |
| [第二章](#第二章-蒸馏的核心数学框架) | Softmax、温度、KL 散度完整推导 | ⭐⭐ 基础 |
| [第三章](#第三章-蒸馏技术的分类体系) | Response/Feature/Relation，Offline/Online/Self | ⭐⭐ 基础 |
| [第四章](#第四章-蒸馏进入强化学习) | Policy Distillation → RL 中的知识迁移 | ⭐⭐⭐ 进阶 |
| [第五章](#第五章-从蒸馏到在线策略蒸馏的桥梁) | 🚪 前往 OPD 正篇的过渡 | ⭐⭐⭐ 桥梁 |

> 💡 **建议**：通读全文约 40 分钟。若已熟悉 Hinton 2015 蒸馏原理，可直接跳至[第四章](#第四章-蒸馏进入强化学习)。

---

## 第一章：知识蒸馏的起源

### 📜 1.1 模型压缩的史前时代

在深度学习统治世界之前，研究者们就面临一个朴素的问题：

> <span style="color:#e74c3c">**一个大模型（ensemble）学到的知识，能不能"灌"进一个小模型里？**</span>

**🟡 1997 — Pedro Domingos** 最早探索了从多个模型中提取知识的思想。他研究了如何将多个模型的"知识"合并到一个单一模型中。

**🟡 2006 — Bucilă, Caruana & Niculescu-Mizil** 发表了题为 *"Model Compression"* 的论文。这是现代知识蒸馏的**直接前身**。他们证明：

```
一个大型模型集成（ensemble）
        ↓  可以被压缩为
一个轻量级单一模型
        ↓  且
几乎不损失准确率
```

这项工作的核心思想是：**让学生模型去拟合教师模型的输出概率分布**，而非直接学习原始标签。

**🟡 2014 — Ba & Caruana** 更进一步，在论文 *"Do Deep Nets Really Need to be Deep?"* 中提出：**浅层网络可以通过模仿深层网络来达到相近性能**。他们训练浅层网络去拟合深层网络的 logit 输出（而非最终分类结果）。

这三篇论文共同铺就了知识蒸馏的基石。但真正的革命，发生在 2015 年。

---

### 🏛️ 1.2 Hinton et al. 2015 —— 蒸馏时代的开山之作

<blockquote style="background:#fff3e0;padding:16px;border-left:6px solid #ff9800">

**📄 论文信息**

**标题**: *"Distilling the Knowledge in a Neural Network"*
**作者**: Geoffrey Hinton, Oriol Vinyals, Jeff Dean（Google）
**发表**: NIPS 2015 Deep Learning Workshop
**arXiv**: 1503.02531

</blockquote>

这篇论文做了三件开创性的事情：

1. ✨ **提出了"暗知识"（Dark Knowledge）的概念**
2. 🌡️ **引入温度参数 T，将 softmax 推广为可调节软度的概率输出**
3. 📐 **定义了标准的"蒸馏损失"框架**

---

#### 1.2.1 🔮 暗知识（Dark Knowledge）—— 被丢弃的宝藏

考虑一个图像分类任务。传统训练中，我们给模型看一张"猫"的图片：

| 训练方式 | 标签信息 |
|----------|----------|
| 硬标签（Hard Label） | 🐱 猫 = 100%，其他 = 0% |
| 软标签（Soft Label） | 🐱 猫 = 90%，🐶 狗 = 8%，🐯 虎 = 2% |

**关键洞察**：

> <span style="color:#c0392b">**硬标签丢弃了"猫和狗长得像，猫和汽车完全不像"这类类间相似性信息。这些被丢弃的信息就是"暗知识"。**</span>

Hinton 用一个极端的例子展示了暗知识的力量：

```
MNIST 数据集训练一个教师模型
    ↓
用教师预测出的"3"这个类概率最高的样本
    ↓
发现：虽然答案是"3"，但教师对"3"的置信度是 99.8%
        对"8"的置信度是 0.15%  ← 这不是噪音！
        对"2"的置信度是 0.03%  ← 这也不是噪音！
    ↓
这些微小的概率差异，编码了教师对"3 和 8 长得像"的理解
```

在 MNIST 实验中，使用软标签训练，学生只需要 **3% 的训练数据** 就能达到使用全部硬标签数据训练的模型水平。暗知识充当了强大的**正则化器**。

---

#### 1.2.2 🌡️ 温度参数 T —— 控制知识的"软硬"

这是知识蒸馏中最核心的数学工具。Hinton 对标准的 softmax 做了一个精妙的推广：

<center>

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

</center>

其中 $z_i$ 是类别 $i$ 的 logit，$T$ 是**温度**。

| T 值 | 效果 | 直观理解 |
|------|------|----------|
| $T = 1$ | 标准 softmax | 常温——原始概率分布 |
| $T \to 0$ | 趋近 argmax | 极冷——硬标签 |
| $T \to \infty$ | 趋近均匀分布 | 极热——所有类别等概率 |
| $T > 1$（典型 2~20） | **软目标** | 温暖——显露出暗知识 |

**🔥 温度的工作原理**（以 logits = [3.0, 1.0, 0.2] 为例）：

| T | 类别1 | 类别2 | 类别3 | 说明 |
|---|-------|-------|-------|------|
| 1 | 85.3% | 11.5% | 3.2% | 标准 softmax，类别1 碾压 |
| 2 | 64.4% | 23.7% | 11.9% | 变软，类别2/3 露出来了 |
| 5 | 44.3% | 29.7% | 26.0% | 更软，三类别分布接近 |
| 10 | 36.6% | 30.0% | 33.4% | 极软，几近平坦 |

> 💡 **直觉**：温度越高，logit 之间的差异被"抹平"越多，模型关于类间关系的信息就暴露得越充分。

---

#### 1.2.3 📐 联合损失函数 —— 两种知识都要

Hinton 设计的训练目标同时利用了两种信号：

<center>

$$\mathcal{L}_{\text{total}} = \underbrace{(1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y_{\text{true}}, p_S^{T=1})}_{\text{硬标签损失（地面真相）}} + \underbrace{\alpha \cdot T^2 \cdot \mathcal{L}_{\text{KL}}(p_T^{T=\tau} \;\|\; p_S^{T=\tau})}_{\text{蒸馏损失（教师暗知识）}}$$

</center>

这里有几个需要仔细理解的细节：

**① 为什么蒸馏损失要乘 $T^2$？**

当 $T$ 很大时，softmax 输出的梯度会被缩小约 $1/T^2$ 倍。乘上 $T^2$ 可以保持梯度的尺度与硬标签损失一致。

**证明（简化版）**：对 softmax 求梯度：

$$\frac{\partial p_i}{\partial z_k} = \frac{1}{T} \cdot p_i \cdot (\delta_{ik} - p_k)$$

在 $T$ 较大时，$p_i \approx 1/N$（接近均匀），所以梯度约为 $\frac{1}{T \cdot N}(\delta_{ik} - \frac{1}{N})$。梯度量与 $1/T$ 成正比。对于交叉熵损失，整体梯度约与 $1/T^2$ 成正比。因此需要在蒸馏损失上乘以 $T^2$ 来补偿。

**② $\alpha$ 控制两种信号的权重**

- $\alpha \approx 0.9$ 时：以蒸馏信号为主，硬标签为辅
- $\alpha = 1.0$ 时：完全不用硬标签（纯蒸馏）
- 实践中 $\alpha$ 通常是经验调参的超参数

---

#### 1.2.4 🧪 实验结论

Hinton 在论文中展示了几个关键实验：

| 实验 | 结论 |
|------|------|
| **MNIST 全连接** | 仅用 3% 数据 + 软标签蒸馏 → 达到全数据硬标签训练水平 |
| **JFT 语音识别** | 10 个模型的 ensemble 蒸馏为一个模型，性能几乎不降 |
| **温度消融** | 中等温度（2~5）效果最好；太高导致信息稀释 |
| **不加硬标签** | 纯蒸馏也工作得很好（暗示软目标中有足够信息） |

> 💡 **核心结论**：**软目标 = 硬目标 + 类间关系信息（暗知识）**。类间关系信息的价值，不亚于地面真相本身。

---

### 🎯 1.3 知识蒸馏的直观理解

在进入数学细节之前，你可以这样理解蒸馏：

```
      🏫 教师模型                         🎓 学生模型
   （庞大、缓慢、准确）              （轻量、快速、可能不够准）
          │                                    │
          │  不仅仅是"答案是猫"                  │
          │  还告诉你：                         │
          │  "猫和狗有 8% 相似度"                │
          │  "猫和老虎有 2% 相似度"              │
          │  "猫和汽车几乎无关"                  │
          └─────────────┬──────────────────────┘
                        │
                   📦 暗知识传递
              （通过软概率分布承载）
                        │
                        ▼
                  🎓 学生学会了：
               ✓ 正确答案
               ✓ 类间关系
               ✓ 泛化能力
```

> 📌 **一句话总结**：知识蒸馏不是简单的"抄答案"，而是让学生理解教师的**判断过程**。

---

[➡️ 下一章：蒸馏的核心数学框架](#第二章-蒸馏的核心数学框架)

---

---

# 📐 第二章：蒸馏的核心数学框架

> *"If you want to understand distillation, you must first understand softmax, entropy, and divergence."*

本章将从第一性原理出发，完整推导知识蒸馏涉及的每一个数学组件。

---

### 2.1 🔤 硬标签 vs 软标签 —— 信息的编码方式

#### 硬标签（Hard Label / One-hot Encoding）

对于一个 $K$ 类分类问题，硬标签是一个 $K$ 维向量：

$$\mathbf{y}_{\text{hard}} = [0, 0, 0, \underbrace{1}_{\text{正确类别}}, 0, ..., 0]$$

**特点**：
- ✅ 包含"哪个是正确答案"的信息
- ❌ 丢弃了所有"哪些错误答案比另一些错得更离谱"的信息
- ❌ 极端稀疏、信息密度极低

#### 软标签（Soft Label / Probability Distribution）

$$\mathbf{y}_{\text{soft}} = [0.02, 0.85, 0.08, 0.03, 0.01, 0.01]$$

**特点**：
- ✅ 包含正确答案（0.85）
- ✅ 包含类间关系（类别 2 比类别 0 更像正确答案）
- ✅ 密度丰富——每个位置都承载信息

> <span style="color:#c0392b">**核心洞察**：软标签的熵（entropy）高于硬标签。更高的熵 = 更多的信息。蒸馏的本质，就是让学生从高熵的教师输出中，学到比硬标签更多的信息。</span>

---

### 2.2 🌡️ Softmax + Temperature —— 完整的数学推导

#### 标准 Softmax

给定 logits $\mathbf{z} = [z_1, z_2, ..., z_K]$，标准 softmax 为：

$$\sigma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

#### 温度推广

Hinton 的关键创新：在指数中引入温度参数 $T$：

<center>

$$\boxed{p_i(\mathbf{z}, T) = \frac{\exp(z_i / T)}{\sum_{j=1}^{K} \exp(z_j / T)}}$$

</center>

#### 温度如何影响概率分布 —— 完整推导

**引理**：对于任意两个类别 $i$ 和 $j$，其概率比值为：

$$\frac{p_i}{p_j} = \exp\left(\frac{z_i - z_j}{T}\right)$$

**证明**：

$$\frac{p_i}{p_j} = \frac{\exp(z_i/T)}{\exp(z_j/T)} = \exp\left(\frac{z_i - z_j}{T}\right) \quad \blacksquare$$

**推论**：

| 温度范围 | 概率比行为 | 含义 |
|----------|-----------|------|
| $T \to 0^+$ | $\frac{p_i}{p_j} \to \begin{cases} \infty & z_i > z_j \\ 0 & z_i < z_j \end{cases}$ | 退化为 argmax（硬标签） |
| $T = 1$ | $\frac{p_i}{p_j} = e^{z_i - z_j}$ | 标准 softmax |
| $T \to \infty$ | $\frac{p_i}{p_j} \to 1$ | 均匀分布——所有信息被"加热蒸发" |

**可视化**（以二维 logits $[2.0, 0.5]$ 为例）：

```
T = 0.1   →   [ ████████████████████████████████ | ▏ ]    ≈ [1.00, 0.00]
T = 1     →   [ ████████████████████████ | ██████ ]      ≈ [0.82, 0.18]
T = 2     →   [ ██████████████████ | █████████████ ]     ≈ [0.68, 0.32]
T = 5     →   [ █████████████ | █████████████████ ]      ≈ [0.57, 0.43]
T = 10    →   [ ████████████ | ███████████████████ ]     ≈ [0.54, 0.46]
T = 100   →   [ ██████████████ | █████████████████ ]     ≈ [0.50, 0.50]
```

---

### 2.3 📊 信息论基础：熵、交叉熵与 KL 散度

知识蒸馏的损失函数建立在一组紧密关联的信息论概念之上。

#### 2.3.1 熵（Entropy）—— 不确定性的度量

$$H(P) = -\sum_{i} P(i) \cdot \log P(i)$$

- 熵衡量一个概率分布的**不确定性**
- 均匀分布熵最大 → 最不确定
- 硬标签熵 = 0 → 完全确定
- 软标签熵 > 0 → 包含了"不确定哪些类别也有关联"的信息

#### 2.3.2 交叉熵（Cross-Entropy）—— 用 Q 去编码 P 的代价

$$H(P, Q) = -\sum_{i} P(i) \cdot \log Q(i)$$

**直观理解**：
- 真实分布是 $P$，但我们的模型输出的是 $Q$
- 交叉熵 = 用 $Q$ 的概率去编码实际从 $P$ 中采样的数据所需要的平均 bit 数
- $Q$ 越接近 $P$，交叉熵越小

#### 2.3.3 KL 散度（Kullback-Leibler Divergence）—— 两个分布的距离

<center>

$$\boxed{D_{KL}(P \;\|\; Q) = \sum_{i} P(i) \cdot \log\frac{P(i)}{Q(i)} = \underbrace{H(P, Q)}_{\text{交叉熵}} - \underbrace{H(P)}_{\text{P 的熵}}}$$

</center>

**关键性质**：
- $D_{KL}(P \| Q) \geq 0$，当且仅当 $P = Q$ 时等于 0
- **非对称性**：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$（这是理解 OPD 的关键！）

---

### 2.4 ↔️ Forward KL vs Reverse KL —— 截然不同的行为

这是本章最重要的一个概念，它会直接影响你对 OPD 篇的理解。

#### Forward KL（前向 KL）

$$D_{KL}(P_{\text{teacher}} \;\|\; P_{\text{student}})$$

行为特征（**"均值追寻 / Mean-Seeking"**）：

| 特性 | 描述 |
|------|------|
| 学生学到什么 | 教师的所有模式（包括不重要的） |
| 对教师分布 low-prob 区域 | 用极小的 student prob 去拟合 → 惩罚巨大 |
| 结果 | 学生分布比教师更**宽泛**、更"安全" |
| 类比 | 学生试图**覆盖**教师所有的知识 |

#### Reverse KL（反向 KL）

$$D_{KL}(P_{\text{student}} \;\|\; P_{\text{teacher}})$$

行为特征（**"模式追寻 / Mode-Seeking"**）：

| 特性 | 描述 |
|------|------|
| 学生学到什么 | 教师的主要模式（忽略不重要的模式） |
| 对学生分布 low-prob 区域 | 学生选择不给该区域分配概率 → 无惩罚 |
| 结果 | 学生分布比教师更**尖锐**、更"冒险" |
| 类比 | 学生只学教师**最擅长的部分** |

**🎨 直观对比**：

```
真实分布（教师）是一个双峰分布：
         ┌──┐             ┌──┐
         │  │             │  │
     ────┘  └─────────────┘  └────
         A 点             B 点

Forward KL (教师→学生)：             Reverse KL (学生→教师)：
         ┌──┐        ┌──┐                 ┌──┐
         │  │   +    │  │                 │  │ 
     ────┘  └──  ────┘  └────     ────────┘  └────────────
      两个都学到（宽泛）              只学主要模式（尖锐）
```

> <span style="color:#c0392b">**对于 OPD 篇的关键预告**：在线策略蒸馏（OPD）使用 **Reverse KL**。这意味着学生只关注教师认为最可能的动作，忽略教师不确定的区域。这既是 OPD 高效的原因，也是它容易"熵崩塌"的根源。</span>

---

### 2.5 🔗 蒸馏损失函数的完整形式

现在我们可以写出知识蒸馏的**完整损失函数**：

<center>

$$\boxed{\mathcal{L}_{\text{KD}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{y}_{\text{true}}, \sigma(\mathbf{z}_S; T=1)) + \alpha \cdot T^2 \cdot D_{KL}\big(\sigma(\mathbf{z}_T; T=\tau) \;\big\|\; \sigma(\mathbf{z}_S; T=\tau)\big)}$$

</center>

其中：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $\mathbf{z}_T$ | 教师 logits | — |
| $\mathbf{z}_S$ | 学生 logits | — |
| $\sigma(\cdot; T)$ | 温度 $T$ 下的 softmax | — |
| $\tau$ | 蒸馏温度 | 2 ~ 20 |
| $\alpha$ | 蒸馏权重 | 0.5 ~ 0.9 |
| $T^2$ | 梯度尺度补偿 | — |

#### 两种损失的关系

当硬标签损失采用交叉熵（实际上就是 $T=1$ 的 softmax + KL）：

$$\mathcal{L}_{\text{total}} = (1-\alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{y}_{\text{true}}, \mathbf{p}_S^{T=1}) + \alpha \cdot T^2 \cdot D_{KL}(\mathbf{p}_T^{T=\tau} \;\|\; \mathbf{p}_S^{T=\tau})$$

#### 训练时的技巧

- **推理时设置 $T=1$**：蒸馏只发生在训练阶段。部署学生模型时不需要教师，也不使用高温
- **梯度分离**：教师模型在蒸馏过程中 **不参与梯度计算**（frozen/`torch.no_grad()`）
- **温度调参指南**：
  - 教师和学生容量差距大 → 更高温度
  - 任务简单 → 较低温度
  - 类别多 → 通常在 2~5 之间

---

### 2.6 🔬 数学本质：蒸馏到底在做什么？

从优化的角度看，蒸馏损失实际上是让学生输出的 logits 去**逼近**教师的 logits。

**对于高温极限**（$T \to \infty$），将 softmax 泰勒展开：

$$p_i = \frac{1 + z_i/T + O(1/T^2)}{K + \sum_j z_j/T + O(1/T^2)} \approx \frac{1}{K}\left(1 + \frac{z_i - \bar{z}}{T}\right)$$

此时 KL 散度简化为：

$$D_{KL}(p_T \| p_S) \approx \frac{1}{2T^2} \cdot \frac{1}{K} \sum_i (z_i^T - z_i^S)^2$$

> 💡 **结论**：当温度足够高时，蒸馏退化为**在 logit 空间的均方误差回归**。学生模型实际上在"模仿"教师的 logit 输出，而非仅仅匹配概率。

这解释了为什么蒸馏比从头训练更好：**教师的 logit 包含了关于特征空间的拓扑结构信息，这种结构信息无法从硬标签中获取。**

---

[➡️ 下一章：蒸馏技术的分类体系](#第三章-蒸馏技术的分类体系)

---

---

# 📊 第三章：蒸馏技术的分类体系

> *"Knowledge distillation is not one technique — it is a family of techniques united by a single idea."*

自 Hinton 2015 以来，蒸馏领域已发展出多个正交的技术维度。本章建立完整的分类框架。

---

### 3.1 🏗️ 按"蒸馏什么"分类 —— 知识的三重形态

根据学生从教师那里学习**什么类型的信息**，蒸馏分为三大类：

<center>

| 类别 | 蒸馏目标 | 典型方法 | 年代 |
|------|----------|----------|------|
| **Response-based** | 教师最终输出（logits / soft labels） | Hinton 2015 | 2015 |
| **Feature-based** | 教师中间层特征表示 | FitNets (2015) | 2015 |
| **Relation-based** | 样本间/层间的关系结构 | RKD (2017), SP (2019) | 2017 |

</center>

#### 🅰️ Response-based（响应式蒸馏）

```
教师： [输入] → [隐藏层...] → [logits] → [softmax] → 🎯 软标签
                                                  ↓
学生： [输入] → [隐藏层...] → [logits] → [softmax] → 📖 学习匹配
```

这是 Hinton 的经典范式。学生只模仿教师的**最终输出**。

**代表方法**：
- **Hinton 2015**（原始蒸馏）
- **DKD**（Decoupled KD, Zhao 2022）：将 KL 散度解耦为"目标类"和"非目标类"两部分，发现非目标类的知识才是蒸馏效果的关键
- **CTKD**（Curriculum Temperature KD, Li 2023）：温度不是固定的，而是随训练进程动态调整
- **Logit Standardization**（Sun 2024）：对 logit 进行标准化后再蒸馏，解决 logit 范围差异问题

#### 🅱️ Feature-based（特征式蒸馏）

```
教师： [输入] → [💎 中间特征 T₁] → [💎 中间特征 T₂] → ... → [输出]
                  ↓ 匹配              ↓ 匹配
学生： [输入] → [📖 中间特征 S₁] → [📖 中间特征 S₂] → ... → [输出]
```

学生不仅模仿教师的输出，还要模仿教师**中间层的特征表示**。

**🏛️ FitNets (Romero et al., 2015)** —— 特征蒸馏的开山之作：

教师是宽而浅的网络，学生是窄而深的网络。在学生中间层插入一个**适配层（regressor）**，将学生的特征映射到教师特征空间，然后最小化均方误差：

$$\mathcal{L}_{\text{feature}} = \| \mathbf{W}_r \cdot \mathbf{F}_S - \mathbf{F}_T \|_2^2$$

其中 $\mathbf{W}_r$ 是可学习的适配矩阵，$\mathbf{F}_S, \mathbf{F}_T$ 分别是学生和教师的中间层特征。

**其他里程碑**：
- **Attention Transfer**（Zagoruyko & Komodakis, 2017）：匹配教师的空间注意力图
- **FSP**（Yim et al., 2017）：匹配特征间的 Gram 矩阵（二阶关系）
- **PKT**（Passalis & Tefas, 2018）：用概率分布建模特征空间

#### 🅲️ Relation-based（关系式蒸馏）

不匹配绝对值，而是匹配**关系结构**。

```
样本间关系蒸馏（RKD, Park et al. 2017）：
        教师空间中：样本A和样本B的距离 = d_T
        学生空间中：样本A和样本B的距离 = d_S
        损失：‖ d_T - d_S ‖²  （让学生保持同样的"空间感"）
```

```
层间关系蒸馏（SP, Tung & Mori 2019）：
        教师：第3层和第5层的相似度矩阵
        学生：第2层和第4层的相似度矩阵
        损失：让这两个相似度矩阵匹配
```

---

### 3.2 ⏱️ 按"何时蒸馏"分类 —— 训练时序的三条路径

<center>

| 模式 | 图示 | 特点 |
|------|------|------|
| **Offline KD** | 教师预训练完 → 蒸馏 → 学生 | 教师固定，最常用 |
| **Online KD** | 教师和学生同时训练 | 教师也在进化，知识实时更新 |
| **Self-Distillation** | 教师 = 学生自己的深拷贝/历史版本 | 无需外部教师 |

</center>

#### Offline KD（离线蒸馏）

```
阶段1：训练教师     →  🏫 教师（大模型，固定）
阶段2：蒸馏        →  🎓 学生（小模型）
阶段3：部署        →  🎓 学生独立运行
```

- ✅ 简单、成熟、最广泛使用
- ❌ 教师知识是"静态快照"，无法适应学生需求
- ❌ **暴露偏差（Exposure Bias）**：学生训练时看到的是教师分布下的数据，但推理时面对的是自己的分布

#### Online KD（在线蒸馏）

```
  🏫 教师（同步更新）         🎓 学生（同步更新）
       ↕  互相学习  ↕
  教师和学生一起训练，知识双向流动
```

- ✅ 教师知识实时更新
- ✅ 可以互相学习（Deep Mutual Learning, Zhang et al. 2018）
- ❌ 需要同时维护教师和学生，计算开销大
- ❌ 教师变大时难以并行

> 📌 **注意区分**：这里的 "Online KD" 指监督学习中的概念（教师和学生同时训练），**不等于**强化学习中的 "Online Policy Distillation"。后者指学生用自己生成的数据学习。

#### Self-Distillation（自蒸馏）

```
  🎓 学生（深层） → 用自己更深的层/更后的epoch去教浅层/早期版本
```

- ✅ 无需额外教师模型
- ✅ 可提升自身性能（甚至超越原始模型）
- 代表：**Be Your Own Teacher** (Zhang et al., 2019)，**Born-Again Networks** (Furlanello et al., 2018)

---

### 3.3 🏆 里程碑方法速览

| 年份 | 方法 | 类型 | 贡献 |
|------|------|------|------|
| 2015 | **Hinton KD** | Response | 开创知识蒸馏范式 |
| 2015 | **FitNets** | Feature | 特征蒸馏，让学生更深而非更浅 |
| 2017 | **Attention Transfer** | Feature | 注意力图匹配 |
| 2017 | **RKD** | Relation | 关系蒸馏 |
| 2019 | **DistilBERT** | Response | NLP 蒸馏标杆：BERT → 40% 更小，60% 更快，97% 性能 |
| 2021 | **DeiT** | Response (hard) | 视觉 Transformer 的硬标签蒸馏 |
| 2022 | **DKD** | Response | 解耦 KD，揭示非目标类知识的重要性 |
| 2024 | **DeepSeek-R1 蒸馏** | Response（RL 场景） | 推理能力蒸馏的革命性突破 |

---

[➡️ 下一章：蒸馏进入强化学习](#第四章-蒸馏进入强化学习)

---

---

# 🤖 第四章：蒸馏进入强化学习

> *"Policy Distillation is knowledge distillation applied to control — where the 'classes' are actions, and the 'knowledge' is how to act."*

将知识蒸馏的思想迁移到强化学习（RL）中，面临一个根本性的挑战：**RL 要学的是"如何行动"，而不是"如何分类"。** 本章讲述这一迁移的完整过程。

---

### 4.1 🎯 为什么 RL 需要蒸馏？

强化学习面临几个独特的挑战，这些挑战使得蒸馏变得格外重要：

| 挑战 | 描述 | 蒸馏的作用 |
|------|------|-----------|
| **模型庞大** | DQN/Policy Network 参数量巨大，难以部署 | 压缩为轻量模型 |
| **多任务困境** | 每个任务训一个模型 → 存储爆炸 | 多教师蒸馏为一个多任务学生 |
| **训练成本** | 每个新任务从零训练 → 时间成本高 | 用已有教师 kickstart 新任务 |
| **连续学习（Continual Learning）** | 学新任务时遗忘旧任务 | 用旧教师策略约束学生 |

---

### 4.2 🏛️ Policy Distillation (Rusu et al., 2015) —— RL 蒸馏的奠基之作

<blockquote style="background:#fff3e0;padding:16px;border-left:6px solid #ff9800">

**📄 论文信息**

**标题**: *"Policy Distillation"*
**作者**: Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, Raia Hadsell
**机构**: Google DeepMind
**发表**: arXiv 1511.06295 (Nov 2015), ICLR 2016
**背景**: 紧接在 DQN (Mnih et al. 2015) 征服 Atari 游戏之后

</blockquote>

#### 核心问题

DQN 在 Atari 游戏上取得了超人类表现，但：
- 每个游戏需要一个专用的大网络
- 多个游戏之间无法共享知识
- 想玩 50 个游戏就需要 50 个模型

Policy Distillation 的目标：**把多个教师策略"压缩"进一个学生策略中。**

#### 关键挑战：Q 值不是概率分布

在监督学习中，教师的输出天然是一个概率分布（softmax 输出）。但在 RL 中，DQN 输出的是 **Q 值**——实数范围的无界标量。

$$
Q(s, a) \in \mathbb{R} \quad \text{（可以是任意实数）}
$$

Rusu 等人的核心技巧：**将 Q 值通过温度 softmax 转换为动作概率分布**：

<center>

$$\boxed{p_T(a|s) = \frac{\exp(Q_T(s, a) / \tau)}{\sum_{a'} \exp(Q_T(s, a') / \tau)}}$$

</center>

然后让学生模仿这个概率分布：

$$\mathcal{L}_{\text{distill}} = D_{KL}\big(p_T(\cdot|s) \;\big\|\; p_S(\cdot|s)\big)$$

---

#### 三种蒸馏模式

Policy Distillation 提出了三种操作模式，直接为后续的 OPD 研究奠定了基础：

<center>

| 模式 | 描述 | 数据来源 | 示意图 |
|------|------|----------|--------|
| ① **单任务蒸馏** | 一个教师 → 一个学生（纯压缩） | 教师轨迹 | 🏫 → 🎓（~15× 压缩） |
| ② **多任务蒸馏** | 多个教师 → 一个学生 | 所有教师轨迹混合 | 🏫₁ + 🏫₂ + ... → 🎓 |
| ③ **在线蒸馏** | 教师一边训练一边输出信号 | **学生自己生成轨迹** | 🏫 ⇄ 🎓（同时进行） |

</center>

> 💡 **关键洞察**：第三种模式——**在线蒸馏（Online Distillation）**——正是 OPD（在线策略蒸馏）的雏形。学生用自己的轨迹去训练，而非依赖教师预先生成的数据。

---

#### 实验发现

| 实验 | 结果 |
|------|------|
| **单任务** | 将 DQN 压缩 ~15 倍（参数减少 93%），**性能几乎不降** |
| **多任务** | 一个学生同时学会 10 个 Atari 游戏，**部分游戏的性能甚至超过了单任务专家** |
| **在线蒸馏** | 在训练过程中持续蒸馏比训练完再蒸馏效果更好 |

这些发现暗示了一个深刻的事实：**蒸馏不仅是压缩手段，还能促进知识的泛化和迁移。**

---

### 4.3 🎭 Actor-Mimic (Parisotto et al., 2016) —— 学生用自己的眼睛看世界

<blockquote style="background:#e8f5e9;padding:16px;border-left:6px solid #4caf50">

**📄 论文信息**
**标题**: *"Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning"*
**作者**: Emilio Parisotto, Jimmy Lei Ba, Ruslan Salakhutdinov
**机构**: University of Toronto
**发表**: ICLR 2016

</blockquote>

Actor-Mimic 对 Policy Distillation 做了关键扩展：

#### 改进一：学生用自己的轨迹

```
Policy Distillation:   教师生成数据 → 学生学习
Actor-Mimic:           学生自己行动 → 教师给信号 → 学生学习
                            ↑
                    这是 OPD 雏形的关键一步！
```

学生使用 **ε-greedy 策略**（自己的策略 + 一定随机探索）与环境交互，产生轨迹。对于每个状态-动作对，教师（预训练的 DQN）给出"什么是好动作"的指导。

#### 改进二：添加特征回归

除了模仿教师的动作概率，学生还模仿教师中间层的**特征表示**：

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{policy}}}_{\text{策略蒸馏（动作）}} + \underbrace{\mathcal{L}_{\text{feature}}}_{\text{特征蒸馏（表示）}}$$

这与 FitNets 的思想一致：不仅模仿"做什么"，还要模仿"怎么看"。

#### 改进三：Q 值引导的截断

对于学生策略访问到但教师不擅长的状态（教师的 Q 值很低），Actor-Mimic 使用 **Q 值阈值**截断蒸馏信号——不强迫学生模仿教师也在瞎猜的状态。

---

### 4.4 🧩 监督学习蒸馏 vs. RL 策略蒸馏 —— 核心差异总结

对于即将深入 OPD 领域的读者，这里建立一个关键的对照表：

<center>

| 维度 | 监督学习蒸馏 | RL 策略蒸馏 |
|------|-------------|------------|
| **教师的输出** | 类别概率分布（天然 softmax） | Q 值 / 动作 logits（需要转换） |
| **数据来源** | 静态数据集（i.i.d.） | 轨迹（序列，非 i.i.d.） |
| **数据由谁生成** | 任意（教师/预存） | 教师轨迹 vs. **学生轨迹** ← 关键区别 |
| **学生面对的数据** | 和训练时一致 | **训练时和推理时分布可能不同**（暴露偏差） |
| **损失函数** | Forward KL（教师 → 学生） | Forward KL 或 Reverse KL（学生 → 教师） |
| **额外挑战** | — | 探索-利用平衡 / 时序依赖性 / 奖励稀疏性 |

</center>

> <span style="color:#c0392b">**最关键的区别是最后一行**：RL 中，学生用谁的数据训练，直接决定了蒸馏的性质。这正是 OPD（Online Policy Distillation）与普通离线蒸馏的根本分野。</span>

---

### 4.5 🌉 通往 OPD 的桥梁 —— 离线蒸馏的极限

离线策略蒸馏（用教师轨迹训练学生）存在一个**根本性的问题**：

#### 暴露偏差（Exposure Bias）

```
训练时：
  学生看到  →  教师访问过的状态分布 d_T(s)
               ↓
  学生学习  →  "如果我在教师访问过的状态，我应该做什么"
  
推理时：
  学生处于  →  由自己策略产生的状态分布 d_S(s)
               ↓
  学生面临  →  从未见过的状态！（因为自己的策略和教师不同）
               ↓
  学生犯错  →  小错误累积 → 进入更陌生的状态 → 更大错误 → 💥 崩塌
```

#### 类比：学车

```
离线蒸馏 = 坐在副驾看师傅开 → 没真正握过方向盘 → 上路就懵
在线蒸馏 = 自己握方向盘，师傅在旁边纠正 → 真正学会了
```

这就是为什么我们需要 **Online Policy Distillation（在线策略蒸馏）**。

---

[➡️ 下一章：前往 OPD 正篇](#第五章-从蒸馏到在线策略蒸馏的桥梁)

---

---

# 🌉 第五章：从蒸馏到在线策略蒸馏的桥梁

> *"You now have all the tools to understand OPD. Let's cross the bridge."*

### 你已掌握的核心概念

在进入正篇之前，确认你已经理解了以下基石：

| ✅ | 概念 | 在 OPD 中对应的角色 |
|----|------|-------------------|
| | 知识蒸馏（Hinton 2015） | OPD 的理论祖先 |
| | 温度参数 T | 控制教师指导信号的"软硬度" |
| | KL 散度（Forward vs Reverse） | OPD 使用 **Reverse KL**（学生 → 教师） |
| | Forward KL = 均值追寻 | 离线蒸馏的特征（覆盖所有模式） |
| | Reverse KL = 模式追寻 | 在线蒸馏的特征（专注主要模式） |
| | Policy Distillation（Rusu 2015） | OPD 的直接前身 |
| | 在线蒸馏模式 | OPD 的雏形——学生用自己的轨迹 |
| | Actor-Mimic | 学生自生成轨迹 + 教师信号 |
| | 暴露偏差（Exposure Bias） | OPD 要解决的核心问题 |

### OPD 的核心思想（一句话预览）

> <span style="color:#c0392b">**让学生用自己的策略去探索世界，同时让教师实时提供"什么是好动作"的密集信号——这就是 Online Policy Distillation。**</span>

### 🚪 入口

现在，请打开正篇：

> 📖 [**02-OPD-在线策略蒸馏发展全历程.md**](./02-OPD-在线策略蒸馏发展全历程.md)

---

*📘 前置篇 · 完*

