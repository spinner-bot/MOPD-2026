# 🔥 在线策略蒸馏（Online Policy Distillation）—— 发展全历程

> *"Let the student explore, but never let it get lost. OPD is the art of providing dense, real-time guidance without constraining discovery."*
>
> —— The OPD Paradigm, 2015–2026

---

## 🗺️ 阅读导航

| 章节 | 时期 | 主题 | 难度 |
|------|------|------|------|
| [第一章](#第一章-萌芽期20152017) | 2015–2017 | 在线蒸馏概念的诞生 | ⭐⭐ 基础 |
| [第二章](#第二章-理论深化期20182019) | 2018–2019 | 梯度场分析与理论奠基 | ⭐⭐⭐ 进阶 |
| [第三章](#第三章-算法成熟与扩展期20202023) | 2020–2023 | 多智能体/离线RL/机器人应用 | ⭐⭐⭐ 进阶 |
| [第四章](#第四章-llm-时代的-opd-革命20242026) | 2024–2026 | LLM 推理蒸馏革命 ★ | ⭐⭐⭐⭐ 深入 |
| [第五章](#第五章-统一框架的建立2026) | 2026 | 前缀×KL 解耦与统一理论 | ⭐⭐⭐⭐ 深入 |
| [第六章](#第六章-行业落地与生产实践) | — | DeepSeek/Qwen 工业级实践 | ⭐⭐⭐ 进阶 |
| [第七章](#第七章-开放问题与未来方向) | — | 失效模式、开放挑战、研究前沿 | ⭐⭐⭐⭐ 前沿 |

> 📌 **前置依赖**：建议先阅读 [**01-Knowledge-Distillation-知识蒸馏入门.md**](./01-Knowledge-Distillation-知识蒸馏入门.md)，尤其是第四章和第五章。

---

## 🕰️ OPD 发展全景时间线

```
2015 ─┬─ Policy Distillation (Rusu)  ── "在线蒸馏"概念首次出现
      │
2016 ─┼─ Actor-Mimic (Parisotto)      ── 学生自生成轨迹 + 教师信号
      │
2017 ─┼─ Distral (Teh)                ── 蒸馏作为多任务通信机制
      │
2018 ─┼─ Kickstarting (Schmitt)       ── 时间衰减蒸馏 + PBT
      │
2019 ─┼─ Distilling Policy Dist.      ── 梯度向量场理论分析
      │   (Czarnecki)
      │
2020 ─┼─                              ── DPO 等对齐技术兴起
2021 ─┤                               
2022 ─┤                               
2023 ─┤                               
      │
2024 ─┼─                              ── 准备阶段：LLM 推理能力爆发
      │
2025 ─┼─ DeepSeek-R1 (Jan)           ── 🔥 RL 蒸馏革命
      ├─ REOPOLD (Ko)                 ── 松弛在线策略蒸馏
      ├─ PW-OPSD (Liu)                ── 位置加权在线自蒸馏
      ├─ GAR (Liu)                    ── 生成式对抗推理器
      │
2026 ─┼─ 统一框架：前缀×KL 解耦        ── 四种蒸馏目标的完整矩阵
      ├─ CoPD                          ── 师生协同进化蒸馏
      ├─ CR-OPD / GAE-OPD             ── 反事实路由/GAE 增强 OPD
      ├─ StepOPSD                      ── 步骤感知在线偏好蒸馏
      └─ UIUC OPD 失效模式研究         ── 三大失效模式与修复方案
```

---

# 🌱 第一章：萌芽期（2015–2017）

> *在线策略蒸馏的概念并非一夜之间诞生。它是从策略蒸馏、模仿学习、多任务RL的交汇处自然生长出来的。*

---

### 1.1 🌅 黎明：Rusu et al. (2015) 中的"Online Distillation"

<blockquote style="background:#fff3e0;padding:16px;border-left:6px solid #ff9800">

**📄 Policy Distillation (Rusu et al., 2015)**
**arXiv**: 1511.06295 | **机构**: Google DeepMind
**引用量**: 1000+

</blockquote>

Policy Distillation 论文提出了三种蒸馏模式，其中第三种——**"Online Distillation"（在线蒸馏）**——是 OPD 领域的起源。

#### 在线蒸馏模式的定义

在原文中，Rusu 等人是这样描述的：

```
     ┌─────────────────────────────────────────┐
     │                                         │
     ▼                                         │
  🏫 DQN 正在训练（权重持续更新）               │
     │                                         │
     │  每个训练步，教师输出当前 Q 值             │
     ▼                                         │
  🎓 学生网络                                 │
     │                                         │
     │  学生 N 步后更新，使用累积的蒸馏损失         │
     └─────────────────────────────────────────┘
        学生更新后，教师也更新（异步）
```

**关键设计**：

1. **教师与学生并行训练**：DQN 教师一边与环境交互（在线 Q-learning），一边向学生提供蒸馏信号
2. **异步更新**：学生每 $N$ 步更新一次，教师持续训练
3. **目标任务仍然是核心**：蒸馏是辅助信号，主要目标还是最大化预期回报

#### 在线蒸馏 vs 离线蒸馏的区别

```
离线蒸馏：
  教师训练完 → 冻结 → 生成大量数据 → 学生训练 → 部署

在线蒸馏：
  教师 ───→ 训练中 ───→ 训练中 ───→ 训练中
     ↘         ↘         ↘
      学生 ←── 学生 ←── 学生 ←── （持续接收信号）
```

> 💡 **本质区别**：在线蒸馏中，**教师的知识是活的、进化的**。学生学到的不是教师的"最终答案"，而是"教师在不同训练阶段如何思考问题"。

---

### 1.2 🏛️ Distral (Teh et al., 2017) —— 蒸馏作为多任务通信机制

<blockquote style="background:#e3f2fd;padding:16px;border-left:6px solid #2196f3">

**📄 Distral: Robust Multitask Reinforcement Learning**
**作者**: Yee Whye Teh, Victor Bapst, Wojciech M. Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, Razvan Pascanu
**机构**: 全部来自 DeepMind | **发表**: NeurIPS 2017
**arXiv**: 1707.04175

</blockquote>

Distral 提出了一个优雅的框架：**将蒸馏视为任务特定策略与共享策略之间的"通信协议"**。

#### 问题背景：多任务 RL 的梯度冲突

在多任务 RL 中，如果简单地用一个共享网络学多个任务（硬参数共享），不同任务的梯度会互相干扰：

```
任务A（需要精确控制）：梯度 → 更新权重的这个方向
任务B（需要快速反应）：梯度 → 更新权重的那个方向
                          ↓
                    梯度冲突 → 哪个都学不好
```

#### Distral 的核心设计

Distral 引入两级策略：

<center>

| 策略 | 角色 | 更新方式 |
|------|------|----------|
| **$\pi_0$**（共享/中心策略） | 捕获通用的、可迁移的行为 | 从所有任务策略蒸馏 |
| **$\pi_i$**（任务特定策略） | 专门解决第 $i$ 个任务 | RL 奖励 + KL 正则化（靠近 $\pi_0$） |

</center>

#### 完整的数学目标函数

$$J(\pi_0, \{\pi_i\}_{i=1}^n) = \sum_{i=1}^n \mathbb{E}_{\pi_i}\left[\sum_{t \geq 0} \gamma^t R_i(a_t, s_t) - \underbrace{c_{\text{KL}} \cdot \gamma^t \cdot \log\frac{\pi_i(a_t|s_t)}{\pi_0(a_t|s_t)}}_{\text{KL 正则化项}} - \underbrace{c_{\text{Ent}} \cdot \gamma^t \cdot \log \pi_i(a_t|s_t)}_{\text{熵正则化项}}\right]$$

**逐个拆解**：

1. **$\sum_{t \geq 0} \gamma^t R_i(a_t, s_t)$**：标准的强化学习目标——最大化累积奖励
2. **$-c_{\text{KL}} \cdot \gamma^t \cdot \log\frac{\pi_i(a_t|s_t)}{\pi_0(a_t|s_t)}$**：KL 惩罚——让任务策略不要离共享策略太远。这等效于**用 $\pi_0$ 作为学习的先验**
3. **$-c_{\text{Ent}} \cdot \gamma^t \cdot \log \pi_i(a_t|s_t)$**：熵奖励——鼓励探索，防止过早收敛

#### 交替优化过程

```
阶段1：固定 π₀，优化所有 πᵢ
  ├─ 每个 πᵢ 通过熵正则化的 RL 优化自己的任务
  └─ KL 惩罚使 πᵢ 不会偏离 π₀ 太远
        ↓
阶段2：固定所有 πᵢ，优化 π₀
  └─ π₀ 通过蒸馏从所有 πᵢ 中提取共性：
      π₀ = argmin Σᵢ D_KL(πᵢ || π₀)
        ↓
重复...
```

#### 软 Q-learning 视角

对 Distral 做 RL 改造，可以推导出任务策略的显式形式：

$$\pi_i(a|s) \propto \pi_0(a|s)^{\alpha} \cdot \exp\left(\frac{Q_i(s,a)}{\beta}\right)$$

这个公式有着优美的解释：

> 任务策略 = **共享先验** $\times$ **任务特定的 Q 值**

- $\alpha$ 越大 → 任务策略越偏向共享策略（越保守）
- $\beta$ 越大 → 任务策略越偏向 Q 值驱动（越激进）

#### 为什么叫"蒸馏作为通信"

```
  π₁ ──蒸馏──→ π₀ ←──蒸馏── π₂
                   │
          π₀ 学到的通用行为被"广播"回所有任务策略
          （通过 KL 正则化）
                   │
                   ▼
  π₁ ←──约束── π₀ ──约束──→ π₂
```

> 💡 **核心创新**：Distral 让蒸馏成为**双向的**——不只是教师教学生，而是多个策略通过一个共享"语言"（$\pi_0$）互相通信。这是 OPD 思想的**社会化延伸**。

---

### 1.3 🔬 Actor-Mimic (Parisotto et al., 2016) 中在线信号的精炼

在前置篇第四章中介绍了 Actor-Mimic 的基本架构。这里从 OPD 的角度深入分析它的两个核心技术贡献。

#### 技术贡献一：学生自生成轨迹中的"筛子"机制

Actor-Mimic 的学生使用 **$\varepsilon$-greedy** 与环境交互，这意味着：

```
学生的每一步：
  ├─ 以 1-ε 的概率：选择学生认为最优的动作
  └─ 以 ε 的概率：随机探索
        ↓
  教师对每个 (s, a) 对提供 Q 值信号
        ↓
  但！如果教师对这个状态的 Q 值也很低（教师也不确定）
        ↓
  截断蒸馏 —— "这个地方老师也不会，别硬学了"
```

**截断阈值**：

$$\mathcal{L}_{\text{distill}}(s) = \begin{cases}
D_{KL}(p_T(\cdot|s) \;\|\; p_S(\cdot|s)) & \text{if } \max_a Q_T(s,a) > \theta \\
0 & \text{otherwise（老师也不确定，跳过）}
\end{cases}$$

这个设计预见了后来 2026 年 UIUC 研究中发现的"学生前缀扭曲教师行为"问题。

#### 技术贡献二：特征空间的"结构化蒸馏"

```
教师网络：  Conv1 → Conv2 → Conv3 → FC → Q-values
               ↓       ↓
              匹配    匹配         （特征回归）
               ↑       ↑
学生网络：  Conv1 → Conv2 → Conv3 → FC → Q-values
```

特征回归损失：

$$\mathcal{L}_{\text{feature}} = \| f_T(s) - W_r \cdot f_S(s) \|_2^2$$

其中 $W_r$ 是可学习的映射矩阵，解决学生和教师特征维度不一致的问题。

---

### 1.4 📋 萌芽期（2015–2017）总结

| 时间 | 工作 | 核心贡献 | 与 OPD 的关系 |
|------|------|----------|-------------|
| 2015 | Policy Distillation | 首次提出"在线蒸馏"概念 | OPD 的**词汇和概念**都来源于此 |
| 2016 | Actor-Mimic | 学生自生成轨迹 + 特征回归 + Q 截断 | OPD 的**技术组件**：on-policy 采样 + teacher scoring |
| 2017 | Distral | 蒸馏作为通信 + KL 正则化交替优化 | OPD 的**理论框架**：KL 约束作为先验 |

> <span style="color:#c0392b">**萌芽期的共识**：蒸馏不仅仅是训练后的压缩手段，更可以成为一种**在训练中持续发生的知识传递过程**。但在这一时期，OPD 还只是"在线蒸馏"一词的简单使用，真正的理论分析和算法设计，要等到 2018–2019 年。</span>

---

[➡️ 下一章：理论深化期（2018–2019）](#第二章-理论深化期20182019)

---

---

# 🔬 第二章：理论深化期（2018–2019）

> *"We need to understand WHY on-policy distillation works — and when it doesn't."*

这一阶段是 OPD 理论基础的建立期。两篇 DeepMind 论文从截然不同的角度推进了我们对在线蒸馏的理解：一篇从工程角度（怎么用好），一篇从数学角度（为什么这样）。

---

### 2.1 🚀 Kickstarting Deep Reinforcement Learning (Schmitt et al., 2018)

<blockquote style="background:#e3f2fd;padding:16px;border-left:6px solid #2196f3">

**📄 Kickstarting Deep Reinforcement Learning**
**作者**: Simon Schmitt, Jonathan J. Hudson, Augustin Zidek, Simon Osindero, Carl Doersch, Wojciech M. Czarnecki, Joel Z. Leibo, Heinrich Kuttler, Andrew Zisserman, Karen Simonyan, S. M. Ali Eslami
**机构**: 全部来自 DeepMind | **发表**: arXiv 1803.03835 (March 2018)
**实验平台**: DeepMind Lab (DMLab-30) —— 30 个不同的 3D 导航/操作任务

</blockquote>

#### 核心动机：不要让教师成为学生的天花板

Policy Distillation 和 Actor-Mimic 都有一个隐含的假设：**教师比学生好，学生应该尽可能模仿教师。**

但 Kickstarting 提出了一个反直觉的观点：

> <span style="color:#c0392b">**如果学生只模仿教师，它永远无法超越教师。蒸馏应该像学骑自行车时父母的手——开始时扶着，然后慢慢松开。**</span>

#### Kickstarting 的核心设计

完整的损失函数：

<center>

$$\boxed{\mathcal{L}_{\text{kick}}(\omega, x, t) = \underbrace{\mathcal{L}_{\text{RL}}(\omega, x, t)}_{\text{标准 RL 目标（最大化回报）}} + \underbrace{\lambda_k \cdot H\big(\pi_T(\cdot|x_t) \;\|\; \pi_S(\cdot|x_t, \omega)\big)}_{\text{逐渐衰减的蒸馏信号}}}$$

</center>

**关键创新：$\lambda_k \to 0$**

蒸馏权重 $\lambda_k$ 不是一个固定值，而是随时间衰减：

```
训练初期：λ_k 大
   🏫 教师信号强 → 学生快速找到正确方向
   （避免了 RL 中痛苦的冷启动阶段）

训练中期：λ_k 中等
   🏫 教师信号减弱 → 学生开始自主探索
   （学生已经开始"自己骑车"）

训练后期：λ_k → 0
   🏫 教师信号消失 → 学生完全自主
   （学生可能超越教师！）
```

#### Population Based Training (PBT) —— 自动调参

Kickstarting 的第二个关键创新是使用 **PBT** 来自动调度 $\lambda_k$ 的衰减曲线，而不是手工设计。

```
PBT 的工作原理：

1. 初始化一个"种群"（population），每个成员有各自的超参数
   （λ_k 初始值、衰减率、学习率、熵正则化系数）

2. 在训练中周期性地：
   ├─ 评估每个成员的表现
   ├─ 表现差的成员：复制表现好的成员的参数 + 超参数
   └─ 对复制后的超参数加一点随机扰动（explore）

3. 最终，PBT 自动发现每个任务的最佳 λ_k 衰减曲线
```

#### 与 IMPALA 的结合

Kickstarting 与 IMPALA 架构（Espeholt et al., 2018）结合，利用 V-trace 处理 on-policy 采样的偏差：

```
Actor（与环境交互）:
  使用学生策略 π_S + 蒸馏信号 → 产生轨迹
       ↓
Learner（更新参数）:
  使用 V-trace 校正 off-policy 偏差
  + Kickstarting 损失
```

> 💡 **关键细节**：学生用**自己的策略**与环境交互（on-policy），但 V-trace 允许 Learner 高效地使用这些轨迹进行训练。这确保了蒸馏信号始终基于学生访问过的状态。

#### 实验结果

| 实验设置 | 结果 |
|----------|------|
| **单教师** | 训练速度提升 **1.5 倍**，最终性能持平或更好 |
| **多教师（每个任务一个专家）** | 用 **1/10 的训练步数**达到从头训练的最终性能，最终性能**超越专家 42.2%** |
| **学生 vs 教师** | 在多个任务上，学生**显著超越**教师 |

**最关键的数字**（多教师设置）：

```
从头训练需要： 10,000,000,000 步 → 达到性能 P
Kickstarting： 1,043,000,000 步   → 达到性能 P
                          ↓
                 9.58 倍加速！
            最终还超越了教师 42.2%
```

> <span style="color:#c0392b">**Kickstarting 证明了一个深刻的事实**：教师的角色不是"正确答案的提供者"，而是"学习过程的加速器"。最好的蒸馏不是让学生复制教师，而是让学生**借助教师超越教师**。</span>

---

### 2.2 📐 Distilling Policy Distillation (Czarnecki et al., 2019) —— OPD 的理论基石

<blockquote style="background:#e8f5e9;padding:16px;border-left:6px solid #4caf50">

**📄 Distilling Policy Distillation**
**作者**: Wojciech M. Czarnecki, Razvan Pascanu, Simon Osindero, Siddhant M. Jayakumar, Grzegorz Swirszcz, Max Jaderberg
**机构**: 全部来自 DeepMind | **发表**: AISTATS 2019 (PMLR Vol. 89, pp. 1331–1340)
**arXiv**: 1902.02186

</blockquote>

如果说 Kickstarting 回答了"怎么用好 OPD"，这篇论文回答了**"OPD 到底是什么"**。它是整个 OPD 领域最重要的理论文献之一。

#### 核心贡献：OPD 不是一个梯度向量场

这是本文最核心、最反直觉的数学发现。

**定义**：一个更新规则构成**梯度向量场**，当且仅当存在某个损失函数 $L(\theta)$，使得更新方向恰好等于其梯度的负方向：$\Delta\theta = -\nabla_\theta L$。

**Theorem 1（梯度场不存在定理）**：

> 标准的 on-policy 蒸馏更新规则：
>
> $$\Delta\theta = \mathbb{E}_{s \sim d_{\pi_\theta}} \left[\nabla_\theta H(\pi_T(\cdot|s) \;\|\; \pi_\theta(\cdot|s))\right]$$
>
> **不对应于任何损失函数的梯度**。这意味着标准的 OPD 不是一个保守力场（conservative vector field），它可能：
> - 在参数空间中产生循环轨迹
> - 不收敛到任何不动点
> - 与 RL 奖励信号结合时产生不可预测的行为

---

#### 为什么不是梯度场？（直观理解）

考虑两个状态 $s_1$ 和 $s_2$，学生在状态 $s_1$ 比 $s_2$ 更接近教师。

```
标准的梯度下降直觉：
  "我应该朝着减少损失的方向走"
  ∇_θ L 的方向 = 损失下降最快的方向

但 OPD 的问题在于：
  学生的参数 θ 变了
  → 学生的策略变了
  → 学生访问的状态分布 d_{π_θ}(s) 也变了
  → 教师在这些"新状态"上的表现未知
  → 蒸馏损失的定义域已经漂移了

  你永远在追一个移动的目标——而且这个目标是因为你自己的移动而移动的。
```

这是强化学习中典型的**"移动目标问题"（moving target problem）**的蒸馏版本。

---

#### Theorem 2（梯度场重建定理）—— OPD 的"修复"

好消息是：**可以通过添加一个奖励修正项来恢复梯度场性质**。

**修复方案**：在轨迹的每个时间步添加一个"未来蒸馏损失"作为奖励：

$$\hat{r}_t = -\ell(\pi_T(\cdot|s_{t+1}) \;\|\; \pi_\theta(\cdot|s_{t+1}))$$

其中 $\ell$ 是某步的蒸馏损失（如交叉熵）。

**修正后的完整更新**：

<center>

$$\boxed{\mathbb{E}_{\pi_\theta}\left[\sum_t -\nabla_\theta \log\pi_\theta(a_t|s_t) \cdot \underbrace{\sum_{i=t}^{\infty} \hat{r}_i}_{\text{累积未来修正奖励}} + \underbrace{\nabla_\theta \ell_t}_{\text{直接蒸馏梯度}}\right]}$$

</center>

**直观解释**：
- 第一项（累积 $\hat{r}_i$）：像一个 RL 奖励，鼓励学生走向后续状态也能被教师认可的方向
- 第二项（$\nabla_\theta \ell_t$）：直接的蒸馏梯度，在当前状态上模仿教师

两者结合 = 一个真正有意义的梯度向量场。

---

#### 由此诞生的新方法

基于上述理论，论文提出了几种具有梯度场性质的蒸馏方法：

##### ① **N-distill**（N-step Corrected Distillation）

使用 $N$ 步的未来损失作为修正奖励：

$$\hat{r}_t^{\text{N-distill}} = -\frac{1}{N} \sum_{k=0}^{N-1} \ell(\pi_T(\cdot|s_{t+k+1}) \;\|\; \pi_\theta(\cdot|s_{t+k+1}))$$

- N 越大 → 修正越准确 → 但方差也越大
- N=0 → 退化为标准 OPD（无梯度场）

##### ② **Expected Entropy Regularised Distillation**

在损失中显式地加入对学生行为熵的考量。这是作者推荐的**最通用的方法**。

##### ③ **Teacher V Reward**（当教师的价值函数可用时）

如果教师不仅提供策略 $\pi_T$，还提供价值函数 $V_T$（这在 Actor-Critic 架构中很常见），可以用一个巧妙的技巧：

$$\hat{r}_t^V = V_T(s_{t+1}) - V_T(s_t) + r_t^{\text{env}}$$

这给出了**理论保证**（Proposition 3）：使用此奖励的 TD 学习保证对学生是策略改进（相对于教师的价值函数）。

---

#### 决策流程图 —— 蒸馏方法选择的"地图"

论文提供了蒸馏方法选择的决策树：

```
                    ┌─ 教师提供 V 函数？ ─┐
                    │ YES                  │ NO
                    ▼                      ▼
            教师强？                  任务 horizon 短？
           ┌─YES──NO─┐              ┌─YES──NO─┐
           ▼         ▼              ▼         ▼
      Expected      Teacher      Expected     N-distill
      Entropy +R    V Reward     Entropy      +R
      (最佳选择)    (安全选择)    Regularised  (长 horizon)
```

---

#### 实验启示

| 发现 | 含义 |
|------|------|
| **学生驱动蒸馏比教师驱动蒸馏快 ~3×** | On-policy 采样天然适合探索 |
| **学生驱动蒸馏探索更广的状态空间** | 不容易陷入教师的"舒适区" |
| **学生驱动蒸馏对教师错误更鲁棒** | 学生只学教师擅长的部分（reverse KL 的模式追寻特性） |
| **Tabular 场景下无奖励时 OPD 收敛** | OPD 的收敛问题主要体现在与 RL 奖励结合时 |
| **有奖励时 OPD 可能在反例 MDP 中震荡** | 当教师能区分但学生无法区分的状态出现时 |

---

### 2.3 📋 理论深化期（2018–2019）总结

| 时间 | 工作 | 核心贡献 | 对 OPD 的意义 |
|------|------|----------|-------------|
| 2018 | **Kickstarting** | $\lambda_k \to 0$ 衰减 + PBT 自动调参 | 证明了 OPD 可以**加速训练**且**不限制学生上限** |
| 2019 | **Distilling Policy Distillation** | 梯度场分析 + 修正项 | 建立了 OPD 的**数学基础**：OPD 不是梯度场，但可以被修复 |
| 2019 | **N-distill** | N 步修正 | 第一个**理论上正确的**在线蒸馏算法 |
| 2019 | **Expected Entropy Reg.** | 熵正则化蒸馏 | 最通用的、有梯度场保证的蒸馏方法 |

> <span style="color:#c0392b">**这一时期的核心发现**：OPD 是一个强大的工具，但它不是一个"免费午餐"。标准的 OPD 公式存在数学上的缺陷（非梯度场），需要通过特定的修正来恢复其理论保证。理解这些修正，是深入 OPD 的必经之路。</span>

此外，2019 年的研究还揭示了一个重要的**工程实践准则**：

| 情况 | 推荐方法 |
|------|----------|
| 教师很强 + V 函数可用 | Expected Entropy Reg. + Teacher V Reward |
| 教师很强 + 无 V 函数 + 长 horizon | N-distill +R |
| 教师较弱 + V 函数可用 | 直接用 Teacher V Reward（不完全克隆） |
| 短 horizon 简单任务 | Expected Entropy Reg. |

---

[➡️ 下一章：算法成熟与扩展期（2020–2023）](#第三章-算法成熟与扩展期20202023)

---

---

# 🧬 第三章：算法成熟与扩展期（2020–2023）

> *"OPD moved from a niche technique to a standard tool in the RL practitioner's toolbox."*

在理论地基打好之后，2020–2023 年间 OPD 向多个方向扩展：从单一的策略蒸馏扩展到偏好对齐、离线 RL、多智能体系统和机器人控制。同时，大语言模型的兴起开始对这一领域施加引力。

---

### 3.1 🎯 DPO（Direct Preference Optimization）与策略对齐

<blockquote style="background:#fce4ec;padding:16px;border-left:6px solid #e91e63">

**📄 Direct Preference Optimization (Rafailov et al., 2023)**
**arXiv**: 2305.18290 | **发表**: NeurIPS 2023

</blockquote>

虽然 DPO 不是直接属于 OPD 的论文，但它与 OPD 有着深层的数学联系，并在 2025–2026 年间与 OPD 发生了融合。

#### DPO 的核心公式

DPO 将 RLHF（RL from Human Feedback）简化为一个分类问题：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \cdot \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

#### DPO 与 OPD 的数学联系

DPO 可以视为一种特殊形式的 **off-policy 隐式蒸馏**：

| 维度 | DPO | OPD |
|------|-----|-----|
| **参考模型** | $\pi_{\text{ref}}$（通常是 SFT 模型） | $\pi_T$（教师模型） |
| **优化目标** | 偏好胜出回答 | 模仿教师分布 |
| **数据来源** | 静态偏好对 | **学生自生成轨迹** |
| **数学形式** | 隐式奖励下的对比损失 | 显式 KL 最小化 |

> 💡 **关键联系**：DPO 和 OPD 共享同一个核心——使用一个参考模型（ref/teacher）来定义"好行为"的基准，然后让学生在这个基准之上优化。区别在于 **数据来源**（静态 vs. 在线）和 **信号形式**（成对偏好 vs. 逐 token 概率）。

**2025 年的融合**：在 LLM 推理蒸馏的实践中，DPO 式的偏好对比信号开始与 OPD 的在线采样结合，催生了 **Online DPO** 和 **StepOPSD**（见第四章和第五章）。

---

### 3.2 🗄️ 离线 RL 中的蒸馏应用

离线强化学习（Offline RL / Batch RL）在 2020–2022 年间经历了快速发展。这一领域与蒸馏有着天然的亲和性。

#### 离线 RL 的蒸馏契机

```
传统在线 RL：     传统离线 RL：
  环境交互 → 数据   固定数据集 → 学习
       ↓                  ↓
  探索是免费的        无法探索
       ↓                  ↓
  可以纠正错误        错误可能是永久性的
```

**离线策略蒸馏**在离线 RL 中扮演了关键角色：
- **CQL + 蒸馏**：使用 Conservative Q-Learning 训练教师，然后蒸馏到更小的学生，避免学生在部署时的不稳定
- **IQL + 蒸馏**：Implicit Q-Learning 教师的期望回归风格与蒸馏的 soft target 天然匹配
- **多数据集融合**：在多个离线数据源上训练教师 → 蒸馏到一个统一策略

---

### 3.3 🤝 多智能体蒸馏

多智能体强化学习（MARL）中，蒸馏成为一种重要的知识共享机制。

#### 核心挑战

在 MARL 中，每个智能体面临：
- **局部观测**（看不到全局）
- **非平稳性**（其他智能体也在学习）
- **通信限制**（不能随时共享信息）

蒸馏提供了一种优雅的解决方案：**让一个集中式教师（可以看到全局信息）在训练时指导分散式学生（只有局部观测）**。

#### 典型架构：Centralized Teacher → Decentralized Students

```
训练阶段：
         ┌──────────────────────┐
         │   🏫 集中式教师        │
         │  （全局观测 + 全连接）  │
         └──────┬───────────────┘
                │ 蒸馏信号（KL / CE）
      ┌─────────┼─────────┐
      ▼         ▼         ▼
   🎓 智能体1  🎓 智能体2  🎓 智能体3
   （局部观测）（局部观测）（局部观测）

部署阶段：
   只有分散式学生运行（不需要教师）
```

**关键论文**：
- **MADDPG + 蒸馏**（Lowe et al., 2017 扩展）：集中式 Critic 的 Q 值蒸馏到分散式 Actor
- **QMIX + 蒸馏**：将 QMIX 的混合网络知识转移到轻量级策略
- **RODE + 蒸馏**：角色分解 + 蒸馏，实现高效的多智能体协作

---

### 3.4 🤖 机器人控制中的在线蒸馏

机器人领域是 OPD 最自然的应用场景之一——你不可能在每个机器人上训练一个巨型策略网络。

#### 为什么机器人需要 OPD？

| 挑战 | OPD 的应对 |
|------|-----------|
| **Sim-to-Real Gap** | 仿真环境中训练的教师提供"理想世界"的信号，真实机器人学生在线适应 |
| **硬件异构** | 不同机器人有不同的自由度/传感器，但共享"任务理解" |
| **实时性要求** | 部署时需要小模型，训练时可以用大模型 |
| **安全性** | 教师提供安全约束信号，防止学生做出危险动作 |
| **Dexterous Manipulation** | 灵巧手操作空间巨大，教师引导可大幅加速 |

#### 代表性工作

| 工作 | 蒸馏目标 | 关键方法 |
|------|----------|----------|
| **DexPilot** (OpenAI, 2019) | 人手 → 机器人手 | 从人类示范蒸馏到 RL 策略 |
| **Generalizable Dexterity** (2023) | 单一灵巧策略 → 多任务 | 多教师蒸馏 + 在线自适应 |
| **AMP + Distill** (2022) | 运动风格 → 任务策略 | 对抗运动先验 + 蒸馏约束 |
| **R3M → Distill** (2022) | 通用视觉表示 → 控制策略 | 预训练表示 + 在线蒸馏微调 |

---

### 3.5 📊 2020–2023 的方法论成熟

到了 2023 年底，OPD 已经积累了足够多的实践经验和理论工具。以下是这一时期确立的**工程最佳实践**：

#### 蒸馏温度的选择策略

```
任务类型                 推荐温度 T      原因
──────────────────────────────────────────────────
离散控制（Atari）         T = 0.01~0.1   Q 值差异大，低温度可以保留峰值
连续控制（MuJoCo）        T = 1~5        连续动作需要更软的概率分布
多任务蒸馏                T = 2~10       需要暴露更多类间结构
机器人 Sim-to-Real        T = 0.5~2      平衡信号强度与鲁棒性
```

#### KL 系数调度

```
               λ（蒸馏权重）
               │
初期（warmup）   │  ████         大 → 快速引导
中期（explore）  │    ██▌        中 → 允许探索
后期（refine）   │      █        小 → 自主优化
终期（surpass）  │       ▏       0 → 完全自主
               └──────────────────→ 训练步数
```

---

[➡️ 下一章：LLM 时代的 OPD 革命（2024–2026）★](#第四章-llm-时代的-opd-革命20242026)

---

---

# 🔥 第四章：LLM 时代的 OPD 革命（2024–2026）★

> *"This is the moment when online policy distillation stopped being a niche RL technique and became the engine behind the world's most capable reasoning models."*

2024–2026 年是 OPD 历史上最重要的时期。大语言模型（LLM）的推理能力蒸馏，将 OPD 从一个小众的 RL 技术推向 AI 领域的中心舞台。

---

### 4.1 🧠 背景：为什么 LLM 时代重新定义了 OPD？

#### 传统的 RL 策略蒸馏（2015–2023）

```
输入：像素 / 传感器数据（几十到几百维）
动作空间：几个到几十个离散动作
网络大小：几百万参数
"知识"的含义：在特定环境中"怎么动"
```

#### LLM 时代的策略蒸馏（2024–2026）

```
输入：自然语言（词汇表 ~100K tokens，序列长度可达数万）
动作空间：词汇表中的每个 token（~100K 个"动作"）
网络大小：数十亿参数
"知识"的含义：如何一步步推理、计算、验证
```

**关键变化**：

| 维度 | 传统 RL | LLM 推理 |
|------|---------|----------|
| **状态空间** | ~10²–10⁴ 维 | ~10⁵ tokens × 序列长度 |
| **动作空间** | ~10¹ | ~10⁵（整个词汇表） |
| **"正确动作"的定义** | 最大化累积奖励 | 最大化**正确推理答案的概率** |
| **轨迹长度** | 100–1000 步 | 数千到数万 tokens |
| **蒸馏对教师的依赖** | 教师策略 | 教师 log-probabilities + 最终答案正确性 |

> <span style="color:#c0392b">**根本差异**：在 LLM 推理中，"动作"是生成下一个 token。一个推理链可能有 5000 个 tokens——这意味着有 5000 个"决策点"需要指导。如果没有密集的 token 级信号（这正是 OPD 提供的），纯 RL 的稀疏奖励无法有效训练小模型。</span>

---

### 4.2 🐋 DeepSeek-R1 (2025.01) —— 蒸馏革命的引爆点

<blockquote style="background:#fff3e0;padding:16px;border-left:6px solid #ff9800">

**📄 DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
**作者**: DeepSeek-AI (Daya Guo, Dejian Yang, et al.)
**发表**: arXiv 2501.12948 (January 2025)
**影响力**: 🌟🌟🌟🌟🌟 定义了 LLM 推理蒸馏的新范式

</blockquote>

DeepSeek-R1 并不是一篇"蒸馏论文"——它的核心贡献是**通过 RL 激发 LLM 的推理能力**。但它证明了一个震撼性的结论，永久改变了人们对蒸馏的看法。

#### DeepSeek-R1 的完整 Pipeline

```
┌──────────────────────────────────────────────────┐
│                    Stage 1                        │
│            ❄️ COLD-START SFT                      │
│  ┌──────────────────────────────────────┐         │
│  │ 人工精心编写数千条"长思维链"示例       │         │
│  │ 格式：<think> 推理过程 </think> <answer> │       │
│  │ 用这些数据 SFT 基础模型                │         │
│  └──────────────────────────────────────┘         │
│                     ↓                             │
│                    Stage 2                        │
│            🎯 REASONING RL (GRPO)                 │
│  ┌──────────────────────────────────────┐         │
│  │ 使用 GRPO 强化学习                     │         │
│  │ 奖励信号：数学答案正确性 + 格式合规性    │         │
│  │ 模型开始涌现"自我反思""自我验证"行为    │         │
│  │ → 这就是 R1-Zero                      │         │
│  └──────────────────────────────────────┘         │
│                     ↓                             │
│                    Stage 3                        │
│         🔄 REJECTION SAMPLING + SFT               │
│  ┌──────────────────────────────────────┐         │
│  │ R1-Zero 生成大量推理链                │         │
│  │ 筛选：只保留答案正确的                │         │
│  │ 用精选数据 SFT → 更强的推理模型        │         │
│  └──────────────────────────────────────┘         │
│                     ↓                             │
│                    Stage 4                        │
│                🛡️ ALIGNMENT RL                    │
│  ┌──────────────────────────────────────┐         │
│  │ 加入安全/有用性/指令遵循奖励           │         │
│  │ → 最终 R1 模型                        │         │
│  └──────────────────────────────────────┘         │
│                     ↓                             │
│                    Stage 5                        │
│              🎓 DISTILLATION                      │
│  ┌──────────────────────────────────────┐         │
│  │ 从 R1 教师 + 精选推理链                │         │
│  │ → 蒸馏到 Qwen/Llama 1.5B~70B         │         │
│  │ → R1-Distill 系列                     │         │
│  └──────────────────────────────────────┘         │
└──────────────────────────────────────────────────┘
```

#### GRPO (Group Relative Policy Optimization) —— OPD 的"隐形存在"

GRPO 是 DeepSeek 使用的 RL 算法，它在一个**关键设计**上与 OPD 产生了交集：

**GRPO 的损失函数**：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(\rho_t(\theta) \hat{A}_{i,t}, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t}\right) - \beta \cdot \underbrace{D_{KL}(\pi_\theta \;\|\; \pi_{\text{ref}})}_{\text{KL 参考正则化}}$$

这个 $D_{KL}(\pi_\theta \;\|\; \pi_{\text{ref}})$ 项——**策略对参考模型的 reverse KL 散度**——本质上是 OPD 的核心。$\pi_{\text{ref}}$ 在这里是初始 SFT 模型，充当"不让模型跑偏"的锚点。

> 💡 **OPD 的隐性角色**：在 R1 的 Stage 2（GRPO）中，KL 惩罚项 $\beta \cdot D_{KL}(\pi_\theta \;\|\; \pi_{\text{ref}})$ 确保模型在探索推理策略时不会偏离基础语言能力太远。这正是 OPD 在 LLM 时代的典型形态——**蒸馏不是"额外的步骤"，而是 RL 训练本身的内在组成部分。**

---

#### 蒸馏结果 —— "RL vs 蒸馏"的决定性对比

这是 DeepSeek-R1 论文中最震撼的发现：

<center>

| 模型 | 大小 | AIME 2024 pass@1 | MATH-500 | GPQA Diamond |
|------|------|:-----------------:|:---------:|:-----------:|
| **R1-Distill-Qwen-1.5B** | 1.5B | 28.9 | 83.9 | 33.8 |
| **R1-Distill-Qwen-7B** | 7B | 55.5 | 92.8 | 49.1 |
| **R1-Distill-Qwen-14B** | 14B | 69.7 | 93.9 | 59.1 |
| **R1-Distill-Qwen-32B** | 32B | **72.6** | **94.3** | **62.1** |
| **R1-Distill-Llama-70B** | 70B | 70.0 | 94.5 | 65.2 |

</center>

**关键对比（相同架构，RL 训练 vs 蒸馏）**：

<center>

| 模型 | 训练方式 | AIME 2024 | MATH-500 |
|------|----------|:---------:|:---------:|
| R1-Zero-Qwen-32B | **纯 RL (GRPO)** | 47.0 | 91.6 |
| R1-Distill-Qwen-32B | **蒸馏 (from R1)** | **72.6** | **94.3** |
| 差距 | | **+25.6 分！** | **+2.7 分** |

</center>

> <span style="color:#c0392b;font-size:1.1em">**结论**：对于小模型（<70B），从强教师蒸馏的知识远远优于用 RL 从零探索。纯 RL 受限于小模型的表示能力——它根本无法通过稀疏奖励自主发现复杂的推理策略。蒸馏将大模型"发现"的推理能力，高效地"压缩"到小模型中。</span>

---

#### 一句话改变了整个领域

DeepSeek-R1 论文中的这句话，成为了 2025 年 LLM 蒸馏研究的"指导思想"：

> *"RL discovers capabilities; distillation makes them stable and efficient."*
>
> **"RL 发现能力，蒸馏使之稳定高效。"**

#### 更深层的含义：OPD 的角色转变

R1 论文隐含地把 OPD 从"训练后的压缩步骤"提升为"训练中不可分割的核心环节"。这体现在：

1. **GRPO 中的 KL 正则化 = 隐式 OPD**：约束策略不偏离基础模型
2. **Rejection Sampling + SFT = 离线 OPD**：使用强模型的精选输出进行监督
3. **最终蒸馏 = Hybrid OPD**：融合了 RL 筛选 + 知识转移

---

### 4.3 🔬 REOPOLD —— 松弛在线策略蒸馏 (Ko et al., 2025)

<blockquote style="background:#e8f5e9;padding:16px;border-left:6px solid #4caf50">

**📄 REOPOLD: Relaxed On-Policy Distillation for Efficient Reasoning**
**作者**: Sumin Ko, et al. | **发表**: arXiv 2025
**核心改进**: 使 OPD 在 LLM 推理场景下更加稳定和高效

</blockquote>

REOPOLD 直接面对标准 OPD 在 LLM 推理场景中的三个核心问题。

#### 问题一：重尾负奖励（Heavy-Tailed Negative Rewards）

在 OPD 中，每步的奖励信号是教师和学生对同一 token 的 log-probability 差：

$$R_t = \log \pi_T(a_t|s_t) - \log \pi_\theta(a_t|s_t)$$

**问题**：当学生生成的某个 token 在教师看来几乎不可能时（$\pi_T(a_t|s_t) \approx 0$），$R_t \to -\infty$。这导致：
- 梯度爆炸
- 训练不稳定
- 学生被"吓到"不敢探索

**REOPOLD 的解决**：**奖励裁剪（Reward Clipping）**

$$\hat{R}_{i,t}^\lambda = \max\left(R_{i,t}, \frac{\log \lambda}{1-\lambda}\right)$$

其中 $\lambda$ 是一个混合参数，裁剪阈值从"教师分布与学生分布的混合分布"中推导出来。这个阈值有严格的概率基础，不是简单的启发式裁剪。

**直观理解**：

```
原始 OPD：  学生说了"哈密顿量" → 教师在数学上下文中认为
            π_T("哈密顿量"|"求解") ≈ 0.000001
            → R = log(0.000001) - log(0.1) ≈ -11.5
            → 梯度爆炸 💥

REOPOLD：   R = max(-11.5, 裁剪阈值 ≈ -3.0)
            → R = -3.0
            → 可控的梯度 ✅
```

#### 问题二：近零奖励（Near-Zero Rewards）

**问题**：在 LLM 推理中，大多数 token 都是常识性的"填充词"（"the", "is", "因此", "我们"），教师和学生对这些 token 的概率差异极小。这些 token 产生了大量 **~0 的奖励信号**，稀释了关键推理步骤的信号。

**REOPOLD 的解决**：**熵引导的动态 Token 采样**

```
原理：
  - 高熵 token = 学生对它不确定 = 值得学习
  - 低熵 token = 学生很确定 = 教师大概率也差不多 = 跳过

算法：
  1. 计算每个 token 的学生策略熵 H(π_θ(·|s_t))
  2. 在一个 batch 内，选择熵最高的 top-p% 的 tokens
  3. 只在这些 token 上计算蒸馏损失
```

这带来了两个好处：
- **计算效率**：跳过 70–90% 的 token，大幅降低计算量
- **信号质量**：把注意力集中在"真正有信息量"的决策点

#### 问题三：熵崩塌（Entropy Collapse）

**问题**：随着训练的进行，OPD 的 reverse KL 模式追寻特性会使学生的输出分布越来越尖锐——最终陷入一个"只说安全话"的死循环。

```
训练初期：  学生的 token 分布：较均匀（~100 个候选 token 都有概率）
训练中期：  学生的 token 分布：变窄（~20 个候选）
训练后期：  学生的 token 分布：极窄（~3 个候选）
                    ↓
              熵崩塌！学生只会说几种固定表达
                    ↓
              无法进行新的推理 💀
```

**REOPOLD 的解决**：**两阶段训练**

```
Phase 1: Exploration（探索阶段）
  - 过滤掉强负奖励（保留探索的动力）
  - 动力学类似 SFT → 保持分布多样性
  - 目标：拓宽学生的知识面

Phase 2: Refinement（精炼阶段）
  - 聚焦高熵 token（只关心学生不确定的决策）
  - 动力学类似 RL → 锐化关键决策
  - 目标：提升关键推理步骤的准确度
```

#### 实验结果

| 实验 | 改进 |
|------|:----:|
| R1-Distill-Qwen-1.5B → REOPOLD | AIME24: 28.6 → **41.6** (+13.0) |
| 采样效率 vs 纯 RL | **6.7–12×** 提升 |
| 7B 学生在视觉推理上 | 匹配 32B 教师水平 (~3.32× 推理加速) |
| 训练稳定性 | 消除了梯度爆炸和熵崩塌 |

---

[➡️ 下一节：PW-OPSD 与 GAR](#43-位置加权在线自蒸馏-pw-opsd)

---

### 4.4 📍 PW-OPSD —— 位置加权在线自蒸馏 (Liu et al., 2025)

<blockquote style="background:#e3f2fd;padding:16px;border-left:6px solid #2196f3">

**📄 When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning**
**作者**: Liu et al. (SaFo-Lab) | **发表**: arXiv May 2025
**代码**: [github.com/SaFo-Lab/PW-OPSD](https://github.com/SaFo-Lab/PW-OPSD)

</blockquote>

PW-OPSD 提出了一个精妙的问题：**在教师的推理链中，所有 token 的指导价值是一样的吗？**

答案是：**远非如此！**

#### 核心发现：位置是最强预测器

研究者系统地分析了教师 token 的可靠性，发现：

<center>

| 预测器 | AUROC（区分可靠 vs 不可靠 token） |
|--------|:---:|
| Token 在序列中的**位置** | **0.83** |
| 教师概率（entropy-based） | 0.57 |
| 学生的不确定性 | ~0.60 |
| 随机猜测 | 0.50 |

</center>

> 💡 **直观含义**：推理链中靠后的 token 远比靠前的 token 更可靠。因为推理是累积的——前面的步骤可能有多种可能性，但后面的步骤是在前面已经确定的基础上进行的。

#### 推理链的"可靠性梯度"

```
推理链示例（数学证明）：

"首先，考虑方程 x² + 5x + 6 = 0"          ← 早期 token，不确定性高
                                            （可能有多种解法）
"我们可以因式分解为 (x+2)(x+3) = 0"         ← 中期 token，逐步确定
"因此 x = -2"                               ← 后期 token，高度确定
"验证：(-2)² + 5(-2) + 6 = 4 - 10 + 6 = 0" ← 最可靠！已通过验证
  
  低可靠性 ←──────────────────────────→ 高可靠性
  权重: 0.5      权重: 0.7      权重: 0.9      权重: 1.0
```

#### 方法：递增位置权重

PW-OPSD 的核心修改出奇简单（这也是它的优雅之处）：

**标准 OPSD**：

$$\mathcal{L} = \sum_{t=1}^T D_{KL}(\pi_\theta(\cdot|s_t) \;\|\; \pi_T(\cdot|s_t))$$

**PW-OPSD**：

$$\mathcal{L}_{\text{PW}} = \sum_{t=1}^T \underbrace{w(t)}_{\text{位置权重}} \cdot D_{KL}(\pi_\theta(\cdot|s_t) \;\|\; \pi_T(\cdot|s_t))$$

其中：

$$w(t) = \alpha + (1 - \alpha) \cdot \frac{t}{T}$$

- $\alpha \in [0.5, 0.7]$：最小权重（防止开头权重太低）
- $t/T$：归一化位置（0 到 1）
- 最简单的线性递增，效果已经很好

#### 实验结果

<center>

| 方法 | MATH-500 | AIME 2024 | AIME 2025 | HMMT 2025 |
|------|:--------:|:---------:|:---------:|:---------:|
| OPSD（基准） | 89.02 | 40.74 | 31.57 | 20.93 |
| **PW-OPSD** | **88.72** | **41.20** | **32.22** | **21.48** |
| 提升 | -0.3 | +0.46 | +0.65 | +0.55 |

</center>

> 💡 **洞察**：PW-OPSD 的改进虽然看起来微小（+0.5~1 分），但重要的是它只改变了蒸馏权重的分配方式——没有额外的模型、没有额外的计算、没有复杂的新损失。这证明了**位置感知的重要性**是 OPD 中的一个基本设计原则。

---

### 4.5 ⚔️ GAR —— 生成式对抗推理器 (Liu et al., 2025)

<blockquote style="background:#fce4ec;padding:16px;border-left:6px solid #e91e63">

**📄 Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial RL**
**作者**: Liu et al. | **发表**: arXiv 2025

</blockquote>

GAR 将对抗训练引入推理蒸馏，创造了一个"学生写推理 + 判别器找茬 + 学生改进"的动态博弈。

#### 架构：双人博弈的蒸馏

```
┌─────────────────────────────────────────────┐
│                                             │
│   🎓 Reasoner（推理器 = 学生）               │
│   ┌─────────────────────────────────┐       │
│   │ 输入：问题                       │       │
│   │ 输出：推理链 + 答案              │       │
│   │ 奖励：                           │       │
│   │  ✓ 判别器认为逻辑连贯            │       │
│   │  ✓ 答案正确                      │       │
│   └─────────────────────────────────┘       │
│         │ 生成推理链               │ 评估    │
│         ▼                          ▼        │
│   ┌─────────────────────────────────┐       │
│   │ 🕵️ Discriminator（判别器）       │       │
│   │ 输入：推理链的"切片"             │       │
│   │ 输出：逻辑是否连贯？是否有错误？ │       │
│   │ 奖励：                           │       │
│   │  ✓ 正确检测出错误               │       │
│   │  ✓ 区分"真实推理"和"生成推理"   │       │
│   └─────────────────────────────────┘       │
│                                             │
│   两者同时训练 → 对抗 → 共同进步             │
└─────────────────────────────────────────────┘
```

#### 判别器的"切片"评估

GAR 的关键创新是判别器不评估完整的推理链，而是评估**推理链的局部切片**：

```
完整推理链：
  Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Answer

判别器评估的切片：
  切片1:  [Step 1, Step 2]     → 逻辑是否连贯？
  切片2:  [Step 2, Step 3, Step 4] → 这一步推导是否合理？
  切片3:  [Step 4, Step 5]     → 结论是否从前提合理推出？
```

**为什么是切片？**
- 长推理链的全局评估太难（参数太多、组合爆炸）
- 局部逻辑错误更容易检测
- 切片让判别器的反馈更**密集**、更**具体**（类似于 OPD 的分步指导）

#### 训练过程：三方博弈

```
每个训练迭代：

1️⃣ Reasoner 生成推理链
   → 输出一组 (问题, 推理链, 答案)

2️⃣ Discriminator 评估
   → 对每个切片：真实/伪造？逻辑正确/错误？
   → 产生对抗奖励信号

3️⃣ Reasoner 更新
   → 使用判别器奖励 + 答案正确性奖励
   → 改进推理质量

4️⃣ Discriminator 更新
   → 使用 Reasoner 的最新推理链作为"伪造样本"
   → 提升检测能力
```

#### 实验结果

<center>

| 基础模型 | 基准 | +GAR | 提升 |
|----------|:----:|:----:|:----:|
| R1-Distill-Qwen-7B | 54.0 (AIME24) | **61.3** | +7.3 |
| R1-Distill-Llama-8B | 43.7 (AIME24) | **53.7** | +10.0 |

</center>

> 💡 **与 OPD 的关系**：GAR 的判别器可以视为一种**可学习的、自适应的蒸馏信号源**。它比固定的教师更灵活，因为它可以针对学生的弱点进行定制化的批评。这种"对抗性蒸馏"是 OPD 的一个重要发展方向。

---

[➡️ 下一章：统一框架的建立（2026）](#第五章-统一框架的建立2026)