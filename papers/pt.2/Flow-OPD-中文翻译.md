# Flow-OPD: On-Policy Distillation for Flow Matching Models
# Flow-OPD：面向流匹配模型的在线策略蒸馏

**arXiv:2605.08063v5 [cs.CV] 2026年5月24日**

---

**方震<sup>1*</sup> &emsp; 黄文轩<sup>1*</sup> &emsp; 曾宇<sup>1</sup> &emsp; 赵一鸣<sup>1</sup> &emsp; 陈爽<sup>2</sup> &emsp; 冯凯拓<sup>3</sup> &emsp; 林云龙<sup>3</sup> &emsp; 陈林<sup>1</sup> &emsp; 陈泽辉<sup>1†</sup> &emsp; 曹少成<sup>4†</sup> &emsp; 赵峰<sup>1†</sup>**

<sup>1</sup> 中国科学技术大学 &emsp;
<sup>2</sup> 加州大学洛杉矶分校 &emsp;
<sup>3</sup> 香港中文大学 &emsp;
<sup>4</sup> 小红书公司

fazii@mail.ustc.edu.cn (方震) &emsp; wxhuang0616@gmail.com (黄文轩)

<sup>*</sup>: 同等贡献 &emsp; <sup>†</sup>: 项目负责人 &emsp; <sup>†</sup>: 通讯作者

**项目主页:** https://costaliya.github.io/Flow-OPD/

---

## 摘要

现有的流匹配（Flow Matching, FM）文本到图像模型在多任务对齐下面临两个关键瓶颈：由标量值奖励引发的奖励稀疏性，以及由联合优化异质目标导致的梯度干扰——这两者共同造成了竞争指标的"跷跷板效应"和普遍的奖励黑客行为。受在线策略蒸馏（On-Policy Distillation, OPD）在大语言模型社区中成功应用的启发，我们提出了 Flow-OPD，这是首个将在线策略蒸馏融入流匹配模型的统一后训练框架。Flow-OPD 采用两阶段对齐策略：首先通过单奖励 GRPO 微调培养领域专用教师模型，使每位专家在隔离状态下达到其性能上限；随后通过基于流的冷启动（Flow-based Cold-Start）方案建立鲁棒的初始策略，并通过三步编排——在线策略采样、任务路由标注和密集轨迹级监督——将异构专业知识无缝整合至单一学生模型。我们进一步引入了流形锚定正则化（Manifold Anchor Regularization, MAR），利用任务无关教师提供全数据监督，将生成过程锚定于高质量流形，有效缓解了纯 RL 驱动对齐中普遍出现的美学退化。基于 Stable Diffusion 3.5 Medium，Flow-OPD 将 GenEval 分数从 63 提升至 92，OCR 准确率从 59 提升至 94，整体性能相比原始 GRPO 提升约 10 分，同时保持了图像保真度和人类偏好对齐，并展现出涌现的"超越教师"效应。这些结果使 Flow-OPD 成为构建通用文本到图像模型的可扩展对齐范式。代码和权重将在以下地址发布：https://github.com/CostaliyA/Flow-OPD。

---

## 1 引言

流匹配（Flow Matching, FM）[1, 2, 3, 4] 已成为生成建模的优越范式，通过学习连续时间速度场，在采样效率和高保真合成方面均超越了传统扩散模型。然而，随着研究前沿从无约束图像合成转向高度可控的多维对齐，当前后训练方法的局限性已愈发明显。现代应用要求单一模型在统一生成空间中掌握多种任务——从精确文本渲染和复杂组合推理 [5, 6, 7, 8, 9, 10] 到严格遵循细腻的人类美学偏好 [11, 12, 13, 14]。

近期进展尝试通过将强化学习（Reinforcement Learning, RL）算法——如分组相对策略优化（Group Relative Policy Optimization, GRPO）[15]——移植到流匹配领域来弥合这一差距。

预印本。

这些方法在单奖励场景中已展现出显著潜力，在线策略探索使模型能够优化其采样轨迹并改善特定指标，如 PickScore 或美学评分。然而，不同任务需要异质且相互冲突的特征表示。正如大语言模型对齐 [19] 中所指出的，稀疏标量奖励缺乏协调这些目标的粒度，从而导致零和的"跷跷板效应"——优化某些特征（如 OCR）不可避免地通过奖励黑客行为损害美学质量。这迫切需要转向密集的轨迹级蒸馏，以提供解耦的专家监督。

这一问题最近在大语言模型（Large Language Models, LLMs）领域找到了令人信服的解决方案：在线策略蒸馏（On-Policy Distillation, OPD）。受益于 OPD，DeepSeek-V4 [9]、Mimo v2 [20] 和 GLM-5 [19] 等模型通过从专业专家中蒸馏，成功协调了复杂的多领域能力。这一范式转变为视觉社区提出了一个关键问题：流匹配模型是否能类似地利用 OPD 将多个教师模型的多样化优势整合到单一鲁棒的学生模型中？为回答这一关键问题，我们引入了 Flow-OPD——首个将 OPD 融入 FM 模型后训练流程的框架。

我们提出了一种两阶段对齐策略，首先通过单奖励 GRPO 微调培养领域专用教师，确保每位专家在隔离状态下达到其性能上限。为促进学生模型的平滑过渡，我们开发了基于流的冷启动（Flow-based Cold-Start）策略，包含两种不同变体——基于监督微调（SFT）的初始化和模型合并（Model Merging）——旨在建立能够进行多任务学习的鲁棒基础策略。在此基础上，我们通过三步编排将 OPD 应用于流匹配过程：（1）执行在线策略采样以捕获学生模型当前的速度场；（2）执行任务路由标注，由不同专家为各自领域提供密集监督；（3）引入流形锚定正则化（Manifold Anchor Regularization, MAR），纳入任务无关教师提供全数据监督，有效将生成过程锚定于高质量流形，进一步提升合成图像的美学完整性。

跨多个基准和指标的实验结果表明，Flow-OPD 相比使用稀疏奖励的原始 GRPO 实现了 10% 的性能提升，为基于流的生成模型中的规模化对齐建立了新前沿。综上所述，我们的贡献包括以下三个方面：

- **FM 多任务训练分析**：我们对流匹配模型中基于 GRPO 的多任务训练的失败模式进行了实证分析，特别识别了奖励稀疏性和梯度干扰的挑战。为解决这些问题，据我们所知，我们首次将 OPD 范式引入 FM 模型的后训练中。

- **Flow-OPD 框架**：我们提出了 Flow-OPD，一个将专业知识获取与模型统一解耦的两阶段后训练框架。我们的框架引入了基于流的冷启动策略（SFT 和合并变体）、用于细粒度监督的任务路由密集标注机制，以及通过任务无关引导确保全局生成质量的新型流形锚定正则化（MAR）。

- **卓越的性能和泛化能力**：通过在四个主流基准上的广泛实验，我们证明 Flow-OPD 相比 GRPO 基线实现了约 10 分的显著提升。值得注意的是，统一学生模型在域内匹配甚至超越了专业教师的性能，同时展现出卓越的分布外（Out-of-Distribution, OOD）泛化能力。

---

## 2 相关工作

### 文本到图像模型的强化学习

基于 RL 的对齐在大语言模型中的成功最近启发了文本到图像（Text-to-Image, T2I）生成的强化学习。早期方法如 DDPO [23]、DPOK [24] 和 ImageReward/ReFL [25] 将扩散生成建模为策略优化，以美学、人类偏好或图文对齐作为奖励，而 Diffusion-DPO [26] 则利用偏好对来对齐扩散模型。更新的 GRPO 风格方法将 RL 扩展到现代视觉生成器，包括流模型 [27, 17] 和自回归范式 [28, 29, 30, 31, 32] 中的那些。然而，T2I 生成需要多个奖励来覆盖美学、对齐、保真度和组合正确性。现有解决方案仍然难以控制：DanceGRPO [17] 直接混合 HPS 和 CLIP 等奖励，往往在指标之间相互权衡；Flow-GRPO [27] 使用分阶段奖励/数据集课程，使结果对顺序和阶段设计敏感；GDPO [33] 表明 GRPO [9] 在多奖励设置下可能出现奖励归一化崩溃。这促使需要一种更可控的多奖励协调机制。

### 在线策略蒸馏

传统的离线蒸馏依赖于固定数据集，无法适应学生不断演化的轨迹。相比之下，在线策略蒸馏（On-Policy Distillation, OPD）将教师的监督信号与学生的探索空间动态耦合。在大语言模型领域，OPD 发展迅速：GKD [34] 建立了缓解曝光偏差的规范框架；MiniLLM [35] 和 DistiLLM [36] 引入反向和偏斜 KL 散度以改进模式寻求和优化稳定性；G-OPD [37] 在 KL 约束 RL 理论下统一了 OPD；Entropy-Aware OPD [38] 通过自适应散度函数保持多样性；Fast OPD [39] 通过前缀截断显著加速计算；PACED [40] 基于梯度信噪比分析实现了能力感知课程。尽管大语言模型取得了这些进展，OPD 在视觉流匹配模型中仍未被充分探索——后者需要在高维速度场内进行密集监督。我们提出 Flow-OPD，这是在线策略蒸馏在流匹配中的首次系统性迁移，利用多教师密集监督克服奖励稀疏性瓶颈。

---

## 3 预备知识

### 流匹配模型

流匹配（Flow Matching, FM）通过常微分方程 $dx_t = v_t(x_t, t)dt$ 将噪声分布 $p_0$ 映射到数据分布 $p_{\text{data}}$。在最优传输（Optimal Transport, OT）公式 [3] 下，路径为 $x_t = (1-t)x_0 + tx_1$，模型 $v_\theta$ 通过下式学习恒定速度 $(x_1 - x_0)$：

$$ \mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \tag{1} $$

遵循 Flow-GRPO [16]，我们将离散化的 ODE 积分概念化为序列马尔可夫去噪过程。通过将每个转移 $x_t \rightarrow x_{t-\Delta t}$ 建模为马尔可夫状态步骤，这一视角连接了连续生成动态与强化学习，为逐步策略优化定义了形式化轨迹。

### 在线策略蒸馏

知识蒸馏旨在通过最小化输出差异将教师能力压缩到学生模型中。为缓解分布偏移，在线策略蒸馏（OPD）[41] 要求学生模型 $f_\theta$ 在实时教师监督的引导下生成轨迹 $\tau \sim p_\theta(\tau)$。对于自回归（Autoregressive, AR）模型，该优化被形式化为最小化学生与教师分布之间的反向 KL 散度：

$$ \mathcal{L}_{\text{OPD}} = -\mathbb{E}_{y \sim p_\theta(y|x)} \left[ \log p_{\text{teacher}}(y|x) \right] = D_{\text{KL}}(p_\theta \parallel p_{\text{teacher}}) \tag{2} $$

通过在模型自身生成的分布上进行对齐，OPD 有效抑制了曝光偏差，确保了交互式或迭代生成任务中的鲁棒泛化。

---

## 4 动机

### 4.1 问题一：GRPO 为何有效？

标准 FM 依赖离线重建，从根本上将性能限制在静态数据集质量上，无法优化不可微的偏好。GRPO [9, 16, 17] 通过在线探索克服了这一问题。通过从当前策略 $\pi_\theta$ 主动采样 $G$ 个输出，它利用分组相对优势 $A(\mathbf{x}_1^{(i)}) = (r(\mathbf{x}_1^{(i)}) - \bar{r}) / \sigma$ 评估自身生成的状态。策略梯度于是由这些在线经验显式驱动：

$$ \mathcal{J}(\theta) \approx \frac{1}{G} \sum_{i=1}^{G} A(\mathbf{x}_1^{(i)}) \nabla_\theta \log p_\theta(\mathbf{x}_1^{(i)} | c) \tag{3} $$

这种对自身动态分布的持续探索使模型能够发现新颖的高奖励轨迹，成功突破了离线 SFT 的性能上限。

### 4.2 问题二：GRPO 为何失败？——多任务视角

尽管单奖励 GRPO 对特定目标有效，但它会在正交能力上造成严重退化（图 2）。这种灾难性遗忘源于共享参数空间 $\theta$ 中由稀疏标量奖励驱动的无约束梯度干扰。

对于由目标任务 $T_1$ 驱动（具有优势 $A_1$）的参数更新 $\Delta\theta$，其对未被监控的能力 $T_k$（$k \neq 1$）的附带影响可通过一阶泰勒展开近似：

$$ \Delta J_k \approx \nabla_\theta J_k \cdot \Delta\theta \propto \mathbb{E}_x \left[ A_1(x) \cdot \nabla_\theta J_k \cdot \nabla_\theta \log \pi_\theta(x|c) \right] \tag{4} $$

在高维空间中，发散的任务梯度经常相互冲突（$\nabla_\theta J_k \cdot \nabla_\theta J_1 < 0$）。由于缺少 $T_k$ 的监督信号，优化器会积极利用这些未被监控的自由度来最大化 $A_1$，从而破坏预训练的协同效应并导致流形崩溃。

这引出了一个自然的问题：我们能否通过简单混合多个数据集和奖励进行联合优化来解决这种退化？

### 4.3 问题三：混合训练能否解决问题？

为探索混合训练方法的可行性，我们在 Stable Diffusion 3.5 Medium（SD-3.5-M）[2] 上进行了受控实证实验。遵循 Flow-GRPO，我们逐步叠加四种不同的奖励函数：GenEval、OCR、PickScore 和 DeQA。如表 1 所示，混合标量奖励无法构建稳定的认知基础。

**表 1：多奖励优化中的能力退化**

| 模型 | GenEval | OCR |
|---|---|---|
| SD-3.5-M | 0.63 | 0.59 |
| +GenEval | 0.94 | 0.65 |
| +OCR | 0.89 (-5%) | 0.91 |
| +PickScore | 0.82 (-7%) | 0.86 (-5%) |
| +DeQA | 0.73 (-9%) | 0.83 (-3%) |

虽然初始奖励（+GenEval）取得了成功，但后续的添加却触发了灾难性遗忘（例如，+OCR 使 GenEval 下降了 5%）。这验证了我们关于梯度干扰（$\nabla_\theta J_i \cdot \nabla_\theta J_j < 0$）的假设。将多维冲突压缩为标量优势会强制形成零和博弈；例如，适应美学风格化（PickScore）会积极地覆盖精确的几何表示。因此，由于这种稀疏的信息瓶颈，标量奖励混合在根本上不可扩展。

为避免参数相互蚕食，我们需要一种同时具备在线策略性（保持探索）和密集解耦性（防止干扰）的监督信号。受大语言模型中多教师在线策略蒸馏（Multi-Teacher OPD）的启发，我们提出了 Flow-OPD。该框架将多教师范式无缝引入连续基础模型，实现由密集监督引导的主动在线策略探索。

---

## 5 方法：Flow-OPD

Flow-OPD 通过对自生成轨迹进行密集监督来重新定义多任务对齐。我们首先使用 Flow-GRPO 训练领域专家教师。在冷启动初始化之后，学生经历多教师在线蒸馏，将在线样本动态路由到特定教师以获得细粒度引导。最后，流形锚定正则化解耦功能对齐与美学崩塌，保留固有的生成先验。

### 5.1 冷启动

为确保稳定的初始化 $\pi_{\theta_0}$ 并防止早期展开中的轨迹发散，我们探索了两种冷启动策略：基于 SFT 的初始化和模型合并初始化。我们的 SFT 协议遵循 Flow-GRPO，但利用从专业教师采样的轨迹，确保学生从一开始就继承专家级的知识分布。作为替代，模型合并将不同教师的各向异性先验叠加到统一参数状态中。这种"合并即初始化"的方法将学生定位在损失景观中的高能力区域，该区域的多任务协同效应已初步形成，为后续蒸馏提供了鲁棒基础。

### 5.1.1 多教师在线策略蒸馏

**架接 OPD 与流匹配** 如公式 2 所示，ThinkingMachines 的 OPD [41] 通过使用相对教师分布 $p_{\text{teacher}}$ 的反向 KL 散度作为在自主生成轨迹 $\tau$ 上的环境奖励来优化学生策略 $\pi_\theta$。为将此策略梯度（Policy Gradient, PG）范式转换到连续时间 FM 框架中，我们将离散的 token 序列映射到连续的潜在轨迹 $x_t \in \mathbb{R}^d$。自回归预测转化为由速度场 $v_\theta(x_t, t)$ 参数化的瞬时转移策略。关键在于，我们不是通过监督回归直接最小化向量场之间的距离，而是推导出精确的连续时间 KL 散度，并将其作为密集奖励信号通过 PG 引导策略探索。

**在线策略采样** Flow-OPD 的基本前提要求学生暴露其自身的特定分布偏移。为促进充分的状态空间探索——这是 RL 中逃离局部最优的必要条件——我们通过将确定性概率流 ODE 转换为等效的随机微分方程（Stochastic Differential Equation, SDE）来注入随机性 [16]：

$$ dx_t = \left[ v_\theta(x_t, t) + \frac{\sigma_t^2}{2t} \left( x_t + (1 - t) v_\theta(x_t, t) \right) \right] dt + \sigma_t d\mathbf{w} \tag{5} $$

在时间步长 $\Delta t$ 上应用 Euler-Maruyama 离散化，学生的转移行为表现为局部各向同性高斯策略：

$$ \pi_\theta(x_{t-\Delta t} | x_t, c) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 \Delta t \mathbf{I}) \tag{6} $$

通过对每个提示采样 $G$ 条独立轨迹，生成在线策略边际分布 $x_t \sim \pi_\theta(\cdot | c)$，作为随机行为策略。

**任务特定教师标注** 在每个探索状态 $x_t$，学生查询专家教师集成以获得局部监督。为消除域间梯度干扰，我们实现了硬路由机制 $\mathcal{R}(c) = k$，将文本条件 $c$ 映射到集成中唯一的对应领域专家 $k$。该机制选择性地激活单一教师提供参考速度场 $v_{\phi_k}^*(x_t, t, c)$。目标流因此定义为：

$$ v_{\text{target}}(x_t, t, c) = v_{\phi_k}^*(x_t, t, c), \quad \text{其中 } k = \mathcal{R}(c) \tag{7} $$

其中 $\mathcal{R}(\cdot)$ 表示确定性的任务到教师路由函数。这产生了任务特定的目标转移策略 $\pi_{\text{target}} = \mathcal{N}(\mu_{\text{target}}(x_t, t), \sigma_t^2 \Delta t \mathbf{I})$，作为评估学生在线策略轨迹的最终黄金标准。

**推导流模型中的解析 KL** 在大语言模型中，OPD 将密集优势定义为负反向 KL 散度：$A_t = -D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{target}})$。由于离散 token 词汇表阻止梯度直接通过采样过程流动，优化严格依赖于得分函数估计器（即策略梯度定理）：

$$ \nabla_\theta \mathcal{J}_{\text{LLM}} = \mathbb{E}_{s_t} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right] \tag{8} $$

这种离散公式不可避免地引入了高梯度方差，需要计算昂贵的 logprob 跟踪、重要性采样和复杂的代理裁剪机制（如 PPO）来稳定训练。为将此框架迁移到流匹配模型，我们必须在数学上将 KL 计算适应到连续状态空间。根据 SDE 公式，学生转移策略 $\pi_\theta(\cdot | x_t)$ 和目标策略 $\pi_{\text{target}}(\cdot | x_t)$ 都是 $d$ 维多元高斯分布。关键的是，它们共享由前向 SDE 噪声调度严格导出的完全相同各向同性协方差矩阵 $\Sigma_t = \sigma_t^2 \Delta t \mathbf{I}$。对于两个一般高斯分布 $\mathcal{N}_1(\mu_1, \Sigma_1)$ 和 $\mathcal{N}_2(\mu_2, \Sigma_2)$，期望 KL 散度解析定义为：

$$ D_{\text{KL}}(\mathcal{N}_1 \parallel \mathcal{N}_2) = \frac{1}{2} \left[ \text{tr}(\Sigma_2^{-1} \Sigma_1) - d + (\mu_1 - \mu_2)^T \Sigma_2^{-1} (\mu_1 - \mu_2) + \ln \frac{\det \Sigma_2}{\det \Sigma_1} \right] \tag{9} $$

由于我们的策略共享相同的协方差（$\Sigma_1 = \Sigma_2 = \Sigma_t$），迹项求值为数据维度 $\text{tr}(\mathbf{I}) = d$，与 $-d$ 常数无缝抵消。同时，相同行列式的对数比求值为 $\ln(1) = 0$。因此，复杂的散度精简为仅包含马氏距离。代入具体的逆协方差 $\Sigma_t^{-1} = (\sigma_t^2 \Delta t)^{-1} \mathbf{I}$，KL 散度优雅地约化为其均值之间的 L2 距离 [16]：

$$ D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{target}}) = \frac{\|\mu_\theta(x_t, t) - \mu_{\text{target}}(x_t, t)\|^2}{2\sigma_t^2 \Delta t} \tag{10} $$

根据 Euler-Maruyama 离散化，策略均值 $\mu_\theta(x_t, t)$ 由预测向量场 $v_\theta$ 确定性参数化：

$$ \mu_\theta(x_t, t) = x_t + \left[ v_\theta(x_t, t, c) + \frac{\sigma_t^2}{2t} \left( x_t + (1 - t) v_\theta(x_t, t, c) \right) \right] \Delta t \tag{11} $$

当减去目标均值 $\mu_{\text{target}}$ 时，状态相关的基础项完全相同地消去。KL 散度优雅地简化为严格在向量场之间的缩放差异：

$$ D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{target}}) = \frac{\Delta t \left[ t(1 - t) + 1 \right]}{2 \sigma_t^2} \cdot \frac{1}{\Delta t} \left\| v_\theta(x_t, t, c) - v_{\text{target}}(x_t, t, c) \right\|^2 \tag{12} $$

**将 OPD 引入流匹配模型** 在大语言模型中，由于离散 token 词汇表禁止通过采样过程直接微分，OPD 必须依赖得分函数估计器（即策略梯度定理）以优化密集奖励 $r_t = -D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{target}})$：

$$ \nabla_\theta \mathcal{J}_{\text{LLM-PG}} = \mathbb{E}_{s_t} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left( - D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{target}}) \right) \right] \tag{13} $$

将此策略梯度公式直接迁移到流模型的连续状态空间会产生结构上相同的 Flow-OPD PG 更新：

$$ \nabla_\theta \mathcal{J}_{\text{Flow-OPD}} = \mathbb{E}_{x_t \sim \text{SDE}} \left[ \nabla_\theta \log \pi_\theta(x_{t-\Delta t} | x_t) \cdot \left( - D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{target}}) \right) \right] \tag{14} $$

在离散空间中，评估公式 13 需要高方差蒙特卡洛估计。然而，对数导数技巧的一个基本性质规定了 KL 散度的期望策略梯度在数学上等于期望 KL 散度本身的直接梯度。具体而言，在无限样本上计算 PG 期望在数学上收敛为：$\mathbb{E}[\nabla \log \pi \cdot (-D_{\text{KL}})] \equiv \nabla_\theta (-D_{\text{KL}})$。

虽然这种直接梯度对离散大语言模型在解析上不可处理，但我们的连续流公式具有完全可微的闭式解。因此，我们可以完全绕过高方差 PG 估计器、$\log \pi$ 计算和 PPO 代理边界。从根本上说，在流模型中直接最小化这一精确的 MSE 损失在数学上严格等价于在大语言模型中执行策略梯度 OPD，但有效地将梯度方差降至零。因此，公式 14 可以通过直接对标准时间加权 MSE 损失进行反向传播来简化：

$$ \mathcal{L}_{\text{Flow-OPD}} = \mathbb{E}_{x_t \sim \text{SDE}} \left[ w(t) \left\| v_\theta(x_t, t, c) - \text{SG}\left( v_{\text{target}}(x_t, t, c) \right) \right\|^2 \right] \tag{15} $$

其中 $v_\theta$ 保持连接到计算图以接收梯度，$\text{SG}(\cdot)$ 表示教师目标的停止梯度算子，且 $w(t) = \frac{\Delta t \left[ t(1 - t) + 1 \right]}{2 \sigma_t^2} \cdot \frac{1}{\Delta t}$。最终，该公式将在线策略 RL 的严格状态空间探索与回归的确定性优化效率无缝统一。

**流形锚定正则化** 激进地优化功能目标（如精确文本渲染或严格空间布局）经常引发奖励黑客行为，表现为视觉美学和生成多样性的严重退化 [16]。为将功能对齐与风格崩塌解耦，我们引入了一种受 Flow-GRPO 中 KL 惩罚启发的连续时间美学保持机制。

然而，不同于锚定到通用预训练模型，我们维持一个冻结的美学教师（例如，通过 DeQA 优化）以提供高保真正则化向量场 $v_{\text{aesthetic}}$。如先前推导的，SDE 框架中的反向 KL 散度优雅地转化为向量场之间的时间加权 L2 距离。在我们的实现中，优化公式化为最小化总损失 $\mathcal{L}_{\text{Total}}(\theta)$，即策略损失 $\mathcal{L}_{\text{Policy}}(\theta)$（定义为替代目标 $-\mathcal{J}(\theta)$ 的负值）与该密集 KL 惩罚的直接求和：

$$ \mathcal{L}_{\text{Total}}(\theta) = \mathcal{L}_{\text{Flow-OPD}}(\theta) + \beta \cdot \mathbb{E}_{c, t, x_t \sim \pi_t} \left[ w(t) \left\| v_\theta(x_t, t, c) - v_{\text{aesthetic}}(x_t, t, c) \right\|^2 \right] \tag{16} $$

该 KL 正则化作为连续的弹性锚运作。它保证在学生策略贪婪吸收多教师集成的功能智能的同时，始终被严格约束在高质量视觉流形上，完全避免单目标 RL 中典型的美学退化。

---

## 6 实验

### 6.1 实验设置

遵循 Flow-GRPO [16]，我们在四个任务上评估方法：GenEval [21]、OCR [22]、PickScore [42] 和 DeQA [43]。前三个任务我们采用官方检查点作为专家教师。DeQA 教师通过以 4:6 的比例混合 DeQA 和 PickScore 奖励在三个数据集上专门训练。所有训练和测试数据严格遵循 Flow-GRPO 的划分。训练在 4 个节点（每个节点 8 块 H800 GPU）上执行，评估在单个 8 块 H800 节点上进行。

我们主要将 Flow-OPD 与两类基线进行比较：（1）单一奖励 GRPO，记为 GRPO-[奖励名称]，其中模型使用 Flow-GRPO 在单一奖励目标上微调；（2）混合奖励 GRPO，记为 GRPO-Mix，采用固定比例的加权奖励组合（GenEval : OCR : PickScore = 3 : 1 : 1）。这些基线旨在突出传统基于标量的对齐在扩展到多维专家能力时的局限性。

### 6.2 主要结果

表 2 中的定量结果表明，Flow-OPD 在所有基准上始终匹配或超越专业教师模型，特别是在文本渲染和 DeQA 图像质量方面。关键的是，它解决了专业化固有的严重跨域干扰（例如 PickScore 教师的 GenEval 分数降至 0.51），并克服了稀疏奖励多任务 GRPO 的优化瓶颈。通过利用密集多专家监督，Flow-OPD 无缝整合多样化专业知识而无能力退化。

**表 2：模型在组合图像生成、视觉文本渲染和图像质量基准上的性能比较。avg 值通过平均四个 0-1 归一化指标计算。教师模型分数以粗体和下划线标出以表示性能上限，不参与比较。最优分数以蓝色标注，次优分数以绿色标注。**

| 模型 | GenEval | OCR Acc. | DEQA | PickScore | Avg |
|---|---|---|---|---|---|
| SD-3.5-M | 0.63 | 0.59 | 4.07 | 21.64 | 0.7165 |
| +GRPO-Geneval | 0.94 | 0.65 | 4.01 | 21.53 | 0.8050 |
| +GRPO-OCR | 0.64 | 0.92 | 4.06 | 21.69 | 0.8015 |
| +GRPO-deqa | 0.64 | 0.66 | 4.23 | 23.02 | 0.7578 |
| +GRPO-Pickscore | 0.51 | 0.69 | 4.22 | 23.19 | 0.7340 |
| GRPO-Mix | 0.73 | 0.83 | 4.33 | 21.84 | 0.8165 |
| SFT+GRPO-Mix | 0.85 | 0.86 | 4.29 | 21.79 | 0.8515 |
| Merge+GRPO-Mix | 0.84 | 0.86 | 4.18 | 21.87 | 0.8442 |
| **Ours (SFT)** | **0.91** | **0.92** | **4.29** | **21.83** | **0.8820** |
| **Ours (Merge)** | **0.93** | **0.93** | **4.31** | **23.05** | **0.9021** |

图 3 中的定性结果表明，Flow-OPD 实现了最优的多任务权衡，在高提示保真度与优越视觉美学之间取得平衡。值得注意的是，Flow-OPD 在所有个别教师都失败的某些边缘情况中取得了成功——我们将此现象称为"超越教师"（Teacher-Surpassing）。我们假设这种涌现的优越性源于潜在流流形中的知识交叉授粉。虽然个别教师受限于领域特定偏差，但同时进行的密集引导迫使学生学习更全面、更平滑的表示。这种集体监督弥合了认知差距，使学生能够合成最终超越任何单一监督者的新颖轨迹。

### 6.3 分析

#### 6.3.1 冷启动消融

如图 4 所示，冷启动初始化快速为后续训练建立了鲁棒基础。在两种方案中，SFT 作为一种广泛采用且高度可扩展的策略；值得注意的是，其固有的灵活性为未来从异构教师中提取能力提供了有前景的途径。相反，模型合并在没有任何额外训练成本的情况下，最优地利用可用的同构教师实现优越的功能对齐。关键的是，Flow-OPD 始终优于从零开始和冷启动的多任务 GRPO。当 GRPO 因稀疏标量奖励引起的任务间冲突而收敛到次优状态时，Flow-OPD 利用密集多专家监督解决了梯度干扰。因此，我们的方法在所有基线上实现了显著且均匀的增益，成功匹配或超越了单个专业教师的性能上限。

#### 6.3.2 分布外泛化

为进一步研究我们方法的泛化能力，我们在 T2I-CompBench 基准上进行了额外评估。Flow-OPD 展示了相比多任务 GRPO 更优越的分布外泛化能力，在多个组合指标上达到了最先进（SOTA）性能。值得注意的是，当从相同的冷启动基线初始化时，标准 GRPO 在特定能力维度上遭受灾难性遗忘，如形状渲染和 3D 空间关系。相比之下，通过利用密集多专家监督和任务风格解耦正则化，Flow-OPD 有效缓解了这些退化问题，产生了鲁棒且全面的性能提升。

**表 3：T2I-CompBench++ 结果。最优分数以蓝色标注。**

| 模型 | Color | Shape | Texture | Complex | 3D-Spatial | Numeracy | Non-Spatial |
|---|---|---|---|---|---|---|---|
| SD3.5-M [2] | 0.7994 | 0.5669 | 0.7338 | 0.3800 | 0.3739 | 0.5927 | 0.3146 |
| GRPO-mix | 0.7966 | 0.5803 | 0.7392 | 0.3677 | 0.3681 | 0.6388 | 0.3130 |
| Cold Start | 0.8173 | 0.6126 | 0.7342 | 0.3870 | 0.4249 | 0.6458 | 0.3145 |
| Cold Start+GRPO | 0.8031 | 0.5985 | 0.7409 | 0.3842 | 0.4017 | 0.6269 | 0.3136 |
| **Ours (Merge)** | **0.8298** | **0.6292** | **0.7446** | **0.3943** | **0.4565** | **0.6837** | **0.3163** |

#### 6.3.3 流形锚定正则化

流形锚定正则化（MAR）是一种任务无关约束，旨在维持生成完整性和美学对齐。如图 5 所示，基于原始 GRPO 的优化经常触发背景模式崩塌——模型过拟合于单调环境——以及语义冗余，由于粗糙的奖励粒度导致多个实体出现相同特征。虽然像 DeQA 这样的教师提供多样化的样本，但它们往往在指令遵循方面存在困难。MAR 通过将优化锚定于高保真流形来解决这些问题，在结构多样性与精确语义遵循之间取得平衡。

**表 4：通用图像质量和对齐指标的性能比较。最优分数以蓝色标注。**

| 模型 | ImageReward | Aesthetic | UnifiedReward | HPS-v2.1 | QwenVL Score |
|---|---|---|---|---|---|
| SD-3.5-M | 1.02 | 5.87 | 3.339 | 0.2982 | 3.45 |
| GRPO-DeQA | 1.33 | 5.97 | 3.456 | 0.2846 | 3.68 |
| GRPO-mix | 1.23 | 5.93 | 3.501 | 0.3101 | 3.88 |
| w.o. MAR | 1.26 | 5.89 | 3.518 | 0.2998 | 3.82 |
| **Ours (Merge)** | **1.36** | **6.23** | **3.659** | **0.3302** | **4.05** |

表 4 进一步提供了我们的方法在图像质量和人类偏好对齐方面优越性的定量证据。MAR 的整合在整个数据集上利用额外监督，显著增强了生成图像的视觉质量和表现力。

---

## 7 结论

我们引入了 Flow-OPD——首个将在线策略蒸馏融入流匹配模型的框架，有效解决了奖励稀疏性和梯度干扰问题。通过用密集的轨迹级监督替代标量奖励，Flow-OPD 打破了竞争指标的"跷跷板效应"。我们在 SD-3.5-M 上的结果表明，Flow-OPD 成功整合了组合和排版方面的专业知识，同时实现了涌现的"超越教师"效应。通过流形锚定正则化（MAR），该框架通过解耦功能对齐与美学保持来维持高视觉保真度。最终，Flow-OPD 为开发具有优越生成完整性的通用文本到图像模型提供了可扩展的对齐范式。

---

## 参考文献

[1] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. *arXiv e-prints*, pages arXiv–2506, 2025.

[2] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024.

[3] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

[4] Zhen Fang, Zhuoyang Liu, Jiaming Liu, Hao Chen, Yu Zeng, Shiting Huang, Zehui Chen, Lin Chen, Shanghang Zhang, and Feng Zhao. Dualvla: Building a generalizable embodied agent via partial decoupling of reasoning and action. *arXiv preprint arXiv:2511.22134*, 2025.

[5] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2026.

[6] Wenxuan Huang, Yu Zeng, Qiuchen Wang, Zhen Fang, Shaosheng Cao, Zheng Chu, Qingyu Yin, Shuang Chen, Zhenfei Yin, Lin Chen, et al. Vision-deepresearch: Incentivizing deepresearch capability in multimodal large language models. *arXiv preprint arXiv:2601.22060*, 2026.

[7] Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. *arXiv preprint arXiv:2506.04207*, 2025.

[8] Shuang Chen, Yue Guo, Yimeng Ye, Shijue Huang, Wenbo Hu, Haoxi Li, Manyuan Zhang, Jiayu Chen, Song Guo, and Nanyun Peng. Ares: Multimodal adaptive reasoning via difficulty-aware token-level entropy shaping. *arXiv preprint arXiv:2510.08457*, 2025.

[9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

[10] Shuang Chen, Kaituo Feng, Hangting Chen, Wenxuan Huang, Dasen Dai, Quanxin Shou, Yunlong Lin, Xiangyu Yue, Shenghua Gao, and Tianyu Pang. Opensearch-vl: An open recipe for frontier multimodal search agents, 2026.

[11] Ruiyan Han, Zhen Fang, XinYu Sun, Yuchen Ma, Ziheng Wang, Yu Zeng, Zehui Chen, Lin Chen, Wenxuan Huang, Wei-Jie Xu, et al. Unicorn: Towards self-improving unified multimodal models through self-generated supervision. *arXiv preprint arXiv:2601.03193*, 2026.

[12] Shuang Chen, Quanxin Shou, Hangting Chen, Yucheng Zhou, Kaituo Feng, Wenbo Hu, Yi-Fan Zhang, Yunlong Lin, Wenxuan Huang, Mingyang Song, et al. Unify-agent: A unified multimodal agent for world-grounded image synthesis. *arXiv preprint arXiv:2603.29620*, 2026.

[13] Kaituo Feng, Manyuan Zhang, Shuang Chen, Yunlong Lin, Kaixuan Fan, Yilei Jiang, Hongyu Li, Dian Zheng, Chenyang Wang, and Xiangyu Yue. Gen-searcher: Reinforcing agentic search for image generation. *arXiv preprint arXiv:2603.28767*, 2026.

[14] Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, et al. Interleaving reasoning for better text-to-image generation. *arXiv preprint arXiv:2509.06945*, 2025.

[15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

[16] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. *arXiv preprint arXiv:2505.05470*, 2025.

[17] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. Dancegrpo: Unleashing grpo on visual generation, 2025.

[18] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. *arXiv preprint arXiv:2507.21802*, 2025.

[19] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. *arXiv preprint arXiv:2602.15763*, 2026.

[20] Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. Mimo-v2-flash technical report. *arXiv preprint arXiv:2601.02780*, 2026.

[21] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. *Advances in Neural Information Processing Systems*, 36:52132–52152, 2023.

[22] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters. *Advances in Neural Information Processing Systems*, 36:9353–9387, 2023.

[23] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning, 2024.

[24] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models, 2023.

[25] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In *Proceedings of the 37th International Conference on Neural Information Processing Systems*, pages 15903–15935, 2023.

[26] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization, 2023.

[27] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. *arXiv preprint arXiv:2505.05470*, 2025.

[28] Shihao Yuan, Yahui Liu, Yang Yue, Jingyuan Zhang, Wangmeng Zuo, Qi Wang, Fuzheng Zhang, and Guorui Zhou. Ar-grpo: Training autoregressive image generation models via reinforcement learning. *arXiv preprint arXiv:2508.06924*, 2025.

[29] Guohui Zhang, Hu Yu, Xiaoxiao Ma, JingHao Zhang, Yaning Pan, Mingde Yao, Jie Xiao, Linjiang Huang, and Feng Zhao. Group critical-token policy optimization for autoregressive image generation. *arXiv preprint arXiv:2509.22485*, 2025.

[30] Xiaoxiao Ma, Haibo Qiu, Guohui Zhang, Zhixiong Zeng, Siqi Yang, Lin Ma, and Feng Zhao. Stage: Stable and generalizable grpo for autoregressive image generation. *arXiv preprint arXiv:2509.25027*, 2025.

[31] Guohui Zhang, Hu Yu, Xiaoxiao Ma, Yaning Pan, Hang Xu, and Feng Zhao. Maskfocus: Focusing policy optimization on critical steps for masked image generation. *arXiv preprint arXiv:2512.18766*, 2025.

[32] Xiaoxiao Ma, Jiachen Lei, Tianfei Ren, Jie Huang, Siming Fu, Aiming Hao, Jiahong Wu, Xiangxiang Chu, and Feng Zhao. Mar-grpo: Stabilized grpo for ar-diffusion hybrid image generation. *arXiv preprint arXiv:2604.06966*, 2026.

[33] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization, 2026.

[34] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In *The twelfth international conference on learning representations*, 2024.

[35] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In *The twelfth international conference on learning representations*, 2024.

[36] Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and Se-Young Yun. Distillm-2: A contrastive approach boosts the distillation of llms. *arXiv preprint arXiv:2503.07067*, 2025.

[37] Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. *arXiv preprint arXiv:2602.12125*, 2026.

[38] Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. *arXiv preprint arXiv:2603.07079*, 2026.

[39] Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler II, Qian Qian, Gregory D Lyng, Sanjit Singh Batra, and Robert E Tillman. Fast and effective on-policy distillation from reasoning prefixes. *arXiv preprint arXiv:2602.15260*, 2026.

[40] Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. Paced: Distillation and self-distillation at the frontier of student competence. *arXiv e-prints*, pages arXiv–2603, 2026.

[41] Kevin Lu and Thinking Machines Lab. On-policy distillation. *Thinking Machines Lab: Connectionism*, 2025. https://thinkingmachines.ai/blog/on-policy-distillation.

[42] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. *Advances in Neural Information Processing Systems*, 36:36652–36663, 2023.

[43] Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. *arXiv preprint arXiv:2501.11561*, 2025.

[44] Christoph Schuhmann. Laion aesthetics, Aug 2022.

[45] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. *Advances in Neural Information Processing Systems*, 36, 2024.

[46] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. *arXiv preprint arXiv:2503.05236*, 2025.

[47] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. *arXiv preprint arXiv:2306.09341*, 2023.

[48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.

[49] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. *arXiv preprint arXiv:2509.16117*, 2025.

---

## 附录 A 更多细节

遵循 Flow-GRPO 的数据和奖励配置，我们使用 GenEval、OCR 和 PickScore 的轮次比例 3:1:1 对 GRPO-mix 进行多任务混合训练。在每个轮次中，奖励由当前数据分区对应的奖励模型独占提供。训练在四个节点的分布式集群上执行，每个节点配备八块 H800 GPU，约 50 小时。对于 GenEval、OCR 和 PickScore 教师，我们使用了官方的 Flow-GRPO 检查点。此外，为纳入仅关注图像质量的 DeQA 教师，其奖励信号以 1:1 的求和比集成到标准 GRPO-mix 训练中。

**超参数说明** 除 $\beta$ 外，GRPO 超参数在各任务间保持一致。我们使用采样时间步 $T = 10$ 和评估时间步 $T = 40$。其他设置包括组大小 $G = 24$、噪声水平 $\sigma = 0.7$ 和图像分辨率 512。MAR KL 比率 $\beta$ 设置为 0.02。我们使用 LoRA，$\alpha = 64$ 和 $r = 32$。

| 模型 | 链接 |
|---|---|
| SD3.5-M [2] | https://huggingface.co/stabilityai/stable-diffusion-3.5-medium |
| Aesthetic Score [44] | https://github.com/LAION-AI/aesthetic-predictor |
| PickScore [42] | https://huggingface.co/yuvalkirstain/PickScore_v1 |
| DeQA score [43] | https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3 |
| ImageReward [45] | https://huggingface.co/THUDM/ImageReward |
| UnifiedReward [46] | https://huggingface.co/CodeGoat24/UnifiedReward-7b-v1.5 |
| HPS-v2.1 [47] | https://github.com/tgxs002/HPSv2 |
| Qwenvl Score [48] | https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507 |

关于 Qwenvl Score，我们采用了 Flow-GRPO [16] 中使用的提示词，如图 6 所示。我们使用 Qwen3-30B-A3B-Instruct-2507。

**图 6：Qwenvl Score 的结构化评估提示词**

```
[ROLE]（角色）
You are a professional visual verification assistant. Evaluate the generated image based on the
provided instruction across three specific dimensions.
（你是一名专业的视觉验证助手。根据提供的指令，在三个特定维度上评估生成的图像。）

[CRITERIA]（标准）
1. Aesthetic Quality（美学质量）
   - 1-2 (Low/低): Blurry, poor lighting, or chaotic composition.（模糊、光线差或构图混乱。）
   - 3 (Fair/中等): In focus, adequate lighting, but lacks creativity.（对焦清晰、光线充足，但缺乏创意。）
   - 4-5 (High/高): Sharp, vibrant colors, masterful composition and impact.（清晰、色彩鲜艳、构图精湛且富有冲击力。）
2. Instruction Following（指令遵循）
   - 1-2 (Low/低): Ignores or contradicts the instruction; misses key elements.（忽略或违反指令；缺少关键元素。）
   - 3 (Fair/中等): Partially follows, but distorts some important elements.（部分遵循，但扭曲了一些重要元素。）
   - 4-5 (High/高): Faithful representation of all elements in the prompt.（忠实呈现提示中的所有元素。）
3. Overall Score (Priority: Alignment > Aesthetics)（总体分数，优先级：对齐 > 美学）
The overall score must primarily reflect Instruction Following. A fair image that perfectly follows
the prompt scores higher than a beautiful image that misses it.
（总体分数必须主要反映指令遵循程度。一幅完美遵循提示但质量一般的图像，其得分应高于一幅美观但偏离提示的图像。）

[EXECUTION RULES]（执行规则）
   - Strictness: Be rigorous; required details must be explicitly supported.（严格性：需严谨；要求的细节必须有明确支持。）
   - Reasoning: You MUST analyze keyword-by-keyword in the <Thought> tag.（推理：必须在 <Thought> 标签中逐关键词分析。）
   - Output: Provide the analysis first, then the scores in XML tags.（输出：先提供分析，再以 XML 标签形式提供分数。）

[INPUT DATA]（输入数据）
Instruction: {prompt}
（指令：{prompt}）

[OUTPUT FORMAT]（输出格式）
<Thought> [Detailed analysis of quality and adherence] </Thought>
<QualityScore>X</QualityScore>
<InstructionScore>Y</InstructionScore>
<OverallScore>Z</OverallScore>
```

---

## 附录 B 更多结果

### B.1 定性结果

更多定性结果如图 7、8 和 9 所示。我们的方法不仅确保了精确的内容生成，还提供了卓越的图像质量和连贯的结构布局。通过与人类偏好实现更强的对齐，Flow-OPD 展示了在功能准确性与美学卓越之间架桥的显著新颖性。

*（图 7-9 中的文本提示和图像均为英文原版引用，此处不再逐条翻译。）*

### B.2 与 DiffusionNFT 的比较

DiffusionNFT [49] 引入了一种在线强化学习框架，将奖励反馈直接集成到前向扩散过程中，实现在噪声注入阶段的策略优化。尽管在基准分数上具有竞争力，DiffusionNFT 仍存在若干关键局限性。首先，它与无分类器引导（Classifier-Free Guidance, CFG）根本不兼容，这严重限制了其性能上限。其次，它遭受明显的奖励黑客行为。如图 10 所示，虽然模型正确生成了目标文本和"日落"元素，但同时幻觉出畸形的手部和多余的物体（如橙子），并伴有严重的过平滑、类塑料纹理伪影。当前的基准在很大程度上忽视了这些局部结构和美学失败。为解决这一评估盲点，我们采用 Qwenvl Score 进行更全面的评估。通过利用连续时间密集多专家监督和任务风格解耦，Flow-OPD 有效规避了这些奖励黑客行为，获得了比 DiffusionNFT 显著更高的 Qwen-VL 分数。这些发现也凸显了社区亟需开发更鲁棒、细粒度的文本到图像生成评估范式。

**表 5：人类偏好对齐的比较。我们的 Flow-OPD 在复杂视觉推理和布局连贯性方面始终获得卓越分数（由 Qwen-VL 评估）。**

| 模型 | Qwen-VL Score |
|---|---|
| DiffusionNFT | 3.74 |
| **Ours (Flow-OPD)** | **4.05** |

### B.3 失败案例与局限性

尽管 Flow-OPD 在主观和客观基准上均表现出优越性能，但某些局限性仍然存在。一个主要限制是教师模型所施加的性能上限。如图 11 所示，当专业教师未能合成语义正确的图像时，这些不准确性会通过密集监督信号传播。这种错误引导会将噪声引入蒸馏目标，最终阻碍学生超越教师集成的固有限制。另一个固有局限性是教师和学生模型之间需要架构同质性方可实现细粒度的逐步监督。展望未来，我们旨在通过若干有前景的方向探索 Flow-OPD 的更广泛潜力，包括：（1）协同演化蒸馏（Co-evolutionary Distillation），其中教师和学生迭代地相互优化；（2）自蒸馏（Self-Distillation）机制，无需外部教师即可提升性能；（3）跨词汇表蒸馏（Cross-Vocabulary Distillation），以弥合异构模型架构之间的差距。

---

## 更广泛的影响

我们关于 Flow-OPD 的工作引入了一个面向生成模型中多任务对齐的鲁棒框架，既带来了积极的社会贡献，也带来了需要审慎考虑的潜在风险。

### 积极的社会影响

本工作的主要贡献是增强了 AI 生成内容的功能可靠性。通过改进布局连贯性和 OCR 准确率，Flow-OPD 可以显著造福专业领域，如自动化平面设计、教育内容创作以及面向视障人士的辅助技术。此外，我们的多教师范式推动了更平衡的优化目标，这减轻了单奖励强化学习中固有的"赢者通吃"偏差，有望催生更多样化和更具代表性的生成系统。

### 消极的社会影响

尽管有这些益处，生成高质量、遵循指令的图像的能力提升也可能被滥用于制造复杂的虚假信息或欺骗性视觉内容。虽然我们的模型继承了基础模型的安全过滤器，但增强的结构真实感可能被利用来生成更具说服力的伪造文件或误导性社交媒体资产。为缓解这一问题，我们倡导在下游应用中集成数字水印和来源追踪（如 C2PA）。此外，与所有大规模生成模型一样，存在专业教师可能隐藏训练数据中潜在偏差的风险，这些偏差可能被无意中蒸馏到学生模型中。我们鼓励社区在采用本框架的同时使用偏差检测基准，以确保在所有人群中的公平性能。

---

*本文为完整翻译版本，保留了所有公式、表格结构、图表说明和参考文献原文。翻译力求学术严谨与专业术语准确。*
