# Prune-OPD：面向长程推理的高效可靠在策略蒸馏

# Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning

**arXiv:2605.07804v3 [cs.LG] 2026年6月1日**

---

## 作者信息

**杨志成¹&emsp;郭志江¹,²&emsp;宋一帆¹&emsp;许敏锐¹&emsp;王永新³&emsp;王一伟⁴&emsp;梁小丹³,⁵&emsp;唐靖¹,²**

¹ 香港科技大学（广州）
² 香港科技大学
³ 穆罕默德·本·扎耶德人工智能大学（MBZUAI）
⁴ 加州大学默塞德分校
⁵ 中山大学

**项目仓库：** https://github.com/yangzhch6/Prune-OPD

---

## 摘要

在策略蒸馏（On-Policy Distillation, OPD）利用密集的教师奖励信号来增强推理模型。然而，将OPD扩展到长程任务时暴露了一个关键缺陷：随着学生模型生成的前缀不可避免地偏离教师的思维过程，教师的密集奖励失去了局部可利用性。在这些"漂移"轨迹上继续生成和评估token不仅降低了奖励质量，还造成了大量的计算浪费。为解决这一问题，我们提出了**Prune-OPD**，一个动态将训练预算与监督质量对齐的框架。通过持续监测学生与教师预测之间的局部兼容性（例如，通过top-k重叠度），Prune-OPD实时检测前缀漂移事件。一旦检测到严重漂移，它会单调地降低后续不可靠奖励的权重，并触发动态生成长度截断。这使得训练过程能够停止无意义的生成，并将计算资源严格重新分配到可靠的教师监督上。在多种教师-学生组合中，Prune-OPD始终将计算与监督可靠性对齐。当前缀漂移使密集教师奖励变得不可靠时，它将训练时间减少37.6%至68.0%，同时在具有挑战性的基准测试（AMC、AIME、HMMT）上保持甚至提升性能。当学生-教师兼容性保持较高时，它通过扩展训练窗口自动保留长上下文监督。这些结果表明，Prune-OPD并非通过盲目缩短生成长度来改进OPD，而是通过将计算重新分配到局部可利用的教师奖励上来实现改进。

---

## 1 引言

在策略蒸馏（OPD）已成为大语言模型后训练的核心技术。最近的系统如Qwen3 [36]、MiMo [34]和GLM-5 [45]在后训练流程中采用OPD风格的密集监督。Thinking Machines Lab [24]报告称，OPD方案能够以远低于结果奖励强化学习的计算量复现强大的推理增益。这使得OPD成为监督微调和可验证强化学习的有力补充：它在长推理轨迹上提供token级学习信号，而非等待稀疏的最终答案奖励。

OPD的吸引力在于其在策略特性。离策略蒸馏在固定的教师生成序列上训练学生，这会引发曝光偏差问题：在推理时，学生必须在并非由教师或参考策略生成的前缀下行动[2]。而OPD则从当前学生模型采样生成长度，并在相同的学生访问前缀上评估教师。这一设计也启发了最近的自蒸馏方法，其中模型在特权信息或反馈下充当自身的教师，并从自身的在策略经验中改进[14, 47, 30]。在长程数学推理中，这一区别尤为重要：单个解答可能包含数千个token，因此即使是微小的训练-推理不匹配也可能在整个轨迹上累积放大。

**预印本。**

![图1]()

> **图1：Prune-OPD概念概述。** Prune-OPD沿学生生成长度监测局部学生-教师兼容性，在低重叠漂移事件后单调衰减OPD奖励，并在可靠的教师监督耗尽后截断响应。

然而，同样的在策略设计也带来了一个新的可靠性问题。教师不仅在被查询的前缀上其局部分布能提供有用的纠正，也在已经偏离教师推理过程的前缀上被查询。最近的OPD动力学分析指出，OPD的成功依赖于局部思维模式兼容性，并且奖励质量可能随轨迹深度而恶化[22]。我们将这一诊断视为设计约束：长程OPD不应将每个生成的token视为同等值得监督。

缺失的一环是在线可靠性控制。当前的OPD方案通常是在生成前选择一个生成长度预算，然后在采样的响应上均匀地施加密集教师奖励。这造成了一个糟糕的权衡。短的固定预算可能丢弃有用的长推理，而长的固定预算则花费大量计算来生成和评分那些教师信号可能不再局部可利用的后缀。实际上，单次生成长度可以同时包含两种状态：早期的前缀，其中学生和教师仍共享合理的下一token支持；以及随后的后缀，其中学生已经承诺了教师不会采取的推理选择。因此，OPD所需的不是简单增加或减少密集奖励，而是一种判断密集奖励在何处足够可靠以用于训练的方法。

我们提出**Prune-OPD**，一个面向长程OPD的可靠性感知修饰器，在训练过程中做出这一决策。Prune-OPD不过滤整个响应，也不施加固定的响应长度，而是提出一个位置级别的问题：在相同的学生前缀下，学生和教师是否仍然共享一个高概率候选区域？主要信号是每位置的top-k重叠率，教师top-p/top-k动作接受度作为一个更严格的变体。当选定的兼容性指标低于其阈值时，Prune-OPD记录一个前缀漂移事件。累积的漂移事件定义了一个单调可靠性权重，用于衰减后续的OPD奖励，而同样的原始可靠性信号则定义了动态生成长度控制的有效响应长度。预期的转变是从固定预算OPD转向可靠性预算OPD：密集监督应被分配到教师在学生轨迹上仍然可操作的位置。

我们进行了广泛的实验，以评估可靠性感知剪枝是否在实践中改善了长程OPD。结果验证了Prune-OPD的有效性：当密集监督变得不可靠时，它将训练时间减少了37.6%至68.0%，同时保持或提升了下游准确率。额外的实验表明，当可靠性信号保持较高时，控制器保留长上下文监督，支持了Prune-OPD重新分配计算而非盲目缩短生长度的论断。总之，我们的贡献如下：

- 我们将长程OPD形式化为学生访问前缀上的可靠性分配问题，将轨迹深度退化与局部兼容性和奖励可利用性联系起来。
- 我们提出了Prune-OPD，它将累积前缀漂移转换为单调的OPD奖励衰减和动态响应长度控制器，在禁用时不改变基线OPD路径。
- 我们提供了初步证据，表明可靠性感知剪枝可以在前缀漂移限制可靠监督时将OPD训练时间减少37.6%至68.0%，同时保持基准性能，并且我们指定了固定长度剪枝对照，以测试收益是否来自可靠性而非简单的泛化缩短。

---

## 2 背景与问题设置

### 2.1 在策略蒸馏

设 $x$ 表示输入提示，$\mathbf{y} = (y_1, \ldots, y_T)$ 是从学生策略 $\pi_\theta$ 自回归采样的响应。在位置 $t$，记 $\mathbf{y}_{<t} = (y_1, \ldots, y_{t-1})$ 为前缀，并定义学生和教师的下一token分布为：

$$p_t(v) \triangleq \pi_\theta(v \mid x, \mathbf{y}_{<t}), \quad q_t(v) \triangleq \pi_T(v \mid x, \mathbf{y}_{<t}), \tag{1}$$

在词汇表token $v \in \mathcal{V}$ 上。OPD在当前学生采样的轨迹上计算监督，通常通过最小化学生生成轨迹上的逆KL目标[9, 1, 22]：

$$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x, \mathbf{y} \sim \pi_\theta(\cdot|x)} \left[ \sum_{t=1}^{T} D_{\text{KL}}(p_t \parallel q_t) \right]. \tag{2}$$

这种token级分解驱动了OPD的样本效率：每个位置提供密集的教师信号，而不是等待最终答案奖励。这也暴露了核心假设：$q_t$ 应在学生访问的状态 $(x, \mathbf{y}_{<t})$ 上是有用的局部目标。在实践中，我们在学生的top-k候选token上近似此奖励，并获得每位置的奖励向量 $\mathbf{r}_{t,\theta}$。由于奖励已经按位置局部化，可靠性修饰器可以在策略损失之前简单地将位置 $t$ 的所有奖励乘以一个标量权重。这是Prune-OPD使用的接口：

可靠的前缀保留其OPD奖励，而漂移的前缀则获得衰减的奖励。

### 2.2 动态指标

遵循OPD机制分析[22]，设 $\mathcal{S}_t(p) = \text{TopK}(p_t, k)$ 和 $\mathcal{S}_t(q) = \text{TopK}(q_t, k)$。该工作提出的重叠率为：

$$\mathcal{M}_{\text{overlap}} = \mathbb{E}_t\left[ \frac{|\mathcal{S}_t(p) \cap \mathcal{S}_t(q)|}{k} \right], \tag{3}$$

它衡量学生和教师是否将高概率分配给共享的候选区域。

熵差：

$$\Delta H_t = |H(q_t) - H(p_t)|, \tag{4}$$

跟踪两个模型在相同访问状态上是否具有相似的不确定性。这些指标通常作为训练后的诊断报告。Prune-OPD将其转化为在线控制信号。当前主要指标是每位置的重叠率，它衡量学生和教师在相同学生前缀上是否仍共享高概率候选区域。我们还评估了一个更严格的动作级别变体，询问采样的学生token是否被教师的可观察top-p/top-k区域接受。

### 2.3 长程可靠性问题

位置 $t$ 的奖励依赖于 $q_t(v) = \pi_T(v \mid x, \mathbf{y}_{<t})$，其中前缀 $\mathbf{y}_{<t}$ 由学生生成。随着 $t$ 增长，前缀可能偏离教师自然生成的轨迹。先前的OPD分析报告了后缀到前缀的不稳定性，以及随着学生前缀深度增加，教师延续优势递减[22]。因此，在中等长度轨迹上有帮助的同一密集奖励，在长轨迹上可能变得不可靠。这促使了位置级别的可靠性控制。我们估计每个学生前缀是否保持教师兼容，然后将密集监督分配给可靠的前缀，并在前缀漂移后衰减奖励。

---

## 3 Prune-OPD

Prune-OPD将局部学生-教师兼容性转换为OPD奖励上的位置级权重。该方法被设计为OPD的即插即用修饰器：它不改变教师奖励的计算方式，但改变每个访问位置应该信任该奖励的程度。它包含三个部分：每位置兼容性指标、累积可靠性权重和可选的动态响应长度控制器。

### 3.1 兼容性指标

在位置 $\tau$，我们的目标是判断教师在学生当前轨迹上是否仍然提供局部可利用的监督。因此，学生和教师在相同的由学生生成的前缀 $\mathbf{s}_\tau = (x, \mathbf{y}_{<\tau})$ 上进行评估。

我们首先将比较限制在每个模型的高置信度候选token上。设

$$\mathcal{K}_S = \text{TopK}\big(\pi_\theta(\cdot \mid \mathbf{s}_\tau), k\big), \quad \mathcal{K}_T = \text{TopK}\big(\pi_T(\cdot \mid \mathbf{s}_\tau), k\big) \tag{5}$$

为返回的top-k候选集。top-k大小是一个实验超参数。然后，我们用重叠率测量局部兼容性：

$$\mathcal{O}_\tau = \frac{|\mathcal{K}_S \cap \mathcal{K}_T|}{k}. \tag{6}$$

低重叠意味着教师的密集奖励是在其偏好的下一token支持已与学生支持分离的前缀下计算的。因此，我们定义前缀漂移事件为：

$$\mathcal{B}_\tau = \mathbb{1}\left[\mathcal{O}_\tau < \delta\right], \tag{7}$$

其中 $\delta$ 是兼容性阈值。这些漂移事件随后被累积，以衰减后续OPD奖励并定义动态生成长度控制的有效响应长度。

该指标直接遵循OPD机制分析：成功的OPD与逐渐增加的高概率重叠相关联，而失败的运行则在学生访问状态上表现出停滞或不稳定的重叠[22]。它衡量学生和教师是否仍共享合理的局部动作空间。低重叠意味着教师的密集奖励正在其偏好的下一token支持已与学生支持分离的区域中计算。我们还设计了一个替代的top-p动作接受度指标，其定义见附录A.10。

### 3.2 累积可靠性与损失加权

坏事件在响应上累积：

$$C_\tau = \sum_{i=1}^{\tau} \mathcal{B}_i. \tag{8}$$

原始可靠性权重是一个截断的线性衰减：

$$\tilde{R}_\tau = \text{clip}(1 - w_{\text{drop}} \cdot C_\tau, \; 0, \; 1), \tag{9}$$

其中 $w_{\text{drop}} \geq 0$ 控制每次兼容性失败的惩罚。原始权重沿每个有效响应单调非增。填充位置被强制设为零。

实际损失权重添加一个基础底线：

$$\mathcal{L}_\tau = \tilde{R}_\tau + w_{\text{base}} \tag{10}$$

对有效响应位置，且 $\mathcal{L}_\tau = 0$ 对填充位置。缩放后的OPD奖励为：

$$\tilde{\mathbf{r}}_{\tau,j} = \mathcal{L}_\tau \cdot \mathbf{r}_{\tau,j}. \tag{11}$$

基础权重有两个实际作用。第一，它防止Prune-OPD在 $\tilde{R}_\tau$ 达到零时变成硬开关。第二，它允许方法在去噪和保留弱教师信号之间进行权衡。具体的超参数值在实验设置中指定。

### 3.3 动态响应预算

Prune-OPD还可以在训练期间调整生成长度。定义一个样本的有效响应长度为：

$$\mathcal{E} = \sum_{\tau=1}^{T} \mathbb{1}\left[\tilde{R}_\tau > \varepsilon\right], \tag{12}$$

对于一个小容差 $\varepsilon$。此定义刻意使用 $\tilde{R}_\tau$ 而非 $\mathcal{L}_\tau$。因此，具有 $\tilde{R}_\tau = 0$ 和 $w_{\text{base}} > 0$ 的token仍可贡献少量的OPD奖励，但不被计为可靠长度。

---

**算法1：Prune-OPD——可靠性感知奖励缩放与动态响应预算**

**输入：** 提示分布 $\mathcal{D}$，学生策略 $\pi_\theta$，教师策略 $\pi_T$，OPD奖励例程，初始响应预算 $M_1$

**输出：** 更新后的学生策略 $\pi_\theta$ 和响应预算序列 $\{M_t\}$

1. **for** 训练步 $t = 1, 2, \ldots$ **do**
2. &emsp;采样提示 $\{x^{(b)}\}_{b=1}^{B} \sim \mathcal{D}$，并以最大长度 $M_t$ 生成学生生成长度 $\{\mathbf{y}_{1:T_b}^{(b)}\}_{b=1}^{B}$。
3. &emsp;在学生访问前缀上计算原始OPD奖励张量 $\mathbf{r}^{(b,j)}$。
4. &emsp;**for** 每个生成长度 $b = 1, \ldots, B$ **do**
5. &emsp;&emsp;使用公式(6)在学生访问前缀上计算重叠率序列。
6. &emsp;&emsp;标记低兼容性位置，并使用公式(7)–(10)形成累积可靠性权重。
7. &emsp;&emsp;使用公式(11)缩放OPD奖励，并使用公式(12)计算有效可靠长度 $\mathcal{E}^{(b)}$。
8. &emsp;**end for**
9. &emsp;使用缩放后的奖励 $\tilde{\mathbf{r}}^{(b,j)}$ 以OPD目标更新 $\pi_\theta$。
10. &emsp;$h_t \leftarrow B^{-1} \sum_{b=1}^{B} \mathbb{1}\left[\mathcal{E}^{(b)} \geq M_t - m\right]$。
11. &emsp;**if** $h_t \geq \eta$ **then**
12. &emsp;&emsp;$M_{t+1} \leftarrow \min(M_t + \Delta, M_{\max})$ 并重置低命中计数器。
13. &emsp;**else if** $h_t < \eta$ 连续 $P$ 步 **then**
14. &emsp;&emsp;$M_{t+1} \leftarrow \max(M_t - \Delta, M_{\min})$ 并重置低命中计数器。
15. &emsp;**else**
16. &emsp;&emsp;$M_{t+1} \leftarrow M_t$。
17. &emsp;**end if**
18. **end for**

---

令 $M_t$ 为训练步 $t$ 用于生成长度生成的最大响应长度。控制器监测批次命中率：有效可靠长度达到当前限制（减去边距）的样本比例。如果命中率较大，限制按固定步长扩展，上限为 $M_{\max}$。如果命中率在耐心窗口内保持较小，限制按相同步长收缩，下限为 $M_{\min}$。初始限制、上下界、更新步长、边距、命中率阈值和耐心在实验设置中指定。算法1总结了完整的Prune-OPD流程，包括可靠性感知奖励缩放和动态响应预算自适应。为保持训练循环的可读性，位置级兼容性和加权规则直接引用上述定义。

---

## 4 实验

我们评估可靠性感知奖励缩放是否能在保持下游推理准确率的同时显著降低OPD训练成本。除最终基准准确率和训练时间外，我们还测量学生访问状态上的局部学生-教师动力学，包括重叠率、top-p接受度和有效可靠长度。

### 4.1 实验设置

**训练数据。** 我们在DAPO-Math-17K [44]上训练，遵循近期推理实验中使用的OPD设置。训练提示在每次运行前打乱。主要的强化学习结果奖励是最终答案正确性。然而，OPD提供密集的token级蒸馏奖励。

**模型。** 我们评估五对教师-学生组合，以在不同OPD动力学下测试可靠性信号。DeepSeek-R1-Distill-Qwen-1.5B从DeepSeek-R1-Distill-Qwen-7B蒸馏，后者是一个更大的同族推理模型，其更高的基准得分可能并不意味着局部可利用的token级反馈[10]。同一学生还从JustRL-DeepSeek-1.5B蒸馏，后者是一个同规模的RL改进版教师，由DeepSeek-R1-Distill-Qwen-1.5B通过RL获得[11]。Qwen3-1.7B-Base和Qwen3-4B-Base从Qwen3-4B（非思考模式）蒸馏，隔离了思考模式转变如何在同一模型族内与学生规模交互[36]。最后，DeepSeek-R1-Distill-Qwen-7B从Skywork-OR1-7B蒸馏，提供了一个高初始重叠的设置，其中Prune-OPD预期保留长上下文监督而非缩短生成长度。总体而言，这些组合在不同条件下测试同一论断：当可靠性衰退时控制器减少计算，当兼容性保持较高时保留密集监督。

**基线。** 我们比较基线OPD、固定长度截断基线和Prune-OPD变体。固定长度OPD使用与基线OPD相同的蒸馏目标，但将最大响应长度限制为固定预算。随机剪枝匹配Prune-OPD保留的token比例或保留的OPD奖励质量，同时独立于学生-教师兼容性选择剪枝位置。这些对照测试了收益是否来自学生-教师兼容性信号，而非来自简单的更短生成长度或更低奖励密度。主要Prune-OPD配置使用重叠率，阈值为 $\delta = 0.7$。我们还报告了教师top-p动作接受度变体，$p = 0.95$。两种变体均使用 $\text{top-k} = 16$、$w_{\text{drop}} = 0.01$、$w_{\text{base}} = 0.5$，以及动态长度命中率 $0.1$。除非另有说明，动态响应长度使用初始值 $2048$、最小值 $1024$、最大值 $12288$、更新步长 $100$、边距 $100$ 和收缩耐心 $3$。

**评估。** 我们在AMC23、AIME24、AIME25、HMMT24和HMMT25上进行评估。主要指标是固定评估协议下的pass@1准确率。我们还报告相对训练时间减少、响应长度统计、重叠率、top-p接受率和有效可靠长度。训练时间相对于同一教师-学生对下的基线OPD测量。

### 4.2 主要结果

主要结果是在不显著降低下游准确率的情况下实现一致的效率增益。在主要低兼容性运行中，重叠率变体将训练时间分别减少了35.7%、68.0%、37.6%和52.6%，而基准准确率保持接近相应的OPD基线。准确率差异在各个基准上混合存在，应视为次要的：Prune-OPD在若干AIME/HMMT得分上略有提升，但也降低了少数条目，例如DeepSeek-R1-Distill-Qwen-7B教师下的AIME24和JustRL-DeepSeek-1.5B教师下的HMMT24。这一模式与Prune-OPD作为可靠性感知效率机制而非独立的准确率提升方法的角色一致。top-p动作接受度变体通常更保守，且往往弱于重叠率变体，这表明精确采样token的可接受性对于推理轨迹可能过于严格，而候选空间重叠更好地捕捉了教师是否仍然提供可利用的局部监督。

**表1：Prune-OPD主要结果。** 准确率以百分比报告。时间减少相对于同一学生-教师对下的基线OPD。

| 方法 | AMC23 | AIME24 | AIME25 | HMMT24 | HMMT25 | 时间减少 |
|------|-------|--------|--------|--------|--------|----------|
| **DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B** |
| OPD | 77.5 | 43.5 | 32.7 | 23.5 | 18.1 | 6.9h / 0% |
| OPD (截断4k) | 77.8 | 43.1 | 31.0 | 22.7 | 17.9 | 4.8h / 30.4% |
| PRUNE-OPD (top-p) | 77.3 | 44.0 | 30.8 | 23.3 | 19.8 | 4.3h / 37.7% |
| PRUNE-OPD (overlap) | 78.4 | 45.2 | 33.5 | 22.7 | 18.8 | 4.1h / 40.6% |
| **DeepSeek-R1-Distill-Qwen-1.5B / DeepSeek-R1-Distill-Qwen-7B** |
| OPD | 66.4 | 33.5 | 25.4 | 14.4 | 15.0 | 12.5h / 0% |
| OPD (截断4k) | 66.3 | 31.7 | 25.8 | 14.2 | 14.4 | 6.6h / 47.2% |
| PRUNE-OPD (top-p) | 65.5 | 29.6 | 24.2 | 14.6 | 14.0 | 4.8h / 61.6% |
| PRUNE-OPD (overlap) | 65.9 | 33.3 | 25.8 | 14.6 | 15.0 | 4.0h / 68.0% |
| **Qwen3-1.7B-Base / Qwen3-4B (非思考)** |
| OPD | 28.0 | 7.3 | 2.9 | 2.1 | 0.6 | 16.5h / 0% |
| OPD (截断4k) | 28.6 | 6.3 | 2.9 | 2.3 | 1.0 | 12.9h / 21.8% |
| PRUNE-OPD (top-p) | 28.4 | 5.6 | 2.9 | 1.9 | 0.6 | 11.5h / 30.3% |
| PRUNE-OPD (overlap) | 29.1 | 7.3 | 4.2 | 2.9 | 1.0 | 10.3h / 37.6% |
| **Qwen3-4B-Base / Qwen3-4B (非思考)** |
| OPD | 42.7 | 14.8 | 10.6 | 5.4 | 2.9 | 28.5h / 0% |
| OPD (截断4k) | 41.9 | 15.2 | 13.5 | 4.8 | 2.7 | 22.3h / 21.8% |
| PRUNE-OPD (top-p) | 40.5 | 13.1 | 12.7 | 4.8 | 4.0 | 20.0h / 29.8% |
| PRUNE-OPD (overlap) | 42.7 | 16.3 | 14.6 | 6.7 | 5.6 | 13.5h / 52.6% |

![图2]()

> **图2：四个DeepSeek学生-教师对的准确率随墙钟时间变化。** 每个面板以墙钟时间为x轴，基准准确率为y轴，比较OPD和Prune-OPD。成功的曲线应在匹配或超过OPD准确率的同时，以更短的时间达到可比检查点。

### 4.3 高兼容性下的自适应行为

该设置测试Prune-OPD是否均匀地缩短每次运行。当教师和学生具有高top-k重叠时，有效长度命中率很快得到满足，动态控制器将训练窗口打开到12288个token。由此产生的训练时间和准确率与OPD几乎不变。图3应从三个视图展示相应的机制：重叠率保持高位，动态响应预算跟踪长可靠上下文而非强制提前截断，AMC23学习曲线比较Prune-OPD是否匹配OPD级别的准确率，同时避免固定截断的刚性。这支持了预期的解释：Prune-OPD是一个自适应可靠性控制器，而非硬性的长度惩罚。

**表2：高兼容性行为。** DeepSeek-R1-Distill-Qwen-7B和Skywork-OR1-7B初始重叠率约为0.94。

| 方法 | AMC23 | AIME24 | AIME25 | HMMT24 | HMMT25 | 时间减少 |
|------|-------|--------|--------|--------|--------|----------|
| **DeepSeek-R1-Distill-Qwen-7B / Skywork-OR1-7B** |
| OPD | 88.3 | 67.1 | 52.5 | 33.5 | 32.1 | 17.5h / 0% |
| OPD (截断4k) | 88.9 | 66.7 | 52.1 | 33.1 | 31.5 | 11.3h / 35.4% |
| PRUNE-OPD (overlap) | 87.2 | 66.0 | 48.8 | 31.0 | 30.4 | 17.0h / 2.9% |

![图3]()

> **图3：DeepSeek-R1-Distill-Qwen-7B / Skywork-OR1-7B的高兼容性训练动力学。** 左：有效响应长度和最大OPD长度随训练步变化。中：重叠率随训练步变化。右：AMC23准确率随训练变化，比较OPD、OPD（截断4k）和Prune-OPD。

### 4.4 训练动力学

图4检验了Prune-OPD是否改变了学习动力学，还是仅仅减少了生成长度计算。对于JustRL-DeepSeek-1.5B教师，各基准的步数-准确率曲线应显示Prune-OPD在AMC23、AIME24、AIME25、HMMT24和HMMT25上遵循与OPD相同的学习轨迹，而非以牺牲准确率为代价获得速度。图5解释了节省的计算来自何处：按位置的权重图应显示累积前缀漂移后的后缀奖励衰减，有效长度图应显示动态最大OPD长度自适应于可靠监督窗口，重叠率曲线应确认剪枝由测量的局部兼容性信号驱动。

![图4]()

> **图4：DeepSeek-R1-Distill-Qwen-1.5B从JustRL-DeepSeek-1.5B蒸馏的训练步准确率动力学。** 五个面板报告AMC23、AIME24、AIME25、HMMT24和HMMT25上随训练步的基准准确率，比较OPD和Prune-OPD。

![图5]()

> **图5：DeepSeek-R1-Distill-Qwen-1.5B从JustRL-DeepSeek-1.5B蒸馏的训练动力学诊断。** 面板报告按token位置的Prune-OPD平均权重，每20个训练步绘制曲线（从0到200）；有效响应长度和最大OPD长度随训练变化；以及重叠率随训练变化。

### 4.5 墙钟效率

图2在墙钟时间轴上评估核心效率主张：Prune-OPD应在不降低准确率的情况下减少经过的训练时间，在某些情况下甚至可以略微提升。如果Prune-OPD按预期工作，其曲线应在可比准确率水平上位于OPD左侧，或在相同墙钟时间上高于OPD。这一呈现直接测试了Prune-OPD是否在两种DeepSeek教师-学生组合中以更少的经过训练时间达到OPD级别（且偶尔更好）的准确率。结合图4和图5的动力学，这些结果表明：可靠性感知后缀降权缩短了非生产性生成长度计算，同时保留了驱动最终性能的有用前缀监督。

**表3：DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B上的重叠阈值消融。**

| 阈值 $\delta$ | AMC23 | AIME24 | AIME25 | HMMT24 | HMMT25 | 时间减少 |
|---------------|-------|--------|--------|--------|--------|----------|
| 0.6 | 78.9 | 45.4 | 33.1 | 22.0 | 18.8 | 4.8h / 30.4% |
| 0.7 | 78.4 | 45.2 | 33.5 | 22.7 | 18.8 | 4.1h / 40.6% |
| 0.8 | 77.8 | 44.5 | 31.5 | 21.9 | 18.0 | 3.1h / 55.1% |
| 0.9 | 76.8 | 43.9 | 30.8 | 21.8 | 17.9 | 2.4h / 65.2% |

### 4.6 消融研究

**OPD与简单截断。** 我们在表1和表2中包含了OPD（截断4k）作为固定预算基线。该基线使用相同的OPD目标，但均匀地缩短每次生成长度，而不观察学生-教师兼容性。尽管简单截断可以降低成本，并且对某些教师-学生对可能有效，但它并非Prune-OPD的可靠替代。特别是，对于DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B和DeepSeek-R1-Distill-Qwen-7B / Skywork-OR1-7B，在固定4K预算下截断会降低性能，因为它丢弃了有用的长前缀监督。这一失败模式突显了刚性长度规则的局限：它无法区分漂移的后缀与长但仍兼容的推理轨迹。

**剪枝指标：重叠率 vs. top-p接受度。** 指标消融在表1中报告。在已完成的运行中，重叠率是更强的默认选择：它提供与top-p接受度相等或更大的时间减少，同时通常更好地保持准确率。我们将此归因于其支持级别特性：它询问学生和教师是否仍共享高概率候选区域，而非要求采样的学生token落入教师的可见核内。这使得重叠率成为长推理轨迹的平滑可靠性信号，其中多个局部合理的延续可能保持有效。

**Prune-OPD阈值消融。** 我们在表3中消融了DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B上的重叠率阈值。已完成的 $\delta = 0.7$ 设置提供了一个强大的工作点：它在保持OPD级别基准性能的同时将训练时间减少了35.7%。较低的阈值应更宽容，产生更少的坏事件和更长的有效蒸馏长度；较高的阈值应更早剪枝，如果设置过于激进可能移除有用的长推理。核心结论是 $\delta = 0.7$ 同时提供了性能保护和显著的训练效率提升。

---

## 5 相关工作

**在策略蒸馏。** MiniLLM将OPD形式化为在逆KL目标下由策略梯度优化的大语言模型方法，强调了逆KL的模式寻求行为[9]。GKD通过在多种散度下插值在策略和离策略数据，拓宽了这一视角[1]。更近期的理论将OPD框架化为密集KL约束的强化学习，其中教师的每token对数比可以被解释为隐式奖励，甚至可以外推到教师标准强度之外[38]。OPD此后出现在主要的推理和后训练方案中[36, 24, 45, 34, 20, 17, 15, 7, 38, 8]。它还被扩展到自蒸馏和在线经验学习，其中模型在特权信息、反馈或样本路由下充当自身的教师[14, 47, 12, 30, 43, 26, 18, 42, 37, 21, 46, 6, 31]。这些工作确立了OPD作为强大的后训练原始语。Prune-OPD解决了一个互补的可靠性问题：即使密集教师奖励在策略上可用，长生成长度可能包含教师局部信号不再可利用的前缀。关于OPD的更多讨论见附录A.5。

---

## 6 结论

OPD为长推理链提供密集监督；然而，当学生的推理模式偏离教师时，产生的密集奖励变得不可靠。Prune-OPD将其视为位置级可靠性问题：它检查学生和教师在共享前缀下是否保持局部兼容，将累积兼容性失败转换为漂移信号，重新缩放OPD奖励，并基于可靠而非原始token计数控制响应长度。在多种蒸馏设置中，基于重叠的Prune-OPD在低兼容性情境下将训练时间减少了37.6%至68.0%，同时保持基准性能；当兼容性保持较高时，它保持长训练窗口打开。这些结果表明，高效的长程OPD不应仅由教师强度或固定生成长度预算主导。相反，密集教师监督应根据其对学生实际访问前缀的局部可利用性来分配。

---

## 参考文献

[1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In *International Conference on Learning Representations*, 2024.

[2] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. *Advances in Neural Information Processing Systems*, 28, 2015.

[3] Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russell Webb. Distillation scaling laws. In *International Conference on Machine Learning*, pages 5977–6045. PMLR, 2025.

[4] Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4794–4802, 2019.

[5] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. *Journal of Machine Learning Research*, 25(70):1–53, 2024.

[6] Ken Ding. HDPO: Hybrid distillation policy optimization via privileged self-distillation, 2026.

[7] Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes, 2026.

[8] Naibin Gu, Chenxu Yang, Qingyi Si, Chuanyu Qin, Dingyu Yao, Peng Fu, Zheng Lin, Weiping Wang, Nan Duan, and Jiaqi Wang. Co-evolving policy distillation, 2026.

[9] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In *The Twelfth International Conference on Learning Representations*, 2024.

[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. *Nature*, 645(8081):633–638, 2025.

[11] Bingxiang He, Zekai Qu, Zeyuan Liu, Yinghao Chen, Yuxin Zuo, Cheng Qian, Kaiyan Zhang, Weize Chen, Chaojun Xiao, Ganqu Cui, et al. JustRL: Scaling a 1.5B LLM with a simple RL recipe, 2025.

[12] Bingxiang He, Yuxin Zuo, Zeyuan Liu, Shangziqi Zhao, Zixuan Fu, Junlin Yang, Cheng Qian, Kaiyan Zhang, Yuchen Fan, Ganqu Cui, et al. How far can unsupervised RLVR scale LLM training?, 2026.

[13] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.

[14] Jonas Hubotter, Frederike Lubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation, 2026.

[15] Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggu Lim, and Taesup Kim. Stable on-policy distillation through adaptive target reformulation, 2026.

[16] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. TinyBERT: Distilling BERT for natural language understanding. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 4163–4174, 2020.

[17] Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models, 2026.

[18] Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of LLMs?, 2026.

[19] Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 1317–1327, 2016.

[20] Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning efficiently via relaxed on-policy distillation, 2026.

[21] Gengsheng Li, Tianyu Yang, Junfeng Fang, Mingyang Song, Mao Zheng, Haiyun Guo, Dan Zhang, Jinqiao Wang, and Tat-Seng Chua. Unifying group-relative and self-distillation policy optimization via sample routing, 2026.

[22] Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. *arXiv preprint arXiv:2604.13016*, 2026.

[23] Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, and Radha Poovendran. Small models struggle to learn from strong reasoners. In *Findings of the Association for Computational Linguistics: ACL 2025*, pages 25366–25394, 2025.

[24] Kevin Lu and Thinking Machines Lab. On-policy distillation. *Thinking Machines Lab: Connectionism*, 2025. https://thinkingmachines.ai/blog/on-policy-distillation.

[25] Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 5191–5198, 2020.

[26] Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. CRISP: Compressed reasoning via iterative self-policy distillation, 2026.

[27] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter, 2019.

[28] Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al. Multitask prompted training enables zero-shot task generalization. In *International Conference on Learning Representations*, 2021.

[29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

[30] Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. Self-distillation enables continual learning, 2026.

[31] Yifan Song, Xingjian Tao, Zhicheng Yang, Yihong Luo, and Jing Tang. EHRAG: Bridging semantic gaps in lightweight GraphRAG via hybrid hypergraph construction and retrieval. *arXiv preprint arXiv:2604.17458*, 2026.

[32] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural Information Processing Systems*, 33:5776–5788, 2020.

[33] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In *International Conference on Learning Representations*, 2021.

[34] Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. MiMo-v2-flash technical report, 2026.

[35] Minrui Xu, Zilin Wang, Mengyi Deng, Zhiwei Li, Zhicheng Yang, Xiao Zhu, Yinhong Liu, Boyu Zhu, Baiyu Huang, Chao Chen, Heyuan Deng, Fei Mi, Lifeng Shang, Xingshan Zeng, and Zhijiang Guo. EnvFactory: Scaling tool-use agents via executable environments synthesis and robust RL, 2026.

[36] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report, 2025.

[37] Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled RLVR, 2026.

[38] Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation, 2026.

[39] Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Wenlei Shi, Yiwei Wang, Xiaodan Liang, and Jing Tang. Accordion-thinking: Self-regulated step summaries for efficient and readable LLM reasoning. In *Forty-Third International Conference on Machine Learning*, 2026.

[40] Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Dongchun Xie, Hanhui Li, Yiwei Wang, Xiaodan Liang, and Jing Tang. Depth-breadth synergy in RLVR: Unlocking LLM reasoning gains with adaptive exploration. In *Forty-Third International Conference on Machine Learning*, 2026.

[41] Zhicheng Yang, Yiwei Wang, Yinya Huang, Zhijiang Guo, Wei Shi, Xiongwei Han, Liang Feng, Linqi Song, Xiaodan Liang, and Jing Tang. OptiBench meets ReSocratic: Measure and improve LLMs for optimization modeling. In *The Thirteenth International Conference on Learning Representations*, 2025.

[42] Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Online experiential learning for language models, 2026.

[43] Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models, 2026.

[44] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale, 2025.

[45] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, et al. GLM-5: From vibe coding to agentic engineering, 2026.

[46] Guoliang Zhao, Ruobing Xie, An Wang, Shuaipeng Li, Huaibing Xie, and Xingwu Sun. Self-distillation for multi-token prediction, 2026.

[47] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models, 2026.

---

## 附录

### A.1 更广泛影响

本研究为开发专注于复杂推理的更高效、更可靠的大语言模型训练协议做出了贡献。通过引入Prune-OPD，我们提供了一种机制来缓解密集在策略蒸馏中固有的"奖励黑客"和训练不稳定风险——否则模型可能从低质量或漂移的教师信号中学习。训练计算量的大幅减少（最多68%）直接回应了大语言模型后训练中高能耗带来的环境关切。通过使复杂的推理模型能够以更少的资源进行训练，本工作使开发高性能AI的能力更加民主化，可能加速科学发现的进展。

尽管有这些益处，我们认识到增强大语言模型的推理能力可能在不经意间降低生成复杂错误信息或双重用途内容的门槛。虽然Prune-OPD关注的是学习过程而非数据内容，但由此产生的模型如果被用于欺骗目的，可能更具说服力或更难检测。为减轻这些风险，我们强调将Prune-OPD与安全对齐的教师模型和严格的可验证强化学习框架结合使用的重要性。我们致力于通过提供详细的实验设置和超参数配置来实现透明和可复现性，使社区能够审计这些训练动力学的可靠性，并确保推理能力的提升不以模型安全性或伦理对齐为代价。

### A.2 大语言模型使用声明

本手稿严格将大语言模型用于语言编辑和文本润色，以提升呈现质量。我们声明，新颖的想法、方法论框架、实验执行和数据分析均为作者的原创工作。所有经AI工具修改的内容均已由作者仔细审查和验证以确保准确性。

### A.3 局限性

Prune-OPD依赖于top-k局部分布统计量。主要的重叠指标可能遗漏学生和教师共享候选token但排名或评分差异很大的情况，而top-p动作接受度变体对于风格不同但有效的推理路径可能过于严格。当前的衰减是累积坏事件的线性函数，并假设局部兼容性与奖励可利用性相关；当低重叠前缀后来返回到教师兼容状态时，这一假设可能失败。实现上也仅重新缩放OPD奖励，且不为零可靠性token附加GRPO回退。最后，当前实验仅限于使用DeepSeek、Qwen和Skywork风格模型的数学推理；智能体和多轮设置仍然是未来工作。

### A.4 未来工作

一个自然的扩展是将Prune-OPD与GRPO结合在可靠性门控目标中。在可靠性权重衰减到零之前，模型仍优化OPD损失并从局部兼容前缀上的密集教师指导中受益。在可靠性权重达到零之后，不可靠的后缀将切换到GRPO损失，允许学生使用结果级反馈继续探索，而非强迫其遵循不再局部可利用的教师分布。这种混合目标可以将教师引导蒸馏的样本效率与强化学习的探索优势结合起来，并可能进一步改进长程推理，超越仅依赖可靠性感知截断的方法。

### A.5 附加相关工作

**知识蒸馏与离策略蒸馏。** 知识蒸馏通过匹配教师的软输出分布将信息从教师迁移到学生[13]。对于自回归序列模型，序列级蒸馏在学生上训练教师生成的输出[19]，这种离策略范式是许多紧凑语言模型蒸馏方法的基础[27, 16, 32, 8]。监督微调和指令调优同样通过在被管理的响应或示范上训练来改进下游行为[5, 28, 33]。共同的局限是训练-推理分布不匹配：学生在教师或参考轨迹上优化，但在推理时必须在其自身策略采样的前缀上条件化——一种在长生成中累积的曝光偏差问题[2]。Prune-OPD遵循OPD的动机，即蒸馏应在学生访问的状态上计算，但提出了一个进一步的问题：一旦学生处于在策略状态，哪些访问状态仍然允许可靠的密集教师监督？

**OPD动力学与机制。** 最近的OPD分析表明，OPD的成功由思维模式一致性以及教师是否贡献了超出学生已获知识的新知识所支配[22]。在token层面，成功的OPD表现出高概率token上的渐进对齐：top-k重叠上升、重叠token优势改善，且熵差缩小。同一分析进一步表明，奖励质量随轨迹深度退化，并且奖励可能在全局上与最终正确性相关，但无法在学生当前策略周围提供局部有用的梯度。Prune-OPD直接操作化这些诊断发现。它不是将重叠率、熵差或教师可接受性作为事后测量，而是将每前缀兼容性转化为在线可靠性信号，衰减后续OPD奖励并控制生成长度。

**容量差距与可蒸馏性。** 另一条研究线索探讨了教师何时过于强大、差异过大或复杂到学生无法模仿。研究表明，过于强大的教师可能损害学生表现[4]，并且引入了教师助手以弥合大的容量差距[25]。蒸馏缩放定律识别了教师质量、学生规模和数据量之间的非单调交互[3]。对于推理模型，小学生可能难以从具有长思维链轨迹的强大推理者那里学习，表明蒸馏复杂推理行为存在可学习性差距[23]。这些结果警示不应假设教师基准强度单独决定蒸馏质量。Prune-OPD从在策略、位置级的视角研究同一问题：教师可能在全局上是有用的，但在学生生成轨迹的特定深度上不可靠。

**推理强化学习与长程监督。** 现代推理模型依赖可验证强化学习、长思维链生成长度和大型数学训练集[29, 10, 44, 11, 39, 40, 41, 35]。长响应为搜索和自我纠正创造空间，但也使密集监督变得昂贵且更难以信任。先前的OPD结果识别了响应长度的最佳点：短响应提供的密集信号太少，而非常长的响应可能触发后期崩溃和后缀到前缀的不稳定性[22]。Prune-OPD专为此长程设置而设计。它不施加固定的响应预算；它估计哪些前缀仍承载可靠的教师信号，并使用该估计进行奖励缩放和动态响应长度控制。

### A.6 可靠性理由

Prune-OPD的动机在于区分全局信息性教师奖励与学生访问前缀上局部可利用的token级监督[22]。因此，重叠率和相关兼容性指标被在线使用：重复的低兼容性事件被视为学生轨迹已漂移到统一密集OPD奖励不再可信赖的区域的证据。该方法不假设每个低重叠token是错误的；它仅假设累积兼容性失败是衰减后续后缀奖励的有用路径依赖信号。

单调可靠性衰减是一个故意保守的设计。后续token可能因为教师适应了已经漂移的前缀而变得局部兼容，而非因为完整的推理轨迹已返回到教师兼容状态。使用累积衰减避免了在重复漂移事件后重新放大后缀奖励，而基础权重 $w_{\text{base}}$ 将可靠性估计 $\tilde{R}_\tau$ 与优化尺度 $\mathcal{L}_\tau$ 分离。这使Prune-OPD能够降低不可靠后缀监督的权重，而不将方法变成脆弱的硬性过滤器。

### A.7 实现细节

Prune-OPD作为OPD奖励张量上的后处理步骤实现。基线奖励张量具有形状 $[B, T, k]$，其中 $B$ 是批次大小，$T$ 是响应长度，$k$ 是top-k token维度。对于每个有效响应位置，该方法计算标量损失权重 $\mathcal{L}_\tau$ 并将该位置的每个候选奖励乘以相同的标量：

$$\tilde{\mathbf{r}}_{b,\tau,j} = \mathcal{L}_{b,\tau} \cdot \mathbf{r}_{b,\tau,j}. \tag{13}$$

填充位置具有 $\tilde{R}_\tau = 0$ 和 $\mathcal{L}_\tau = 0$。优势估计器保持为 `token_reward_direct`，因此缩放后的奖励直接用作token级优势。

当前代码路径保持基线隔离。当 `prune_opd.enable=False` 时，原始OPD/GRPO行为不变。当启用时，主要实验使用 `overlap_ratio`；`teacher_top_p_accept` 作为动作级变体保留。教师top-k大小由 `actor_rollout_ref.rollout.log_prob_top_k` 控制，在当前实验中设置为16。

### A.8 提示模板

我们使用与OPD动力学研究[22]相同的DAPO-Math提示模板。对于每个数学问题，原始问题被插入到以下指令中。

**DAPO-Math提示模板：**
```
{问题} 请逐步推理，并将最终答案放在 \boxed{} 内。
```

### A.9 主要配置

表4总结了我们在Prune-OPD实验中使用的超参数。除非另有说明，论文中报告的所有基准性能均为Avg@16准确率：对于每个基准问题，评估使用固定评估协议下的16个采样响应，并报告这些样本的平均正确性。对于高兼容性对（DeepSeek-R1-Distill-Qwen-7B / Skywork-OR1-7B），初始动态长度设置为6144；对于其他对，初始动态长度设置为1024。

**表4：Prune-OPD默认超参数。**

| 项目 | 值 |
|------|-----|
| 训练数据集 | DAPO-Math-17K |
| 训练温度 | 1.0 |
| 教师温度 | 1.0 |
| 生成长度数量 | 4 |
| 小批次大小 | 64 |
| LogProb top-K | 16 |
| Top-K策略 | Student Top-K |
| 奖励权重模式 | Student probability |
| 最大提示长度 | 1,024 |
| 最大响应长度 | 8,192–12,288 |
| 验证最大响应长度 | 31,744 |
| 学习率 | $1 \times 10^{-6}$ |
| 训练步数 | 203 |
| KL系数 | 0.0 |
| 损失聚合 | token-mean |
| 重叠阈值 $\delta$ | 0.7 |
| Top-p阈值 $p$ | 0.95 |
| $w_{\text{drop}}$ | 0.01 |
| $w_{\text{base}}$ | 0.5 |
| 动态长度 初始/最小/最大 | 1024 / 1024 / 12288 |
| 动态长度步长 | 100 |
| 命中率阈值 | 0.1 |
| 长度边距 | 100 |
| 收缩耐心 | 3 |

### A.10 Top-p动作接受度指标

除重叠率外，我们还评估了一个更严格的动作级兼容性指标。在位置 $\tau$，学生采样实际的下一token $y_\tau$，而教师在同一前缀下返回top-k token和概率：

$$\mathcal{K} = \{v_{\tau,1}, \ldots, v_{\tau,k}\}, \quad q_{\tau,1} \geq q_{\tau,2} \geq \cdots \geq q_{\tau,k}. \tag{14}$$

给定top-p阈值 $p$，定义累积可见质量：

$$S_{\tau,m} = \sum_{j=1}^{m} q_{\tau,j}. \tag{15}$$

如果存在最小 $m$ 满足 $S_{\tau,m} \geq p$，则可见核为：

$$\mathcal{A}_\tau(p, k) = \{v_{\tau,1}, \ldots, v_{\tau,m}\}. \tag{16}$$

如果返回的top-k质量仍低于 $p$，真实核超出观察集，因此我们保守地使用 $\mathcal{A}_\tau(p, k) = \mathcal{K}$。坏事件为 $\mathcal{B}_\tau = \mathbb{1}[y_\tau \notin \mathcal{A}_\tau(p, k)]$。在报告的top-p运行中，$p = 0.95$。

### A.11 为什么Prune-OPD能为低重叠Qwen对提升准确率？

表1显示，两个Qwen3教师-学生对——Qwen3-1.7B-Base / Qwen3-4B（非思考）和Qwen3-4B-Base / Qwen3-4B（非思考）——是Prune-OPD同时提升训练效率和基准准确率的最清晰案例。我们将此解释为低重叠情境，其中可靠性感知剪枝不仅充当节省计算的机制，还充当梯度去噪机制。如图6所示，这些对表现出低学生-教师重叠，因此Prune-OPD将有效OPD长度保持为仅几百个token。相比之下，相应的OPD基线使用最多12,288个token的最大响应预算。

这种不匹配为未剪枝的OPD创造了一个合理的失败模式。只有早期前缀包含局部可利用的教师监督，而长后缀贡献了大量在漂移、低重叠前缀上产生的token级梯度。由于这些后缀token在数量上远超有用前缀token，即使每个单独的后缀奖励很弱或充满噪声，它们的梯度仍可能主导更新。Prune-OPD通过衰减和截断低可靠性后缀来消除这种不平衡，将优化集中在教师仍然可操作的短前缀上。这解释了为什么这两个Qwen3对可以在节省时间的同时获得准确率增益。对于其他教师-学生组合，兼容性更高或有用监督窗口已经足够长，使得剪枝主要减少计算量同时保持准确率，而非显著改变有效训练信号。

![图6]()

> **图6：低重叠Qwen3蒸馏对中的短有效OPD窗口。** 对于Qwen3-1.7B-Base / Qwen3-4B（非思考）和Qwen3-4B-Base / Qwen3-4B（非思考），低重叠导致Prune-OPD将OPD监督集中在几百个可靠token内，而OPD基线继续在最多12,288个token的响应上训练。

### A.12 附加诊断

附录诊断聚焦于驱动Prune-OPD的兼容性信号。图7报告了在未剪枝OPD下重叠率如何跨深度带演化，图8显示了重叠阈值如何控制后缀降权和动态响应预算。

**OPD下按深度的重叠。** 图7隔离了未剪枝OPD基线的兼容性动力学。浅层0–1K带作为早期前缀监督的参考，而后续带测试了教师监督在长学生生成推理前缀后是否保持局部对齐。关键观察是兼容性是深度依赖的：后续token位置带是重叠最脆弱、统一密集奖励最不可能提供有用的局部梯度的区域。这支持了Prune-OPD的核心前提：长程OPD的低效不仅在于后缀计算昂贵，还在于后缀奖励通常是在教师和学生已经漂移分离的前缀上产生的。

![图7]()

> **图7：DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B的OPD基线重叠率训练动力学。** 每个面板绘制不同token位置带（0–1K、2–3K、4–6K和7–8K）的重叠率随训练步的变化。此诊断显示了在未剪枝OPD下局部学生-教师兼容性如何在不同轨迹深度上演变。

**阈值敏感性。** 图8解释了可靠性阈值如何改变干预。更大的阈值更早宣布兼容性失败，使token级权重衰减更快出现并产生更短的动态OPD预算；更小的阈值更宽容，保留更长的后缀监督。主设置 $\delta = 0.7$ 介于这些极端之间：它移除大量低可靠性后缀计算，同时避免了硬性固定长度截断的过度刚性行为。因此，阈值主要控制Prune-OPD在可靠性-计算轴上的位置，而非改变底层的OPD目标。

![图8]()

> **图8：DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B的Prune-OPD阈值诊断。** 左：三个重叠阈值（$\delta = 0.6, 0.7, 0.8$）下Prune-OPD平均权重作为token位置的函数；对于每个阈值，曲线取训练步100、120、140、160、180和200。右：相同阈值下最大OPD响应长度随训练步的变化。通过结合，这些诊断显示了可靠性阈值如何改变后缀降权和动态生成长度预算。

### A.13 计算量报告

我们工作的主要计算成本来自使用OPD风格教师评分对学生模型进行后训练以及基准评估。对于每个教师-学生对，OPD、固定截断和Prune-OPD在相同的训练栈和匹配的硬件分配下运行，因此报告的墙钟时间比较隔离了可靠性感知剪枝的效果，而非基础设施变化的影响。因此，主要计算证据以表1、表2、表3和图2中经过的训练时间报告。

---

*（翻译完成。本文翻译严格遵循原文结构，保留所有数学公式、表格数据和图表引用。如有任何翻译不准确之处，欢迎指正。）*
