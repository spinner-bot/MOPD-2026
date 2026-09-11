# 对齐度 Metric 选型

> 创建时间：2026-09-02
> 用途：详细记录 idea2 中对齐度量化指标的选型依据与实现要点

---

## 一、目标

在多模态 OPD 训练过程中，追踪每个 token 上 VL 教师与 Text 教师输出分布的"对齐度"（即相似程度），分析其随训练步数的演化。

---

## 二、候选 Metric

### 2.1 JS 散度（Jensen-Shannon Divergence）—— 主 Metric

**公式**：
$$
D_{JS}(p \parallel q) = \frac{1}{2} D_{KL}(p \parallel m) + \frac{1}{2} D_{KL}(q \parallel m), \quad m = \frac{p + q}{2}
$$

**优点**：
- **对称**：$D_{JS}(p \parallel q) = D_{JS}(q \parallel p)$，不依赖教师顺序
- **Bounded**：$D_{JS} \in [0, \log 2]$，便于跨 token / 跨 step 比较
- **数值稳定**：不会像 KL 散度那样在 $q \to 0$ 时爆炸
- **标准**：文献中广泛使用（Vision-OPD 也用 JSD 作为蒸馏目标）

**缺点**：
- 计算需要完整的概率分布，对大词表有开销
- 对尾部概率敏感（但比 KL 好）

**实现要点**：
```python
import torch
import torch.nn.functional as F

def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    计算两个 logit 分布的 JS 散度
    Args:
        p_logits: [batch, seq, vocab]
        q_logits: [batch, seq, vocab]
    Returns:
        js_div: [batch, seq] 每个 token 的 JS 散度
    """
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    
    eps = 1e-8
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    m = torch.clamp(m, min=eps)
    
    kl_p_m = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
    kl_q_m = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)
    
    js = 0.5 * (kl_p_m + kl_q_m)
    return js
```

**使用场景**：
- 主分析：整体对齐度 vs 训练步数曲线
- 跨任务类型比较
- 相变点检测

---

### 2.2 Top-K Jaccard —— 辅助 Metric 1

**公式**：
$$
J_K(p, q) = \frac{|\text{TopK}(p) \cap \text{TopK}(q)|}{|\text{TopK}(p) \cup \text{TopK}(q)|}
$$

其中 $\text{TopK}(p)$ 是分布 $p$ 下概率最高的 $K$ 个 token 的集合。

**优点**：
- **直观**：两个教师"意见一致"的 token 占比
- **Robust**：不受概率分布尾部影响
- **可解释**：Jaccard = 1 表示两教师 top-k 完全一致

**缺点**：
- 丢失概率信息（只看集合，不看概率大小）
- $K$ 的选择影响结果（建议 $K = 10$ 或 $K = 100$）

**实现要点**：
```python
def topk_jaccard(p_logits: torch.Tensor, q_logits: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    计算两个 logit 分布的 top-k Jaccard 相似度
    Args:
        p_logits: [batch, seq, vocab]
        q_logits: [batch, seq, vocab]
        k: top-k 大小
    Returns:
        jaccard: [batch, seq] 每个 token 的 Jaccard 相似度
    """
    _, p_topk = torch.topk(p_logits, k, dim=-1)
    _, q_topk = torch.topk(q_logits, k, dim=-1)
    
    # 对每个 token，计算 top-k 集合的交集大小
    # 由于 topk 返回的是 index，可以用 broadcasting 比较
    # 这里简化实现：假设 batch 和 seq 维度已经 flatten
    p_set = p_topk.unsqueeze(-1)  # [N, k, 1]
    q_set = q_topk.unsqueeze(-2)  # [N, 1, k]
    intersection = (p_set == q_set).sum(dim=(-1, -2))  # [N]
    
    # Jaccard = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    union = 2 * k - intersection
    jaccard = intersection.float() / union.float()
    
    return jaccard
```

**使用场景**：
- 辅助分析：验证 JS 散度的趋势是否一致
- 可视化：画"两教师 top-k 一致率"曲线

---

### 2.3 Top-1 一致率 —— 辅助 Metric 2

**公式**：
$$
\text{Top1Agree}(p, q) = \mathbb{1}[\arg\max p = \arg\max q]
$$

**优点**：
- **最简单**：两教师最高概率 token 是否相同
- **易解释**：一致率 = 1 表示两教师"完全同意"

**缺点**：
- 太 coarse，丢失大量信息
- 对噪声敏感（概率接近时容易跳变）

**实现要点**：
```python
def top1_agreement(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    计算两个 logit 分布的 top-1 一致率
    Args:
        p_logits: [batch, seq, vocab]
        q_logits: [batch, seq, vocab]
    Returns:
        agree: [batch, seq] 每个 token 是否一致（0 或 1）
    """
    p_argmax = torch.argmax(p_logits, dim=-1)
    q_argmax = torch.argmax(q_logits, dim=-1)
    return (p_argmax == q_argmax).float()
```

**使用场景**：
- 快速 sanity check
- 与 JS 散度 / Top-K Jaccard 对比

---

### 2.4 Attention 相似度 —— Supplementary

**公式**：
$$
\text{AttnSim}(A_p, A_q) = \cos(\text{flatten}(A_p), \text{flatten}(A_q))
$$

其中 $A_p, A_q$ 是两个教师的 attention map。

**优点**：
- **机制层面洞察**：看两教师是否"看同一个地方"
- 与分布层面的 metric 互补

**缺点**：
- 需要从教师模型提取 attention map（可能增加显存开销）
- Attention map 维度高（num_heads × seq_len × seq_len），需要降维或聚合
- 不是所有模型都返回 attention（需要 `output_attentions=True`）

**实现要点**：
```python
def attention_similarity(p_attn: torch.Tensor, q_attn: torch.Tensor) -> torch.Tensor:
    """
    计算两个 attention map 的余弦相似度
    Args:
        p_attn: [batch, num_heads, seq, seq]
        q_attn: [batch, num_heads, seq, seq]
    Returns:
        sim: [batch, num_heads] 每个 head 的相似度
    """
    # Flatten last two dims
    p_flat = p_attn.flatten(start_dim=-2)  # [batch, heads, seq*seq]
    q_flat = q_attn.flatten(start_dim=-2)
    
    # Cosine similarity
    sim = F.cosine_similarity(p_flat, q_flat, dim=-1)
    return sim
```

**使用场景**：
- Supplementary analysis：放在附录或 supplement
- 机制解释：如果分布层面的对齐度上升，attention 层面是否也上升？

---

## 三、Metric 选择策略

**主 metric**：JS 散度
- 理由：对称、bounded、数值稳定、文献标准

**辅助 metric**：Top-K Jaccard + Top-1 一致率
- 理由：直观、robust、与 JS 散度互相印证

**Supplementary metric**：Attention 相似度
- 理由：机制层面洞察，但不作为主 story

**建议**：
- 同时计算所有 metric，但论文中主要展示 JS 散度
- 如果 JS 散度和 Top-K Jaccard 趋势一致 → story 更强
- 如果不一致 → 分析为什么（可能是 metric 的 sensitivity 不同）

---

## 四、超参数选择

### 4.1 Top-K 的 K 值

- **Top-K Jaccard**：建议 $K = 10$ 或 $K = 100$
  - $K = 10$：看"最可能的 token"是否一致
  - $K = 100$：看"概率较高的 token 集合"是否一致（Vision-OPD 用 $K = 100$ 做 top-k distillation）
  - **建议**：两个都试，看哪个更有区分度

### 4.2 JS 散度的温度

- 标准 JS 散度不需要温度
- 但如果想控制对尾部概率的敏感度，可以用 temperature scaling：
  $$
  p_T = \text{softmax}(\text{logits} / T)
  $$
  然后计算 $D_{JS}(p_T \parallel q_T)$
- **建议**：先用 $T = 1$（标准），如果需要再调

---

## 五、计算开销估算

假设 probe batch 大小 = 100 样本，平均 seq_len = 200，vocab_size = 150k（Qwen3.5）：

- **JS 散度**：需要对完整 vocab 维度做 softmax + KL，开销 $\sim O(\text{vocab\_size})$
- **Top-K Jaccard**：只需要 top-k，开销 $\sim O(k \log \text{vocab\_size})$
- **Top-1 一致率**：只需要 argmax，开销 $\sim O(\text{vocab\_size})$
- **Attention 相似度**：需要提取 attention map，开销较大

**建议**：
- JS / Top-1 / Top-K 都可以高效计算，不用担心
- Attention 相似度只在 supplementary 分析时用，且只在少量 step 上计算

---

## 六、下一步

1. 在 verl 框架里实现上述 metric（参考 `mcd_distillation.py` 中的 `compute_js_divergence`）
2. 在 dry run 中验证 metric 的正确性
3. 根据 dry run 结果调整超参数（如 Top-K 的 K 值）
