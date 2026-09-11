"""
MCDC (Multimodal Confidence-aware Dynamic Control) Distillation

核心模块：实现模态可信度感知的动态控制策略
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_modality_confidence(teacher_logits: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """
    计算教师的预测置信度（top-k 概率和）

    Args:
        teacher_logits: 教师模型的 logits [batch_size, seq_len, vocab_size]
        top_k: 计算 top-k 概率和

    Returns:
        confidence: 置信度分数 [batch_size, seq_len]
    """
    probs = F.softmax(teacher_logits, dim=-1)
    top_k_probs, _ = torch.topk(probs, top_k, dim=-1)
    confidence = top_k_probs.sum(dim=-1)
    return confidence


def compute_advantage_gap(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """
    计算教师相对于学生的优势差

    Args:
        teacher_logits: 教师模型的 logits [batch_size, seq_len, vocab_size]
        student_logits: 学生模型的 logits [batch_size, seq_len, vocab_size]

    Returns:
        advantage_gap: 优势差 [batch_size, seq_len]
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)

    # 计算每个 token 处教师相对于学生的概率差
    prob_diff = teacher_probs - student_probs

    # 取最大差值作为优势差
    advantage_gap = prob_diff.max(dim=-1).values

    return advantage_gap


def compute_dynamic_weights(
    vl_scores: torch.Tensor,
    text_scores: torch.Tensor,
    tau: float = 1.0,
    epsilon: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算动态权重（softmax + 门控）

    Args:
        vl_scores: VL 教师的可信度分数 [batch_size, seq_len]
        text_scores: Text 教师的可信度分数 [batch_size, seq_len]
        tau: 温度参数（控制权重分布的尖锐程度）
        epsilon: 最小权重阈值（门控机制）

    Returns:
        w_vl: VL 教师的权重 [batch_size, seq_len]
        w_text: Text 教师的权重 [batch_size, seq_len]
    """
    # Softmax
    w_vl = torch.exp(vl_scores / tau)
    w_text = torch.exp(text_scores / tau)
    total = w_vl + w_text
    w_vl = w_vl / total
    w_text = w_text / total

    # 门控机制：防止权重极端化
    w_vl = torch.clamp(w_vl, min=epsilon)
    w_text = torch.clamp(w_text, min=epsilon)

    # 归一化
    total = w_vl + w_text
    w_vl = w_vl / total
    w_text = w_text / total

    return w_vl, w_text


def compute_js_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    计算 JS 散度（Jensen-Shannon Divergence）

    Args:
        p_logits: 分布 P 的 logits [batch_size, seq_len, vocab_size]
        q_logits: 分布 Q 的 logits [batch_size, seq_len, vocab_size]
        reduction: 归约方式 ('batchmean', 'mean', 'sum', 'none')

    Returns:
        js_div: JS 散度标量
    """
    # KL(P || M) + KL(Q || M)，其中 M = (P + Q) / 2
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    m_probs = 0.5 * (p_probs + q_probs)

    # 避免 log(0)
    eps = 1e-8
    p_probs = torch.clamp(p_probs, min=eps)
    q_probs = torch.clamp(q_probs, min=eps)
    m_probs = torch.clamp(m_probs, min=eps)

    kl_p_m = F.kl_div(m_probs.log(), p_probs, reduction=reduction)
    kl_q_m = F.kl_div(m_probs.log(), q_probs, reduction=reduction)

    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div


def mcd_distillation_loss(
    student_logits: torch.Tensor,
    vl_teacher_logits: torch.Tensor,
    text_teacher_logits: torch.Tensor,
    config: Dict
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    MCDC 蒸馏损失

    Args:
        student_logits: 学生模型的 logits [batch_size, seq_len, vocab_size]
        vl_teacher_logits: VL 教师的 logits [batch_size, seq_len, vocab_size]
        text_teacher_logits: Text 教师的 logits [batch_size, seq_len, vocab_size]
        config: 配置字典，包含 alpha, beta, tau, epsilon

    Returns:
        loss: MCDC 蒸馏损失标量
        metrics: 指标字典，包含 w_vl, w_text
    """
    # 超参数
    alpha = config.get('alpha', 0.5)
    beta = config.get('beta', 0.5)
    tau = config.get('tau', 1.0)
    epsilon = config.get('epsilon', 0.1)

    # 计算置信度和优势差
    vl_conf = compute_modality_confidence(vl_teacher_logits)
    vl_adv = compute_advantage_gap(vl_teacher_logits, student_logits)
    vl_scores = alpha * vl_conf + beta * vl_adv

    text_conf = compute_modality_confidence(text_teacher_logits)
    text_adv = compute_advantage_gap(text_teacher_logits, student_logits)
    text_scores = alpha * text_conf + beta * text_adv

    # 计算动态权重
    w_vl, w_text = compute_dynamic_weights(vl_scores, text_scores, tau, epsilon)

    # 计算加权 JS 散度
    js_vl = compute_js_divergence(student_logits, vl_teacher_logits, reduction='none')
    js_text = compute_js_divergence(student_logits, text_teacher_logits, reduction='none')

    # 对 vocab 维度求和，保留 batch 和 seq 维度
    js_vl = js_vl.sum(dim=-1)  # [batch_size, seq_len]
    js_text = js_text.sum(dim=-1)  # [batch_size, seq_len]

    # 加权损失
    weighted_loss = w_vl * js_vl + w_text * js_text  # [batch_size, seq_len]

    # 对 batch 和 seq 维度求平均
    loss = weighted_loss.mean()

    # 指标
    metrics = {
        'w_vl': w_vl.mean(),
        'w_text': w_text.mean(),
        'vl_conf': vl_conf.mean(),
        'vl_adv': vl_adv.mean(),
        'text_conf': text_conf.mean(),
        'text_adv': text_adv.mean(),
    }

    return loss, metrics


class MCDCDistillation:
    """
    MCDC 蒸馏模块（封装类，方便在 dp_actor.py 中使用）
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典，包含 alpha, beta, tau, epsilon
        """
        self.config = config

    def __call__(
        self,
        student_logits: torch.Tensor,
        vl_teacher_logits: torch.Tensor,
        text_teacher_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 MCDC 蒸馏损失

        Args:
            student_logits: 学生模型的 logits
            vl_teacher_logits: VL 教师的 logits
            text_teacher_logits: Text 教师的 logits

        Returns:
            loss: MCDC 蒸馏损失
            metrics: 指标字典
        """
        return mcd_distillation_loss(
            student_logits=student_logits,
            vl_teacher_logits=vl_teacher_logits,
            text_teacher_logits=text_teacher_logits,
            config=self.config
        )
