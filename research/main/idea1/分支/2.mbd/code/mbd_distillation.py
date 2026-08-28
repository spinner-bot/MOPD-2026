"""
MBD (Modal Balanced Distillation) Distillation

核心模块：强制两个模态教师平衡贡献，解决"模态坍缩"问题。

与 MCDC 的关系：MCDC 动态加权但允许一个教师主导；MBD 通过平衡损失
主动拉近两个教师权重，防止模态坍缩。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_modality_confidence(teacher_logits: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """计算教师的预测置信度（top-k 概率和）[batch, seq]"""
    probs = F.softmax(teacher_logits, dim=-1)
    top_k_probs, _ = torch.topk(probs, top_k, dim=-1)
    return top_k_probs.sum(dim=-1)


def compute_advantage_gap(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """计算教师相对于学生的优势差 [batch, seq]"""
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    return (teacher_probs - student_probs).max(dim=-1).values


def compute_dynamic_weights(
    vl_scores: torch.Tensor,
    text_scores: torch.Tensor,
    tau: float = 1.0,
    epsilon: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算动态权重（softmax + 门控）"""
    w_vl = torch.exp(vl_scores / tau)
    w_text = torch.exp(text_scores / tau)
    total = w_vl + w_text
    w_vl = w_vl / total
    w_text = w_text / total
    # 门控
    w_vl = torch.clamp(w_vl, min=epsilon)
    w_text = torch.clamp(w_text, min=epsilon)
    total = w_vl + w_text
    w_vl = w_vl / total
    w_text = w_text / total
    return w_vl, w_text


def compute_js_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    reduction: str = 'none',
) -> torch.Tensor:
    """计算 JS 散度（逐元素）"""
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    m_probs = 0.5 * (p_probs + q_probs)
    eps = 1e-8
    p_probs = torch.clamp(p_probs, min=eps)
    q_probs = torch.clamp(q_probs, min=eps)
    m_probs = torch.clamp(m_probs, min=eps)
    kl_p_m = F.kl_div(m_probs.log(), p_probs, reduction=reduction)
    kl_q_m = F.kl_div(m_probs.log(), q_probs, reduction=reduction)
    return 0.5 * (kl_p_m + kl_q_m)


def compute_modal_balance_penalty(
    w_vl: torch.Tensor,
    w_text: torch.Tensor,
) -> torch.Tensor:
    """
    计算模态平衡惩罚项：|w_vl - w_text|

    Args:
        w_vl: VL 教师权重 [batch, seq]
        w_text: Text 教师权重 [batch, seq]

    Returns:
        penalty: 平衡惩罚 [batch, seq]
    """
    return torch.abs(w_vl - w_text)


def mbd_distillation_loss(
    student_logits: torch.Tensor,
    vl_teacher_logits: torch.Tensor,
    text_teacher_logits: torch.Tensor,
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    MBD 蒸馏损失（模态平衡蒸馏）

    L_total = L_distill + λ * L_balance
    L_distill = w_vl * JS(student||VL) + w_text * JS(student||Text)
    L_balance = |w_vl - w_text|

    Args:
        student_logits: 学生 logits [batch, seq, vocab]
        vl_teacher_logits: VL 教师 logits
        text_teacher_logits: Text 教师 logits
        config: 包含 alpha, beta, tau, epsilon, balance_lambda

    Returns:
        loss: MBD 蒸馏损失
        metrics: 指标字典
    """
    alpha = config.get('alpha', 0.5)
    beta = config.get('beta', 0.5)
    tau = config.get('tau', 1.0)
    epsilon = config.get('epsilon', 0.1)
    balance_lambda = config.get('balance_lambda', 1.0)

    # 模态可信度
    vl_conf = compute_modality_confidence(vl_teacher_logits)
    vl_adv = compute_advantage_gap(vl_teacher_logits, student_logits)
    vl_scores = alpha * vl_conf + beta * vl_adv

    text_conf = compute_modality_confidence(text_teacher_logits)
    text_adv = compute_advantage_gap(text_teacher_logits, student_logits)
    text_scores = alpha * text_conf + beta * text_adv

    # 动态权重
    w_vl, w_text = compute_dynamic_weights(vl_scores, text_scores, tau, epsilon)

    # 蒸馏损失（加权 JS）
    js_vl = compute_js_divergence(student_logits, vl_teacher_logits, reduction='none').sum(dim=-1)
    js_text = compute_js_divergence(student_logits, text_teacher_logits, reduction='none').sum(dim=-1)
    distill_loss = w_vl * js_vl + w_text * js_text

    # 模态平衡惩罚
    balance_penalty = compute_modal_balance_penalty(w_vl, w_text)

    # 总损失
    loss = distill_loss.mean() + balance_lambda * balance_penalty.mean()

    metrics = {
        'w_vl': w_vl.mean(),
        'w_text': w_text.mean(),
        'balance_penalty': balance_penalty.mean(),
        'distill_loss': distill_loss.mean(),
        'vl_conf': vl_conf.mean(),
        'vl_adv': vl_adv.mean(),
        'text_conf': text_conf.mean(),
        'text_adv': text_adv.mean(),
    }

    return loss, metrics


class MBDDistillation:
    """MBD 蒸馏模块（封装类）"""

    def __init__(self, config: Dict):
        self.config = config

    def __call__(
        self,
        student_logits: torch.Tensor,
        vl_teacher_logits: torch.Tensor,
        text_teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return mbd_distillation_loss(
            student_logits=student_logits,
            vl_teacher_logits=vl_teacher_logits,
            text_teacher_logits=text_teacher_logits,
            config=self.config,
        )
