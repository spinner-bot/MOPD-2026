"""
AMF (Adaptive Modal Fusion) Distillation

核心模块：融合两个模态教师的 logits（而非选择），基于模态置信度自适应。

与 MCDC 的关系：MCDC 加权两个独立 JS 损失（偏向一个）；AMF 融合两个
logits 得到一个分布（利用全部信息），基于贝叶斯融合理论。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_modality_confidence(teacher_logits: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """计算教师的预测置信度（top-k 概率和）[batch, seq]"""
    probs = F.softmax(teacher_logits, dim=-1)
    top_k_probs, _ = torch.topk(probs, top_k, dim=-1)
    return top_k_probs.sum(dim=-1)


def compute_fusion_weight(
    vl_confidence: torch.Tensor,
    text_confidence: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    计算融合权重 alpha（贝叶斯融合）

    alpha = sigmoid((conf_vl - conf_text) / tau)

    Args:
        vl_confidence: VL 教师置信度 [batch, seq]
        text_confidence: Text 教师置信度 [batch, seq]
        tau: 温度参数

    Returns:
        alpha: 融合权重 [batch, seq]，VL 教师权重
    """
    diff = vl_confidence - text_confidence
    return torch.sigmoid(diff / tau)


def fuse_logits(
    vl_logits: torch.Tensor,
    text_logits: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    融合两个教师的 logits（贝叶斯加权和）

    fused_logits = alpha * VL_logits + (1-alpha) * Text_logits

    Args:
        vl_logits: VL 教师 logits [batch, seq, vocab]
        text_logits: Text 教师 logits [batch, seq, vocab]
        alpha: 融合权重 [batch, seq, 1]

    Returns:
        fused_logits: 融合后的 logits
    """
    alpha = alpha.unsqueeze(-1)  # [batch, seq, 1]
    return alpha * vl_logits + (1.0 - alpha) * text_logits


def compute_js_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    reduction: str = 'none',
) -> torch.Tensor:
    """计算 JS 散度"""
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


def amf_distillation_loss(
    student_logits: torch.Tensor,
    vl_teacher_logits: torch.Tensor,
    text_teacher_logits: torch.Tensor,
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    AMF 蒸馏损失（自适应模态融合）

    L = JS(student || fused)
    其中 fused = alpha * VL + (1-alpha) * Text

    Args:
        student_logits: 学生 logits [batch, seq, vocab]
        vl_teacher_logits: VL 教师 logits
        text_teacher_logits: Text 教师 logits
        config: 配置，包含 tau

    Returns:
        loss: AMF 蒸馏损失
        metrics: 指标字典
    """
    tau = config.get('tau', 1.0)

    # 模态置信度
    vl_conf = compute_modality_confidence(vl_teacher_logits)
    text_conf = compute_modality_confidence(text_teacher_logits)

    # 融合权重
    alpha = compute_fusion_weight(vl_conf, text_conf, tau)

    # 融合 logits
    fused_logits = fuse_logits(vl_teacher_logits, text_teacher_logits, alpha)

    # 蒸馏损失（学生 vs 融合分布）
    js = compute_js_divergence(student_logits, fused_logits, reduction='none').sum(dim=-1)
    loss = js.mean()

    metrics = {
        'alpha_mean': alpha.mean(),
        'alpha_vl': alpha.mean(),
        'vl_conf': vl_conf.mean(),
        'text_conf': text_conf.mean(),
        'fused_loss': loss.detach(),
    }

    return loss, metrics


class AMFDistillation:
    """AMF 蒸馏模块（封装类）"""

    def __init__(self, config: Dict):
        self.config = config

    def __call__(
        self,
        student_logits: torch.Tensor,
        vl_teacher_logits: torch.Tensor,
        text_teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return amf_distillation_loss(
            student_logits=student_logits,
            vl_teacher_logits=vl_teacher_logits,
            text_teacher_logits=text_teacher_logits,
            config=self.config,
        )
