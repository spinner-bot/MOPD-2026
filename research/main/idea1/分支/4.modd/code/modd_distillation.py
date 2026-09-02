"""
MODD (Modal Decoupled Distillation) Distillation

核心模块：分阶段解耦训练，分别学习视觉和语言能力，避免"只学语言捷径"。

核心思想（课程学习）：
- Stage 1 视觉学习：用 VL 教师视觉位置输出，让学生学会"看"
- Stage 2 语言学习：用 Text 教师文本输出，让学生学会"想"
- Stage 3 整合：两个教师同时指导，融合视觉+语言能力
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# 阶段定义
STAGE_VISION = "vision"
STAGE_LANGUAGE = "language"
STAGE_INTEGRATE = "integrate"


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


def modd_stage_vision_loss(
    student_logits: torch.Tensor,
    vl_teacher_logits: torch.Tensor,
    vision_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Stage 1: 视觉能力学习
    只对视觉位置（vision_mask=1）用 VL 教师监督

    Args:
        student_logits: 学生 logits [batch, seq, vocab]
        vl_teacher_logits: VL 教师 logits
        vision_mask: 视觉位置掩码 [batch, seq]，1=视觉位置

    Returns:
        loss: 视觉阶段蒸馏损失
    """
    js = compute_js_divergence(student_logits, vl_teacher_logits, reduction='none').sum(dim=-1)
    # 只对视觉位置计算
    masked_js = js * vision_mask.float()
    loss = masked_js.sum() / (vision_mask.sum() + 1e-8)
    return loss


def modd_stage_language_loss(
    student_logits: torch.Tensor,
    text_teacher_logits: torch.Tensor,
    language_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Stage 2: 语言能力学习
    只对文本位置（language_mask=1）用 Text 教师监督

    Args:
        student_logits: 学生 logits [batch, seq, vocab]
        text_teacher_logits: Text 教师 logits
        language_mask: 文本位置掩码 [batch, seq]，1=文本位置

    Returns:
        loss: 语言阶段蒸馏损失
    """
    js = compute_js_divergence(student_logits, text_teacher_logits, reduction='none').sum(dim=-1)
    masked_js = js * language_mask.float()
    loss = masked_js.sum() / (language_mask.sum() + 1e-8)
    return loss


def modd_stage_integrate_loss(
    student_logits: torch.Tensor,
    vl_teacher_logits: torch.Tensor,
    text_teacher_logits: torch.Tensor,
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Stage 3: 整合阶段
    两个教师同时指导所有位置

    Args:
        student_logits: 学生 logits
        vl_teacher_logits: VL 教师 logits
        text_teacher_logits: Text 教师 logits
        config: 配置，包含整合权重

    Returns:
        loss: 整合阶段蒸馏损失
        metrics: 指标
    """
    w_vl = config.get('integrate_w_vl', 0.5)
    w_text = 1.0 - w_vl

    js_vl = compute_js_divergence(student_logits, vl_teacher_logits, reduction='none').sum(dim=-1)
    js_text = compute_js_divergence(student_logits, text_teacher_logits, reduction='none').sum(dim=-1)

    loss = (w_vl * js_vl + w_text * js_text).mean()

    metrics = {
        'w_vl': torch.tensor(w_vl),
        'w_text': torch.tensor(w_text),
        'integrate_loss': loss.detach(),
    }
    return loss, metrics


class MODDDistillation:
    """
    MODD 蒸馏模块（封装类）
    支持分阶段训练：vision -> language -> integrate
    """

    def __init__(self, config: Dict, current_stage: str = STAGE_INTEGRATE):
        self.config = config
        self.current_stage = current_stage

    def set_stage(self, stage: str):
        """设置当前训练阶段"""
        assert stage in (STAGE_VISION, STAGE_LANGUAGE, STAGE_INTEGRATE)
        self.current_stage = stage

    def __call__(
        self,
        student_logits: torch.Tensor,
        vl_teacher_logits: torch.Tensor,
        text_teacher_logits: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        根据当前阶段计算蒸馏损失
        """
        if self.current_stage == STAGE_VISION:
            assert vision_mask is not None, "vision stage requires vision_mask"
            loss = modd_stage_vision_loss(student_logits, vl_teacher_logits, vision_mask)
            metrics = {'stage': STAGE_VISION, 'loss': loss.detach()}
        elif self.current_stage == STAGE_LANGUAGE:
            assert language_mask is not None, "language stage requires language_mask"
            loss = modd_stage_language_loss(student_logits, text_teacher_logits, language_mask)
            metrics = {'stage': STAGE_LANGUAGE, 'loss': loss.detach()}
        else:  # integrate
            loss, metrics = modd_stage_integrate_loss(
                student_logits, vl_teacher_logits, text_teacher_logits, self.config
            )
            metrics['stage'] = STAGE_INTEGRATE
        return loss, metrics
