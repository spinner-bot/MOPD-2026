"""
TAMR (Task-Aware Modal Routing) Distillation

核心模块：根据任务类型实现任务级模态路由，避免 token 级路由的高开销。

核心思想：
- 不同任务（视觉推理/语言推理/混合）对模态的依赖不同
- 用轻量任务分类器预测任务类型
- 根据任务类型设置教师权重（任务级，非 token 级）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# 任务类型定义
VISUAL_REASONING = "visual_reasoning"
LANGUAGE_REASONING = "language_reasoning"
MIXED = "mixed"


class LightTaskClassifier(nn.Module):
    """
    轻量任务分类器：基于 prompt 嵌入预测任务类型
    用一个小型 MLP，计算开销 <1%
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """输入 prompt 嵌入 [batch, input_dim]，输出任务类型 logits [batch, 3]"""
        x = F.gelu(self.fc1(prompt_embeddings))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def task_routing_weights(
    task_type: str,
    vl_weight: float = 0.7,
    text_weight: float = 0.3,
    mixed_weight: float = 0.5,
) -> Tuple[float, float]:
    """
    根据任务类型返回 (w_vl, w_text) 权重

    Args:
        task_type: 任务类型 (visual_reasoning / language_reasoning / mixed)
        vl_weight: 视觉任务时 VL 教师权重
        text_weight: 语言任务时 Text 教师权重
        mixed_weight: 混合任务时两个教师权重

    Returns:
        (w_vl, w_text): 教师权重元组
    """
    if task_type == VISUAL_REASONING:
        return vl_weight, 1.0 - vl_weight
    elif task_type == LANGUAGE_REASONING:
        return 1.0 - text_weight, text_weight
    elif task_type == MIXED:
        return mixed_weight, 1.0 - mixed_weight
    else:
        # 默认均衡
        return 0.5, 0.5


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


def tamr_distillation_loss(
    student_logits: torch.Tensor,
    vl_teacher_logits: torch.Tensor,
    text_teacher_logits: torch.Tensor,
    task_type: str,
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    TAMR 蒸馏损失（任务级路由）

    L = w_vl * JS(student||VL) + w_text * JS(student||Text)

    Args:
        student_logits: 学生 logits [batch, seq, vocab]
        vl_teacher_logits: VL 教师 logits
        text_teacher_logits: Text 教师 logits
        task_type: 任务类型
        config: 配置，包含路由权重参数

    Returns:
        loss: TAMR 蒸馏损失
        metrics: 指标字典
    """
    # 任务级权重
    w_vl, w_text = task_routing_weights(
        task_type=task_type,
        vl_weight=config.get('vl_weight', 0.7),
        text_weight=config.get('text_weight', 0.3),
        mixed_weight=config.get('mixed_weight', 0.5),
    )
    # 转成张量
    w_vl_t = torch.full_like(student_logits[..., 0], w_vl)
    w_text_t = torch.full_like(student_logits[..., 0], w_text)

    # 蒸馏损失
    js_vl = compute_js_divergence(student_logits, vl_teacher_logits, reduction='none').sum(dim=-1)
    js_text = compute_js_divergence(student_logits, text_teacher_logits, reduction='none').sum(dim=-1)
    loss = (w_vl_t * js_vl + w_text_t * js_text).mean()

    metrics = {
        'w_vl': torch.tensor(w_vl),
        'w_text': torch.tensor(w_text),
        'task_type': task_type,
        'distill_loss': loss.detach(),
    }

    return loss, metrics


class TAMRDistillation:
    """TAMR 蒸馏模块（封装类）"""

    def __init__(self, config: Dict, task_classifier: Optional[nn.Module] = None):
        self.config = config
        self.task_classifier = task_classifier

    def predict_task_type(
        self,
        prompt_embeddings: torch.Tensor,
    ) -> List[str]:
        """用任务分类器预测任务类型"""
        assert self.task_classifier is not None, "task_classifier not set"
        logits = self.task_classifier(prompt_embeddings)
        preds = logits.argmax(dim=-1).tolist()
        label_map = [VISUAL_REASONING, LANGUAGE_REASONING, MIXED]
        return [label_map[p] for p in preds]

    def __call__(
        self,
        student_logits: torch.Tensor,
        vl_teacher_logits: torch.Tensor,
        text_teacher_logits: torch.Tensor,
        task_type: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return tamr_distillation_loss(
            student_logits=student_logits,
            vl_teacher_logits=vl_teacher_logits,
            text_teacher_logits=text_teacher_logits,
            task_type=task_type,
            config=self.config,
        )
