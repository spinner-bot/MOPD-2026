"""
MODD 蒸馏模块单元测试

验证：分阶段（视觉/语言/整合）蒸馏损失，掩码过滤。
用法：python test_modd_distillation.py（需服务器环境，含 torch）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from modd_distillation import (
    MODDDistillation,
    STAGE_INTEGRATE,
    STAGE_LANGUAGE,
    STAGE_VISION,
    modd_stage_integrate_loss,
    modd_stage_language_loss,
    modd_stage_vision_loss,
)


def make_logits(batch=2, seq=8, vocab=100, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, vocab)


def test_vision_stage():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    vision_mask = torch.zeros(2, 8)
    vision_mask[:, 0:4] = 1  # 前4位置是视觉
    loss = modd_stage_vision_loss(student, vl_teacher, vision_mask)
    assert loss.item() > 0, f"视觉损失应>0, 实际 {loss.item()}"
    print(f"[OK] vision stage, loss={loss.item():.4f}")


def test_language_stage():
    student = make_logits()
    text_teacher = make_logits(seed=2)
    language_mask = torch.zeros(2, 8)
    language_mask[:, 4:8] = 1  # 后4位置是语言
    loss = modd_stage_language_loss(student, text_teacher, language_mask)
    assert loss.item() > 0
    print(f"[OK] language stage, loss={loss.item():.4f}")


def test_integrate_stage():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"integrate_w_vl": 0.5}
    loss, metrics = modd_stage_integrate_loss(student, vl_teacher, text_teacher, config)
    assert loss.item() > 0
    assert "w_vl" in metrics and "w_text" in metrics
    print(f"[OK] integrate stage, loss={loss.item():.4f}")


def test_modd_class():
    """验证分阶段切换"""
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    vision_mask = torch.zeros(2, 8)
    vision_mask[:, 0:4] = 1
    language_mask = torch.zeros(2, 8)
    language_mask[:, 4:8] = 1
    config = {"integrate_w_vl": 0.5}

    # 视觉阶段
    module = MODDDistillation(config, current_stage=STAGE_VISION)
    loss, metrics = module(student, vl_teacher, text_teacher, vision_mask=vision_mask)
    assert metrics["stage"] == STAGE_VISION
    # 语言阶段
    module.set_stage(STAGE_LANGUAGE)
    loss, metrics = module(student, vl_teacher, text_teacher, language_mask=language_mask)
    assert metrics["stage"] == STAGE_LANGUAGE
    # 整合阶段
    module.set_stage(STAGE_INTEGRATE)
    loss, metrics = module(student, vl_teacher, text_teacher)
    assert metrics["stage"] == STAGE_INTEGRATE
    print("[OK] MODDDistillation class stage switching")


if __name__ == "__main__":
    print("=== MODD 单元测试 ===")
    test_vision_stage()
    test_language_stage()
    test_integrate_stage()
    test_modd_class()
    print("\n=== 全部测试通过 ===")
