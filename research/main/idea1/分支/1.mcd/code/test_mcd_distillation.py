"""
MCDC 蒸馏模块单元测试

验证：可信度计算、动态权重、蒸馏损失能正确运行，形状正确。
用法：python test_mcd_distillation.py
"""

import os
import sys

import torch

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl", "workers", "actor"))
sys.path.insert(0, os.path.dirname(__file__))

from mcd_distillation import (
    MCDCDistillation,
    compute_advantage_gap,
    compute_dynamic_weights,
    compute_modality_confidence,
    mcd_distillation_loss,
)


def make_logits(batch=2, seq=8, vocab=100, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, vocab)


def test_compute_modality_confidence():
    logits = make_logits()
    conf = compute_modality_confidence(logits)
    assert conf.shape == (2, 8), f"置信度形状错误: {conf.shape}"
    assert (conf >= 0).all() and (conf <= 1).all(), "置信度应在 [0,1]"
    print("[OK] compute_modality_confidence")


def test_compute_advantage_gap():
    student = make_logits(seed=1)
    teacher = make_logits(seed=2)
    adv = compute_advantage_gap(teacher, student)
    assert adv.shape == (2, 8), f"优势差形状错误: {adv.shape}"
    print("[OK] compute_advantage_gap")


def test_compute_dynamic_weights():
    vl_scores = torch.randn(2, 8)
    text_scores = torch.randn(2, 8)
    w_vl, w_text = compute_dynamic_weights(vl_scores, text_scores, tau=1.0, epsilon=0.1)
    # 权重和为 1
    assert torch.allclose(w_vl + w_text, torch.ones_like(w_vl), atol=1e-5), "权重和应=1"
    # 门控保证最小值
    assert (w_vl >= 0.09).all(), f"w_vl 最小值应≥0.09, 实际 {w_vl.min()}"
    assert (w_text >= 0.09).all(), f"w_text 最小值应≥0.09, 实际 {w_text.min()}"
    print("[OK] compute_dynamic_weights")


def test_mcd_loss():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"alpha": 0.5, "beta": 0.5, "tau": 1.0, "epsilon": 0.1}
    loss, metrics = mcd_distillation_loss(student, vl_teacher, text_teacher, config)
    assert loss.item() > 0, f"损失应>0, 实际 {loss.item()}"
    assert "w_vl" in metrics and "w_text" in metrics, "指标缺少权重"
    print(f"[OK] mcd_distillation_loss, loss={loss.item():.4f}")


def test_mcd_class():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"alpha": 0.5, "beta": 0.5, "tau": 1.0, "epsilon": 0.1}
    module = MCDCDistillation(config)
    loss, metrics = module(student, vl_teacher, text_teacher)
    assert loss.item() > 0
    print(f"[OK] MCDCDistillation class, loss={loss.item():.4f}")


if __name__ == "__main__":
    print("=== MCDC 单元测试 ===")
    test_compute_modality_confidence()
    test_compute_advantage_gap()
    test_compute_dynamic_weights()
    test_mcd_loss()
    test_mcd_class()
    print("\n=== 全部测试通过 ===")
