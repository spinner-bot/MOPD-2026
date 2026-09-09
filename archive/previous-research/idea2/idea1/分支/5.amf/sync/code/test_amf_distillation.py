"""
AMF 蒸馏模块单元测试

验证：融合权重计算、logits 融合、融合蒸馏损失。
用法：python test_amf_distillation.py（需服务器环境，含 torch）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from amf_distillation import (
    AMFDistillation,
    amf_distillation_loss,
    compute_fusion_weight,
    fuse_logits,
)


def make_logits(batch=2, seq=8, vocab=100, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, vocab)


def test_compute_fusion_weight():
    # VL 置信度高 → alpha 接近 1
    vl_conf = torch.tensor([0.9, 0.8])
    text_conf = torch.tensor([0.1, 0.2])
    alpha = compute_fusion_weight(vl_conf, text_conf, tau=1.0)
    assert (alpha > 0.5).all(), f"VL 置信度高时 alpha 应>0.5, 实际 {alpha}"
    # alpha 在 [0,1]
    assert (alpha >= 0).all() and (alpha <= 1).all()
    print(f"[OK] compute_fusion_weight, alpha={alpha.tolist()}")


def test_fuse_logits():
    vl_logits = make_logits(seed=1)
    text_logits = make_logits(seed=2)
    alpha = torch.full((2, 8), 0.5)
    fused = fuse_logits(vl_logits, text_logits, alpha)
    assert fused.shape == vl_logits.shape, "融合 logits 形状错误"
    # alpha=0.5 时，fused 应为两者平均
    expected = 0.5 * vl_logits + 0.5 * text_logits
    assert torch.allclose(fused, expected, atol=1e-5), "融合计算错误"
    print("[OK] fuse_logits")


def test_amf_loss():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"tau": 1.0}
    loss, metrics = amf_distillation_loss(student, vl_teacher, text_teacher, config)
    assert loss.item() > 0, f"损失应>0, 实际 {loss.item()}"
    assert "alpha_mean" in metrics and "fused_loss" in metrics
    assert 0 <= metrics["alpha_mean"].item() <= 1, "融合权重应在[0,1]"
    print(f"[OK] amf_distillation_loss, loss={loss.item():.4f}, alpha={metrics['alpha_mean'].item():.3f}")


def test_amf_class():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"tau": 1.0}
    module = AMFDistillation(config)
    loss, metrics = module(student, vl_teacher, text_teacher)
    assert loss.item() > 0
    print(f"[OK] AMFDistillation class, loss={loss.item():.4f}")


if __name__ == "__main__":
    print("=== AMF 单元测试 ===")
    test_compute_fusion_weight()
    test_fuse_logits()
    test_amf_loss()
    test_amf_class()
    print("\n=== 全部测试通过 ===")
