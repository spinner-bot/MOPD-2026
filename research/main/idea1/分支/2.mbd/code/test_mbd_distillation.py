"""
MBD 蒸馏模块单元测试

验证：模态平衡惩罚、平衡蒸馏损失能正确运行，权重被拉近。
用法：python test_mbd_distillation.py（需服务器环境，含 torch）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mbd_distillation import (
    MBDDistillation,
    compute_modal_balance_penalty,
    mbd_distillation_loss,
)


def make_logits(batch=2, seq=8, vocab=100, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, vocab)


def test_balance_penalty():
    w_vl = torch.tensor([0.8, 0.2])
    w_text = torch.tensor([0.2, 0.8])
    penalty = compute_modal_balance_penalty(w_vl, w_text)
    assert torch.allclose(penalty, torch.tensor([0.6, 0.6]), atol=1e-5), "平衡惩罚计算错误"
    # 平衡时惩罚=0
    w_vl2 = torch.tensor([0.5, 0.5])
    w_text2 = torch.tensor([0.5, 0.5])
    penalty2 = compute_modal_balance_penalty(w_vl2, w_text2)
    assert torch.allclose(penalty2, torch.zeros_like(penalty2), atol=1e-5), "平衡时惩罚应=0"
    print("[OK] compute_modal_balance_penalty")


def test_mbd_loss():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"alpha": 0.5, "beta": 0.5, "tau": 1.0, "epsilon": 0.1, "balance_lambda": 1.0}
    loss, metrics = mbd_distillation_loss(student, vl_teacher, text_teacher, config)
    assert loss.item() > 0, f"损失应>0, 实际 {loss.item()}"
    assert "balance_penalty" in metrics, "指标缺少平衡惩罚"
    assert "w_vl" in metrics and "w_text" in metrics
    print(f"[OK] mbd_distillation_loss, loss={loss.item():.4f}, penalty={metrics['balance_penalty'].item():.4f}")


def test_balance_effect():
    """验证：加平衡项后，不平衡权重被拉近"""
    # 模拟极端情况：VL 置信度远高于 Text
    batch, seq, vocab = 2, 8, 100
    torch.manual_seed(0)
    student = torch.randn(batch, seq, vocab)
    # VL 教师置信度高（top1 概率接近 1）
    vl_teacher = torch.zeros(batch, seq, vocab)
    vl_teacher[..., 0] = 5.0  # 强置信
    # Text 教师置信度低（均匀）
    text_teacher = torch.randn(batch, seq, vocab) * 0.1

    config = {"alpha": 1.0, "beta": 0.0, "tau": 1.0, "epsilon": 0.1, "balance_lambda": 5.0}
    loss, metrics = mbd_distillation_loss(student, vl_teacher, text_teacher, config)
    w_vl, w_text = metrics["w_vl"].item(), metrics["w_text"].item()
    # 有平衡项，w_text 不应趋近 0
    assert w_text > 0.1, f"有平衡项时 w_text 应>0.1, 实际 {w_text}"
    print(f"[OK] 平衡效果: w_vl={w_vl:.3f}, w_text={w_text:.3f}")


def test_mbd_class():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"alpha": 0.5, "beta": 0.5, "tau": 1.0, "epsilon": 0.1, "balance_lambda": 1.0}
    module = MBDDistillation(config)
    loss, metrics = module(student, vl_teacher, text_teacher)
    assert loss.item() > 0
    print(f"[OK] MBDDistillation class, loss={loss.item():.4f}")


if __name__ == "__main__":
    print("=== MBD 单元测试 ===")
    test_balance_penalty()
    test_mbd_loss()
    test_balance_effect()
    test_mbd_class()
    print("\n=== 全部测试通过 ===")
