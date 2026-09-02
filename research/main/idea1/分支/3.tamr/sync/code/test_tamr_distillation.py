"""
TAMR 蒸馏模块单元测试

验证：任务级路由权重、任务分类器、路由蒸馏损失。
用法：python test_tamr_distillation.py（需服务器环境，含 torch）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from tamr_distillation import (
    LANGUAGE_REASONING,
    MIXED,
    VISUAL_REASONING,
    LightTaskClassifier,
    TAMRDistillation,
    tamr_distillation_loss,
    task_routing_weights,
)


def make_logits(batch=2, seq=8, vocab=100, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, vocab)


def test_task_routing_weights():
    # 视觉任务 → 侧重 VL
    w_vl, w_text = task_routing_weights(VISUAL_REASONING)
    assert w_vl > w_text, f"视觉任务 VL 应更强: w_vl={w_vl}, w_text={w_text}"
    # 语言任务 → 侧重 Text
    w_vl, w_text = task_routing_weights(LANGUAGE_REASONING)
    assert w_text > w_vl, f"语言任务 Text 应更强"
    # 混合任务 → 均衡
    w_vl, w_text = task_routing_weights(MIXED)
    assert abs(w_vl - w_text) < 0.01, f"混合任务应均衡"
    # 权重和为 1
    for task in (VISUAL_REASONING, LANGUAGE_REASONING, MIXED):
        w_vl, w_text = task_routing_weights(task)
        assert abs(w_vl + w_text - 1.0) < 1e-5, "权重和应=1"
    print("[OK] task_routing_weights")


def test_tamr_loss():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"vl_weight": 0.7, "text_weight": 0.3, "mixed_weight": 0.5}
    for task in (VISUAL_REASONING, LANGUAGE_REASONING, MIXED):
        loss, metrics = tamr_distillation_loss(student, vl_teacher, text_teacher, task, config)
        assert loss.item() > 0, f"损失应>0, 实际 {loss.item()}"
        assert "task_type" in metrics and metrics["task_type"] == task
        print(f"  [OK] task={task}, loss={loss.item():.4f}, w_vl={metrics['w_vl'].item():.2f}")


def test_task_classifier():
    # 测试任务分类器前向
    classifier = LightTaskClassifier(input_dim=64, hidden_dim=32, num_classes=3)
    prompt_emb = torch.randn(4, 64)
    logits = classifier(prompt_emb)
    assert logits.shape == (4, 3), f"分类器输出形状错误: {logits.shape}"
    print("[OK] LightTaskClassifier forward")


def test_tamr_class():
    student = make_logits()
    vl_teacher = make_logits(seed=1)
    text_teacher = make_logits(seed=2)
    config = {"vl_weight": 0.7, "text_weight": 0.3, "mixed_weight": 0.5}
    module = TAMRDistillation(config)
    loss, metrics = module(student, vl_teacher, text_teacher, VISUAL_REASONING)
    assert loss.item() > 0
    print(f"[OK] TAMRDistillation class, loss={loss.item():.4f}")


if __name__ == "__main__":
    print("=== TAMR 单元测试 ===")
    test_task_routing_weights()
    test_tamr_loss()
    test_task_classifier()
    test_tamr_class()
    print("\n=== 全部测试通过 ===")
