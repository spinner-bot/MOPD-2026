"""
链顺序测试 (Chain Order Test)

按实验方案 §13 实现：
检验 teacher 利用的是推理结构，而不只是额外事实。

对于 G1 → G2 → G3，构造 G3, G1, G2：
- 保持文字内容相同
- token 总量相近
- 中间答案相同
- 仅破坏 reasoning order

如果 Perf(G_ordered) > Perf(G_shuffle)，则证明 teacher 利用了推理结构。
"""

import numpy as np
import random
from typing import Callable
from dataclasses import dataclass


@dataclass
class ChainOrderResult:
    """链顺序测试结果"""
    ordered_logp: float
    shuffled_logp: float
    advantage: float  # ordered - shuffled
    significant: bool  # 是否显著
    n_trials: int


def shuffle_guides(
    guides: list[dict],
    seed: int = None
) -> list[dict]:
    """
    打乱引导顺序

    Args:
        guides: 原始引导列表
        seed: 随机种子

    Returns:
        打乱后的引导列表
    """
    if seed is not None:
        random.seed(seed)

    shuffled = guides.copy()
    random.shuffle(shuffled)
    return shuffled


def run_chain_order_test(
    guides: list[dict],
    logp_fn: Callable[[list[dict]], float],
    n_shuffles: int = 10,
    seed: int = 42
) -> ChainOrderResult:
    """
    运行链顺序测试（方案 §13）

    Args:
        guides: 原始有序引导链
        logp_fn: 函数，输入引导列表，输出 log probability
        n_shuffles: 打乱次数
        seed: 随机种子

    Returns:
        ChainOrderResult
    """
    random.seed(seed)

    # 计算有序链的 log probability
    ordered_logp = logp_fn(guides)

    # 多次打乱，计算平均 log probability
    shuffled_logps = []
    for i in range(n_shuffles):
        shuffled = shuffle_guides(guides, seed=seed + i)
        shuffled_logp = logp_fn(shuffled)
        shuffled_logps.append(shuffled_logp)

    # 平均 shuffled log probability
    avg_shuffled_logp = np.mean(shuffled_logps)
    std_shuffled_logp = np.std(shuffled_logps)

    # 计算优势
    advantage = ordered_logp - avg_shuffled_logp

    # 显著性检验（简单 t-test）
    if std_shuffled_logp > 0:
        t_stat = advantage / (std_shuffled_logp / np.sqrt(n_shuffles))
        # 粗略判断：t > 2 大约对应 p < 0.05
        significant = abs(t_stat) > 2.0
    else:
        significant = advantage > 0.01  # 如果方差为 0，只看方向

    return ChainOrderResult(
        ordered_logp=ordered_logp,
        shuffled_logp=avg_shuffled_logp,
        advantage=advantage,
        significant=significant,
        n_trials=n_shuffles
    )


def run_order_ablation(
    guides: list[dict],
    logp_fn: Callable[[list[dict]], float]
) -> dict:
    """
    顺序消融实验

    测试不同排列顺序的性能

    Args:
        guides: 原始引导链
        logp_fn: log probability 函数

    Returns:
        {
            "ordered": logp,
            "reversed": logp,
            "random_1": logp,
            "random_2": logp,
            ...
        }
    """
    results = {}

    # 1. 原始顺序
    results["ordered"] = logp_fn(guides)

    # 2. 反转顺序
    reversed_guides = list(reversed(guides))
    results["reversed"] = logp_fn(reversed_guides)

    # 3. 随机顺序（3次）
    for i in range(3):
        shuffled = shuffle_guides(guides, seed=i)
        results[f"random_{i+1}"] = logp_fn(shuffled)

    return results


def analyze_order_sensitivity(results: dict) -> dict:
    """
    分析顺序敏感性

    Returns:
        {
            "ordered_best": bool,  # 有序是否最佳
            "max_difference": float,  # 最大差异
            "order_matters": bool  # 顺序是否重要
        }
    """
    values = list(results.values())
    ordered_value = results.get("ordered", 0)

    max_value = max(values)
    min_value = min(values)
    max_diff = max_value - min_value

    ordered_best = ordered_value >= max_value - 0.01  # 允许微小误差

    # 如果差异大于阈值，认为顺序重要
    order_matters = max_diff > 0.1

    return {
        "ordered_best": ordered_best,
        "max_difference": max_diff,
        "order_matters": order_matters,
        "interpretation": (
            "Teacher uses reasoning structure" if ordered_best and order_matters
            else "Teacher may just use content, not structure"
        )
    }


def create_balanced_shuffle(
    guides: list[dict],
    original_order: list[int],
    target_order: list[int]
) -> list[dict]:
    """
    创建特定的重排顺序

    Args:
        guides: 原始引导
        original_order: 原始索引 [0, 1, 2, ...]
        target_order: 目标索引 [2, 0, 1, ...]

    Returns:
        重排后的引导列表
    """
    if len(original_order) != len(target_order):
        raise ValueError("Order lengths must match")

    # 创建映射
    reorder_map = {orig: target for orig, target in zip(original_order, target_order)}

    # 重排
    result = [None] * len(guides)
    for i, guide in enumerate(guides):
        new_pos = reorder_map.get(i, i)
        result[new_pos] = guide

    return result
