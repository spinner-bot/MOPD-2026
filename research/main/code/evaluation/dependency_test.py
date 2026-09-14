"""
依赖性测试 (Dependency Test)

按实验方案 §12 实现：
测量冻结 guide 在特定实验模型上的利用贡献。

定义 guide g_i 的删除贡献：
Contribution(g_i) = log P(A*|I,G,Q) - log P(A*|I,G\{g_i},Q)

如果 Contribution(g_i) ≈ 0，只说明该模型没有体现正利用；guide 的
语义必要性由结果盲 DAG/反事实审核 D_sem 决定。

IMPORTANT: 本模块是 analysis-only。返回值不得用于 G0/G1/test 成员、
family/depth 标签、guide 重写或第一版主训练数据的准入（1082）。

测试步骤：
1. 完整链 G1, G2, G3
2. 删除 G1: G2, G3
3. 删除 G2: G1, G3
4. 删除 G3: G1, G2
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class DependencyResult:
    """依赖性测试结果"""
    guide_id: str
    contribution: float  # 删除该 guide 后的 log-prob 下降
    relative_contribution: float  # 相对贡献比例
    is_critical: bool  # Legacy model-utilization threshold flag, not D_sem.
    analysis_only: bool = True
    selection_safe: bool = False


def compute_contribution(
    full_chain_logp: float,
    reduced_chain_logp: float
) -> float:
    """
    计算单个 guide 的贡献

    Contribution(g_i) = log P(A*|I,G,Q) - log P(A*|I,G\\{g_i},Q)

    Args:
        full_chain_logp: 完整链的 log probability
        reduced_chain_logp: 删除某个 guide 后的 log probability

    Returns:
        贡献值（正值表示该 guide 有帮助）
    """
    return full_chain_logp - reduced_chain_logp


def run_dependency_test(
    guides: list[dict],
    logp_fn: Callable[[list[dict]], float],
    threshold: float = 0.1
) -> list[DependencyResult]:
    """
    运行冻结样本的模型利用分析（方案 §12，1082 勘误）

    Args:
        guides: 引导问题列表 [{"question": ..., "answer": ...}, ...]
        logp_fn: 函数，输入引导列表，输出正确答题的 log probability
            即 logp_fn(guides) = log P(A*|I,guides,Q)
        threshold: 判定为"关键节点"的最小贡献阈值

    Returns:
        每个 guide 的模型利用结果。结果始终标记 analysis_only，不能
        反馈到确认性样本选择。
    """
    n_guides = len(guides)

    if n_guides == 0:
        return []

    # 计算完整链的 log probability
    full_logp = logp_fn(guides)

    results = []

    # 对每个 guide，计算删除后的贡献
    for i in range(n_guides):
        # 构建删除第 i 个 guide 的子集
        reduced_guides = [g for j, g in enumerate(guides) if j != i]

        # 计算减少链的 log probability
        reduced_logp = logp_fn(reduced_guides)

        # 计算贡献
        contribution = compute_contribution(full_logp, reduced_logp)

        # 计算相对贡献（占总贡献的比例）
        # 暂时标记，稍后计算
        results.append(DependencyResult(
            guide_id=f"G{i+1}",
            contribution=contribution,
            relative_contribution=0.0,  # 稍后计算
            is_critical=contribution >= threshold,
            analysis_only=True,
            selection_safe=False,
        ))

    # 计算相对贡献
    total_contribution = sum(r.contribution for r in results)
    if total_contribution > 0:
        for r in results:
            r.relative_contribution = r.contribution / total_contribution

    return results


def analyze_dependency_pattern(
    results: list[DependencyResult]
) -> dict:
    """
    分析依赖性模式

    Returns:
        {
            "all_critical": bool,  # 所有 guide 都关键
            "any_redundant": bool,  # 存在冗余 guide
            "weakest_guide": str,  # 最弱的 guide ID
            "strongest_guide": str,  # 最强的 guide ID
            "chain_quality": str  # "good", "mixed", "poor"
        }
    """
    if not results:
        return {
            "chain_quality": "no_guides",
            "analysis_only": True,
            "selection_safe": False,
        }

    n_critical = sum(1 for r in results if r.is_critical)
    n_total = len(results)

    # 找最强/最弱
    contributions = [(r.guide_id, r.contribution) for r in results]
    weakest = min(contributions, key=lambda x: x[1])
    strongest = max(contributions, key=lambda x: x[1])

    # 判定链质量
    if n_critical == n_total:
        chain_quality = "good"  # 所有 guide 都关键
    elif n_critical >= n_total * 0.5:
        chain_quality = "mixed"  # 部分关键
    else:
        chain_quality = "poor"  # 大多数不关键（可能是伪引导）

    return {
        "all_critical": n_critical == n_total,
        "any_redundant": n_critical < n_total,
        "weakest_guide": weakest[0],
        "weakest_contribution": weakest[1],
        "strongest_guide": strongest[0],
        "strongest_contribution": strongest[1],
        "n_critical": n_critical,
        "n_total": n_total,
        "chain_quality": chain_quality,
        "analysis_only": True,
        "selection_safe": False,
    }


def run_partial_chain_analysis(
    guides: list[dict],
    logp_fn: Callable[[list[dict]], float]
) -> dict:
    """
    部分链分析（方案 §45）

    训练不同长度的链，观察 Gain(k) 随 chain depth 的变化。

    Args:
        guides: 完整引导链
        logp_fn: log probability 函数

    Returns:
        {
            "k_0": logp 无引导,
            "k_1": logp 1-hop,
            "k_2": logp 2-hop,
            ...
            "gains": {1: gain1, 2: gain2, ...}
        }
    """
    n_guides = len(guides)
    results = {}

    # k=0: 无引导
    results["k_0"] = logp_fn([])

    # k=1, 2, ..., n: 使用第一个 k 个引导
    for k in range(1, n_guides + 1):
        partial_guides = guides[:k]
        results[f"k_{k}"] = logp_fn(partial_guides)

    # 计算增益（相对于 k=0）
    baseline = results["k_0"]
    gains = {}
    for k in range(1, n_guides + 1):
        gains[k] = results[f"k_{k}"] - baseline

    results["gains"] = gains
    results["analysis_only"] = True
    results["selection_safe"] = False
    return results
