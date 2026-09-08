"""
统计检验框架

按实验方案 §64-67 实现：
- Clustered paired bootstrap（方案 §64）
- McNemar test（方案 §65）
- Holm correction（方案 §65）
- Factorial regression（方案 §66）
"""

import numpy as np
from typing import Literal
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


def clustered_paired_bootstrap(
    student_correct: np.ndarray,
    teacher_correct: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> dict:
    """
    以 source_image_id 为 cluster 的 paired bootstrap（方案 §64）

    同一 image 派生的多个问题并不独立，因此以 image 为单位抽样。

    Args:
        student_correct: 学生每题是否正确 [0/1]
        teacher_correct: 教师每题是否正确 [0/1]
        cluster_ids: 每题所属的 cluster (source_image_id)
        n_bootstrap: bootstrap 次数
        confidence: 置信水平
        seed: 随机种子

    Returns:
        {
            "mean_diff": 平均差距,
            "ci_lower": CI 下界,
            "ci_upper": CI 上界,
            "p_value": 单侧 p 值 (teacher > student),
            "significant": 是否显著
        }
    """
    np.random.seed(seed)

    # 获取唯一 clusters
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # 按 cluster 组织数据
    cluster_data = {}
    for cid in unique_clusters:
        mask = cluster_ids == cid
        cluster_data[cid] = {
            "student": student_correct[mask],
            "teacher": teacher_correct[mask]
        }

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        # 以 cluster 为单位有放回抽样
        sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

        # 计算本次 bootstrap 的准确率差
        student_scores = []
        teacher_scores = []
        for cid in sampled_clusters:
            student_scores.extend(cluster_data[cid]["student"])
            teacher_scores.extend(cluster_data[cid]["teacher"])

        acc_s = np.mean(student_scores)
        acc_t = np.mean(teacher_scores)
        diffs.append(acc_t - acc_s)

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)

    # 置信区间
    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    # 单侧 p 值 (teacher > student)
    p_value = np.mean(diffs <= 0)

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": ci_lower > 0,  # CI 下界 > 0 表示显著
        "n_bootstrap": n_bootstrap,
        "confidence": confidence
    }


def mcnemar_test(
    student_correct: np.ndarray,
    teacher_correct: np.ndarray
) -> dict:
    """
    McNemar test（方案 §65）

    用于比较配对二分类结果的差异。

    构造 2x2 列联表:
                    Teacher
                Correct  Wrong
    Student Correct    a      b
            Wrong      c      d

    McNemar 统计量 = (b - c)^2 / (b + c)

    Args:
        student_correct: 学生每题是否正确 [0/1]
        teacher_correct: 教师每题是否正确 [0/1]

    Returns:
        {
            "statistic": McNemar 统计量,
            "p_value": p 值,
            "significant": 是否显著 (α=0.05),
            "contingency_table": 列联表
        }
    """
    # 构造列联表
    a = np.sum((student_correct == 1) & (teacher_correct == 1))  # 都对
    b = np.sum((student_correct == 1) & (teacher_correct == 0))  # 学生对，教师错
    c = np.sum((student_correct == 0) & (teacher_correct == 1))  # 学生错，教师对
    d = np.sum((student_correct == 0) & (teacher_correct == 0))  # 都错

    # McNemar 统计量（带连续性校正）
    if b + c == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "contingency_table": {"a": a, "b": b, "c": c, "d": d}
        }

    # 使用连续性校正
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "contingency_table": {"a": a, "b": b, "c": c, "d": d}
    }


def holm_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = "holm"
) -> dict:
    """
    Holm correction for multiple comparisons（方案 §65）

    大量 subgroup comparison 使用 Holm correction 控制多重比较。

    Args:
        p_values: 原始 p 值数组
        alpha: 显著性水平
        method: 校正方法 ("holm", "holm-sidak", "bonferroni", etc.)

    Returns:
        {
            "reject": 是否拒绝原假设,
            "pvals_corrected": 校正后 p 值,
            "alphacSidak": Sidak 校正 alpha,
            "alphacBonferroni": Bonferroni 校正 alpha
        }
    """
    reject, pvals_corrected, alphacSidak, alphacBonferroni = multipletests(
        p_values, alpha=alpha, method=method
    )

    return {
        "reject": reject,
        "pvals_corrected": pvals_corrected,
        "alphacSidak": alphacSidak,
        "alphacBonferroni": alphacBonferroni,
        "method": method
    }


def factorial_regression(
    correct: np.ndarray,
    v_privilege: np.ndarray,
    t_privilege: np.ndarray,
    depth: np.ndarray = None,
    clarity: np.ndarray = None,
    target_size: np.ndarray = None,
    cluster_ids: np.ndarray = None
) -> dict:
    """
    Factorial regression（方案 §66）

    Correct ~ V + T + V:T
    进一步: Correct ~ V + T + V:T + Depth + Clarity + TargetSize

    研究 T:Depth 和 V:TargetSize 交互效应。

    Args:
        correct: 是否正确 [0/1]
        v_privilege: 是否有视觉特权 [0/1]
        t_privilege: 是否有文本特权 [0/1]
        depth: 推理深度 (可选)
        clarity: 视觉清晰度 (可选)
        target_size: 目标大小 (可选)
        cluster_ids: cluster ID (可选，用于 cluster-robust SE)

    Returns:
        {
            "coefficients": 系数,
            "std_errors": 标准误,
            "p_values": p 值,
            "summary": 模型摘要
        }
    """
    # 构建设计矩阵
    X_data = [
        np.ones(len(correct)),  # 截距
        v_privilege,
        t_privilege,
        v_privilege * t_privilege  # 交互项 V:T
    ]

    feature_names = ["Intercept", "V", "T", "V:T"]

    if depth is not None:
        X_data.append(depth)
        feature_names.append("Depth")
        # T:Depth 交互
        X_data.append(t_privilege * depth)
        feature_names.append("T:Depth")

    if clarity is not None:
        X_data.append(clarity)
        feature_names.append("Clarity")

    if target_size is not None:
        X_data.append(target_size)
        feature_names.append("TargetSize")
        # V:TargetSize 交互
        X_data.append(v_privilege * target_size)
        feature_names.append("V:TargetSize")

    X = np.column_stack(X_data)

    # 逻辑回归
    try:
        model = sm.Logit(correct, X)
        result = model.fit(disp=0)

        return {
            "coefficients": dict(zip(feature_names, result.params)),
            "std_errors": dict(zip(feature_names, result.bse)),
            "p_values": dict(zip(feature_names, result.pvalues)),
            "summary": result.summary().as_text(),
            "aic": result.aic,
            "bic": result.bic
        }
    except Exception as e:
        return {
            "error": str(e),
            "coefficients": None,
            "p_values": None
        }


def compute_interaction_effect(
    perf_00: float,
    perf_V: float,
    perf_T: float,
    perf_VT: float
) -> dict:
    """
    计算 V/T 交互效应（方案 §35）

    Interaction = Perf(VT) - Perf(V) - Perf(T) + Perf(00)

    Interaction > 0: V/T 协同
    Interaction ≈ 0: 近似独立
    Interaction < 0: 冗余或干扰

    Args:
        perf_00: 无特权性能
        perf_V: 仅视觉特权性能
        perf_T: 仅文本特权性能
        perf_VT: 双模态特权性能

    Returns:
        {
            "interaction": 交互效应值,
            "v_main": V 主效应,
            "t_main": T 主效应,
            "interpretation": 解释
        }
    """
    # 主效应（方案 §35）
    v_main = (perf_V + perf_VT - perf_00 - perf_T) / 2
    t_main = (perf_T + perf_VT - perf_00 - perf_V) / 2

    # 交互效应
    interaction = perf_VT - perf_V - perf_T + perf_00

    # 解释
    if interaction > 0.02:
        interpretation = "synergy (V and T complement each other)"
    elif interaction < -0.02:
        interpretation = "interference or redundancy"
    else:
        interpretation = "approximately independent"

    return {
        "interaction": interaction,
        "v_main_effect": v_main,
        "t_main_effect": t_main,
        "interpretation": interpretation
    }
