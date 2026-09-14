"""
评估指标实现

按实验方案 §21 实现核心指标：
- Accuracy Gap
- Correct-Answer Log Probability Gap
- Margin Improvement
- Entropy
- Student-Teacher JSD
"""

import numpy as np
from typing import Literal


def accuracy_gap(
    acc_teacher: float,
    acc_student: float
) -> float:
    """
    准确率差距（方案 §21）

    ΔAcc_T = Acc_T - Acc_S
    """
    return acc_teacher - acc_student


def log_prob_gap(
    logp_teacher: float,
    logp_student: float
) -> float:
    """
    正确答案对数概率差距（方案 §21）

    ΔLP_T = log P_T(A*) - log P_S(A*)

    这是非常重要的指标，因为可能出现二者 accuracy 都为 1，
    但 teacher 的 log-prob 明显更高的情况。
    """
    return logp_teacher - logp_student


def margin(probs: np.ndarray, correct_idx: int) -> float:
    """
    置信度边际（方案 §21）

    M = P(A*) - max_{a≠A*} P(a)
    """
    correct_prob = probs[correct_idx]
    # 排除正确答案后取最大
    other_probs = np.delete(probs, correct_idx)
    if len(other_probs) == 0:
        return 1.0  # 只有一个选项
    max_other = np.max(other_probs)
    return correct_prob - max_other


def margin_improvement(
    margin_teacher: float,
    margin_student: float
) -> float:
    """
    边际改进（方案 §21）

    ΔM = M_T - M_S
    """
    return margin_teacher - margin_student


def entropy(probs: np.ndarray, eps: float = 1e-10) -> float:
    """
    信息熵（方案 §21）

    H(p) = -Σ p(a) log p(a)

    检查 T 是否只是让模型"更确定"，还是"更正确"。
    """
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    KL 散度

    KL(P || Q) = Σ P(x) log(P(x) / Q(x))
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Jensen-Shannon 散度（方案 §21）

    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    其中 M = 0.5 * (P + Q)

    Student-Teacher JSD: D_ST = JSD(P_S, P_T)
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_all_metrics(
    probs_student: np.ndarray,
    probs_teacher: np.ndarray,
    correct_idx: int
) -> dict:
    """
    计算所有指标

    Args:
        probs_student: 学生模型的概率分布 [P(A), P(B), P(C), P(D)]
        probs_teacher: 教师模型的概率分布
        correct_idx: 正确答案索引 (0-3)

    Returns:
        包含所有指标的字典
    """
    # 基础指标
    acc_s = int(np.argmax(probs_student) == correct_idx)
    acc_t = int(np.argmax(probs_teacher) == correct_idx)

    logp_s = np.log(probs_student[correct_idx] + 1e-10)
    logp_t = np.log(probs_teacher[correct_idx] + 1e-10)

    margin_s = margin(probs_student, correct_idx)
    margin_t = margin(probs_teacher, correct_idx)

    entropy_s = entropy(probs_student)
    entropy_t = entropy(probs_teacher)

    return {
        # 基础
        "acc_student": acc_s,
        "acc_teacher": acc_t,
        "logp_student": logp_s,
        "logp_teacher": logp_t,
        "margin_student": margin_s,
        "margin_teacher": margin_t,
        "entropy_student": entropy_s,
        "entropy_teacher": entropy_t,

        # 差距指标（方案 §21）
        "accuracy_gap": acc_t - acc_s,
        "log_prob_gap": logp_t - logp_s,
        "margin_improvement": margin_t - margin_s,

        # 分布差异
        "jsd": jsd(probs_student, probs_teacher),
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """
    汇总多样本指标

    Args:
        results: compute_all_metrics 的结果列表

    Returns:
        汇总统计
    """
    if not results:
        return {}

    keys = results[0].keys()
    summary = {}

    for key in keys:
        values = [r[key] for r in results]
        summary[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values),
        }

    # 准确率汇总
    acc_s = np.mean([r["acc_student"] for r in results])
    acc_t = np.mean([r["acc_teacher"] for r in results])

    summary["overall_accuracy_student"] = acc_s
    summary["overall_accuracy_teacher"] = acc_t
    summary["overall_accuracy_gap"] = acc_t - acc_s

    return summary


# === Stage 0 Go/No-Go 判定（方案 §23）===
# === Retired Stage 0 decision entry point ===

def check_stage0_go_criteria(
    metrics_by_condition: dict[str, list[dict]],
    paired_tests: dict[str, dict] | None = None,
    threshold_gap: float = 0.02,
    threshold_logp: float = 0.1,
    threshold_depth3_gain: float = 0.03,
) -> dict:
    """Fail closed: the pre-1017 fixed-label Stage 0 selector is retired.

    This historical interface selected T-Q or T-QA after observing their
    log-probability effects and used one unmatched generic/random condition.
    It cannot represent the natural_rotation contract, the two per-item
    Generic controls, semantic label aggregation, the fixed decision-rule
    hash, or the result-integrity gates.  Keeping a false-returning shim makes
    old imports explicit without allowing this path to authorize an experiment.

    Use ``exploratory_teacher_gap.decide_teacher_gap_label_rotation`` only
    after its locked-manifest/result/asset gates have passed.
    """

    del (metrics_by_condition, paired_tests, threshold_gap, threshold_logp, threshold_depth3_gain)
    return {
        "overall_go": False,
        "status": "retired_unsafe",
        "replacement": "exploratory_teacher_gap.decide_teacher_gap_label_rotation",
        "reason": (
            "pre-1017 selector lacks label rotation, paired Generic-Q/Generic-QA, "
            "locked-result gates and a hashed decision rule"
        ),
    }
