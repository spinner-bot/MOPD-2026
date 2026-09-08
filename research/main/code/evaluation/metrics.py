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

def check_stage0_go_criteria(
    metrics_by_condition: dict[str, list[dict]],
    paired_tests: dict[str, dict] | None = None,
    threshold_gap: float = 0.02,
    threshold_logp: float = 0.1,
    threshold_depth3_gain: float = 0.03
) -> dict:
    """
    检查 Stage 0 Go/No-Go 条件

    Args:
        metrics_by_condition: 各条件的指标列表
            键: "S", "T_Q", "T_QA", "T_random", "T_generic", etc.
        threshold_gap: 最小准确率差距
        threshold_logp: 最小 log-prob 差距
        threshold_depth3_gain: depth>=3 的最小增益

    Returns:
        判定结果
    """
    paired_tests = paired_tests or {}
    required = {"T_Q", "T_QA", "T_random", "T_generic"}
    missing = sorted(key for key in required if not metrics_by_condition.get(key))
    if missing:
        return {"overall_go": False, "status": "incomplete", "missing_conditions": missing}

    def summarize(condition: str) -> dict:
        rows = metrics_by_condition[condition]
        depth3_rows = [row for row in rows if row.get("reasoning_depth", 0) >= 3]
        return {
            "accuracy_gap": float(np.mean([row["accuracy_gap"] for row in rows])),
            "log_prob_gap": float(np.mean([row["log_prob_gap"] for row in rows])),
            "depth3_accuracy_gap": (
                float(np.mean([row["accuracy_gap"] for row in depth3_rows]))
                if depth3_rows else None
            ),
            "n": len(rows),
            "n_depth3": len(depth3_rows),
        }

    summaries = {condition: summarize(condition) for condition in required}
    candidates = ["T_Q", "T_QA"]
    selected = max(candidates, key=lambda condition: summaries[condition]["log_prob_gap"])
    selected_summary = summaries[selected]

    def significant(name: str) -> bool:
        return bool(paired_tests.get(name, {}).get("significant", False))

    controls = {
        "random": summaries[selected]["accuracy_gap"] - summaries["T_random"]["accuracy_gap"],
        "generic": summaries[selected]["accuracy_gap"] - summaries["T_generic"]["accuracy_gap"],
    }
    criteria = {
        "positive_teacher_gap": {
            "value": selected_summary["accuracy_gap"], "threshold": threshold_gap,
            "pass": selected_summary["accuracy_gap"] >= threshold_gap and significant(selected),
        },
        "log_prob_increase": {
            "value": selected_summary["log_prob_gap"], "threshold": threshold_logp,
            "pass": selected_summary["log_prob_gap"] >= threshold_logp and significant(selected),
        },
        "depth3_gain": {
            "value": selected_summary["depth3_accuracy_gap"], "threshold": threshold_depth3_gain,
            "pass": selected_summary["depth3_accuracy_gap"] is not None
            and selected_summary["depth3_accuracy_gap"] >= threshold_depth3_gain,
        },
        "structured_vs_random": {
            "value": controls["random"], "threshold": 0.01,
            "pass": controls["random"] > 0.01 and significant(f"{selected}_vs_T_random"),
        },
        "structured_vs_generic": {
            "value": controls["generic"], "threshold": 0.01,
            "pass": controls["generic"] > 0.01 and significant(f"{selected}_vs_T_generic"),
        },
    }
    return {
        "overall_go": all(item["pass"] for item in criteria.values()),
        "status": "complete",
        "selected_text_method": selected,
        "summaries": summaries,
        "criteria": criteria,
    }
