"""
可视化模板

按实验方案 §61-62, §38-39 实现核心图表：
- Teacher-Student JSD vs training step (Figure A)
- Privilege Gap vs training step (Figure B)
- Diagnostic Accuracy vs training step (Figure C)
- Gain vs Reasoning Depth (Figure 3)
- P/R/PR 分层图 (Figure 4)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os


def setup_style():
    """设置绘图风格"""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def plot_training_curves(
    training_steps: np.ndarray,
    jsd_values: np.ndarray,
    privilege_gap: np.ndarray,
    diagnostic_acc: np.ndarray,
    save_path: Optional[str] = None
):
    """
    绘制训练曲线（方案 §61）

    Figure A: Teacher-Student JSD vs training step
    Figure B: Privilege Gap vs training step
    Figure C: Diagnostic Accuracy vs training step

    Args:
        training_steps: 训练步数
        jsd_values: Teacher-Student JSD
        privilege_gap: Privilege Gap (Teacher - Student)
        diagnostic_acc: Diagnostic 准确率
        save_path: 保存路径（可选）
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Figure A: JSD
    axes[0, 0].plot(training_steps, jsd_values, "b-", linewidth=2)
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Teacher-Student JSD")
    axes[0, 0].set_title("Figure A: Teacher-Student JSD vs Training Step")
    axes[0, 0].grid(True, alpha=0.3)

    # Figure B: Privilege Gap
    axes[0, 1].plot(training_steps, privilege_gap, "r-", linewidth=2)
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("Privilege Gap")
    axes[0, 1].set_title("Figure B: Privilege Gap vs Training Step")
    axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)

    # Figure C: Diagnostic Accuracy
    axes[1, 0].plot(training_steps, diagnostic_acc, "g-", linewidth=2)
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Diagnostic Accuracy")
    axes[1, 0].set_title("Figure C: Diagnostic Accuracy vs Training Step")
    axes[1, 0].grid(True, alpha=0.3)

    # 空白（可添加更多曲线）
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_reasoning_depth_analysis(
    depths: list[int],
    gains_by_depth: dict[int, float],
    model_names: list[str] = None,
    save_path: Optional[str] = None
):
    """
    绘制推理深度分析图（方案 §39, Figure 3）

    Gain_T(D) = Acc_T(D) - Acc_00(D)

    理想情况：Gain 随 reasoning depth 上升

    Args:
        depths: 推理深度列表 [1, 2, 3, 4]
        gains_by_depth: 各模型在各深度的增益
            {depth: {"V": gain, "T": gain, "VT": gain}}
        model_names: 模型名称列表
        save_path: 保存路径
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(gains_by_depth[depths[0]].keys())
    colors = sns.color_palette("husl", len(models))
    markers = ["o", "s", "^", "D"]

    for i, model in enumerate(models):
        gains = [gains_by_depth[d].get(model, 0) for d in depths]
        ax.plot(depths, gains, marker=markers[i % len(markers)],
                color=colors[i], linewidth=2, markersize=8, label=model)

    ax.set_xlabel("Reasoning Depth", fontsize=14)
    ax.set_ylabel("Accuracy Gain over Baseline", fontsize=14)
    ax.set_title("Figure 3: Gain vs Reasoning Depth", fontsize=16)
    ax.set_xticks(depths)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_prp_stratification(
    task_families: list[str],
    results_by_family: dict[str, dict[str, float]],
    save_path: Optional[str] = None
):
    """
    绘制 P/R/PR 分层图（方案 §38, Figure 4）

    Table 格式:
    | Task | Base | 00 | V | T | VT |

    Args:
        task_families: ["P", "R", "PR"]
        results_by_family: {family: {"Base": acc, "00": acc, "V": acc, "T": acc, "VT": acc}}
        save_path: 保存路径
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(results_by_family[task_families[0]].keys())
    x = np.arange(len(task_families))
    width = 0.15
    colors = sns.color_palette("husl", len(models))

    for i, model in enumerate(models):
        accs = [results_by_family[family].get(model, 0) for family in task_families]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=model, color=colors[i])

        # 在柱子上添加数值标签
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Task Family", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Figure 4: P/R/PR Stratification", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(["P (Perception)", "R (Reasoning)", "PR (Dual)"])
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_gap_closure(
    training_steps: np.ndarray,
    student_perf: np.ndarray,
    teacher_perf: np.ndarray,
    save_path: Optional[str] = None
):
    """
    绘制 Gap Closure 图（方案 §62-63）

    理想情况：Student ↑, Gap ↓
    而不是 Teacher ↓ 导致 Gap ↓

    Args:
        training_steps: 训练步数
        student_perf: 学生性能
        teacher_perf: 教师性能
        save_path: 保存路径
    """
    setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Student 和 Teacher 性能
    ax1.plot(training_steps, student_perf, "b-", linewidth=2, label="Student (I,Q)")
    ax1.plot(training_steps, teacher_perf, "r-", linewidth=2, label="Teacher (I,G,Q)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Performance")
    ax1.set_title("Student vs Teacher Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：Gap 变化
    gap = teacher_perf - student_perf
    ax2.plot(training_steps, gap, "g-", linewidth=2)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Privilege Gap")
    ax2.set_title("Reasoning-Privilege Gap Curve")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 添加注释
    ax2.annotate("Initial Gap >> 0",
                 xy=(training_steps[0], gap[0]),
                 xytext=(training_steps[0] + 10, gap[0] + 0.02),
                 arrowprops=dict(arrowstyle="->", color="gray"))

    ax2.annotate("Gap ↓ as Student catches up",
                 xy=(training_steps[-1], gap[-1]),
                 xytext=(training_steps[-1] - 50, gap[-1] - 0.03),
                 arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_teacher_disagreement(
    disagreement_bins: list[str],
    vt_gain: list[float],
    save_path: Optional[str] = None
):
    """
    绘制 Teacher Disagreement vs VT Gain（方案 §49, Figure 5）

    观察 VT 增益是否随 disagreement 下降

    Args:
        disagreement_bins: ["Low", "Medium", "High"]
        vt_gain: 各 bin 的 VT 增益
        save_path: 保存路径
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["green", "orange", "red"]
    bars = ax.bar(disagreement_bins, vt_gain, color=colors, alpha=0.7)

    ax.set_xlabel("Teacher Disagreement (JSD between V and T)", fontsize=12)
    ax.set_ylabel("VT Gain over 00", fontsize=12)
    ax.set_title("Figure 5: VT Gain vs Teacher Disagreement", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 添加数值标签
    for bar, gain in zip(bars, vt_gain):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f'{gain:.1%}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_thinking_budget(
    budgets: list[int],
    accuracy_by_model: dict[str, list[float]],
    save_path: Optional[str] = None
):
    """
    绘制 Thinking-mode Accuracy vs Budget（方案 §53, Figure 6）

    观察不同 inference budget 下各模型的表现

    Args:
        budgets: [512, 1024, 2048, 4096]
        accuracy_by_model: {"Base": [...], "T-model": [...]}
        save_path: 保存路径
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(accuracy_by_model))
    markers = ["o", "s", "^", "D", "v"]

    for i, (model, accs) in enumerate(accuracy_by_model.items()):
        ax.plot(budgets, accs, marker=markers[i % len(markers)],
                color=colors[i], linewidth=2, markersize=8, label=model)

    ax.set_xlabel("Inference Budget (tokens)", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Figure 6: Accuracy vs Inference Budget (Thinking Mode)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
