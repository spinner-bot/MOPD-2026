#!/usr/bin/env python3
"""Analyze response length, correctness, and distribution shape for the paired run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, skew, spearmanr


CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["correct_bool"] = df["correct"].astype(str).str.lower().eq("true")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    stats = []
    for condition in CONDITIONS:
        x = df[df["condition"] == condition]
        correct = x.loc[x["correct_bool"], "response_chars"]
        incorrect = x.loc[~x["correct_bool"], "response_chars"]
        pearson = pearsonr(x["response_chars"], x["correct_bool"].astype(int))
        spearman = spearmanr(x["response_chars"], x["correct_bool"].astype(int))
        mw = mannwhitneyu(correct, incorrect, alternative="two-sided")
        stats.append(
            {
                "condition": condition,
                "n": int(len(x)),
                "mean_chars": float(x["response_chars"].mean()),
                "median_chars": float(x["response_chars"].median()),
                "skewness": float(skew(x["response_chars"])),
                "q90_chars": float(x["response_chars"].quantile(0.90)),
                "q95_chars": float(x["response_chars"].quantile(0.95)),
                "max_chars": int(x["response_chars"].max()),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "correct_n": int(len(correct)),
                "correct_mean_chars": float(correct.mean()),
                "correct_median_chars": float(correct.median()),
                "incorrect_n": int(len(incorrect)),
                "incorrect_mean_chars": float(incorrect.mean()),
                "incorrect_median_chars": float(incorrect.median()),
                "correct_vs_incorrect_mann_whitney_p": float(mw.pvalue),
            }
        )

    # Paired correctness comparisons between the three substantive short-output modes.
    pivot = df.pivot(index="sample_id", columns="condition", values="correct_bool")
    paired = []
    for a, b in [("S1", "S2"), ("S3", "S2"), ("S1", "S3")]:
        a_only = int((pivot[a] & ~pivot[b]).sum())
        b_only = int((~pivot[a] & pivot[b]).sum())
        from scipy.stats import binomtest

        p_value = float(binomtest(min(a_only, b_only), a_only + b_only, p=0.5).pvalue)
        paired.append(
            {
                "comparison": f"{a} vs {b}",
                "accuracy_difference": float(pivot[a].mean() - pivot[b].mean()),
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "p_value": p_value,
            }
        )

    result = {"input": str(args.input), "conditions": stats, "paired_comparisons": paired}
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    colors = {"S0": "#4c78a8", "SC": "#72b7b2", "S1": "#f58518", "S2": "#e45756", "S3": "#54a24b"}

    # Correct versus incorrect output lengths. Log scale keeps the long-tail structure visible.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions, labels, box_data, box_colors = [], [], [], []
    pos = 1
    for condition in CONDITIONS:
        x = df[df["condition"] == condition]
        box_data.extend([x.loc[x["correct_bool"], "response_chars"], x.loc[~x["correct_bool"], "response_chars"]])
        positions.extend([pos, pos + 0.38])
        labels.extend([f"{condition}\ncorrect", f"{condition}\nincorrect"])
        box_colors.extend([colors[condition], "#bdbdbd"])
        pos += 1.65
    bp = ax.boxplot(box_data, positions=positions, widths=0.30, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Response characters (log scale)")
    ax.set_title("Output length by condition and correctness")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "length_by_correctness.png", dpi=180)
    plt.close(fig)

    # Decile-binned accuracy by response length, separately for each condition.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition in CONDITIONS:
        x = df[df["condition"] == condition].copy()
        x["bin"] = pd.qcut(x["response_chars"].rank(method="first"), 10, labels=False)
        grouped = x.groupby("bin", observed=True).agg(length=("response_chars", "median"), accuracy=("correct_bool", "mean"))
        ax.plot(grouped["length"], grouped["accuracy"] * 100, marker="o", label=condition, color=colors[condition])
    ax.set_xscale("log")
    ax.set_xlabel("Median response characters within length decile (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy across response-length deciles")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "accuracy_by_length_decile.png", dpi=180)
    plt.close(fig)

    # Mean, median, and skewness summarize the long-tail structure.
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    summary = pd.DataFrame(stats).set_index("condition").loc[CONDITIONS]
    xpos = np.arange(len(CONDITIONS))
    width = 0.34
    ax1.bar(xpos - width / 2, summary["mean_chars"], width, label="Mean", color="#4c78a8")
    ax1.bar(xpos + width / 2, summary["median_chars"], width, label="Median", color="#f58518")
    ax1.set_xticks(xpos, CONDITIONS)
    ax1.set_ylabel("Response characters")
    ax1.set_title("Mean–median gap and right-skewness")
    ax1.legend(loc="upper left", frameon=False)
    ax2 = ax1.twinx()
    ax2.plot(xpos, summary["skewness"], color="#e45756", marker="o", linewidth=2, label="Skewness")
    ax2.set_ylabel("Skewness")
    ax2.legend(loc="upper right", frameon=False)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "length_distribution_shape.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"output_json": str(args.output_json), "figures": 3}, ensure_ascii=False))


if __name__ == "__main__":
    main()
