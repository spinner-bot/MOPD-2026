#!/usr/bin/env python3
"""Extract authoritative vLLM token usage and analyze correctness/long tails."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, skew, spearmanr


CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()

    per_sample = pd.read_csv(args.per_sample)
    usage_rows = []
    for path in sorted(args.outputs.glob("*.jsonl")):
        condition = path.stem
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                usage = row.get("usage") or {}
                if "completion_tokens" not in usage or "prompt_tokens" not in usage:
                    raise ValueError(f"missing usage token counts: {path}:{line_no}")
                usage_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "condition": row["condition"],
                        "prompt_tokens": int(usage["prompt_tokens"]),
                        "completion_tokens": int(usage["completion_tokens"]),
                        "total_tokens": int(usage["total_tokens"]),
                        "model_id": row.get("model_id"),
                    }
                )
    usage_df = pd.DataFrame(usage_rows)
    if len(usage_df) != 10000 or usage_df.duplicated(["sample_id", "condition"]).any():
        raise ValueError("usage records are not exactly one complete paired record per condition")
    # Token fields in the adjudicated table are convenient for other analyses,
    # but the inference output usage is authoritative. Avoid pandas' automatic
    # _x/_y suffixes when both inputs already contain these columns.
    per_sample = per_sample.drop(
        columns=[column for column in ("prompt_tokens", "completion_tokens", "total_tokens") if column in per_sample]
    )
    merged = per_sample.merge(usage_df, on=["sample_id", "condition"], how="inner", validate="one_to_one")
    if len(merged) != len(per_sample):
        raise ValueError("usage/per-sample join is incomplete")
    merged["correct_bool"] = merged["correct"].astype(str).str.lower().eq("true")
    merged.to_csv(args.output_csv, index=False, encoding="utf-8")

    conditions = []
    for condition in CONDITIONS:
        x = merged[merged.condition == condition]
        correct = x.loc[x.correct_bool, "completion_tokens"]
        incorrect = x.loc[~x.correct_bool, "completion_tokens"]
        p = pearsonr(x.completion_tokens, x.correct_bool.astype(int))
        s = spearmanr(x.completion_tokens, x.correct_bool.astype(int))
        mw = mannwhitneyu(correct, incorrect, alternative="two-sided")
        p95 = float(x.completion_tokens.quantile(.95))
        p99 = float(x.completion_tokens.quantile(.99))
        tail = x[x.completion_tokens >= p95]
        conditions.append({
            "condition": condition, "n": int(len(x)),
            "prompt_tokens_mean": float(x.prompt_tokens.mean()),
            "prompt_tokens_median": float(x.prompt_tokens.median()),
            "completion_tokens_mean": float(x.completion_tokens.mean()),
            "completion_tokens_median": float(x.completion_tokens.median()),
            "completion_tokens_sd": float(x.completion_tokens.std(ddof=1)),
            "completion_tokens_skewness": float(skew(x.completion_tokens)),
            "completion_tokens_q90": float(x.completion_tokens.quantile(.90)),
            "completion_tokens_q95": p95, "completion_tokens_q99": p99,
            "completion_tokens_max": int(x.completion_tokens.max()),
            "pearson_r": float(p.statistic), "pearson_p": float(p.pvalue),
            "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue),
            "correct_n": int(len(correct)), "correct_mean_tokens": float(correct.mean()),
            "correct_median_tokens": float(correct.median()), "incorrect_n": int(len(incorrect)),
            "incorrect_mean_tokens": float(incorrect.mean()), "incorrect_median_tokens": float(incorrect.median()),
            "correct_vs_incorrect_mann_whitney_p": float(mw.pvalue),
            "p95_tail_n": int(len(tail)), "p95_tail_accuracy": float(tail.correct_bool.mean()),
        })

    # For each condition, accuracy over fixed within-condition token deciles.
    deciles = []
    for condition in CONDITIONS:
        x = merged[merged.condition == condition].copy()
        x["length_decile"] = pd.qcut(x.completion_tokens.rank(method="first"), 10, labels=False) + 1
        grouped = x.groupby("length_decile", observed=True).agg(
            median_completion_tokens=("completion_tokens", "median"),
            mean_completion_tokens=("completion_tokens", "mean"),
            accuracy=("correct_bool", "mean"), n=("correct_bool", "size"),
        ).reset_index()
        grouped["condition"] = condition
        deciles.extend(grouped.to_dict("records"))

    result = {
        "schema": "short-output-token-length-analysis-v1",
        "source": "vLLM usage.prompt_tokens/completion_tokens from run 60011513",
        "model_id": sorted(set(usage_df.model_id.dropna()))[0],
        "n_records": int(len(merged)), "conditions": conditions, "deciles": deciles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    colors = {"S0": "#4c78a8", "SC": "#72b7b2", "S1": "#f58518", "S2": "#e45756", "S3": "#54a24b"}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

    fig, ax = plt.subplots(figsize=(10, 5.5))
    data, positions, labels, facecolors = [], [], [], []
    pos = 1
    for condition in CONDITIONS:
        x = merged[merged.condition == condition]
        data.extend([x.loc[x.correct_bool, "completion_tokens"], x.loc[~x.correct_bool, "completion_tokens"]])
        positions.extend([pos, pos + .38]); labels.extend([f"{condition}\ncorrect", f"{condition}\nincorrect"])
        facecolors.extend([colors[condition], "#bdbdbd"]); pos += 1.65
    bp = ax.boxplot(data, positions=positions, widths=.30, patch_artist=True, showfliers=False)
    for box, color in zip(bp["boxes"], facecolors): box.set_facecolor(color); box.set_alpha(.8)
    ax.set_yscale("log"); ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Completion tokens (log scale)"); ax.set_title("Completion-token length by correctness")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(args.figure_dir / "token_length_by_correctness.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ddf = pd.DataFrame(deciles)
    for condition in CONDITIONS:
        x = ddf[ddf.condition == condition]
        ax.plot(x.median_completion_tokens, x.accuracy * 100, marker="o", label=condition, color=colors[condition])
    ax.set_xscale("log"); ax.set_xlabel("Median completion tokens within decile (log scale)")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy across completion-token deciles")
    ax.grid(alpha=.25); ax.legend(ncol=5, frameon=False); fig.tight_layout()
    fig.savefig(args.figure_dir / "token_accuracy_by_decile.png", dpi=180); plt.close(fig)

    summary = pd.DataFrame(conditions).set_index("condition").loc[CONDITIONS]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2)); xpos = np.arange(len(CONDITIONS)); width = .34
    ax1.bar(xpos-width/2, summary.completion_tokens_mean, width, label="Mean", color="#4c78a8")
    ax1.bar(xpos+width/2, summary.completion_tokens_median, width, label="Median", color="#f58518")
    ax1.set_xticks(xpos, CONDITIONS); ax1.set_ylabel("Completion tokens"); ax1.set_title("Completion-token distribution shape")
    ax1.legend(loc="upper left", frameon=False); ax2 = ax1.twinx()
    ax2.plot(xpos, summary.completion_tokens_skewness, color="#e45756", marker="o", linewidth=2, label="Skewness")
    ax2.set_ylabel("Skewness"); ax2.legend(loc="upper right", frameon=False); ax1.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(args.figure_dir / "token_distribution_shape.png", dpi=180); plt.close(fig)

    tail = merged.copy(); p95_map = summary["completion_tokens_q95"].to_dict(); tail["p95_tail"] = [r.completion_tokens >= p95_map[r.condition] for r in tail.itertuples()]
    tail = tail[tail.p95_tail]
    task = pd.crosstab(tail.task, [tail.condition, tail.correct_bool])
    task.to_csv(args.figure_dir / "token_p95_tail_task_table.csv", encoding="utf-8")
    top_tasks = tail.task.value_counts().head(12).index
    plot = tail[tail.task.isin(top_tasks)].groupby(["task", "condition"], observed=True).correct_bool.mean().unstack(fill_value=np.nan).reindex(top_tasks)
    ax = plot.plot(kind="bar", figsize=(11, 5.5), color=[colors[c] for c in CONDITIONS], ylim=(0, 1))
    ax.set_ylabel("Accuracy in condition-specific P95 tail"); ax.set_xlabel("Task (top 12 by tail frequency)")
    ax.set_title("Correctness of long-tail outputs (condition-specific P95)"); ax.legend(title="Condition", ncol=5, frameon=False)
    ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25); ax.figure.tight_layout()
    ax.figure.savefig(args.figure_dir / "token_p95_tail_task_accuracy.png", dpi=180); plt.close(ax.figure)

    print(json.dumps({"records": len(merged), "conditions": len(conditions), "figures": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
