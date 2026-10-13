#!/usr/bin/env python3
"""Render diagnostics comparing the preserved rule labels with LLM adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.per_sample)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    frame["rule_correct"] = frame["rule_correct"].astype(str).str.lower().eq("true")
    frame["llm_correct"] = frame["llm_correct"].astype(str).str.lower().eq("true")

    # Correctness agreement matrix (both labels are measured against the same frozen gold).
    matrix = pd.crosstab(frame["rule_correct"], frame["llm_correct"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks([0, 1], ["LLM wrong", "LLM correct"])
    ax.set_yticks([0, 1], ["Rule wrong", "Rule correct"])
    kappa = summary["rule_llm_correctness_comparison"].get("cohens_kappa_correctness")
    ax.set_title(f"Rule vs LLM correctness labels (κ={kappa:.3f})" if kappa is not None else "Rule vs LLM correctness labels")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix.iloc[i, j]:,}", ha="center", va="center", fontsize=12, color="#14202a")
    fig.colorbar(im, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(args.output_dir / "detector_correctness_agreement.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    by = frame.groupby("condition").apply(
        lambda x: pd.Series({
            "same_correctness": int((x.rule_correct == x.llm_correct).sum()),
            "rule_only_correct": int((x.rule_correct & ~x.llm_correct).sum()),
            "llm_only_correct": int((~x.rule_correct & x.llm_correct).sum()),
            "answer_letter_diff": int((x.rule_parsed_answer.fillna("").astype(str) != x.llm_parsed_answer.fillna("").astype(str)).sum()),
            "n": len(x),
        })
    ).reset_index()
    by["correctness_disagreement_rate"] = 1 - by.same_correctness / by.n
    by["answer_letter_disagreement_rate"] = by.answer_letter_diff / by.n
    by.to_csv(args.output_dir / "detector_disagreement_by_condition.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(by))
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    ax.bar(x - 0.18, by.rule_only_correct, 0.36, label="Rule correct; LLM incorrect", color="#e45756")
    ax.bar(x + 0.18, by.llm_only_correct, 0.36, label="Rule incorrect; LLM correct", color="#54a24b")
    ax.set_xticks(x, by.condition)
    ax.set_ylabel("Samples")
    ax.set_title("Direction of correctness-label changes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "detector_reclassification_direction.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    statuses = frame.pivot_table(index="condition", columns="llm_status", values="sample_id", aggfunc="count", fill_value=0)
    statuses = statuses.reindex(index=["S0", "SC", "S1", "S2", "S3"], columns=["ANSWERED", "NO_ANSWER", "AMBIGUOUS"], fill_value=0)
    ax = statuses.plot(kind="bar", stacked=True, figsize=(8.0, 4.9), color=["#4c78a8", "#f2cf5b", "#e45756"])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Samples")
    ax.set_title("LLM adjudication status by condition")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Status", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(args.output_dir / "llm_status_by_condition.png", dpi=190, bbox_inches="tight")
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
