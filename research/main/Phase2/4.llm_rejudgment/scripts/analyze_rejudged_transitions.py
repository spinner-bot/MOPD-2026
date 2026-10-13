#!/usr/bin/env python3
"""Recompute paired flip/state analyses from LLM-adjudicated correctness labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu


CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]
TARGETS = ["S1", "S2", "S3"]
COLORS = {"S1": "#f58518", "S2": "#e45756", "S3": "#54a24b"}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--include-s2", action="store_true", help="select the expanded S0/S1/S2/S3 report scope")
    parser.add_argument("--case-count", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    per = pd.read_csv(args.per_sample)
    tokens = pd.read_csv(args.tokens)
    token_cols = ["sample_id", "condition", "prompt_tokens", "completion_tokens", "total_tokens"]
    # Usage extracted from the original inference JSONL is authoritative.
    # The adjudicated per-sample table may also carry these fields, so remove
    # its copies before joining to avoid pandas' _x/_y suffixes.
    per = per.drop(columns=[column for column in token_cols[2:] if column in per])
    per = per.merge(tokens[token_cols], on=["sample_id", "condition"], how="inner", validate="one_to_one")
    if len(per) != 10000 or per.duplicated(["sample_id", "condition"]).any():
        raise ValueError("rejudged correctness/token join must contain exactly 10,000 unique rows")
    if per.groupby("condition").size().to_dict() != {condition: 2000 for condition in CONDITIONS}:
        raise ValueError("each condition must contain exactly 2,000 judged rows")
    per["correct_bool"] = per["llm_correct"].astype(str).str.lower().eq("true")
    old_correct_col = "rule_correct"
    per["rule_correct_bool"] = per[old_correct_col].astype(str).str.lower().eq("true")

    input_doc = json.loads(args.input_json.read_text(encoding="utf-8"))
    sample_by_id = {row["sample_id"]: row for row in input_doc["records"]}
    if len(sample_by_id) != 2000:
        raise ValueError("frozen inference input must contain 2,000 samples")

    wide = per.pivot(index="sample_id", columns="condition", values="correct_bool").astype(bool)
    old_wide = per.pivot(index="sample_id", columns="condition", values="rule_correct_bool").astype(bool)
    if set(wide.columns) != set(CONDITIONS) or len(wide) != 2000:
        raise ValueError("paired state matrix is incomplete")

    # Enumerate all mutually exclusive states, retaining zero-count states.
    state_summaries = {}
    for modes in [["S0", "S1", "S3"], ["S0", "S1", "S2", "S3"]]:
        key_name = "states_" + "_".join(modes)
        codes = ["".join("1" if bool(wide.loc[sample_id, mode]) else "0" for mode in modes) for sample_id in wide.index]
        counts = pd.Series(codes).value_counts().to_dict()
        rows = [{"state": format(code, f"0{len(modes)}b"), "n": int(counts.get(format(code, f"0{len(modes)}b"), 0)), "rate": counts.get(format(code, f"0{len(modes)}b"), 0) / len(codes)} for code in range(2 ** len(modes))]
        pd.DataFrame(rows).to_csv(args.output_dir / f"{key_name}.csv", index=False)
        state_summaries[key_name] = {"conditions": modes, "states": rows}

    # Paired baseline transitions for S1/S2/S3.
    raw_by_condition = {}
    for condition in CONDITIONS:
        path = args.raw_output_dir / f"{condition}.jsonl"
        rows = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["sample_id"]] = row
        if len(rows) != 2000:
            raise ValueError(f"raw response coverage failed for {condition}: {len(rows)}")
        raw_by_condition[condition] = rows

    flip_records = []
    target_summaries = []
    task_rate_records = []
    for target in TARGETS:
        target_rows = []
        for sample_id in wide.index:
            baseline = bool(wide.loc[sample_id, "S0"])
            result = bool(wide.loc[sample_id, target])
            category = "gain" if (not baseline and result) else "loss" if (baseline and not result) else "stable_correct" if baseline else "stable_wrong"
            sample = sample_by_id[sample_id]
            rec = {
                "sample_id": sample_id,
                "target": target,
                "category": category,
                "s0_correct": baseline,
                "target_correct": result,
                "task": sample.get("task", ""),
                "source": sample.get("source", ""),
                "split": sample.get("split", ""),
                "question": sample.get("question", sample.get("query", "")),
                "query": sample.get("query", ""),
                "choices": sample.get("choices", []),
                "gold_answer": sample.get("response", ""),
                "S0_completion_tokens": int(wide_token(per, sample_id, "S0")),
                f"{target}_completion_tokens": int(wide_token(per, sample_id, target)),
                "S0_response": raw_by_condition["S0"][sample_id].get("response", ""),
                "target_response": raw_by_condition[target][sample_id].get("response", ""),
                "S0_rule_answer": row_value(per, sample_id, "S0", "rule_parsed_answer"),
                f"{target}_rule_answer": row_value(per, sample_id, target, "rule_parsed_answer"),
                "S0_llm_answer": row_value(per, sample_id, "S0", "llm_parsed_answer"),
                f"{target}_llm_answer": row_value(per, sample_id, target, "llm_parsed_answer"),
            }
            rec["target_minus_s0_completion_tokens"] = rec[f"{target}_completion_tokens"] - rec["S0_completion_tokens"]
            target_rows.append(rec)
        flip_records.extend(target_rows)

        frame = pd.DataFrame(target_rows)
        gains = frame[frame.category == "gain"]
        losses = frame[frame.category == "loss"]
        up = int(len(gains))
        down = int(len(losses))
        p_flip = 1.0 if up + down == 0 else float(binomtest(min(up, down), up + down, 0.5).pvalue)
        target_summaries.append({
            "target": target,
            "n": len(frame),
            "gain_n": up,
            "loss_n": down,
            "stable_correct_n": int((frame.category == "stable_correct").sum()),
            "stable_wrong_n": int((frame.category == "stable_wrong").sum()),
            "gain_rate_of_all": up / len(frame),
            "loss_rate_of_all": down / len(frame),
            "net_gain": up - down,
            "net_accuracy_difference": float(frame.target_correct.mean() - frame.s0_correct.mean()),
            "exact_mcnemar_p": p_flip,
            "gain_mean_token_delta": float(gains.target_minus_s0_completion_tokens.mean()) if up else None,
            "loss_mean_token_delta": float(losses.target_minus_s0_completion_tokens.mean()) if down else None,
            "gain_median_token_delta": float(gains.target_minus_s0_completion_tokens.median()) if up else None,
            "loss_median_token_delta": float(losses.target_minus_s0_completion_tokens.median()) if down else None,
            "gain_vs_loss_mann_whitney_p": float(mannwhitneyu(gains.target_minus_s0_completion_tokens, losses.target_minus_s0_completion_tokens).pvalue) if up and down else None,
        })

        total_by_task = frame.groupby("task", dropna=False).size()
        for task, n_task in total_by_task.items():
            task_frame = frame[frame.task == task]
            task_rate_records.append({
                "target": target,
                "task": task,
                "n_task": int(n_task),
                "gain_n": int((task_frame.category == "gain").sum()),
                "loss_n": int((task_frame.category == "loss").sum()),
                "gain_rate": float((task_frame.category == "gain").mean()),
                "loss_rate": float((task_frame.category == "loss").mean()),
                "net_change_rate": float(task_frame.target_correct.mean() - task_frame.s0_correct.mean()),
            })

    target_adjusted = holm_adjust([row["exact_mcnemar_p"] for row in target_summaries])
    for row, adjusted_p in zip(target_summaries, target_adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)

    flips = pd.DataFrame(flip_records)
    flips.to_csv(args.output_dir / "pairwise_flip_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(task_rate_records).to_csv(args.output_dir / "task_flip_rates.csv", index=False, encoding="utf-8-sig")

    # Logical combinations and S2-specific gains/losses.
    s0 = wide["S0"]
    s1, s2, s3 = wide["S1"], wide["S2"], wide["S3"]
    logical = {
        "s0_wrong_s1s3_both_wrong": int((~s0 & ~s1 & ~s3).sum()),
        "s0_wrong_any_s1s3_recovery": int((~s0 & (s1 | s3)).sum()),
        "s0_wrong_s1s3_both_recover": int((~s0 & s1 & s3).sum()),
        "s0_correct_s1s3_both_correct": int((s0 & s1 & s3).sum()),
        "s0_correct_any_s1s3_loss": int((s0 & (~s1 | ~s3)).sum()),
        "s0_correct_s1s3_both_loss": int((s0 & ~s1 & ~s3).sum()),
        "s2_only_gain_vs_s0": int((~s0 & s2 & ~s1 & ~s3).sum()),
        "s2_only_loss_vs_s0": int((s0 & ~s2 & s1 & s3).sum()),
        "s2_differs_from_both_s1_s3": int((s2 != s1).mul(s2 != s3).sum()),
        "s2_disagrees_with_s1_or_s3": int(((s2 != s1) | (s2 != s3)).sum()),
    }

    # Old versus corrected transition counts establish whether the old S2 rationale survives.
    old_s0, old_s2 = old_wide["S0"], old_wide["S2"]
    old_s2_gain = int((~old_s0 & old_s2).sum())
    old_s2_loss = int((old_s0 & ~old_s2).sum())
    new_s2_gain = int((~s0 & s2).sum())
    new_s2_loss = int((s0 & ~s2).sum())
    s2_pairs = [("S2 vs S0", s2, s0), ("S2 vs S1", s2, s1), ("S2 vs S3", s2, s3)]
    s2_tests = []
    for name, first, second in s2_pairs:
        first_only = int((first & ~second).sum())
        second_only = int((~first & second).sum())
        discordant = first_only + second_only
        p = 1.0 if discordant == 0 else float(binomtest(first_only, discordant, 0.5).pvalue)
        s2_tests.append({"comparison": name, "first_only_correct": first_only, "second_only_correct": second_only, "accuracy_difference": float(first.mean() - second.mean()), "exact_mcnemar_p": p})
    adjusted = holm_adjust([row["exact_mcnemar_p"] for row in s2_tests])
    for row, p_adjusted in zip(s2_tests, adjusted):
        row["holm_adjusted_p"] = float(p_adjusted)

    old_acc_s2 = float(old_wide["S2"].mean())
    new_acc_s2 = float(s2.mean())
    s2_decision = {
        "old_rule_accuracy_s2": old_acc_s2,
        "llm_adjudicated_accuracy_s2": new_acc_s2,
        "accuracy_change_pp": 100 * (new_acc_s2 - old_acc_s2),
        "old_rule_s2_gains_vs_s0": old_s2_gain,
        "old_rule_s2_losses_vs_s0": old_s2_loss,
        "llm_s2_gains_vs_s0": new_s2_gain,
        "llm_s2_losses_vs_s0": new_s2_loss,
        "s2_pairwise_tests": s2_tests,
        "logical_state_counts": logical,
        "decision_method": "Assess corrected S2 accuracy and paired exact tests against S0/S1/S3 together with the number and task distribution of S2-specific transitions; no single unadjusted p-value is used as the sole inclusion rule.",
        "selected_scope": "S0-S1-S2-S3" if args.include_s2 else "S0-S1-S3",
        "selected_scope_requires_reviewer_decision": True,
    }

    # Visual comparisons: preserve 8-state legacy view and offer expanded 16-state view.
    def plot_states(modes: list[str], filename: str, title: str) -> None:
        codes = ["".join("1" if bool(wide.loc[sample_id, mode]) else "0" for mode in modes) for sample_id in wide.index]
        count = pd.Series(codes).value_counts()
        labels = [format(code, f"0{len(modes)}b") for code in range(2 ** len(modes))]
        values = [int(count.get(label, 0)) for label in labels]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.62), 4.7))
        ax.bar(labels, values, color="#4c78a8")
        ax.set_xlabel("Correctness state in listed condition order (1=correct, 0=incorrect)")
        ax.set_ylabel("Samples")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.figure_dir / filename, dpi=190, bbox_inches="tight")
        plt.close(fig)

    plot_states(["S0", "S1", "S3"], "correctness_states_8.png", "Eight paired correctness states: S0–S1–S3")
    plot_states(["S0", "S1", "S2", "S3"], "correctness_states_16.png", "Sixteen paired correctness states: S0–S1–S2–S3")

    targets = ["S1", "S2", "S3"]
    gains = [next(row["gain_n"] for row in target_summaries if row["target"] == target) for target in targets]
    losses = [next(row["loss_n"] for row in target_summaries if row["target"] == target) for target in targets]
    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - 0.18, gains, 0.36, label="S0 wrong → target correct", color="#54a24b")
    ax.bar(x + 0.18, losses, 0.36, label="S0 correct → target wrong", color="#e45756")
    ax.set_xticks(x, targets)
    ax.set_ylabel("Paired samples")
    ax.set_title("Correctness flips from the S0 baseline")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "paired_flip_counts.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    comp = flips[flips.category.isin(["gain", "loss"])].groupby(["target", "task", "category"]).size().rename("n").reset_index()
    comp.to_csv(args.output_dir / "flip_task_composition.csv", index=False, encoding="utf-8-sig")
    pivot = comp.pivot_table(index=["target", "task"], columns="category", values="n", fill_value=0)
    if not pivot.empty:
        pivot["total"] = pivot.get("gain", 0) + pivot.get("loss", 0)
        top_tasks = pivot.groupby("task")["total"].sum().sort_values(ascending=False).head(12).index
        plot_data = pivot.loc[pivot.index.get_level_values("task").isin(top_tasks)]
        plot_data = plot_data[[c for c in ["gain", "loss"] if c in plot_data.columns]]
        ax = plot_data.plot(kind="bar", stacked=True, figsize=(12, 6.5), color=["#54a24b", "#e45756"])
        ax.set_xlabel("Target × task (largest flip groups)")
        ax.set_ylabel("Number of flip samples")
        ax.set_title("Task composition within gain/loss groups")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        ax.figure.tight_layout()
        ax.figure.savefig(args.figure_dir / "flip_task_composition.png", dpi=190, bbox_inches="tight")
        plt.close(ax.figure)

    task_rate_df = pd.DataFrame(task_rate_records)
    task_rate_df.to_csv(args.output_dir / "task_flip_rates.csv", index=False, encoding="utf-8-sig")
    heat = task_rate_df.pivot(index="task", columns="target", values="net_change_rate")
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(7.3, max(4.3, len(heat) * 0.37)))
        im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-0.25, vmax=0.25)
        ax.set_xticks(np.arange(len(heat.columns)), heat.columns)
        ax.set_yticks(np.arange(len(heat.index)), heat.index)
        ax.set_title("Within-task net accuracy change vs S0")
        ax.set_xlabel("Target condition")
        fig.colorbar(im, ax=ax, label="Accuracy difference")
        fig.tight_layout()
        fig.savefig(args.figure_dir / "task_net_accuracy_heatmap.png", dpi=190, bbox_inches="tight")
        plt.close(fig)

    delta_groups = []
    labels, data, box_colors = [], [], []
    for target in TARGETS:
        for category in ["gain", "loss"]:
            vals = flips[(flips.target == target) & (flips.category == category)]["target_minus_s0_completion_tokens"].astype(float)
            if len(vals):
                data.append(vals.to_numpy())
                labels.append(f"{target} {category}")
                box_colors.append("#54a24b" if category == "gain" else "#e45756")
                delta_groups.append({"target": target, "category": category, "n": int(len(vals)), "mean_delta": float(vals.mean()), "median_delta": float(vals.median()), "q25": float(vals.quantile(.25)), "q75": float(vals.quantile(.75))})
    if data:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
        for box, color in zip(bp["boxes"], box_colors):
            box.set_facecolor(color)
            box.set_alpha(0.72)
        ax.axhline(0, color="#555", linewidth=1)
        ax.set_ylabel("Target − S0 completion tokens")
        ax.set_title("Output-token changes among correctness flips")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.figure_dir / "flip_token_delta.png", dpi=190, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(delta_groups).to_csv(args.output_dir / "flip_token_delta_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "schema": "phase2-rejudged-transition-analysis-v1",
        "n_samples": 2000,
        "correctness_label": "LLM answer extraction compared mechanically against frozen gold; NO_ANSWER and AMBIGUOUS count incorrect and are separately reported.",
        "state_summaries": state_summaries,
        "target_flip_summaries": target_summaries,
        "logical_groups": logical,
        "s2_inclusion_evidence": s2_decision,
        "flip_token_delta": delta_groups,
        "selected_report_scope": s2_decision["selected_scope"],
    }
    (args.output_dir / "rejudged_transition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Representative case shortlist; images are fetched later from the pinned benchmark manifest.
    included_targets = TARGETS if args.include_s2 else ["S1", "S3"]
    groups = [(target, category) for target in included_targets for category in ["gain", "loss"]]
    cases = []
    per_group = max(1, args.case_count // len(groups))
    for target, category in groups:
        candidates = flips[(flips.target == target) & (flips.category == category)].copy()
        if candidates.empty:
            continue
        candidates["abs_token_delta"] = candidates.target_minus_s0_completion_tokens.abs()
        candidates = candidates.sort_values(["abs_token_delta", "task"], ascending=[False, True])
        chosen = []
        used_tasks = set()
        for row in candidates.itertuples():
            if row.task not in used_tasks:
                chosen.append(row)
                used_tasks.add(row.task)
                if len(chosen) == per_group:
                    break
        if len(chosen) < per_group:
            for row in candidates.itertuples():
                if row not in chosen:
                    chosen.append(row)
                    if len(chosen) == per_group:
                        break
        for row in chosen:
            cases.append({
                "case_index": len(cases) + 1,
                "target": target,
                "category": category,
                "sample_id": row.sample_id,
                "task": row.task,
                "selection": "large_absolute_token_change",
            })
    (args.output_dir / "rejudged_case_manifest.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target_summaries": target_summaries, "s2_evidence": s2_decision, "case_count": len(cases)}, ensure_ascii=False))


def wide_token(frame: pd.DataFrame, sample_id: str, condition: str):
    row = frame[(frame.sample_id == sample_id) & (frame.condition == condition)]
    return row.iloc[0]["completion_tokens"]


def row_value(frame: pd.DataFrame, sample_id: str, condition: str, column: str):
    row = frame[(frame.sample_id == sample_id) & (frame.condition == condition)]
    return row.iloc[0][column]


if __name__ == "__main__":
    main()
