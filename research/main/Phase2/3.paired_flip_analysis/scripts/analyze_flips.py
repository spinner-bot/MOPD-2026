#!/usr/bin/env python3
"""Build paired S0/S1/S3 flip tables, figures, and a readable HTML audit report."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu


CONDITIONS = ["S0", "S1", "S3"]
TARGETS = ["S1", "S3"]
COLORS = {"S1": "#f58518", "S3": "#54a24b"}


def bool_value(value) -> bool:
    return str(value).lower() == "true"


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    derived = args.source_root / "derived/run-60011513"
    outputs = args.source_root / "runs/60011513/outputs"
    input_json = json.loads((args.source_root / "raw/inference_input.json").read_text(encoding="utf-8"))
    sample_by_id = {row["sample_id"]: row for row in input_json["records"]}
    per = pd.read_csv(derived / "per_sample.csv")
    tokens = pd.read_csv(derived / "token_per_sample.csv")
    token_cols = ["sample_id", "condition", "prompt_tokens", "completion_tokens", "total_tokens"]
    per = per.merge(tokens[token_cols], on=["sample_id", "condition"], how="inner", validate="one_to_one")
    per = per[per.condition.isin(CONDITIONS)].copy()
    per["correct_bool"] = per["correct"].map(bool_value)
    if set(per.condition) != set(CONDITIONS) or any(len(per[per.condition == c]) != 2000 for c in CONDITIONS):
        raise ValueError("S0/S1/S3 completeness gate failed")
    if per.duplicated(["sample_id", "condition"]).any():
        raise ValueError("duplicate sample-condition pair")
    wide = per.pivot(index="sample_id", columns="condition")
    records = []
    raw = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        path = outputs / f"{condition}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                raw[condition][row["sample_id"]] = row
    for sample_id in wide.index:
        s0 = bool(wide.loc[sample_id, ("correct_bool", "S0")])
        item = sample_by_id[sample_id]
        for target in TARGETS:
            tc = bool(wide.loc[sample_id, ("correct_bool", target)])
            if not s0 and tc: category = "gain"
            elif s0 and not tc: category = "loss"
            elif s0 and tc: category = "stable_correct"
            else: category = "stable_wrong"
            rec = {
                "sample_id": sample_id, "target": target, "category": category,
                "s0_correct": s0, "target_correct": tc,
                "split": item.get("split"), "task": item.get("task"), "source": item.get("source"),
                "question": item.get("question", item.get("query", "")),
                "query": item.get("query", ""), "choices": item.get("choices", []),
                "gold_answer": item.get("response"),
            }
            for condition in CONDITIONS:
                rec[f"{condition}_response_chars"] = int(wide.loc[sample_id, ("response_chars", condition)])
                rec[f"{condition}_completion_tokens"] = int(wide.loc[sample_id, ("completion_tokens", condition)])
                rec[f"{condition}_parsed_answer"] = wide.loc[sample_id, ("parsed_answer", condition)]
            rec["target_minus_s0_completion_tokens"] = rec[f"{target}_completion_tokens"] - rec["S0_completion_tokens"]
            rec["raw_s0"] = raw.get("S0", {}).get(sample_id, {}).get("response", "")
            rec["raw_target"] = raw[target][sample_id].get("response", "")
            records.append(rec)
    flip = pd.DataFrame(records)
    args.work_root.joinpath("derived/run-60011513").mkdir(parents=True, exist_ok=True)
    args.work_root.joinpath("figures/run-60011513").mkdir(parents=True, exist_ok=True)
    flip.to_csv(args.work_root / "derived/run-60011513/flip_cases.csv", index=False, encoding="utf-8")
    with (args.work_root / "derived/run-60011513/flip_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cross = wide["correct_bool"].astype(bool)
    cross_summary = {
        "common_gain": int((~cross.S0 & cross.S1 & cross.S3).sum()),
        "either_gain": int((~cross.S0 & (cross.S1 | cross.S3)).sum()),
        "common_loss": int((cross.S0 & ~cross.S1 & ~cross.S3).sum()),
        "either_loss": int((cross.S0 & (~cross.S1 | ~cross.S3)).sum()),
        "s1_only_gain": int((~cross.S0 & cross.S1 & ~cross.S3).sum()),
        "s3_only_gain": int((~cross.S0 & ~cross.S1 & cross.S3).sum()),
        "s1_only_loss": int((cross.S0 & ~cross.S1 & cross.S3).sum()),
        "s3_only_loss": int((cross.S0 & cross.S1 & ~cross.S3).sum()),
        "disagreement": int(((cross.S1 != cross.S3)).sum()),
    }
    summaries = []
    for target in TARGETS:
        x = flip[flip.target == target]
        gain = x[x.category == "gain"]
        loss = x[x.category == "loss"]
        discordant = len(gain) + len(loss)
        summaries.append({
            "target": target, "n": int(len(x)), "gain_n": int(len(gain)), "loss_n": int(len(loss)),
            "gain_rate_of_all": float(len(gain) / len(x)), "loss_rate_of_all": float(len(loss) / len(x)),
            "net_gain": int(len(gain) - len(loss)),
            "gain_ci95": [float(v) for v in binomtest(len(gain), len(x)).proportion_ci().__dict__.values()] if False else None,
            "gain_mean_token_delta": float(gain.target_minus_s0_completion_tokens.mean()),
            "loss_mean_token_delta": float(loss.target_minus_s0_completion_tokens.mean()),
            "gain_median_token_delta": float(gain.target_minus_s0_completion_tokens.median()),
            "loss_median_token_delta": float(loss.target_minus_s0_completion_tokens.median()),
            "gain_vs_loss_mann_whitney_p": float(mannwhitneyu(gain.target_minus_s0_completion_tokens, loss.target_minus_s0_completion_tokens).pvalue),
            "discordant_n": int(discordant),
        })
    task_rows = []
    for target in TARGETS:
        x = flip[flip.target == target]
        for category in ["gain", "loss"]:
            y = x[x.category == category]
            for (task, source, split), group in y.groupby(["task", "source", "split"], dropna=False):
                task_rows.append({"target": target, "category": category, "task": task, "source": source, "split": split, "n": int(len(group)), "mean_token_delta": float(group.target_minus_s0_completion_tokens.mean())})
    summary = {"schema": "paired-flip-analysis-v1", "included_conditions": CONDITIONS, "excluded_conditions": ["SC", "S2"], "n_samples": 2000, "targets": summaries, "cross": cross_summary, "task_rows": task_rows}
    (args.work_root / "derived/run-60011513/flip_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Figure 1: gain/loss counts.
    fig, ax = plt.subplots(figsize=(7.5, 4.7)); x = np.arange(2); width = .35
    gains = [next(s["gain_n"] for s in summaries if s["target"] == c) for c in TARGETS]
    losses = [next(s["loss_n"] for s in summaries if s["target"] == c) for c in TARGETS]
    ax.bar(x-width/2, gains, width, label="S0 wrong → target correct", color="#54a24b")
    ax.bar(x+width/2, losses, width, label="S0 correct → target wrong", color="#e45756")
    ax.set_xticks(x, TARGETS); ax.set_ylabel("Number of paired samples"); ax.set_title("Correctness flips relative to S0")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(args.work_root / "figures/run-60011513/flip_counts.png", dpi=180); plt.close(fig)

    # Figure 2: task/source distribution for gains and losses.
    dist = flip[flip.category.isin(["gain", "loss"])].groupby(["task", "category", "target"]).size().unstack(fill_value=0)
    plot_data = flip[flip.category.isin(["gain", "loss"])].groupby(["task", "category"]).size().unstack(fill_value=0)
    plot_data = plot_data.sort_values(by=["gain", "loss"], ascending=False).head(12)
    ax = plot_data.plot(kind="bar", figsize=(10.5, 5.3), color=["#54a24b", "#e45756"])
    ax.set_xlabel("Task (top categories by flip count)"); ax.set_ylabel("Number of cases"); ax.set_title("Flip cases by task type")
    ax.tick_params(axis="x", rotation=35); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(args.work_root / "figures/run-60011513/flip_task_distribution.png", dpi=180); plt.close(ax.figure)

    # Figure 3: target-minus-baseline completion-token changes for flip directions.
    fig, ax = plt.subplots(figsize=(8.2, 5.0)); data = []
    for target in TARGETS:
        for category in ["gain", "loss"]:
            data.append(flip[(flip.target == target) & (flip.category == category)].target_minus_s0_completion_tokens)
    bp = ax.boxplot(data, patch_artist=True, tick_labels=["S1 gain", "S1 loss", "S3 gain", "S3 loss"], showfliers=False)
    for box, color in zip(bp["boxes"], ["#54a24b", "#e45756", "#54a24b", "#e45756"]): box.set_facecolor(color); box.set_alpha(.75)
    ax.axhline(0, color="#555", linewidth=1); ax.set_ylabel("Target − S0 completion tokens"); ax.set_title("Output-token change among flip cases")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(args.work_root / "figures/run-60011513/flip_token_delta.png", dpi=180); plt.close(fig)

    # Figure 4: common vs condition-specific flips.
    labels = ["common gain", "S1-only gain", "S3-only gain", "common loss", "S1-only loss", "S3-only loss"]
    values = [cross_summary[k] for k in ["common_gain", "s1_only_gain", "s3_only_gain", "common_loss", "s1_only_loss", "s3_only_loss"]]
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(labels, values, color=["#2f855a", "#f6ad55", "#68d391", "#c53030", "#fc8181", "#feb2b2"])
    ax.set_ylabel("Number of samples"); ax.set_title("Common and condition-specific S1/S3 flips"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(args.work_root / "figures/run-60011513/flip_overlap.png", dpi=180); plt.close(fig)

    # A readable report: all flip IDs in tables, plus representative raw-output cases.
    report_dir = args.work_root / "report"; report_dir.mkdir(parents=True, exist_ok=True)
    representative = flip[flip.category.isin(["gain", "loss"])].copy()
    representative["abs_delta"] = representative.target_minus_s0_completion_tokens.abs()
    representative = representative.sort_values(["category", "abs_delta"], ascending=[True, False]).groupby(["target", "category"], sort=False).head(8)
    tables = []
    for target in TARGETS:
        x = flip[flip.target == target]
        rows = "".join(f"<tr><td>{html.escape(str(r.sample_id))}</td><td>{r.category}</td><td>{html.escape(str(r.task))}</td><td>{html.escape(str(r.source))}</td><td>{r.S0_completion_tokens}</td><td>{getattr(r, target+'_completion_tokens')}</td><td>{r.target_minus_s0_completion_tokens:+d}</td><td>{html.escape(str(r.S0_parsed_answer))}</td><td>{html.escape(str(getattr(r, target+'_parsed_answer')))}</td></tr>" for r in x[x.category.isin(["gain","loss"])].itertuples())
        tables.append(f"<h3>{target} relative to S0</h3><table><thead><tr><th>sample_id</th><th>flip</th><th>task</th><th>source</th><th>S0 tokens</th><th>{target} tokens</th><th>delta</th><th>S0 answer</th><th>{target} answer</th></tr></thead><tbody>{rows}</tbody></table>")
    cards = []
    for r in representative.itertuples():
        cards.append(f"<details><summary>{html.escape(str(r.target))} / {html.escape(str(r.category))} / {html.escape(str(r.sample_id))} / {html.escape(str(r.task))}</summary><p><b>Question / 问题:</b> {html.escape(str(r.question))}</p><p><b>Gold:</b> {html.escape(str(r.gold_answer))}</p><h4>S0 raw output</h4><pre>{html.escape(str(r.raw_s0))}</pre><h4>{r.target} raw output</h4><pre>{html.escape(str(r.raw_target))}</pre></details>")
    figs = "".join(f'<figure><img src="{image_uri(args.work_root / "figures/run-60011513" / name)}"><figcaption>{title}</figcaption></figure>' for name, title in [("flip_counts.png", "Flip counts"), ("flip_task_distribution.png", "Flip cases by task"), ("flip_token_delta.png", "Token delta"), ("flip_overlap.png", "Common/specific flips")])
    report = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>S0–S1/S3 Paired Flip Analysis</title><style>body{{font-family:Arial,Microsoft YaHei,sans-serif;max-width:1400px;margin:0 auto;padding:32px;background:#fff;color:#243447;line-height:1.6}}h1,h2{{color:#18324b}}h2{{margin-top:2.6rem;border-left:5px solid #2f6f9f;padding:.5rem .8rem;background:#f7fafc}}h3{{margin-top:2rem;color:#285878}}table{{border-collapse:collapse;width:100%;font-size:.86rem;margin:1rem 0 2rem;display:block;overflow-x:auto}}th,td{{border:1px solid #d7e0e8;padding:.5rem;vertical-align:top;white-space:nowrap}}th{{background:#eaf1f7}}.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}}figure{{margin:0;border:1px solid #d7e0e8;padding:.7rem}}figure img{{width:100%}}figcaption{{text-align:center;font-weight:bold}}details{{margin:1rem 0;padding:.8rem 1rem;border:1px solid #d7e0e8;border-radius:7px;background:#fafcfd}}summary{{cursor:pointer;font-weight:bold}}pre{{white-space:pre-wrap;word-break:break-word;background:#f5f7f9;padding:.8rem;border-left:3px solid #4c78a8}}</style></head><body><h1>S0–S1/S3 配对翻转分析 / Paired flip analysis</h1><p>本分析只包含 S0、S1、S3。SC 和 S2 不进入统计、图表或结论。S0 是基线，S1/S3 是目标条件。</p><h2>摘要 / Summary</h2><ul><li>S1: gain {next(s['gain_n'] for s in summaries if s['target']=='S1')}，loss {next(s['loss_n'] for s in summaries if s['target']=='S1')}，净翻转 {next(s['net_gain'] for s in summaries if s['target']=='S1')}。</li><li>S3: gain {next(s['gain_n'] for s in summaries if s['target']=='S3')}，loss {next(s['loss_n'] for s in summaries if s['target']=='S3')}，净翻转 {next(s['net_gain'] for s in summaries if s['target']=='S3')}。</li><li>共同 gain {cross_summary['common_gain']}，共同 loss {cross_summary['common_loss']}；S1-only gain {cross_summary['s1_only_gain']}，S3-only gain {cross_summary['s3_only_gain']}。</li></ul><div class="figures">{figs}</div><h2>全部翻转题目 / All flip cases</h2>{''.join(tables)}<h2>代表性原始回答 / Representative raw responses</h2>{''.join(cards)}</body></html>"""
    (report_dir / "paired-flip-analysis-60011513.html").write_text(report, encoding="utf-8")
    print(json.dumps({"flip_cases": len(flip[flip.category.isin(['gain','loss'])]), "summary": str(args.work_root / 'derived/run-60011513/flip_summary.json'), "report": str(report_dir / 'paired-flip-analysis-60011513.html')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
