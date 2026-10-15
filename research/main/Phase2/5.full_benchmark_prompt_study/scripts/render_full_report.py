#!/usr/bin/env python3
"""Render a self-contained bilingual HTML report from verified Work5 tables."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("S0", "SC", "S1", "S2", "S3")
PROMPTS = ("SC", "S1", "S2", "S3")
TRACKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
SOURCES = ("native", "extraction")
SOURCE_LABELS = {"native": "Vision-OPD native scorer",
                 "extraction": "Qwen text-only extraction"}
TRACK_LABELS = {
    "vstar": "V*", "zoombench": "ZoomBench", "hrbench-4k": "HRBench-4K",
    "hrbench-8k": "HRBench-8K", "mme-realworld": "MME-RealWorld",
    "mme-realworld-cn": "MME-RealWorld-CN", "all_tracks": "All benchmarks",
}
TABLES = ("condition_summary", "paired_comparisons", "strata_summary", "score_disagreement",
          "judge_source", "native_judge_decision_audit", "extraction_status",
          "length_accuracy_association", "response_length_by_score")
COLORS = {"SC": "#8c8c8c", "S1": "#2474b5", "S2": "#e39c27", "S3": "#c94f58"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_analysis(analysis_dir: Path) -> tuple[dict, dict[str, list[dict[str, str]]]]:
    manifest_path = analysis_dir / "analysis-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase2-work5-statistics-v2" or manifest.get("joined_records") != 160810:
        raise ValueError("Analysis manifest is missing or does not certify all 160,810 joined records")
    if tuple(manifest.get("conditions", ())) != CONDITIONS or tuple(manifest.get("tracks", ())) != TRACKS:
        raise ValueError("Analysis manifest conditions/tracks do not match the frozen Work5 design")
    tables = {}
    for name in TABLES:
        path = analysis_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required aggregate table is missing: {path}")
        tables[name] = read_csv(path)
    summaries = tables["condition_summary"]
    for source in SOURCES:
        for condition in CONDITIONS:
            for track in TRACKS:
                if not any(r.get("scope") == "track" and r.get("condition") == condition
                           and r.get("stratum") == track and r.get(f"{source}_accuracy", "") != ""
                           for r in summaries):
                    raise ValueError(f"Missing {source} summary for {condition}/{track}")
    expected_pairs = 2 * 7 * 10
    if len(tables["paired_comparisons"]) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} paired comparison rows, got {len(tables['paired_comparisons'])}")
    return manifest, tables


def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def accuracy_heatmaps(summaries: list[dict[str, str]]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, source in zip(axes, SOURCES):
        matrix = np.asarray([[float(next(r[f"{source}_accuracy"] for r in summaries
                                          if r.get("scope") == "track" and r.get("condition") == c
                                          and r.get("stratum") == t)) for c in CONDITIONS]
                             for t in TRACKS])
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS)
        ax.set_yticks(range(len(TRACKS)), [TRACK_LABELS[t] for t in TRACKS])
        ax.set_title(SOURCE_LABELS[source], fontsize=10, pad=10)
        ax.set_xlabel("Prompt condition")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, f"{matrix[y, x]:.1%}", ha="center", va="center",
                        color="white" if matrix[y, x] > 0.62 else "#20242a", fontsize=8)
    axes[0].set_ylabel("Evaluation benchmark")
    fig.colorbar(image, ax=axes, shrink=0.78, label="Accuracy")
    fig.suptitle("Absolute accuracy by benchmark and condition", fontsize=13)
    fig.subplots_adjust(left=0.15, right=0.87, bottom=0.12, top=0.84, wspace=0.16)
    return _png(fig)


def paired_effect_forest(rows: list[dict[str, str]]) -> str:
    scopes = [("micro_all", "all_tracks")] + [("track", t) for t in TRACKS]
    fig, axes = plt.subplots(len(scopes), 2, figsize=(13, 17), sharex=True)
    for row_index, (scope, track) in enumerate(scopes):
        for col_index, source in enumerate(SOURCES):
            ax = axes[row_index, col_index]
            selected = [r for r in rows if r["score_source"] == source and r["scope"] == scope
                        and r["stratum"] == track and r["condition_a"] == "S0"
                        and r["condition_b"] in PROMPTS]
            for y, condition in enumerate(PROMPTS):
                record = next((r for r in selected if r["condition_b"] == condition), None)
                if record is None:
                    continue
                estimate = float(record["paired_delta_percentage_points"])
                low, high = float(record["cluster_bootstrap_95ci_low_pp"]), float(record["cluster_bootstrap_95ci_high_pp"])
                ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt="o",
                            color=COLORS[condition], capsize=2, markersize=4)
            ax.axvline(0, color="#535b65", linewidth=0.8, linestyle="--")
            ax.set_yticks(range(len(PROMPTS)), PROMPTS)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title(f"{TRACK_LABELS[track]} · {source}", fontsize=9)
            if col_index == 0:
                ax.set_ylabel("Prompt vs S0")
            if row_index == len(scopes) - 1:
                ax.set_xlabel("Paired accuracy difference (percentage points)")
    fig.suptitle("Paired prompt effects relative to S0 with cluster-bootstrap 95% CI", fontsize=13)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.06, top=0.95, hspace=0.62, wspace=0.24)
    return _png(fig)


def disagreement_heatmap(rows: list[dict[str, str]]) -> str:
    states = ((1, 1), (1, 0), (0, 1), (0, 0))
    matrix = np.zeros((len(CONDITIONS), len(states)), dtype=int)
    for row in rows:
        c = CONDITIONS.index(row["condition"])
        state = (int(row["native_correct"]), int(row["extraction_correct"]))
        matrix[c, states.index(state)] += int(row["n"])
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 4.7))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    labels = ("Both correct", "Native only", "Extraction only", "Both wrong")
    ax.set_xticks(range(4), labels, rotation=12, ha="right")
    ax.set_yticks(range(5), CONDITIONS)
    for y in range(5):
        for x in range(4):
            ax.text(x, y, f"{shares[y, x]:.1%}\n(n={matrix[y, x]:,})", ha="center", va="center",
                    fontsize=8, color="white" if shares[y, x] > 0.55 else "#20242a")
    ax.set_title("Agreement and disagreement between two score sources")
    fig.colorbar(image, ax=ax, label="Within-condition share")
    fig.tight_layout()
    return _png(fig)


def association_heatmaps(rows: list[dict[str, str]]) -> str:
    metrics = ("response_chars", "response_words", "completion_tokens", "total_api_tokens")
    titles = ("Response characters", "Response words", "Completion tokens", "Total API tokens")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for r, source in enumerate(SOURCES):
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[r, col]
            selected = [x for x in rows if x["scope"] == "micro_all" and x["score_source"] == source
                        and x["length_metric"] == metric]
            values = {x["condition"]: float(x["spearman_rho"]) for x in selected if x.get("spearman_rho") not in ("", "None")}
            matrix = np.asarray([[values.get(c, np.nan) for c in CONDITIONS]])
            image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(5), CONDITIONS)
            ax.set_yticks([0], [source])
            ax.set_title(title, fontsize=9)
            for x, c in enumerate(CONDITIONS):
                val = matrix[0, x]
                ax.text(x, 0, "NA" if np.isnan(val) else f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.suptitle("Descriptive Spearman association of response length with correctness", fontsize=13)
    fig.colorbar(image, ax=axes, shrink=0.75, label="Spearman rho")
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.12, top=0.84, hspace=0.48, wspace=0.30)
    return _png(fig)


def token_medians(rows: list[dict[str, str]], source: str) -> str:
    subset = [r for r in rows if r["score_source"] == source and r["metric"] == "completion_tokens"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    x = np.arange(len(CONDITIONS))
    width = 0.34
    for ax, track in zip(axes.flat, TRACKS):
        for offset, correct in enumerate((0, 1)):
            medians = []
            for condition in CONDITIONS:
                row = next((r for r in subset if r["track"] == track and r["condition"] == condition
                            and r["correct"] == str(correct)), None)
                medians.append(float(row["median"]) if row and row.get("median") else 0)
            ax.bar(x + (offset - 0.5) * width, medians, width,
                   label="Incorrect" if correct == 0 else "Correct",
                   color="#9098a1" if correct == 0 else "#3178b5")
        ax.set_title(TRACK_LABELS[track], fontsize=10)
        ax.set_xticks(x, CONDITIONS)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Prompt condition")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median completion tokens")
    fig.suptitle(f"Completion-token medians by correctness · {source}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _png(fig)


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def format_value(value: str, key: str = "") -> str:
    if value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if key.endswith("_accuracy") or key in {"native_accuracy", "extraction_accuracy", "judge_extraction_agreement",
                                             "native_extraction_agreement", "native_judge_exact_binary_rate"}:
        return f"{number:.1%}"
    if "percentage_points" in key or key.endswith("_pp"):
        return f"{number:+.2f}"
    if key.endswith("_p") or key in {"mcnemar_exact_p", "mcnemar_holm_p"}:
        return f"{number:.4g}"
    if key == "spearman_rho":
        return f"{number:.3f}"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:.3f}"


def html_table(rows: list[dict], fields: list[str], titles: dict[str, str] | None = None) -> str:
    if not rows:
        return '<p class="muted">No rows / 无记录。</p>'
    titles = titles or {}
    header = "".join(f"<th>{esc(titles.get(field, field))}</th>" for field in fields)
    body = []
    for row in rows:
        cells = "".join(f"<td>{format_value(row.get(field, ''), field)}</td>" for field in fields)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def build_html(manifest: dict, tables: dict[str, list[dict[str, str]]], charts: dict[str, str]) -> str:
    summaries = tables["condition_summary"]
    overall = [r for r in summaries if r.get("scope") != "track"]
    paired = tables["paired_comparisons"]
    primary = [r for r in paired if r["scope"] == "micro_all" and r["condition_a"] == "S0"]
    source_files = manifest.get("audit", {}).get("source_files", [])
    attempt_audit = manifest.get("audit", {}).get("extraction_attempt_status_counts", [])
    figure_specs = [
        ("absolute", "Absolute accuracy / 各条件准确率", "Accuracy is shown separately by benchmark and score source; the heatmap does not imply cross-benchmark equivalence. / 分评测集与评分来源展示准确率；热图不表示不同评测集可直接等价比较。"),
        ("paired", "Paired effects / 配对效应", "Each estimate compares the same samples under S0 and the named condition. Intervals use image-cluster bootstrap; tests are paired exact McNemar with Holm adjustment. / 每个估计比较S0与相应条件下的同一批样本；区间采用图像聚类Bootstrap，检验为精确配对McNemar并进行Holm校正。"),
        ("disagreement", "Score-source disagreement / 评分来源分歧", "Cell values are within-condition proportions and counts; disagreement is descriptive and is not resolved by treating either source as ground truth. / 单元格展示条件内占比与数量；该分歧分析为描述性结果，不预设任何一方为真值。"),
        ("association", "Length–accuracy association / 长度与准确率关联", "Spearman ρ is descriptive; it is not a causal estimate. / Spearman ρ为描述性关联，不作因果解释。"),
        ("tokens_native", "Output length by native score / 按原生评分分组的输出长度", "Bars show medians only; group differences are descriptive. / 柱形仅表示中位数，组间差异仅作描述。"),
        ("tokens_extraction", "Output length by extracted answer / 按抽取答案分组的输出长度", "Bars show medians only; group differences are descriptive. / 柱形仅表示中位数，组间差异仅作描述。"),
    ]
    figures = "".join(
        f'<figure id="fig-{esc(key)}"><a class="zoom" href="data:image/png;base64,{charts[key]}" target="_blank" rel="noopener"><img alt="{esc(title)}" src="data:image/png;base64,{charts[key]}"></a><figcaption><b>{esc(title)}</b><br>{esc(caption)}</figcaption></figure>'
        for key, title, caption in figure_specs
    )
    summary_fields = ["condition", "n", "native_accuracy", "extraction_accuracy", "macro_track_native_accuracy",
                      "macro_track_extraction_accuracy", "native_extraction_agreement", "native_judge_exact_binary_rate",
                      "completion_tokens_mean", "completion_tokens_median", "completion_tokens_coverage"]
    primary_fields = ["score_source", "condition_b", "n_paired", "b_correct_a_wrong", "a_correct_b_wrong",
                      "paired_delta_percentage_points", "cluster_bootstrap_95ci_low_pp", "cluster_bootstrap_95ci_high_pp",
                      "mcnemar_exact_p", "mcnemar_holm_p"]
    paired_fields = ["score_source", "scope", "stratum", "condition_a", "condition_b", "n_paired",
                     "b_correct_a_wrong", "a_correct_b_wrong", "paired_delta_percentage_points",
                     "cluster_bootstrap_95ci_low_pp", "cluster_bootstrap_95ci_high_pp", "mcnemar_exact_p", "mcnemar_holm_p"]
    paired_titles = {"score_source":"Score source / 评分来源", "scope":"Analysis scope / 分析范围",
                     "stratum":"Benchmark stratum / 评测集分层", "condition_a":"Reference condition / 对照条件",
                     "condition_b":"Comparison condition / 比较条件", "n_paired":"Paired N / 配对样本数",
                     "b_correct_a_wrong":"B correct, A wrong / B对A错", "a_correct_b_wrong":"A correct, B wrong / A对B错",
                     "paired_delta_percentage_points":"Accuracy delta (pp) / 准确率差（百分点）",
                     "cluster_bootstrap_95ci_low_pp":"Cluster-bootstrap 95% CI low / 聚类Bootstrap区间下限",
                     "cluster_bootstrap_95ci_high_pp":"Cluster-bootstrap 95% CI high / 聚类Bootstrap区间上限",
                     "mcnemar_exact_p":"Exact McNemar p / 精确McNemar p值",
                     "mcnemar_holm_p":"Holm-adjusted p / Holm校正p值"}
    strata = tables["strata_summary"]
    strata_fields = ["stratum_type", "track", "stratum", "condition", "n", "native_correct", "native_accuracy",
                     "extraction_correct", "extraction_accuracy"]
    disagree_fields = ["condition", "track", "native_correct", "extraction_correct", "n"]
    source_rows = [{"role": x.get("role"), "condition": x.get("condition", ""), "track": x.get("track", ""),
                    "path": x.get("path"), "sha256": x.get("sha256")} for x in source_files]
    source_fields = ["role", "condition", "track", "path", "sha256"]
    source_titles = {"role":"Input role / 输入用途", "condition":"Condition / 条件",
                     "track":"Benchmark / 评测集", "path":"Source path / 来源路径", "sha256":"SHA-256"}
    audit_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Five-condition full-benchmark evaluation / 五条件全量评测</title>
<style>
:root{{--ink:#182230;--muted:#596579;--line:#d8dee8;--paper:#fff;--wash:#f4f7fb;--blue:#215a91}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:42px max(24px,calc((100vw - 1180px)/2));background:linear-gradient(120deg,#153e68,#347baa);color:#fff}}
header h1{{font-size:clamp(1.8rem,4vw,2.8rem);line-height:1.2;margin:0 0 12px}}header p{{margin:6px 0;max-width:950px}}
main{{max-width:1180px;margin:0 auto;padding:24px}}nav{{background:var(--wash);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:12px 0 30px}}
nav ol{{columns:2;max-width:900px}}a{{color:var(--blue)}}section{{margin:42px 0;scroll-margin-top:20px}}h2{{border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:12px}}h3{{margin-top:28px}}
.meta,.callout{{background:var(--wash);border-left:4px solid #397eac;padding:14px 18px;border-radius:4px;margin:18px 0}}.muted{{color:var(--muted)}}
figure{{margin:26px 0 38px;padding:16px;border:1px solid var(--line);border-radius:12px;background:#fff}}figure img{{display:block;width:100%;height:auto;max-height:900px;object-fit:contain;cursor:zoom-in}}figcaption{{color:var(--muted);font-size:.93rem;padding:12px 4px 0}}
.table-wrap{{overflow:auto;max-height:650px;border:1px solid var(--line);border-radius:8px;margin:12px 0}}table{{border-collapse:collapse;width:100%;font-size:.9rem;white-space:nowrap}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}}th{{position:sticky;top:0;background:#eaf0f6;z-index:1}}tbody tr:nth-child(even){{background:#f8fafc}}
details{{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:16px 0}}summary{{cursor:pointer;font-weight:650}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:14px;border-radius:8px;font-size:.82rem}}
.badge{{display:inline-block;border:1px solid #c8d6e6;border-radius:20px;padding:2px 10px;margin:3px;color:#234d70;font-size:.84rem}}footer{{border-top:1px solid var(--line);padding:24px;color:var(--muted)}}
@media(max-width:700px){{main{{padding:16px}}nav ol{{columns:1}}figure{{padding:8px}}}}
</style></head><body>
<header><h1>Five-condition full-benchmark evaluation<br>五条件全量评测与分层分析</h1>
<p>Vision-OPD evaluation protocol · six benchmark tracks · five prompt conditions / Vision-OPD评测协议 · 六个评测集 · 五种提示条件</p>
<p>Version {esc(manifest.get('schema'))} · {int(manifest.get('joined_records', 0)):,} paired condition–sample records / 配对条件—样本记录</p></header>
<main><nav><b>Contents / 目录</b><ol>
<li><a href="#scope">Scope and protocol / 范围与协议</a></li><li><a href="#overview">Condition overview / 条件总览</a></li>
<li><a href="#figures">Visual analyses / 可视化分析</a></li><li><a href="#paired">Paired tests / 配对检验</a></li>
<li><a href="#strata">Stratified results / 分层结果</a></li><li><a href="#judge">Scorer audit / 评分器审计</a></li>
<li><a href="#methods">Methods and reproducibility / 方法与复现</a></li></ol></nav>
<section id="scope"><h2>Scope and protocol / 范围与评测协议</h2>
<p>This report covers S0, SC, S1, S2 and S3 across V*, ZoomBench, HRBench-4K, HRBench-8K, MME-RealWorld and MME-RealWorld-CN. The same benchmark samples are paired across conditions. / 本报告覆盖S0、SC、S1、S2、S3五种条件及V*、ZoomBench、HRBench-4K、HRBench-8K、MME-RealWorld和MME-RealWorld-CN六个评测集；同一评测集样本在各条件间配对比较。</p>
<p>The report keeps two answer-scoring sources separate: the unmodified Vision-OPD native evaluator and a local Qwen structured, text-only answer extractor. Their agreement and disagreement are shown explicitly. Neither source is silently substituted for the other. / 报告分别保留Vision-OPD未修改的原生评测器与本地Qwen结构化纯文本答案抽取器两种评分来源，并显式展示一致与分歧，不将其中一方静默替代另一方。</p>
<p>For the native evaluator, only the exact final nonempty line “Yes” or “No” is treated as its binary decision; the preceding reasoning text is not parsed as a label. Full raw judge text is preserved in the machine-readable merged audit table, while this report summarizes terminal decisions. / 对原生评测器，仅将最终一个非空行严格等于“Yes”或“No”的文本作为二元判定，不从前面的推理文本中抽取标签。完整原始裁判文本保留在机器可读的逐题合并审计表中，本报告仅汇总最终判定。</p>
<div class="meta">Verified joined records / 已验证合并记录：<b>{int(manifest['joined_records']):,}</b> · Conditions / 条件：{', '.join(CONDITIONS)} · Tracks / 评测集：{', '.join(TRACK_LABELS[t] for t in TRACKS)}</div></section>
<section id="overview"><h2>Condition overview / 条件总览</h2>
<p>Micro accuracy weights every sample equally across the full pooled set; macro-track accuracy gives each benchmark equal weight. Both are reported because benchmark sizes differ substantially. / Micro准确率对合并后的每个样本等权；Macro-track准确率对每个评测集等权。由于评测集规模差异较大，两者并列报告。</p>
{html_table(overall, summary_fields, {"condition":"Condition / 条件","n":"N","native_accuracy":"Native accuracy / 原生准确率","extraction_accuracy":"Extraction accuracy / 抽取准确率","macro_track_native_accuracy":"Macro-track native / 原生宏平均","macro_track_extraction_accuracy":"Macro-track extraction / 抽取宏平均","native_extraction_agreement":"Judge agreement / 评分一致率","native_judge_exact_binary_rate":"Native exact-binary rate / 原生二分类率","completion_tokens_mean":"Mean output tokens / 输出Token均值","completion_tokens_median":"Median output tokens / 输出Token中位数","completion_tokens_coverage":"Token coverage / Token覆盖率"})}</section>
<section id="figures"><h2>Visual analyses / 可视化分析</h2>{figures}</section>
<section id="paired"><h2>Paired comparisons / 配对比较</h2>
<p>Primary contrast: each prompted condition versus S0, using the same sample keys. “B correct, A wrong” and “A correct, B wrong” are discordant pairs for the exact McNemar test. Confidence intervals resample image clusters. / 主要对比为每种提示条件与S0在相同样本键上的配对差异。“B答对、A答错”与“A答对、B答错”构成精确McNemar检验的非一致配对；置信区间按图像簇重抽样。</p>
{html_table(primary, primary_fields, {"score_source":"Score source / 评分来源","condition_b":"Prompt condition / 提示条件","n_paired":"Paired N / 配对样本数","b_correct_a_wrong":"Prompt correct, S0 wrong / 提示对、S0错","a_correct_b_wrong":"S0 correct, prompt wrong / S0对、提示错","paired_delta_percentage_points":"Accuracy delta (pp) / 准确率差（百分点）","cluster_bootstrap_95ci_low_pp":"95% CI low / 区间下限","cluster_bootstrap_95ci_high_pp":"95% CI high / 区间上限","mcnemar_exact_p":"Exact p / 精确p值","mcnemar_holm_p":"Holm-adjusted p / Holm校正p值"})}
<details><summary>All paired comparisons (all tracks and contrasts) / 展开全部配对检验（所有评测集与对比）</summary>
{html_table(paired, paired_fields, paired_titles)}</details></section>
<section id="strata"><h2>Stratified results / 分层结果</h2>
<p>Strata retain benchmark-specific categories, subcategories, question type, language and answer format. Small strata should be read as descriptive estimates; the table reports their denominators. / 分层保留评测集内类别、子类别、题型、语言和答案格式；小样本分层应视为描述性估计，并结合分母解读。</p>
<details><summary>Complete stratified summary / 展开完整分层汇总（{len(strata):,} rows / 行）</summary>
{html_table(strata, strata_fields, {"stratum_type":"Dimension / 维度","track":"Benchmark / 评测集","stratum":"Category / 类别","condition":"Condition / 条件","n":"N","native_correct":"Native correct / 原生答对数","native_accuracy":"Native accuracy / 原生准确率","extraction_correct":"Extraction correct / 抽取答对数","extraction_accuracy":"Extraction accuracy / 抽取准确率"})}</details></section>
<section id="judge"><h2>Scorer audit / 评分器审计</h2>
<p>The embedded disagreement chart shows all four correctness states by condition. Detailed judge-source, raw native label, and extraction-status counts remain available below for audit. / 内嵌分歧图按条件展示四种正误组合；下方保留评分来源、原生标签值及答案抽取状态计数，供审计追溯。</p>
<p>Attempt-level counts include invalid first attempts even when later retries succeeded; the table reports only status and error class, not raw model answers. / 尝试级计数保留后来重试成功的首次无效结果；表格仅汇总状态和错误类别，不展示模型原始回答。</p>
<details><summary>All answer-extraction attempts / 全部答案抽取尝试（含已纠正尝试）</summary>{html_table(attempt_audit, ['condition','track','request_status','error_class','judge_prompt_sha256','attempts'], {'condition':'Condition / 条件','track':'Benchmark / 评测集','request_status':'Request status / 请求状态','error_class':'Error class / 错误类别','judge_prompt_sha256':'Judge prompt SHA-256 / 判别提示词哈希','attempts':'Attempt count / 次数'})}</details>
<details><summary>Native terminal decisions / 原生最终判定计数</summary>{html_table(tables['native_judge_decision_audit'], ['condition','track','terminal_decision','n'], {'condition':'Condition / 条件','track':'Benchmark / 评测集','terminal_decision':'Terminal decision / 最终判定','n':'Count / 数量'})}</details>
<details><summary>Judge source / 评分来源计数</summary>{html_table(tables['judge_source'], ['condition','track','judge_source','n'], {'condition':'Condition / 条件','track':'Benchmark / 评测集','judge_source':'Scorer source / 评分器来源','n':'Count / 数量'})}</details>
<details><summary>Extraction status / 答案抽取状态计数</summary>{html_table(tables['extraction_status'], ['condition','track','status','reason_code','n'], {'condition':'Condition / 条件','track':'Benchmark / 评测集','status':'Status / 状态','reason_code':'Reason code / 原因码','n':'Count / 数量'})}</details>
<details><summary>Four-state disagreement counts / 四状态分歧计数</summary>{html_table(tables['score_disagreement'], disagree_fields, {'condition':'Condition / 条件','track':'Benchmark / 评测集','native_correct':'Native correct / 原生判对','extraction_correct':'Extraction correct / 抽取判对','n':'Count / 数量'})}</details></section>
<section id="methods"><h2>Methods and reproducibility / 方法与可复现性</h2>
<div class="callout"><b>Paired tests / 配对检验：</b>{esc(manifest.get('paired_comparison_method'))}<br><b>Confidence intervals / 置信区间：</b>{esc(manifest.get('confidence_intervals'))}<br><b>Bootstrap iterations / Bootstrap迭代：</b>{int(manifest.get('bootstrap_iterations', 0)):,} · <b>Seed / 随机种子：</b>{int(manifest.get('seed', 0))}</div>
<p>All raw model outputs and score files remain unchanged. The report is generated from the verified output manifest and versioned aggregate tables; exact source paths and SHA-256 hashes are recorded here. / 原始模型输出和评分文件均未修改。本报告由已验证的输出清单及版本化统计表生成，并记录来源路径与SHA-256哈希。</p>
<details><summary>Input file audit / 展开输入文件审计（{len(source_files):,} files / 个文件）</summary>{html_table(source_rows, source_fields, source_titles)}</details>
<details><summary>Analysis manifest / 展开统计清单</summary><pre>{esc(audit_json)}</pre></details>
<p><span class="badge">No causal interpretation / 不作因果解释</span><span class="badge">Benchmark-stratified results retained / 保留评测集分层结果</span><span class="badge">Both scoring sources retained / 保留两种评分来源</span></p></section>
<footer>Versioned Work5 analysis · Self-contained HTML with embedded charts · Generated from verified full-set records / Work5版本化分析 · 图表内嵌的独立HTML · 基于通过验证的全量记录生成</footer>
</main></body></html>'''


def render(analysis_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    manifest, tables = load_analysis(analysis_dir)
    charts = {
        "absolute": accuracy_heatmaps(tables["condition_summary"]),
        "paired": paired_effect_forest(tables["paired_comparisons"]),
        "disagreement": disagreement_heatmap(tables["score_disagreement"]),
        "association": association_heatmaps(tables["length_accuracy_association"]),
        "tokens_native": token_medians(tables["response_length_by_score"], "native"),
        "tokens_extraction": token_medians(tables["response_length_by_score"], "extraction"),
    }
    document = build_html(manifest, tables, charts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete report staging file: {temporary}")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    print(f"PASS: wrote self-contained bilingual HTML with {len(charts)} embedded figures to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis_dir, args.output)


if __name__ == "__main__":
    main()
