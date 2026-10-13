#!/usr/bin/env python3
"""Build the bilingual, self-contained corrected Work 2 + Work 3 report."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import html
import json
import re
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img class="zoom" src="data:{mime};base64,{encoded}" '
            f'alt="{html.escape(caption)}" onclick="window.open(this.src,\'_blank\')">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def table(headers: list[str], rows: list[list[object]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def p(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root
    w2 = root / "research/main/Phase2/2.short_output_prompt_study"
    w4 = root / "research/main/Phase2/4.llm_rejudgment"
    derived = w4 / "derived/run-60011513"
    figures2, figures3, figuresd = w4 / "figures/work2", w4 / "figures/work3", w4 / "figures/detectors"

    summary = read_json(derived / "summary.json")
    lengths = read_json(derived / "length_accuracy_stats.json")
    tokens = read_json(derived / "token_length_stats.json")
    detector = read_json(derived / "detector_comparison.json")
    audit = read_json(derived / "manual_audit_summary.json")
    transitions = read_json(derived / "transitions/rejudged_transition_summary.json")
    flip_cases = read_json(derived / "transitions/rejudged_case_manifest.json")
    flip_images = read_json(w4 / "raw/rejudged_case_images/image_manifest.json")
    per = pd.read_csv(derived / "per_sample_rejudged.csv", keep_default_na=False)
    token_rows = pd.read_csv(derived / "token_per_sample_rejudged.csv", keep_default_na=False)
    token_map = {(r.sample_id, r.condition): r for r in token_rows.itertuples()}
    result_map = {(r.sample_id, r.condition): r for r in per.itertuples()}

    legacy = (w2 / "report/short-output-study-report-final-version.html").read_text(encoding="utf-8")
    input_match = re.search(r"<h2>输入对照.*?(?=<h2>结果摘要)", legacy, re.S)
    if not input_match:
        raise ValueError("could not recover the original five-condition input comparison")
    input_html = input_match.group(0)
    case_blocks = re.findall(r'<article class="case">.*?</article>', legacy, re.S)
    if len(case_blocks) != 10:
        raise ValueError(f"expected ten established Work 2 examples; found {len(case_blocks)}")

    cases = []
    for block in case_blocks:
        match = re.search(r"<h3>Case (\d+).*?—\s*(.*?)</h3>", block, re.S)
        if not match:
            raise ValueError("case title/sample key could not be parsed")
        index, sample_id = int(match.group(1)), html.unescape(match.group(2).strip())
        if not any((sample_id, mode) in result_map for mode in ["S0", "SC", "S1", "S2", "S3"]):
            raise ValueError(f"case missing from corrected judgments: {sample_id}")
        block = re.sub(
            r"<h3>Case \d+.*?</h3>",
            f'<h3>Case {index} / 案例 {index} <small>sample key: {html.escape(sample_id)}</small></h3>',
            block, count=1, flags=re.S,
        )
        for mode in ["S0", "SC", "S1", "S2", "S3"]:
            r = result_map[(sample_id, mode)]
            t = token_map[(sample_id, mode)]
            status = str(r.llm_status)
            answer = str(r.llm_parsed_answer) if str(r.llm_parsed_answer) not in ("", "nan") else "—"
            correct = "正确 / Correct" if str(r.llm_correct).lower() == "true" else "错误 / Incorrect"
            detail = (f"<p class='case-metrics'>判别结果 / Adjudicated: <b>{html.escape(status)}</b> · "
                      f"答案 / Option: <b>{html.escape(answer)}</b> · {correct} · "
                      f"字符 / chars: {int(r.response_chars):,} · "
                      f"Completion tokens: {int(t.completion_tokens):,}</p>")
            pattern = rf'(<div class="mode-card"><h4>{mode}</h4>.*?)(<p>解析答案 / Parsed:.*?</p>)(</div>)'
            block, count = re.subn(pattern, lambda m: m.group(1) + detail + m.group(3), block, count=1, flags=re.S)
            if count != 1:
                raise ValueError(f"could not refresh corrected result for case {index}, {mode}")
        cases.append(block)

    accuracy_rows = []
    for row in summary["summary"]:
        accuracy_rows.append([
            html.escape(row["condition"]), row["n"], p(row["accuracy"]),
            f"[{p(row['accuracy_ci95'][0])}, {p(row['accuracy_ci95'][1])}]",
            p(row["valid_answer_rate"]), p(row["strict_format_rate"]), p(row["reasoning_marker_rate"]),
            f"{row['response_chars_mean']:.1f}", f"{row['response_chars_median']:.1f}", row["api_error_count"],
        ])
    summary_table = table(["模式 / Mode", "N", "准确率 / Accuracy", "有效选项 / Valid option", "严格格式 / Strict format", "推理词标记 / Lexical marker", "均值字符 / Mean chars", "中位字符 / Median chars", "API 错误 / Errors"], accuracy_rows)

    paired_data = []
    for row in summary["paired_comparisons"]:
        comp = row["comparison"]
        p_raw = row.get("mcnemar_exact_p", row.get("mcnemar", {}).get("mcnemar_exact_p", row.get("mcnemar", {}).get("p_value", 1)))
        comp = comp.replace(" vs ", " − ")
        a_only = row.get("mcnemar", {}).get("rule_only_correct", row.get("mcnemar", {}).get("a_only", "—"))
        b_only = row.get("mcnemar", {}).get("llm_only_correct", row.get("mcnemar", {}).get("b_only", "—"))
        ci = row.get("accuracy_difference_ci95", [None, None])
        paired_data.append((comp, row["accuracy_difference"], ci, a_only, b_only, float(p_raw)))
    order = sorted(range(len(paired_data)), key=lambda i: paired_data[i][5])
    adjusted = [0.0] * len(paired_data)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * paired_data[index][5]))
        adjusted[index] = running
    paired_rows = []
    for (comp, difference, ci, s0_only, condition_only, p_raw), p_holm in zip(paired_data, adjusted):
        ci_text = "—" if ci[0] is None else f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp"
        paired_rows.append([html.escape(comp), f"{difference * 100:+.2f} pp", ci_text, s0_only, condition_only, f"{p_raw:.4g}", f"{p_holm:.4g}"])

    length_rows = []
    for x in lengths["conditions"]:
        length_rows.append([x["condition"], f"{x['n']:,}", f"{x['mean_chars']:.1f}", f"{x['median_chars']:.1f}",
                            f"{x['skewness']:+.2f}", f"{x['pearson_r']:+.3f} (p={x['pearson_p']:.2g})",
                            f"{x['spearman_rho']:+.3f} (p={x['spearman_p']:.2g})",
                            f"{x['correct_mean_chars']:.1f} (n={x['correct_n']})",
                            f"{x['incorrect_mean_chars']:.1f} (n={x['incorrect_n']})",
                            f"{x['correct_vs_incorrect_mann_whitney_p']:.2g}"])
    length_table = table(["模式 / Mode", "样本数 / N", "均值字符 / Mean chars", "中位字符 / Median chars", "偏度 / Skewness", "Pearson r (p)", "Spearman ρ (p)", "答对均值 / Correct mean", "答错均值 / Incorrect mean", "MW p (length by correctness)"], length_rows)
    token_rows_html = []
    for x in tokens["conditions"]:
        token_rows_html.append([x["condition"], x["n"], f"{x['completion_tokens_mean']:.1f}", f"{x['completion_tokens_median']:.1f}",
                                f"{x['completion_tokens_skewness']:+.2f}", f"{x['spearman_rho']:+.3f} (p={x['spearman_p']:.2g})",
                                f"{x['correct_mean_tokens']:.1f} (n={x['correct_n']})", f"{x['incorrect_mean_tokens']:.1f} (n={x['incorrect_n']})",
                                f"{x['correct_vs_incorrect_mann_whitney_p']:.2g}", f"{x['p95_tail_n']} / {x['p95_tail_accuracy']:.1%}"])
    token_table = table(["模式 / Mode", "样本数 / N", "平均 Token / Mean", "中位 Token / Median", "偏度 / Skewness", "Spearman ρ (p)", "答对均值 / Correct mean", "答错均值 / Incorrect mean", "MW p (tokens by correctness)", "P95 尾部 n / 准确率"], token_rows_html)

    det_rows = []
    for x in detector["per_condition"]:
        det_rows.append([x["condition"], f"{x['rule_accuracy']:.1%}", f"{x['llm_accuracy']:.1%}",
                         x["old_new_answer_letter_match_n"], f"{x['old_new_answer_letter_match_rate_among_both_answered']:.1%}",
                         x["rule_only_correct"], x["llm_only_correct"], f"{x['cohens_kappa_correctness']:.3f}"])
    detector_table = table(["模式 / Mode", "规则准确率 / Rule accuracy", "模型准确率 / LLM accuracy", "相同选项数 / Same option", "共同作答一致率 / Agreement", "仅规则正确 / Rule-only correct", "仅模型正确 / LLM-only correct", "正确性 κ / Correctness κ"], det_rows)

    work3_rows = []
    for x in transitions["target_flip_summaries"]:
        if x["target"] not in {"S1", "S3"}:
            continue
        work3_rows.append([x["target"], x["gain_n"], f"{x['gain_rate_of_all']:.2%}", x["loss_n"], f"{x['loss_rate_of_all']:.2%}",
                           x["net_gain"], f"{x['net_accuracy_difference'] * 100:+.2f} pp", f"{x['exact_mcnemar_p']:.4g}", f"{x['holm_adjusted_p']:.4g}"])
    work3_table = table(["条件 / Mode", "错→对 / Gain n", "占全部 / All", "对→错 / Loss n", "占全部 / All", "净变化 / Net", "准确率差 / Accuracy Δ", "McNemar p", "Holm 校正 p / Holm p"], work3_rows)
    states = transitions["state_summaries"]["states_S0_S1_S3"]["states"]
    state_table = table(["S0–S1–S3 状态 / Correctness state", "样本数 / N", "总体比例 / Share"], [[f"<code>{x['state']}</code>", x["n"], p(x["rate"])] for x in states])

    task_df = pd.read_csv(derived / "transitions/task_flip_rates.csv", keep_default_na=False)
    task_rows = []
    for mode in ["S1", "S3"]:
        sub = task_df[task_df.target == mode].copy()
        sub["flip_n"] = sub.gain_n + sub.loss_n
        sub = sub.sort_values(["flip_n", "task"], ascending=[False, True]).head(5)
        task_rows.extend([[mode, html.escape(str(x.task)), int(x.n_task), int(x.gain_n), int(x.loss_n),
                           f"{x.gain_rate:.1%}", f"{x.loss_rate:.1%}"] for x in sub.itertuples()])
    task_table = table(["模式 / Mode", "任务 / Task", "该类样本 / Task N", "错→对 / Gain", "对→错 / Loss", "恢复率 / Gain rate", "损失率 / Loss rate"], task_rows)

    per["correct_bool"] = per.llm_correct.astype(str).str.lower().eq("true")
    correctness = per.pivot(index="sample_id", columns="condition", values="correct_bool")
    task_lookup = per.drop_duplicates("sample_id").set_index("sample_id").task
    s2_only_gain = (~correctness.S0 & correctness.S2 & ~correctness.S1 & ~correctness.S3)
    s2_only_loss = (correctness.S0 & ~correctness.S2 & correctness.S1 & correctness.S3)
    s2_task_rows = []
    for label, mask in [("S2-only recovery", s2_only_gain), ("S2-only loss", s2_only_loss)]:
        for task, n in task_lookup[mask].value_counts().head(8).items():
            s2_task_rows.append([label, html.escape(str(task)), int(n), p(n / int(mask.sum()))])
    s2_task_table = table(["结构 / Pattern", "任务 / Task", "样本数 / N", "该结构中的比例 / Within-pattern share"], s2_task_rows)

    flip_rows = pd.read_csv(derived / "transitions/pairwise_flip_cases.csv", keep_default_na=False)
    flip_lookup = {(str(r.sample_id), str(r.target)): r for r in flip_rows.itertuples()}
    image_by_index = {int(row["case_index"]): row for row in flip_images}
    flip_case_cards = []
    for case in flip_cases:
        sample_id, target = case["sample_id"], case["target"]
        r = flip_lookup[(sample_id, target)]
        image_record = image_by_index[int(case["case_index"])]
        try:
            choices = json.loads(r.choices) if isinstance(r.choices, str) else r.choices
        except json.JSONDecodeError:
            choices = ast.literal_eval(r.choices)
        choices_html = "".join(
            f"<li>({html.escape(str(c.get('option', chr(65 + i)) if isinstance(c, dict) else chr(65 + i)) )}) "
            f"{html.escape(str(c.get('text', '') if isinstance(c, dict) else c))}</li>"
            for i, c in enumerate(choices)
        )
        s0_row, target_row = result_map[(sample_id, "S0")], result_map[(sample_id, target)]
        image_markup = image(w4 / "raw/rejudged_case_images" / image_record["file"], f"Flip case {case['case_index']} / 翻转案例 {case['case_index']}")
        category_label = "错转对 / recovery" if case["category"] == "gain" else "对转错 / loss"
        raw_outputs = (
            f"<div class='raw-pair'><div><h4>S0 · {html.escape(str(s0_row.llm_status))} · {html.escape(str(s0_row.llm_parsed_answer))} · "
            f"{'正确 / Correct' if str(s0_row.llm_correct).lower() == 'true' else '错误 / Incorrect'}</h4><pre>{html.escape(str(r.S0_response))}</pre></div>"
            f"<div><h4>{target} · {html.escape(str(target_row.llm_status))} · {html.escape(str(target_row.llm_parsed_answer))} · "
            f"{'正确 / Correct' if str(target_row.llm_correct).lower() == 'true' else '错误 / Incorrect'}</h4><pre>{html.escape(str(r.target_response))}</pre></div></div>"
        )
        flip_case_cards.append(
            f"<article class='flip-case'><h3>Flip {case['case_index']} / 翻转案例 {case['case_index']} · {target} · {category_label} · {html.escape(str(case['task']))}</h3>"
            f"<small class='small'>sample key: {html.escape(sample_id)}</small><div class='flip-intro'>{image_markup}"
            f"<div><p><b>问题 / Question</b></p><p>{html.escape(str(r.question))}</p><ol>{choices_html}</ol>"
            f"<p>标准答案 / Gold: <b>{html.escape(str(r.gold_answer))}</b> · Token 差 / Δ ({target}−S0): <b>{int(r.target_minus_s0_completion_tokens):+d}</b></p></div></div>"
            f"<details><summary>完整原始回答 / Full raw responses</summary>{raw_outputs}</details></article>"
        )
    flip_case_html = "".join(flip_case_cards)

    figures = []
    figures += [image(figures2 / "accuracy_overview.png/accuracy.png", "Accuracy under the five prompting modes / 五种模式准确率")]
    figures += [image(figures2 / "length_distribution_shape.png", "Visible response-length distributions / 可见输出长度分布")]
    figures += [image(figures2 / "length_by_correctness.png", "Character length by correctness / 按正确性分组的字符长度")]
    figures += [image(figures2 / "accuracy_by_length_decile.png", "Accuracy by character-length decile / 字符长度分位与准确率")]
    figures += [image(figures2 / "token_distribution_shape.png", "Tokenizer length distributions / Token 长度分布")]
    figures += [image(figures2 / "token_length_by_correctness.png", "Token length by correctness / 按正确性分组的 Token 长度")]
    figures += [image(figures2 / "token_accuracy_by_decile.png", "Accuracy by token-length decile / Token 长度分位与准确率")]
    figures += [image(figures2 / "token_p95_tail_task_accuracy.png", "P95 long-tail accuracy by task / P95 长尾分任务准确率")]
    figs_work2 = "".join(figures)
    detector_figs = "".join(image(figuresd / f, title) for f, title in [
        ("detector_correctness_agreement.png", "Rule/LLM correctness agreement / 规则与模型正确性一致性"),
        ("detector_reclassification_direction.png", "Reclassification direction / 重判方向"),
        ("llm_status_by_condition.png", "LLM answerability status / 模型判别状态"),
    ])
    figs_work3 = "".join(image(figures3 / f, title) for f, title in [
        ("correctness_states_8.png", "Eight paired correctness states: S0–S1–S3 / 八种配对正确性状态"),
        ("paired_flip_counts.png", "Paired gains and losses / 配对恢复与损失"),
        ("task_net_accuracy_heatmap.png", "Task-wise net accuracy change / 分任务净准确率变化"),
        ("flip_task_composition.png", "Task composition of flips / 翻转任务构成"),
        ("flip_token_delta.png", "Token change among flipped answers / 翻转题目的 Token 变化"),
    ])

    case_html = "".join(cases)
    overall_manual_rate = audit["observed_extraction_match_rate"]
    overall = detector["rule_llm_correctness_comparison"]
    s2 = transitions["s2_inclusion_evidence"]
    direction = f"""
    <p>在修正输出判别后，提示词带来的主要、稳健变化是回答长度下降，而不是准确率提升。S1 与 S3 的平均 completion token 分别约为 S0 的 {tokens['conditions'][2]['completion_tokens_mean']/tokens['conditions'][0]['completion_tokens_mean']:.0%} 与 {tokens['conditions'][4]['completion_tokens_mean']/tokens['conditions'][0]['completion_tokens_mean']:.0%}；但准确率相对 S0 分别为 {summary['summary'][2]['accuracy']-summary['summary'][0]['accuracy']:+.2%} 与 {summary['summary'][4]['accuracy']-summary['summary'][0]['accuracy']:+.2%}，多重比较校正后没有达到统计显著。当前数据支持“可压缩输出且准确率大致接近”的描述，不支持“提示词提高准确率”或“长度导致正确性变化”的因果结论。</p>
    <p>After correcting response adjudication, the clearest robust prompt effect is shorter output, not higher accuracy. Mean completion tokens for S1 and S3 are approximately {tokens['conditions'][2]['completion_tokens_mean']/tokens['conditions'][0]['completion_tokens_mean']:.0%} and {tokens['conditions'][4]['completion_tokens_mean']/tokens['conditions'][0]['completion_tokens_mean']:.0%} of S0. Their accuracy differences from S0 are {100*(summary['summary'][2]['accuracy']-summary['summary'][0]['accuracy']):+.2f} and {100*(summary['summary'][4]['accuracy']-summary['summary'][0]['accuracy']):+.2f} percentage points; neither remains statistically significant after multiplicity correction. The evidence supports “shorter outputs with broadly similar accuracy,” not an accuracy gain or a causal effect of length.</p>
    <p>所有五组的 token 长度分布都有明显右偏，且 P95 长尾样本准确率低于该模式总体准确率；这使长输出样本成为值得继续研究的对象，但需区分复杂样本、犹豫/自我修正、回答质量与截断等可能机制，现有相关性不能识别机制。</p>
    <p>All five token-length distributions are strongly right-skewed, and P95-tail accuracy is below each mode’s overall accuracy. Long-output cases merit further study, while possible mechanisms—task difficulty, hesitation/self-correction, answer quality, or truncation—must remain distinct; the current associations do not identify a mechanism.</p>
    """

    dhtml = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>重判修正版综合报告 / Corrected Integrated Report</title>
<style>
*{{box-sizing:border-box}} body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1320px;margin:0 auto;padding:28px 24px;color:#243447;background:#fff;line-height:1.65}} h1{{font-size:2rem;color:#18324b;margin:.2rem 0}} h2{{font-size:1.42rem;color:#18324b;margin:3rem 0 1rem;padding:.35rem 0;border-bottom:1px solid #dbe3ea}} h3{{color:#285878;margin-top:1.8rem}} h4{{color:#294a62}} p{{max-width:1050px}} .author,.small{{color:#687887;font-size:.88rem}} .lead,.callout{{padding:1rem 1.15rem;border-left:4px solid #4c78a8;background:#f7fafc;margin:1.2rem 0}} .toc{{border:1px solid #d8e1e8;padding:1rem 1.3rem;margin:1.4rem 0;background:#fff}} .toc a{{color:#285878;text-decoration:none}} .toc a:hover{{text-decoration:underline}} .toc ul{{columns:2;margin:.3rem 0}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;margin:1.1rem 0 2rem}} figure{{margin:0;border:1px solid #d9e1e8;padding:.65rem;background:#fff;min-width:0}} figure img{{display:block;width:100%;max-height:620px;object-fit:contain;cursor:zoom-in}} figcaption{{text-align:center;color:#475a6d;font-size:.9rem;padding:.5rem .2rem}} .table-wrap{{overflow-x:auto;margin:1.2rem 0 2rem}} table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{border:1px solid #d6dfe7;padding:.55rem .62rem;text-align:left;vertical-align:top}} th{{background:#f3f6f8;color:#29465d;white-space:nowrap}} .input-grid,.mode-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}} .input-card,.mode-card{{border:1px solid #d8e1e8;background:#fff;padding:.8rem;min-width:0}} .input-card pre,.mode-card pre{{white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;background:#f7f9fb;border:1px solid #e5ebef;border-radius:4px;padding:.7rem;font-size:.81rem;max-height:430px;overflow:auto}} .case{{border:1px solid #ccd8e2;margin:2rem 0;padding:1rem 1.1rem;background:#fff}} .case small{{font-size:.7rem;font-weight:400;color:#83909b;word-break:break-all}} .case-image{{display:block;max-width:100%;max-height:560px;object-fit:contain;margin:1rem auto;background:#fff}} .question,.case-metrics{{padding:.7rem .85rem;background:#f7f9fb;border-left:3px solid #7593aa}} .case-metrics{{font-size:.9rem}} .section-note{{color:#546779;font-size:.94rem}} .figure-stack figure{{margin:1rem 0}} footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid #dce3e9;color:#6a7884;font-size:.86rem}} @media(max-width:650px){{body{{padding:16px 12px}}.toc ul{{columns:1}}.grid,.input-grid,.mode-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>重判修正版：短输出提示词与配对翻转综合分析<br><span class="small">Corrected Rejudgment: Integrated Short-Output Prompt and Paired-Flip Analysis</span></h1>
<p class="author">作者 / Author: <a href="https://github.com/spinner-bot" target="_blank" rel="noopener">spinner-bot</a>（GitHub）</p>
<p class="lead">本报告以本地大模型对 10,000 条原始输出的复判为正确性分析依据，同时保留规则判别结果以量化判别器差异。复判结果不是独立人工金标准；文中明确区分模型判别、人工抽样核查和对照 benchmark 答案。<br>This report uses local-LLM re-adjudication of all 10,000 raw outputs for correctness analyses, while retaining rule-based labels to quantify detector differences. LLM adjudication is not an independent human gold standard; model judgments, sampled manual audit, and benchmark answers are distinguished throughout.</p>
<nav class="toc"><b>目录 / Contents</b><ul>
<li><a href="#protocol">数据与判别方法 / Data and adjudication</a></li><li><a href="#inputs">输入对照 / Input comparison</a></li><li><a href="#w2-summary">五组总体结果 / Five-condition results</a></li><li><a href="#detector">判别器对比与核查 / Detector comparison and audit</a></li><li><a href="#length">字符长度与正确性 / Character length and correctness</a></li><li><a href="#tokens">Token 长度与长尾 / Token length and long tail</a></li><li><a href="#direction">研究方向评估 / Research assessment</a></li><li><a href="#w3">S0/S1/S3 配对翻转 / Paired flips</a></li><li><a href="#flip-cases">翻转样本 / Flip examples</a></li><li><a href="#s2">S2 范围判断 / S2 scope decision</a></li><li><a href="#cases">典型案例 / Concrete cases</a></li><li><a href="#limits">解释边界 / Limitations</a></li></ul></nav>
<h2 id="protocol">数据与判别方法 / Data and adjudication</h2><p>冻结样本共 2,000 题，每题在 S0、SC、S1、S2、S3 五种条件下各有一个原始输出，总计 10,000 条。原规则标签完整保留；本地 Qwen3.8-27B 对每条原始回答独立提取 ANSWERED/AMBIGUOUS/NO_ANSWER 状态及最终选项，正确性随后机械地与冻结 benchmark 答案比对。AMBIGUOUS 和 NO_ANSWER 在主准确率中计为不正确，并单独报告。</p><p>The frozen sample contains 2,000 questions, each with one raw response under S0, SC, S1, S2, and S3 (10,000 responses total). Original rule labels are preserved. Local Qwen3.8-27B independently extracts ANSWERED/AMBIGUOUS/NO_ANSWER and a final option from each response; correctness is then mechanically compared with the frozen benchmark answer. AMBIGUOUS and NO_ANSWER count as incorrect in primary accuracy and are reported separately.</p>
<h2 id="inputs">输入对照 / Input comparison</h2>{input_html.replace('<h2>输入对照 / Input comparison</h2>','',1)}
<h2 id="w2-summary">五组总体结果 / Five-condition results</h2><p>以下答案准确率采用模型复判标签，括号列出 95% bootstrap CI；有效选项率、格式和推理标记按原始输出文本重算。每个条件 n=2,000。</p><p>Accuracy below uses LLM-adjudicated labels, with 95% bootstrap CIs; valid-option, format, and lexical-reasoning indicators are recomputed from raw responses. Each condition has n=2,000.</p>{table(['模式 / Mode','N','准确率 / Accuracy','准确率 95% CI','有效选项 / Valid option','严格格式 / Strict format','推理词标记 / Lexical marker','均值字符 / Mean chars','中位字符 / Median chars','API 错误 / Errors'],accuracy_rows)}
<h3>相对 S0 的配对准确率差 / Paired accuracy differences versus S0</h3><p>差值定义为目标条件减 S0；表中分别列出 S0-only 与目标条件-only 答对数，并给出配对 bootstrap 95% CI、双侧精确 McNemar p 与四项比较的 Holm 校正 p。</p><p>Differences are target condition minus S0. The table reports S0-only and target-only correct counts, paired-bootstrap 95% CIs, two-sided exact McNemar p-values, and Holm-adjusted p-values across the four comparisons.</p>{table(['比较 / Comparison','准确率差 / Difference','95% CI','S0-only correct / S0 独有正确','目标-only correct / Target 独有正确','McNemar p','Holm p'],paired_rows)}
<h2 id="detector">判别器对比与核查 / Detector comparison and audit</h2><p>规则与模型正确性标签在 10,000 条中有 {overall['discordant_correctness']:,} 条不一致；规则判别正确率为 {detector['rule_accuracy']:.2%}，模型复判对应为 {detector['llm_accuracy']:.2%}，正确性 Cohen’s κ={overall['cohens_kappa_correctness']:.3f}。在双方均给出选项的 {detector['old_new_answered_n']:,} 条中，选项字母完全一致 {detector['old_new_same_answer_n']:,} 条（{detector['old_new_same_answer_rate_among_both_answered']:.2%}）。这是显著的判别器差异，而非对模型复判准确性的独立证明。</p><p>Across 10,000 records, the rule and LLM correctness labels disagree on {overall['discordant_correctness']:,}. Rule accuracy is {detector['rule_accuracy']:.2%}, compared with {detector['llm_accuracy']:.2%} for LLM adjudication; correctness Cohen’s κ={overall['cohens_kappa_correctness']:.3f}. Among {detector['old_new_answered_n']:,} records where both supplied an option, exact option-letter agreement is {detector['old_new_same_answer_n']:,} ({detector['old_new_same_answer_rate_among_both_answered']:.2%}). This is a substantial detector difference, not independent proof that LLM adjudication is correct.</p>{detector_table}<div class="grid">{detector_figs}</div>
<h3>分层人工核查 / Stratified manual audit</h3><p>人工逐条检查 209 个预先抽取样本：规则/模型答案一致与不一致各层在五个条件内均抽取 20 条，并纳入全部 {audit['n_nonanswer_census_audited']} 条 AMBIGUOUS/NO_ANSWER 输出（8 条歧义、1 条无答案）。209 条中未观察到模型状态或选项提取错误（观察匹配率 {overall_manual_rate:.1%}）；9 条边界输出均被人工确认适当标为歧义/无答案。由于分层设计刻意过采规则/模型分歧，209 条简单合并比例不是总体误差率的无偏估计；零观察错误也不代表总体误差为零。</p><p>We manually inspected 209 preselected records: within each of the five conditions, 20 same-option and 20 different-option rule/LLM records were sampled, and all {audit['n_nonanswer_census_audited']} AMBIGUOUS/NO_ANSWER outputs were included (eight ambiguous, one unanswered). No state/option extraction error was observed (sample match rate {overall_manual_rate:.1%}); all nine boundary outputs were confirmed as appropriately ambiguous/unanswerable. Because the design deliberately oversamples detector disagreements, the unweighted 209-record rate is not an unbiased population error rate; zero observed errors does not imply zero population error.</p>{table(['条件 / Mode','分层 / Stratum','总体层人数 / Population N','人工核查人数 / Audited n','提取错误 / Extraction errors'],[[x['condition'],html.escape(x['stratum']),x['population_n'],x['sample_n'],x['extraction_errors']] for x in audit['strata']])}
<h2 id="length">字符长度与正确性 / Character length and correctness</h2><p>字符数是原始回答可见字符长度。相关只描述关联，按模型复判的正确/错误分组；图表分别展示分布、分位与正确性。</p><p>Character count is the visible length of each raw response. Correlations describe association only; correctness uses LLM adjudication. Figures show distributions, quantiles, and correctness strata.</p>{length_table}<div class="grid">{image(figures2/'length_distribution_shape.png','Character-length distribution / 字符长度分布')}{image(figures2/'length_by_correctness.png','Character length by correctness / 按正确性分组的字符长度')}{image(figures2/'accuracy_by_length_decile.png','Accuracy by character-length decile / 字符长度分位准确率')}{image(figures2/'accuracy_overview.png/accuracy.png','Accuracy overview / 准确率概览')}</div>
<h2 id="tokens">Token 长度与长尾 / Token length and long tail</h2><p>Token 计数直接取原始 vLLM usage 字段，而不是由替代 tokenizer 估算。五种模式均呈明显右偏（均值高于中位数，偏度为正），P95 长尾组准确率低于各自总体准确率；长输出与错误相关，但该观察不说明因果方向。</p><p>Token counts come directly from original vLLM usage fields rather than a substitute tokenizer. All five modes are strongly right-skewed (mean above median, positive skew); P95-tail accuracy is lower than the corresponding overall accuracy. Long outputs are associated with errors, but this does not establish causal direction.</p>{token_table}<div class="grid">{image(figures2/'token_distribution_shape.png','Token-length distribution / Token 长度分布')}{image(figures2/'token_length_by_correctness.png','Token length by correctness / Token 长度与正确性')}{image(figures2/'token_accuracy_by_decile.png','Accuracy by token-length decile / Token 分位准确率')}{image(figures2/'token_p95_tail_task_accuracy.png','P95 tail accuracy by task / P95 长尾任务准确率')}</div>
<h2 id="direction">研究方向评估 / Research-direction assessment</h2>{direction}
<h2 id="w3">S0/S1/S3 配对翻转 / S0/S1/S3 paired flips</h2><p>状态码顺序为 S0–S1–S3，1=答对、0=答错。主分析保留工作3原定的 S0/S1/S3 范围。基于复判标签，三组全对为 {states[7]['n']:,}（{p(states[7]['rate'])}），三组全错为 {states[0]['n']:,}（{p(states[0]['rate'])}）；基线错题中至少一个提示条件改对共 {transitions['logical_groups']['s0_wrong_any_s1s3_recovery']} 题，基线对题中至少一个提示条件转错共 {transitions['logical_groups']['s0_correct_any_s1s3_loss']} 题。</p><p>State-code order is S0–S1–S3, with 1=correct and 0=incorrect. The primary analysis retains Work 3’s original S0/S1/S3 scope. Under corrected labels, {states[7]['n']:,} samples ({p(states[7]['rate'])}) are correct in all three modes and {states[0]['n']:,} ({p(states[0]['rate'])}) are wrong in all three. At least one prompt condition recovers {transitions['logical_groups']['s0_wrong_any_s1s3_recovery']} baseline errors; at least one prompt condition loses {transitions['logical_groups']['s0_correct_any_s1s3_loss']} baseline-correct answers.</p>{state_table}{work3_table}<div class="grid">{figs_work3}</div><h3>任务类型拆分 / Task-type breakdown</h3><p>下表按各模式中翻转总数较高的任务列出恢复率和损失率；比例分母为该任务在该条件的全部样本，而不是只在翻转样本内部归一化。</p><p>The table lists tasks with the most flips per mode. Rates use all samples of that task-condition as denominators, not only flipped cases.</p>{task_table}
<h2 id="s2">S2 范围判断 / S2 scope decision</h2><p>S2 经复判从旧规则准确率 {s2['old_rule_accuracy_s2']:.2%} 修正为 {s2['llm_adjudicated_accuracy_s2']:.2%}，但这主要反映判别器修正。S2 对 S0/S1/S3 的配对准确率差分别为 {s2['s2_pairwise_tests'][0]['accuracy_difference']*100:+.2f}、{s2['s2_pairwise_tests'][1]['accuracy_difference']*100:+.2f}、{s2['s2_pairwise_tests'][2]['accuracy_difference']*100:+.2f} 个百分点，Holm 校正 p 值分别为 {s2['s2_pairwise_tests'][0]['holm_adjusted_p']:.3f}、{s2['s2_pairwise_tests'][1]['holm_adjusted_p']:.3f}、{s2['s2_pairwise_tests'][2]['holm_adjusted_p']:.3f}。其相对 S0 有 45 次错转对、55 次对转错；S2 特有的基线错题恢复 11 例、特有损失 22 例。S2 与 S1/S3 至少一个不一致的样本为 {s2['logical_state_counts']['s2_disagrees_with_s1_or_s3']}（6.65%），因此它并非逐题完全冗余；但当前准确率差异和唯一翻转规模均不足以显示其产生了不可忽略的总体收益。故工作3主报告维持 S0/S1/S3，不扩展为 S2 主分析。</p><p>LLM adjudication changes S2 accuracy from {s2['old_rule_accuracy_s2']:.2%} under the old rule labels to {s2['llm_adjudicated_accuracy_s2']:.2%}, primarily reflecting detector correction. Paired accuracy differences for S2 versus S0/S1/S3 are {s2['s2_pairwise_tests'][0]['accuracy_difference']*100:+.2f}, {s2['s2_pairwise_tests'][1]['accuracy_difference']*100:+.2f}, and {s2['s2_pairwise_tests'][2]['accuracy_difference']*100:+.2f} pp; Holm-adjusted p-values are {s2['s2_pairwise_tests'][0]['holm_adjusted_p']:.3f}, {s2['s2_pairwise_tests'][1]['holm_adjusted_p']:.3f}, and {s2['s2_pairwise_tests'][2]['holm_adjusted_p']:.3f}. Versus S0, S2 has 45 wrong-to-correct and 55 correct-to-wrong flips; its unique baseline-error recoveries number 11 and unique losses 22. S2 differs from at least one of S1/S3 on {s2['logical_state_counts']['s2_disagrees_with_s1_or_s3']} samples (6.65%), so it is not itemwise redundant. However, neither the accuracy contrasts nor the unique-flip volume show a non-negligible overall benefit. The primary Work 3 report therefore retains S0/S1/S3 and does not expand its core scope to S2.</p>{s2_task_table}<h3>四模式状态诊断（补充） / Four-mode state diagnostic (supplement)</h3><p>该图仅用于透明评估 S2 是否改变总体判断，不改变工作3主分析范围。</p><p>This diagnostic transparently assesses whether S2 changes the overall conclusion; it does not change Work 3’s primary scope.</p>{image(figures3/'correctness_states_16.png','Sixteen paired states / 十六种配对状态')}
<h2 id="cases">典型案例 / Concrete cases</h2><p>保留原工作2的十个精选案例，按原图、问题/选项、五种模式的实际输入与完整原始回答展示。每个模式的长度、模型判别状态、提取选项与正确性均已依据修正数据更新；原始 benchmark 标识以弱化小字保留以便追溯。</p><p>The ten selected Work 2 cases are retained, showing the original image, question/options, actual input, and complete raw output for all five modes. Length, LLM status, extracted option, and correctness are refreshed from corrected data; original benchmark identifiers remain only as subdued traceability labels.</p>{case_html}
<h2 id="limits">解释边界 / Interpretation limits</h2><ul><li>LLM 复判用于修正明显不可靠的规则选项提取，但仍可能出错；人工核查为分层样本审计，不能当作 10,000 条的完整人工金标准。<br>LLM adjudication replaces unreliable rule parsing but can still err; the stratified audit does not constitute full human gold labeling of all 10,000 records.</li><li>准确率依赖冻结 benchmark 答案的正确性；对回答的文字提取正确，不代表原回答视觉判断本身正确。<br>Accuracy depends on frozen benchmark answers; faithful extraction of a response does not mean the response’s visual judgment is itself correct.</li><li>统计显著性、差异大小、分层结构与原始回答需合并解释；相关不能推出因果。<br>Significance, effect size, pairing, and raw responses should be interpreted jointly; association is not causation.</li><li>开放选项题被模型标为歧义或无答案时，在主准确率中计错；边界状态另行展示，避免隐藏分母。<br>Model-labeled ambiguous/unanswered outputs count as incorrect in primary accuracy and are separately shown to keep denominators transparent.</li></ul>
<footer>数据来源 / Data: frozen run 60011513; rejudgment model: Qwen3.8-27B (BF16); inference model: OPD-V-Qwen3.5-4B. Figures are embedded and clickable for full-size viewing. / 图表全部内嵌，点击可查看原尺寸。</footer></body></html>"""
    dhtml = dhtml.replace(
        '<li><a href="#s2">',
        '<li><a href="#flip-cases">翻转样本 / Flip examples</a></li><li><a href="#s2">',
        1,
    )
    flip_report_section = (
        '<h3 id="flip-cases">翻转样本案例 / Paired-flip examples</h3>'
        '<p>按 S1/S3 × 错转对/对转错四个分层，每层选取三个绝对 token 变化较大的案例，并尽可能覆盖不同任务。每个案例展示原图、问题、选项、benchmark 答案、正确性变化与 token 差；完整 S0 与目标模式原始回答可展开查看。<br>'
        'Three cases with large absolute token changes are selected in each of four strata (S1/S3 × recovery/loss), with task diversity where available. Each shows the image, question, options, benchmark answer, correctness transition, and token difference; complete S0 and target-mode raw responses are expandable.</p>'
        + flip_case_html
    )
    dhtml = dhtml.replace('<h2 id="s2">', flip_report_section + '<h2 id="s2">', 1)
    dhtml = dhtml.replace(
        '</style>',
        '.flip-case{border:1px solid #ccd8e2;margin:1.5rem 0;padding:1rem;background:#fff}.flip-intro{display:grid;grid-template-columns:minmax(260px,42%) 1fr;gap:1rem;align-items:start}.flip-intro figure img{max-height:420px}.raw-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}.raw-pair pre{white-space:pre-wrap;word-break:break-word;max-height:620px;overflow:auto;background:#f7f9fb;padding:.8rem;font-size:.82rem;border:1px solid #e5ebef}details{margin:1rem 0;padding:.7rem;border:1px solid #d8e1e8}summary{cursor:pointer;font-weight:600;color:#285878}@media(max-width:760px){.flip-intro,.raw-pair{grid-template-columns:1fr}}'
        + '</style>',
        1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dhtml, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, "cases": len(cases), "figures_embedded": dhtml.count("data:image/")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
