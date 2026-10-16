#!/usr/bin/env python3
"""Summarize the stratified multimodal audit without treating it as population data."""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

from full_campaign_layout import CONDITIONS, TRACKS
from run_visual_audit import make_candidate_excerpt


VERDICTS = ("CORRECT", "INCORRECT", "UNCERTAIN")
SCORE_PATHS = {
    "official_native": "native_correct",
    "terminal_diagnostic": "terminal_correct",
    "text_extraction": "extraction_correct",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze_records(
    inputs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    merged_scores: list[dict[str, Any]],
    sample_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(inputs) != 300:
        raise ValueError(f"Expected 300 frozen audit inputs, got {len(inputs)}")
    input_by_id = {row.get("request_id"): row for row in inputs}
    if len(input_by_id) != len(inputs) or None in input_by_id:
        raise ValueError("Frozen audit inputs contain missing or duplicate request IDs")
    latest: dict[str, dict[str, Any]] = {}
    for row in results:
        request_id = row.get("request_id")
        if request_id not in input_by_id:
            raise ValueError(f"Unexpected visual-audit result ID: {request_id}")
        latest[request_id] = row
    if set(latest) != set(input_by_id):
        raise ValueError(f"Visual-audit result coverage is {len(latest)}/300")
    failures = [row for row in latest.values() if row.get("request_status") != "OK"]
    if failures:
        raise ValueError(f"Visual audit has {len(failures)} non-OK final requests")

    score_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in merged_scores:
        key = (row.get("track"), row.get("sample_uid"), row.get("condition"))
        if key in score_by_key:
            raise ValueError(f"Duplicate score row {key}")
        score_by_key[key] = row

    selection = sample_manifest.get("tracks_selection")
    if not isinstance(selection, dict) or set(selection) != set(TRACKS):
        raise ValueError("Sample manifest does not cover exactly the six Work5 tracks")
    sample_meta: dict[tuple[str, str], dict[str, Any]] = {}
    case_index: dict[tuple[str, str], int] = {}
    index = 0
    for track in TRACKS:
        selected = selection[track].get("selected", [])
        if len(selected) != 10:
            raise ValueError(f"{track}: expected ten selected cases")
        for item in selected:
            index += 1
            uid = item["sample_uid"]
            key = (track, uid)
            if key in sample_meta:
                raise ValueError(f"Duplicate selected case {key}")
            sample_meta[key] = item
            case_index[key] = index

    if len(sample_meta) != 60:
        raise ValueError("Expected 60 unique selected samples")
    expected_score_keys = {
        (track, uid, condition)
        for track, uid in sample_meta for condition in CONDITIONS
    }
    if not expected_score_keys.issubset(score_by_key):
        raise ValueError("Merged v4 scores do not cover all 300 audit rows")

    per_request = []
    for request_id, input_row in input_by_id.items():
        result = latest[request_id]
        source_response = input_row.get("candidate_response")
        if not isinstance(source_response, str):
            raise ValueError(f"Missing original candidate response for {request_id}")
        if hashlib.sha256(source_response.encode("utf-8")).hexdigest() != input_row.get("candidate_response_sha256"):
            raise ValueError(f"Frozen candidate-response hash mismatch for {request_id}")
        excerpt = make_candidate_excerpt(source_response)
        if result.get("candidate_response_excerpt") != excerpt["text"]:
            raise ValueError(f"Auditor excerpt differs from the frozen response policy for {request_id}")
        if result.get("candidate_response_excerpt_sha256") != excerpt["sha256"]:
            raise ValueError(f"Auditor excerpt hash mismatch for {request_id}")
        key = (input_row["track"], input_row["sample_uid"], input_row["condition"])
        score = score_by_key.get(key)
        if score is None:
            raise ValueError(f"Missing v4 score for {key}")
        verdict = result.get("audit", {}).get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"Invalid/missing visual-audit verdict for {request_id}")
        meta = sample_meta[(input_row["track"], input_row["sample_uid"])]
        output = {
            "case_index": case_index[(input_row["track"], input_row["sample_uid"])],
            "track": input_row["track"], "sample_uid": input_row["sample_uid"],
            "condition": input_row["condition"],
            "primary_stratum": meta.get("primary_stratum", ""),
            "all_flags": ";".join(meta.get("all_flags", [])),
            "reference_answer": input_row["reference_answer"],
            "candidate_response": input_row["candidate_response"],
            "candidate_response_original_chars": excerpt["original_chars"],
            "candidate_response_excerpt_chars": len(excerpt["text"]),
            "candidate_response_omitted_chars": excerpt["omitted_chars"],
            "candidate_response_was_truncated": excerpt["omitted_chars"] > 0,
            "visual_verdict": verdict,
            "reference_assessment": result["audit"]["reference_assessment"],
            "candidate_final_answer": result["audit"]["candidate_final_answer"],
            "visual_evidence": result["audit"]["visual_evidence"],
            "visual_reason": result["audit"]["reason"],
            "postprocessing_action": result.get("postprocessing", {}).get("action", "not_recorded"),
            "model_reported_verdict": result.get("postprocessing", {}).get(
                "model_reported_verdict", result["audit"]["verdict"]),
            "image_sha256": input_row["image_sha256"],
            "candidate_response_sha256": input_row["candidate_response_sha256"],
        }
        for metric, source in SCORE_PATHS.items():
            raw = score.get(source)
            if str(raw) not in {"0", "1"}:
                raise ValueError(f"Invalid {source} value for {key}: {raw!r}")
            output[metric] = int(raw)
        per_request.append(output)

    overall = Counter(row["visual_verdict"] for row in per_request)
    by_condition = {
        condition: {
            verdict: sum(row["condition"] == condition and row["visual_verdict"] == verdict
                         for row in per_request)
            for verdict in VERDICTS
        }
        for condition in CONDITIONS
    }
    by_track_condition = {
        f"{track}/{condition}": {
            verdict: sum(row["track"] == track and row["condition"] == condition
                         and row["visual_verdict"] == verdict for row in per_request)
            for verdict in VERDICTS
        }
        for track in TRACKS for condition in CONDITIONS
    }
    cross_tabs = {}
    pairwise_agreement = {}
    for score_name in SCORE_PATHS:
        cells = Counter()
        agreement = 0
        disagreement = 0
        uncertain = 0
        for row in per_request:
            score_label = "SCORE_CORRECT" if row[score_name] else "SCORE_NOT_CORRECT"
            cells[(row["visual_verdict"], score_label)] += 1
            if row["visual_verdict"] == "UNCERTAIN":
                uncertain += 1
            elif (row["visual_verdict"] == "CORRECT") == bool(row[score_name]):
                agreement += 1
            else:
                disagreement += 1
        cross_tabs[score_name] = {
            f"{verdict}|{score_label}": cells[(verdict, score_label)]
            for verdict in VERDICTS for score_label in ("SCORE_CORRECT", "SCORE_NOT_CORRECT")
        }
        definite = agreement + disagreement
        pairwise_agreement[score_name] = {
            "definite_n": definite,
            "agreement_n": agreement,
            "disagreement_n": disagreement,
            "uncertain_n": uncertain,
            "agreement_fraction_among_definite": agreement / definite if definite else None,
        }
    disagreement_cases = []
    for row in per_request:
        if row["visual_verdict"] == "UNCERTAIN":
            continue
        for score_name in SCORE_PATHS:
            judge_correct = row["visual_verdict"] == "CORRECT"
            if judge_correct != bool(row[score_name]):
                disagreement_cases.append({
                    "case_index": row["case_index"], "track": row["track"],
                    "sample_uid": row["sample_uid"], "condition": row["condition"],
                    "comparison": score_name,
                    "visual_verdict": row["visual_verdict"], "score_correct": row[score_name],
                    "all_flags": row["all_flags"], "visual_reason": row["visual_reason"],
                })
    summary = {
        "schema": "phase2-work5-visual-audit-summary-v1",
        "n_samples": len(sample_meta), "n_paired_requests": len(per_request),
        "n_tracks": len(TRACKS), "conditions": list(CONDITIONS),
        "sampling_note": "Purposive risk-stratified qualitative sample; counts are not population prevalence estimates.",
        "verdict_counts": {verdict: overall[verdict] for verdict in VERDICTS},
        "by_condition": by_condition, "by_track_condition": by_track_condition,
        "cross_tabs_visual_vs_score_path": cross_tabs,
        "pairwise_agreement_among_definite_visual_verdicts": pairwise_agreement,
        "sample_counts_by_primary_stratum": dict(Counter(
            meta.get("primary_stratum", "unknown") for meta in sample_meta.values()
        )),
        "definite_disagreement_rows_vs_score_paths": len(disagreement_cases),
        "uncertain_rows": overall["UNCERTAIN"],
        "truncated_candidate_responses": sum(row["candidate_response_was_truncated"] for row in per_request),
        "total_candidate_response_chars_omitted": sum(row["candidate_response_omitted_chars"] for row in per_request),
        "conservatively_normalized_rows": sum(
            row["postprocessing_action"] == "conservative_verdict_normalization" for row in per_request
        ),
        "disagreement_examples": disagreement_cases[:15],
    }
    return summary, per_request


def render_html(summary: dict[str, Any], per_request: list[dict[str, Any]], provenance: dict[str, Any]) -> str:
    colors = {"CORRECT": "#327a54", "INCORRECT": "#bb4848", "UNCERTAIN": "#bd8a2f"}
    max_n = max(1, max(summary["by_condition"][c][v] for c in CONDITIONS for v in VERDICTS))
    bars = []
    for condition in CONDITIONS:
        chunks = []
        for verdict in VERDICTS:
            count = summary["by_condition"][condition][verdict]
            width = count / max_n * 100
            chunks.append(f'<div style="height:28px;width:{width:.2f}%;background:{colors[verdict]}" title="{verdict}: {count}"></div>')
        bars.append(f'<div class="bar-row"><b>{html.escape(condition)}</b><div class="bar">{"".join(chunks)}</div><span>n=60</span></div>')
    rows = []
    for condition, counts in summary["by_condition"].items():
        rows.append("<tr><th>" + html.escape(condition) + "</th>" + "".join(
            f"<td>{counts[verdict]}</td>" for verdict in VERDICTS) + "</tr>")
    cross_rows = []
    for metric, cells in summary["cross_tabs_visual_vs_score_path"].items():
        label = {"official_native": "Vision-OPD official exact-label / 官方精确标签",
                 "terminal_diagnostic": "Terminal diagnostic / 终行诊断",
                 "text_extraction": "Qwen text extraction / Qwen文本抽取"}[metric]
        cross_rows.append(f"<tr><th>{html.escape(label)}</th>" + "".join(
            f"<td>{cells[f'{verdict}|{score}']}</td>"
            for verdict in VERDICTS for score in ("SCORE_CORRECT", "SCORE_NOT_CORRECT")) + "</tr>")
    agreement_rows = []
    for metric, values in summary["pairwise_agreement_among_definite_visual_verdicts"].items():
        fraction = values["agreement_fraction_among_definite"]
        fraction_label = "NA" if fraction is None else f"{fraction:.1%}"
        agreement_rows.append(
            f"<tr><th>{html.escape(metric)}</th><td>{values['definite_n']}</td>"
            f"<td>{values['agreement_n']}</td><td>{values['disagreement_n']}</td>"
            f"<td>{values['uncertain_n']}</td><td>{fraction_label}</td></tr>"
        )
    discrepancy = []
    for row in summary.get("disagreement_examples", []):
        discrepancy.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(k,'')))}</td>"
            for k in ("case_index", "track", "condition", "comparison", "visual_verdict", "score_correct", "all_flags")) + "</tr>")
    table_rows = []
    for row in summary["disagreement_examples"]:
        table_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(k,'')))}</td>" for k in (
                "case_index", "track", "condition", "primary_stratum", "visual_verdict",
                "reference_assessment", "official_native", "terminal_diagnostic", "text_extraction")) + "</tr>"
        )
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Work5 multimodal audit / 工作5多模态辅助审计</title><style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;max-width:1180px;margin:0 auto;padding:28px;color:#222;background:#fff;line-height:1.55}}
h1{{margin-bottom:.3em}}h2{{margin-top:2.3em;border-bottom:1px solid #ddd;padding-bottom:.3em}}.note{{padding:14px;border-left:4px solid #647b91;background:#f5f7f9}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:left;vertical-align:top}}.bar-row{{display:grid;grid-template-columns:70px 1fr 45px;gap:10px;align-items:center;margin:8px 0}}.bar{{display:flex;border-radius:4px;overflow:hidden}}
.legend span{{margin-right:18px}}.muted{{color:#66717c}}details{{margin-top:1em}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f8;padding:12px}}
</style><body>
<h1>Work5: Multimodal audit of stratified cases / 工作5：分层样本多模态辅助审计</h1>
<p>60 samples · 300 paired condition views · six tracks · five conditions / 60个样本、300个配对条件视图、六个评测集、五种条件。</p>
<div class="note"><b>Interpretation boundary / 解释边界：</b>This is a purposive risk-stratified qualitative sample, not a population prevalence estimate. The auditor is Qwen3.8-27B, also used for the text-only extraction path; image-grounded results are supplementary model judgments, not human adjudication or Vision-OPD official scores. Candidate responses over 12,000 characters are presented as a recorded 4,000-character prefix and 8,000-character suffix; full responses remain in frozen inputs, exact excerpts and hashes are saved, and the auditor is instructed to use UNCERTAIN if omitted content could change interpretation. This run excerpted {summary['truncated_candidate_responses']}/300 responses ({summary['total_candidate_response_chars_omitted']:,} characters omitted). / 本材料为风险分层定性样本，不代表总体比例。审计器Qwen3.8-27B也用于文本答案抽取；图像判读是补充模型判断，不是人工裁定或Vision-OPD官方评分。超过12,000字符的回答以有记录的前4,000字符与后8,000字符送审；冻结输入保留完整回答，逐条保存实际截取文本及哈希，并要求在省略内容可能影响判断时标记UNCERTAIN。本次截取{summary['truncated_candidate_responses']}/300条回答，共省略{summary['total_candidate_response_chars_omitted']:,}字符。</div>
<h2>Verdict distribution / 判读分布</h2><p class="legend"><span style="color:{colors['CORRECT']}">■ CORRECT / 正确</span><span style="color:{colors['INCORRECT']}">■ INCORRECT / 错误</span><span style="color:{colors['UNCERTAIN']}">■ UNCERTAIN / 不确定</span></p>
<p>Conservative consistency normalizations / 保守一致性校正：{summary['conservatively_normalized_rows']}. If the model marks the reference as contradicted or unverifiable while returning a definite candidate verdict, the derived verdict is UNCERTAIN; original raw output and model-reported verdict remain preserved. / 若模型将参考答案判为相矛盾或不可核验、却仍给出确定性候选判定，则派生判定保守设为UNCERTAIN；原始输出及模型原判均保留。</p>
{"".join(bars)}
<table><thead><tr><th>Condition / 条件</th><th>CORRECT</th><th>INCORRECT</th><th>UNCERTAIN</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Visual audit × existing scoring / 视觉审计与既有评分交叉</h2>
<p>Cells are counts for the 300 selected paired rows; the visual verdict is compared with each binary score only when it is definite. / 单元格为300条入选配对记录计数；只有视觉判读为确定时，才将其与二元评分比较。</p>
<table><thead><tr><th>Scoring path / 评分路径</th><th>CORRECT × score 1</th><th>CORRECT × score 0</th><th>INCORRECT × score 1</th><th>INCORRECT × score 0</th><th>UNCERTAIN × score 1</th><th>UNCERTAIN × score 0</th></tr></thead><tbody>{"".join(cross_rows)}</tbody></table>
<h2>Agreement with existing score paths / 与既有评分路径的一致性</h2>
<p>Agreement fractions use only definite visual verdicts; UNCERTAIN rows are excluded and shown separately. / 一致比例只在视觉审计作出确定判读时计算；UNCERTAIN单独计数，不纳入分母。</p>
<table><thead><tr><th>Score path / 评分路径</th><th>Definite n</th><th>Agreement n</th><th>Disagreement n</th><th>Uncertain n</th><th>Agreement / 一致</th></tr></thead><tbody>{"".join(agreement_rows)}</tbody></table>
<h2>Disagreement examples / 分歧案例</h2><p>Showing up to 15 rows; the complete 300-row paired result with evidence is in `per-request.csv`. These purposively selected cases are illustrative, not prevalence estimates. / 最多展示15条；含证据的完整300行配对结果见`per-request.csv`。定向抽取样本仅用于案例说明，不代表总体比例。</p>
<table><thead><tr><th>Case</th><th>Track</th><th>Condition</th><th>Compared path</th><th>Visual</th><th>Score</th><th>Audit flags</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table>
<details><summary>Provenance / 来源信息</summary><pre>{html.escape(json.dumps(provenance,ensure_ascii=False,indent=2))}</pre></details>
</body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path,
                        help="Result JSONL to analyze; defaults to the untouched raw model output")
    args = parser.parse_args()
    output_dir = args.run_dir / "analysis"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty analysis directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = read_jsonl(args.run_dir / "inputs.jsonl")
    results_path = args.results or (args.run_dir / "results.jsonl")
    results = read_jsonl(results_path)
    with (args.analysis_dir / "sample_level_merged.csv").open(encoding="utf-8-sig", newline="") as handle:
        merged = list(csv.DictReader(handle))
    sample_manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    summary, per_request = analyze_records(inputs, results, merged, sample_manifest)
    with (args.run_dir / "run-manifest.json").open(encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    provenance = {
        "run_schema": run_manifest.get("schema"),
        "sample_manifest_sha256": run_manifest.get("sample_manifest_sha256"),
        "input_sha256": run_manifest.get("input_sha256"),
        "output_sha256": run_manifest.get("output_sha256"),
        "prompt_sha256": run_manifest.get("prompt_sha256"),
        "runner_commit": run_manifest.get("runner_commit"),
        "model_path": run_manifest.get("model_path"),
        "model_config_sha256": run_manifest.get("model_config_sha256"),
        "model_checkpoint_shards": len(run_manifest.get("model_shards", [])),
        "started_at_utc": run_manifest.get("started_at_utc"),
        "finished_at_utc": run_manifest.get("finished_at_utc"),
        "vllm_version": run_manifest.get("vllm_version"),
        "analysis_results_path": str(results_path.resolve()),
        "analysis_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    }
    normalization_manifest_path = args.run_dir / "normalization-manifest.json"
    if results_path.name == "results-normalized.jsonl" and normalization_manifest_path.is_file():
        normalization_manifest = json.loads(normalization_manifest_path.read_text(encoding="utf-8"))
        provenance["normalization_manifest_sha256"] = hashlib.sha256(
            normalization_manifest_path.read_bytes()
        ).hexdigest()
        provenance["normalized_results_sha256"] = normalization_manifest.get("normalized_results_sha256")
    summary["provenance"] = provenance
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per-request.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_request[0]))
        writer.writeheader()
        writer.writerows(per_request)
    document = render_html(summary, per_request, provenance)
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    print(f"PASS: summarized {summary['n_paired_requests']} paired requests; report={output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
