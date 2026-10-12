#!/usr/bin/env python3
"""Build a self-contained HTML analysis report with embedded figures."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
from pathlib import Path


def image_data(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def pct(value):
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--sample-images", type=int, default=12)
    args = parser.parse_args()
    summary = json.loads((args.derived / "summary.json").read_text(encoding="utf-8"))
    rows = summary["summary"]
    per_sample = list(csv.DictReader((args.derived / "per_sample.csv").open(encoding="utf-8-sig")))

    figure_names = [
        ("accuracy.png", "答案准确率 / Answer accuracy"),
        ("format_compliance.png", "严格输出格式遵从率 / Strict output-format compliance"),
        ("reasoning_markers.png", "词法推理标记率 / Lexical reasoning-marker rate"),
        ("response_length.png", "平均可见输出长度 / Mean visible response length"),
        ("response_length_ci.png", "输出长度及 Bootstrap 置信区间 / Response length with bootstrap confidence intervals"),
    ]
    figures = "\n".join(
        f'<figure><img class="zoomable" onclick="window.open(this.src, \'_blank\')" src="{image_data(args.figures / name)}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>'
        for name, title in figure_names
        if (args.figures / name).exists()
    )
    summary_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['condition'])}</td><td>{row['n']}</td>"
        f"<td>{pct(row['accuracy'])}</td><td>{pct(row['valid_answer_rate'])}</td>"
        f"<td>{pct(row['strict_format_rate'])}</td><td>{pct(row['reasoning_marker_rate'])}</td>"
        f"<td>{row['response_chars_mean']:.1f}</td><td>{row['response_chars_median']:.1f}</td>"
        f"<td>{row['api_error_count']}</td></tr>"
        for row in rows
    )
    paired_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['comparison'])}</td><td>{row['accuracy_difference']:+.4f}</td>"
        f"<td>{row['mcnemar']['a_only']}</td><td>{row['mcnemar']['b_only']}</td>"
        f"<td>{row['mcnemar']['p_value']:.6g}</td></tr>"
        for row in summary["paired_comparisons"]
    )
    input_comparison = ""
    if args.input_json and args.prompt_dir:
        input_payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        preview = input_payload["records"][0]
        prompt_files = {"SC": "SC-neutral.txt", "S1": "S1-concise.txt", "S2": "S2-option-letter.txt", "S3": "S3-single-character.txt"}
        system_prompts = {"S0": None}
        for condition, filename in prompt_files.items():
            system_prompts[condition] = (args.prompt_dir / filename).read_text(encoding="utf-8").strip()
        input_rows = []
        input_cards = []
        for condition in ["S0", "SC", "S1", "S2", "S3"]:
            system = system_prompts[condition]
            system_display = "（无 system message）<br>No system message" if system is None else html.escape(system).replace("\n", "<br>")
            input_rows.append(
                "<tr>"
                f"<td><b>{condition}</b></td>"
                f"<td>{system_display}</td>"
                f"<td>同一图像 / Same image<br><code>{html.escape(preview['image_archive_path'])}</code><br>SHA-256: <code>{html.escape(preview['image_sha256'])}</code></td>"
                f"<td>{html.escape(preview['query'])}</td></tr>"
            )
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:<image-mime>;base64,<same image bytes>"}},
                {"type": "text", "text": preview["query"]},
            ]})
            input_cards.append(f"<div class=\"input-card\"><h3>{condition}</h3><pre>{html.escape(json.dumps(messages, ensure_ascii=False, indent=2))}</pre></div>")
        input_comparison = (
            "<p class=\"note\"><b>真实请求结构 / Actual request structure.</b> "
            "S0 没有 system message，但仍然有包含图像和问题的 user message；五组共享同一图像、问题、模型和推理参数。<br>"
            "S0 has no system message, but it still has a user message containing the image and query. All five conditions share the same image, query, model, and inference settings.</p>"
            "<table><thead><tr><th>条件 / Condition</th><th>System message</th><th>图像 / Image</th><th>User text</th></tr></thead><tbody>"
            + "\n".join(input_rows)
            + "</tbody></table><h3>同一 case 的 messages 结构 / JSON-like messages for the same case</h3><div class=\"input-grid\">"
            + "\n".join(input_cards) + "</div>"
        )
    image_gallery = ""
    if args.image_root and args.image_manifest:
        manifest = json.loads(args.image_manifest.read_text(encoding="utf-8"))
        image_by_id = {row["sample_id"]: row for row in manifest["records"]}
        selected = []
        seen = set()
        for row in per_sample:
            sample_id = row["sample_id"]
            if sample_id not in seen and sample_id in image_by_id:
                selected.append(row)
                seen.add(sample_id)
            if len(selected) >= args.sample_images:
                break
        cards = []
        for row in selected:
            image_path = args.image_root / image_by_id[row["sample_id"]]["archive_path"]
            if not image_path.exists():
                continue
            uri = image_data(image_path)
            cards.append(
                f'<figure class="sample-card"><img class="sample-image zoomable" onclick="window.open(this.src, \'_blank\')" src="{uri}" alt="{html.escape(row["sample_id"])}">'
                f'<figcaption>{html.escape(row["sample_id"])}<br>标准答案 / Gold: {html.escape(row["gold_answer"])}</figcaption></figure>'
            )
        image_gallery = "<div class=\"sample-gallery\">" + "\n".join(cards) + "</div>"
    case_examples = ""
    if args.image_root and args.image_manifest and args.input_json and args.prompt_dir:
        manifest = json.loads(args.image_manifest.read_text(encoding="utf-8"))
        image_by_id = {row["sample_id"]: row for row in manifest["records"]}
        output_by_condition = {
            condition: {row["sample_id"]: row for row in per_sample if row["condition"] == condition}
            for condition in ["S0", "SC", "S1", "S2", "S3"]
        }
        case_cards = []
        for index, sample in enumerate(input_payload["records"][:args.sample_images], start=1):
            sample_id = sample["sample_id"]
            image_record = image_by_id.get(sample_id)
            if not image_record:
                continue
            image_path = args.image_root / image_record["archive_path"]
            if not image_path.exists():
                continue
            choices_html = "".join(f"<li>({chr(65 + i)}) {html.escape(str(choice))}</li>" for i, choice in enumerate(sample.get("choices", [])))
            modes = []
            for condition in ["S0", "SC", "S1", "S2", "S3"]:
                result = output_by_condition[condition].get(sample_id, {})
                system = system_prompts.get(condition)
                system_text = "（无 system message） / No system message" if system is None else system
                user_json = json.dumps({"image": image_record["archive_path"], "image_sha256": image_record["sha256"], "text": sample["query"]}, ensure_ascii=False, indent=2)
                modes.append(
                    f"<div class=\"mode-card\"><h4>{condition}</h4>"
                    f"<p><b>输入 / Input — system</b></p><pre>{html.escape(system_text)}</pre>"
                    f"<p><b>输入 / Input — user</b></p><pre>{html.escape(user_json)}</pre>"
                    f"<p><b>输出 / Output</b></p><pre>{html.escape(result.get('response', ''))}</pre>"
                    f"<p>解析答案 / Parsed: <b>{html.escape(str(result.get('parsed_answer', '')))}</b> · "
                    f"正确 / Correct: <b>{'Yes / 是' if result.get('correct') == 'True' or result.get('correct') is True else 'No / 否'}</b></p></div>"
                )
            uri = image_data(image_path)
            case_cards.append(
                f"<article class=\"case\"><h3>Case {index} / 样本 {index} — {html.escape(sample_id)}</h3>"
                f"<img class=\"case-image zoomable\" onclick=\"window.open(this.src, '_blank')\" src=\"{uri}\" alt=\"{html.escape(sample_id)}\">"
                f"<div class=\"question\"><b>原问题 / Original question</b><p>{html.escape(sample['question'])}</p><ol>{choices_html}</ol>"
                f"<p>标准答案 / Gold answer: <b>{html.escape(str(sample.get('response', '')))}</b></p></div>"
                f"<div class=\"mode-grid\">{' '.join(modes)}</div></article>"
            )
        case_examples = "<p class=\"note\">以下每个案例依次展示原图、原问题与选项，以及五种模式的实际输入和完整原始输出。<br>Each case shows the original image, question and choices, followed by the actual input and complete raw output under all five conditions.</p>" + "\n".join(case_cards)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>短输出提示词研究 / Short-output prompt study</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.45}}
h1,h2{{color:#18324b}} table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem;font-size:.9rem}}
th,td{{border:1px solid #ccd3da;padding:.45rem;vertical-align:top}} th{{background:#edf2f6}}
td:last-child{{max-width:52rem;white-space:pre-wrap;word-break:break-word}} .figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
figure{{margin:0;border:1px solid #d9e0e6;padding:.5rem;background:#fff}} figure img{{width:100%;height:auto}} figcaption{{text-align:center;font-weight:600;padding:.35rem}} .zoomable{{cursor:zoom-in}} .sample-gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}} .sample-card{{font-size:.78rem}} .sample-image{{max-height:260px;object-fit:contain;background:#f2f4f6}}
.note{{background:#f5f7f9;border-left:4px solid #4c78a8;padding:.7rem 1rem}} pre{{background:#f5f7f9;border-left:4px solid #4c78a8;padding:.7rem;white-space:pre-wrap;word-break:break-word}} .input-grid,.mode-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}} .input-card,.mode-card{{border:1px solid #ccd3da;padding:.5rem;background:#fff}} .input-card h3,.mode-card h4{{margin:.2rem 0;color:#18324b}} .case{{border:2px solid #9aafc2;padding:1rem;margin:2rem 0;background:#fff}} .case-image{{display:block;max-width:100%;max-height:520px;margin:0 auto 1rem;object-fit:contain;background:#f2f4f6}} .question{{background:#f7fafc;padding:.75rem;margin-bottom:1rem}}
</style></head><body>
<h1>短输出提示词研究 / Short-output prompt study</h1>
<p class="note">本报告基于冻结的配对样本和保存的模型原始输出生成。所有统计图与代表性样本图像均已内嵌在本 HTML 文件中。<br>This report is generated from the frozen paired sample and preserved raw model responses. All figures and representative sample images are embedded in this HTML file.</p>
<h2>输入对照 / Input comparison</h2>{input_comparison}
<h2>结果摘要 / Summary</h2>
<table><thead><tr><th>条件<br>Condition</th><th>样本数<br>N</th><th>准确率<br>Accuracy</th><th>有效答案率<br>Valid answer</th><th>严格格式遵从率<br>Strict format</th><th>推理标记率<br>Reasoning marker</th><th>平均字符数<br>Mean chars</th><th>字符数中位数<br>Median chars</th><th>API 错误<br>API errors</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>相对 S0 的配对比较 / Paired comparisons against S0</h2>
<table><thead><tr><th>比较<br>Comparison</th><th>准确率差异<br>Accuracy difference</th><th>仅 S0 正确<br>S0 only correct</th><th>仅当前条件正确<br>Condition only correct</th><th>McNemar p 值<br>McNemar p</th></tr></thead><tbody>{paired_rows}</tbody></table>
<h2>统计图 / Figures</h2><div class="figures">{figures}</div>
<h2>具体案例 / Concrete cases</h2>{case_examples}
<h2>解释边界 / Interpretation limits</h2>
<ul><li>词法推理标记率是文本筛查指标，并不等于已证明模型存在可验证的隐藏推理过程。<br>The reasoning-marker rate is a lexical screening measure, not evidence of a hidden reasoning process.</li><li>所有提示词条件共享相同的图像、问题、模型、抽样框、temperature 和 thinking 设置。<br>All prompt conditions share the same image, query, model, sampling frame, temperature, and thinking setting.</li><li>统计结论应结合原始输出检查和配对样本设计共同解释。<br>Statistical conclusions should be interpreted jointly with raw-output inspection and the paired sample design.</li></ul>
</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
