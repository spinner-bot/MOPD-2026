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
    parser.add_argument("--sample-images", type=int, default=12)
    args = parser.parse_args()
    summary = json.loads((args.derived / "summary.json").read_text(encoding="utf-8"))
    rows = summary["summary"]
    per_sample = list(csv.DictReader((args.derived / "per_sample.csv").open(encoding="utf-8-sig")))

    figure_names = [
        ("accuracy.png", "Answer accuracy"),
        ("format_compliance.png", "Strict output-format compliance"),
        ("reasoning_markers.png", "Lexical reasoning-marker rate"),
        ("response_length.png", "Mean visible response length"),
        ("response_length_ci.png", "Response length with bootstrap confidence intervals"),
    ]
    figures = "\n".join(
        f'<figure><img class="zoomable" onclick="window.open(this.src, \'_blank\')" src="{image_data(args.figures / name)}" alt="{html.escape(title)}"><figcaption>{html.escape(title)} · click to enlarge</figcaption></figure>'
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
    representative = []
    for condition in ["S0", "SC", "S1", "S2", "S3"]:
        representative.extend([row for row in per_sample if row["condition"] == condition][:2])
    sample_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['condition'])}</td><td>{html.escape(row['sample_id'])}</td>"
        f"<td>{html.escape(row['gold_answer'])}</td><td>{html.escape(row['parsed_answer'])}</td>"
        f"<td>{html.escape(row['response'][:800])}</td></tr>"
        for row in representative
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
                f'<figcaption>{html.escape(row["sample_id"])}<br>Gold: {html.escape(row["gold_answer"])} · click image for full resolution</figcaption></figure>'
            )
        image_gallery = "<div class=\"sample-gallery\">" + "\n".join(cards) + "</div>"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Short-output prompt study</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.45}}
h1,h2{{color:#18324b}} table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem;font-size:.9rem}}
th,td{{border:1px solid #ccd3da;padding:.45rem;vertical-align:top}} th{{background:#edf2f6}}
td:last-child{{max-width:52rem;white-space:pre-wrap;word-break:break-word}} .figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
figure{{margin:0;border:1px solid #d9e0e6;padding:.5rem;background:#fff}} figure img{{width:100%;height:auto}} figcaption{{text-align:center;font-weight:600;padding:.35rem}} .zoomable{{cursor:zoom-in}} .sample-gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}} .sample-card{{font-size:.78rem}} .sample-image{{max-height:260px;object-fit:contain;background:#f2f4f6}}
.note{{background:#f5f7f9;border-left:4px solid #4c78a8;padding:.7rem 1rem}}
</style></head><body>
<h1>Short-output prompt study</h1>
<p class="note">This report is generated from the frozen paired sample and the preserved raw model responses. All displayed figures are embedded in this HTML file.</p>
<h2>Summary</h2>
<table><thead><tr><th>Condition</th><th>N</th><th>Accuracy</th><th>Valid answer</th><th>Strict format</th><th>Reasoning marker</th><th>Mean chars</th><th>Median chars</th><th>API errors</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Paired comparisons against S0</h2>
<table><thead><tr><th>Comparison</th><th>Accuracy difference</th><th>S0 only correct</th><th>Condition only correct</th><th>McNemar p</th></tr></thead><tbody>{paired_rows}</tbody></table>
<h2>Figures</h2><div class="figures">{figures}</div>
<h2>Embedded sample images</h2><p class="note">The following representative images are embedded at their original resolution. Click any image to open the embedded high-resolution image in a new tab.</p>{image_gallery}
<h2>Representative raw responses</h2>
<table><thead><tr><th>Condition</th><th>Sample</th><th>Gold</th><th>Parsed</th><th>Raw response excerpt</th></tr></thead><tbody>{sample_rows}</tbody></table>
<h2>Interpretation limits</h2>
<ul><li>The reasoning-marker rate is a lexical screening measure, not evidence of a hidden reasoning process.</li><li>All prompt conditions share the same image, query, model, sampling frame, temperature, and thinking setting.</li><li>Statistical conclusions should be interpreted jointly with raw-output inspection and the paired sample design.</li></ul>
</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
