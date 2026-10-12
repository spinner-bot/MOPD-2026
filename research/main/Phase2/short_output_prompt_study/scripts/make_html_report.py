#!/usr/bin/env python3
"""Build a self-contained HTML analysis report with embedded figures."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path


def image_data(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pct(value):
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        f'<figure><img src="{image_data(args.figures / name)}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>'
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
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Short-output prompt study</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.45}}
h1,h2{{color:#18324b}} table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem;font-size:.9rem}}
th,td{{border:1px solid #ccd3da;padding:.45rem;vertical-align:top}} th{{background:#edf2f6}}
td:last-child{{max-width:52rem;white-space:pre-wrap;word-break:break-word}} .figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
figure{{margin:0;border:1px solid #d9e0e6;padding:.5rem;background:#fff}} figure img{{width:100%;height:auto}} figcaption{{text-align:center;font-weight:600;padding:.35rem}}
.note{{background:#f5f7f9;border-left:4px solid #4c78a8;padding:.7rem 1rem}}
</style></head><body>
<h1>Short-output prompt study</h1>
<p class="note">This report is generated from the frozen paired sample and the preserved raw model responses. All displayed figures are embedded in this HTML file.</p>
<h2>Summary</h2>
<table><thead><tr><th>Condition</th><th>N</th><th>Accuracy</th><th>Valid answer</th><th>Strict format</th><th>Reasoning marker</th><th>Mean chars</th><th>Median chars</th><th>API errors</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Paired comparisons against S0</h2>
<table><thead><tr><th>Comparison</th><th>Accuracy difference</th><th>S0 only correct</th><th>Condition only correct</th><th>McNemar p</th></tr></thead><tbody>{paired_rows}</tbody></table>
<h2>Figures</h2><div class="figures">{figures}</div>
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
