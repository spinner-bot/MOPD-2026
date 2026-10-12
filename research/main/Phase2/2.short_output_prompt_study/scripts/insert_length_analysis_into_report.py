#!/usr/bin/env python3
"""Insert the length/correctness analysis before the concrete-case section."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.read_text(encoding="utf-8")
    marker = "<h2>具体案例 / Concrete cases</h2>"
    if marker not in report:
        raise SystemExit("concrete-case marker not found")
    if "输出长度与答题正确性分析 / Response length and correctness" in report:
        raise SystemExit("analysis already present")
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    rows = "\n".join(
        f"<tr><td>{html.escape(r['condition'])}</td><td>{r['mean_chars']:.1f}</td><td>{r['median_chars']:.1f}</td>"
        f"<td>{r['skewness']:.2f}</td><td>{r['pearson_r']:+.3f}</td><td>{r['spearman_rho']:+.3f}</td>"
        f"<td>{r['correct_mean_chars']:.1f}</td><td>{r['incorrect_mean_chars']:.1f}</td></tr>"
        for r in stats["conditions"]
    )
    paired = "\n".join(
        f"<tr><td>{html.escape(r['comparison'])}</td><td>{r['accuracy_difference']:+.4f}</td>"
        f"<td>{r['a_only_correct']}</td><td>{r['b_only_correct']}</td><td>{r['p_value']:.6g}</td></tr>"
        for r in stats["paired_comparisons"]
    )
    figures = []
    for name, title in [
        ("length_by_correctness.png", "答对与答错样本的输出长度 / Output length by correctness"),
        ("accuracy_by_length_decile.png", "输出长度分位组与准确率 / Accuracy by response-length decile"),
        ("length_distribution_shape.png", "均值、中位数与偏度 / Mean, median, and skewness"),
    ]:
        path = args.figure_dir / name
        if path.exists():
            figures.append(f'<figure><img class="zoomable" onclick="window.open(this.src, \'_blank\')" src="{data_uri(path)}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>')
    section = f"""
<section id="length-correctness-analysis">
<h2>输出长度与答题正确性分析 / Response length and correctness</h2>
<p>本节在具体案例之前给出逐样本统计。长度指标为可见输出字符数，不等同于 tokenizer 计算的 token 数。相关系数描述关联，不作因果解释；推理标记率也仅是词法筛查指标。</p>
<p>This section reports per-sample statistics before the concrete cases. Length is measured in visible response characters, not tokenizer-level tokens. Correlations describe association rather than causation; the reasoning-marker rate is lexical screening only.</p>
<table><thead><tr><th>条件<br>Condition</th><th>平均字符数<br>Mean</th><th>中位数<br>Median</th><th>偏度<br>Skewness</th><th>Pearson r<br>Length–accuracy</th><th>Spearman ρ<br>Length–accuracy</th><th>答对平均<br>Correct mean</th><th>答错平均<br>Incorrect mean</th></tr></thead><tbody>{rows}</tbody></table>
<p><b>主要结果 / Main findings.</b> 五组平均字符数均高于中位数，且偏度均为正，支持输出长度呈右偏分布。五组长度—正确性相关均为负，强度总体较弱；S3 的 Spearman ρ 为 −0.301，最明显。每组答对样本的平均输出均短于答错样本，但这不能单独证明缩短输出会导致准确率提升。</p>
<p>All five conditions have means above medians and positive skewness, supporting right-skewed output-length distributions. All length–correctness associations are negative and generally weak; S3 is the strongest by Spearman correlation (ρ = −0.301). Correct samples are shorter on average in every condition, but this does not establish that shortening causes higher accuracy.</p>
<table><thead><tr><th>配对比较<br>Paired comparison</th><th>准确率差<br>Accuracy diff.</th><th>仅前者答对<br>First only</th><th>仅后者答对<br>Second only</th><th>双侧 p<br>Two-sided p</th></tr></thead><tbody>{paired}</tbody></table>
<p>S1 与 S2 的准确率差为 +13.75 个百分点，S3 与 S2 的差为 +11.15 个百分点；两项配对比较均显著。S1 与 S3 的差为 +2.60 个百分点，实际幅度较小。S0 与 SC 的准确率差仅为 +0.25 个百分点，不能据此认为内容中性的 system prompt 带来了准确率改善。</p>
<div class="figures">{''.join(figures)}</div>
</section>
"""
    report = report.replace(marker, section + marker, 1)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(args.report), "bytes": args.report.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
