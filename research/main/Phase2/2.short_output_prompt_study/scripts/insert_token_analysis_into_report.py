#!/usr/bin/env python3
"""Insert the authoritative vLLM token-usage analysis before direction assessment."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path


def uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    text = args.report.read_text(encoding="utf-8")
    marker = '<section id="research-direction-assessment">'
    if marker not in text:
        raise SystemExit("direction-assessment marker not found")
    if "Tokenizer 级长度与长尾分析" in text:
        raise SystemExit("token analysis already present")
    data = json.loads(args.stats.read_text(encoding="utf-8"))
    rows = "\n".join(
        f"<tr><td>{html.escape(r['condition'])}</td><td>{r['prompt_tokens_mean']:.1f}</td><td>{r['prompt_tokens_median']:.1f}</td>"
        f"<td>{r['completion_tokens_mean']:.1f}</td><td>{r['completion_tokens_median']:.1f}</td><td>{r['completion_tokens_skewness']:.2f}</td>"
        f"<td>{r['pearson_r']:+.3f}</td><td>{r['spearman_rho']:+.3f}</td><td>{r['correct_mean_tokens']:.1f}</td><td>{r['incorrect_mean_tokens']:.1f}</td></tr>"
        for r in data["conditions"]
    )
    tail_rows = "\n".join(
        f"<tr><td>{html.escape(r['condition'])}</td><td>{r['completion_tokens_q95']:.1f}</td><td>{r['completion_tokens_q99']:.1f}</td>"
        f"<td>{r['completion_tokens_max']}</td><td>{r['p95_tail_n']}</td><td>{r['p95_tail_accuracy']:.1%}</td></tr>"
        for r in data["conditions"]
    )
    figures = []
    for name, title in [
        ("token_length_by_correctness.png", "答对/答错的 completion token 长度 / Completion-token length by correctness"),
        ("token_accuracy_by_decile.png", "completion token 分位组准确率 / Accuracy by completion-token decile"),
        ("token_distribution_shape.png", "completion token 分布形态 / Completion-token distribution shape"),
        ("token_p95_tail_task_accuracy.png", "P95 长尾样本任务分布 / P95-tail task accuracy"),
    ]:
        path = args.figure_dir / name
        if path.exists():
            figures.append(f'<figure><img class="zoomable" onclick="window.open(this.src, \'_blank\')" src="{uri(path)}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>')
    section = f"""
<section id="token-length-analysis">
<h2>Tokenizer 级长度与长尾分析 / Tokenizer-level length and long-tail analysis</h2>
<p>本节使用正式 vLLM 推理记录中的 <code>usage.prompt_tokens</code>、<code>usage.completion_tokens</code> 和 <code>usage.total_tokens</code>。五组共 10000 条记录全部具有完整 token usage，模型标识均为 <code>OPD-V-Qwen3.5-4B</code>。该口径来自正式推理服务，不是替代 tokenizer 的离线估计。</p>
<p>This section uses <code>usage.prompt_tokens</code>, <code>usage.completion_tokens</code>, and <code>usage.total_tokens</code> recorded by the formal vLLM run. All 10,000 records have complete usage fields and the same model identifier, <code>OPD-V-Qwen3.5-4B</code>. These are authoritative service-side counts, not estimates from a substitute tokenizer.</p>
<table><thead><tr><th>条件<br>Condition</th><th>输入 token 均值<br>Prompt mean</th><th>输入 token 中位数<br>Prompt median</th><th>输出 token 均值<br>Completion mean</th><th>输出 token 中位数<br>Completion median</th><th>输出偏度<br>Skewness</th><th>Pearson r<br>Length–accuracy</th><th>Spearman ρ<br>Length–accuracy</th><th>答对均值<br>Correct mean</th><th>答错均值<br>Incorrect mean</th></tr></thead><tbody>{rows}</tbody></table>
<p><b>主要结果 / Main findings.</b> S1 和 S3 的平均 completion token 数分别为 60.4 和 55.7，低于 S2 的 95.1；五组答对样本的平均 completion token 数均低于答错样本。长度—正确性相关在五组均为负，S3 的 Spearman ρ 为 −0.324，当前最明显。该结果支持关联性判断，不证明缩短输出导致准确率提升。</p>
<p>S1 and S3 have the shortest mean completion lengths (60.4 and 55.7 tokens), below S2 (95.1 tokens). Correct samples are shorter than incorrect samples in every condition. All length–correctness associations are negative; S3 is strongest by Spearman correlation (ρ = −0.324). This supports association, not a causal claim that shortening improves accuracy.</p>
<table><thead><tr><th>条件<br>Condition</th><th>P95 token</th><th>P99 token</th><th>最大值<br>Maximum</th><th>P95 长尾样本数<br>P95-tail N</th><th>P95 长尾准确率<br>P95-tail accuracy</th></tr></thead><tbody>{tail_rows}</tbody></table>
<p>P95/P99 与最大值之间存在明显长尾，尤其 S0、SC 和 S2 的极端值较高。P95 长尾样本的准确率并不统一，不能简单视为“长输出必然错误”；后续应结合任务类型、原始输出内容和是否出现重复/无关解释进行人工或规则审计。</p>
<p>The gap between P95/P99 and the maximum confirms substantial long tails, especially for S0, SC, and S2. P95-tail accuracy is not uniformly low, so long outputs cannot be treated as automatically incorrect; task type and raw-response behavior require separate inspection.</p>
<div class="figures">{''.join(figures)}</div>
</section>
"""
    args.report.write_text(text.replace(marker, section + marker, 1), encoding="utf-8")
    print(f"updated {args.report} ({args.report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
