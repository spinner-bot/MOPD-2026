#!/usr/bin/env python3
"""Insert the research-direction assessment before concrete cases."""

from __future__ import annotations

import argparse
from pathlib import Path


ASSESSMENT = """
<section id="research-direction-assessment">
<h2>研究方向评估 / Research-direction assessment</h2>
<p><b>当前判断 / Current assessment.</b> 本轮结果表明，system prompt 对模型输出行为具有可观测影响，但“严格控制输出长度或 token 数”这一目标尚未实现。因而，当前方向不应被表述为已经解决的短输出控制方法，而应定位为“输入约束如何改变视觉问答输出行为与答题表现”的初步实证研究。</p>
<p>The results show that system prompts measurably affect output behavior, but strict control of output length or token count has not been achieved. The direction should therefore be framed as an initial empirical study of how input constraints alter visual-question-answering behavior and performance, rather than as a validated short-output control method.</p>
<h3>已经得到的证据 / Evidence established in this run</h3>
<ul>
<li>S1、S3 的准确率显著高于 S2；该结论来自相同样本上的配对比较。</li>
<li>五种条件的严格格式遵从率均为 0%，说明输出格式约束仍未稳定生效。</li>
<li>五组输出长度均呈右偏分布，且输出字符数与正确性均呈负相关，但相关强度总体较弱。</li>
<li>内容中性的 SC 与无 system message 的 S0 在准确率和总体输出长度上基本一致。</li>
</ul>
<h3>方向价值 / Research value</h3>
<p>本方向仍具有研究价值，原因是输出长度、格式遵从、答题准确率和答对/答错样本之间已经出现可重复的差异结构。尤其是 S1 与 S3 相对 S2 的优势，值得继续检验其是否来自约束强度、回答形式、视觉判断保留程度或其他输入因素。但这些机制目前尚未被本轮实验单独识别。</p>
<p>The direction remains promising because reproducible differences have appeared across output length, format compliance, accuracy, and correct/incorrect samples. The advantage of S1 and S3 over S2 warrants further testing, while the responsible mechanism has not yet been isolated.</p>
<h3>下一阶段的必要收敛 / Necessary next-stage refinement</h3>
<ol>
<li>增加 tokenizer 级输入和输出 token 统计，避免把字符数直接当作 token 数。</li>
<li>将“格式控制能力”和“准确率变化”分开设计与评价，避免把二者混为同一机制。</li>
<li>对答对/答错样本以及长尾样本按任务类型、题目难度和图像复杂度分层。</li>
<li>用预先定义的长度区间、格式合规指标和配对统计检验复核 S1/S3 的优势。</li>
<li>增加重复运行或不同随机种子，确认当前差异不是单次采样结果。</li>
</ol>
<p><b>阶段性结论 / Stage conclusion.</b> 当前结果支持继续研究“适度输出约束与视觉问答性能的关系”，但不支持宣称已经获得可靠的 token 控制或蒸馏方法。后续工作的核心应从“追求更短输出”转为“识别何种约束在不损害视觉判断的前提下改善输出行为与准确率”。</p>
</section>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.read_text(encoding="utf-8")
    marker = "<h2>具体案例 / Concrete cases</h2>"
    if marker not in report:
        raise SystemExit("concrete-case marker not found")
    if "研究方向评估 / Research-direction assessment" in report:
        raise SystemExit("assessment already present")
    args.report.write_text(report.replace(marker, ASSESSMENT + marker, 1), encoding="utf-8")
    print(f"updated {args.report} ({args.report.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
