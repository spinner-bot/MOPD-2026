"""
引导性问题质量检查脚本

检查生成的引导性问题质量，包括：
- 格式正确性
- 问题数量
- 问题长度
- 重复检测
- 简单统计分析

用法：
    python quality_check.py --input-file data/guiding_questions/raw_generated/batch_001.json
"""

import argparse
import json
from typing import List, Dict, Any
from pathlib import Path
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查引导性问题质量")
    parser.add_argument("--input-file", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-report", help="输出报告路径（可选）")
    return parser.parse_args()


def check_quality(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检查质量并返回统计信息"""

    total = len(results)
    success_count = sum(1 for r in results if r.get("success", False))
    fail_count = total - success_count

    # 统计引导问题
    all_questions = []
    questions_per_sample = []
    question_lengths = []

    for r in results:
        if r.get("success", False):
            questions = r.get("guiding_questions", [])
            questions_per_sample.append(len(questions))
            all_questions.extend(questions)
            question_lengths.extend([len(q) for q in questions])

    # 基本统计
    stats = {
        "total_samples": total,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": success_count / total if total > 0 else 0,
        "total_questions": len(all_questions),
        "avg_questions_per_sample": sum(questions_per_sample) / len(questions_per_sample) if questions_per_sample else 0,
        "min_questions": min(questions_per_sample) if questions_per_sample else 0,
        "max_questions": max(questions_per_sample) if questions_per_sample else 0,
        "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
        "min_question_length": min(question_lengths) if question_lengths else 0,
        "max_question_length": max(question_lengths) if question_lengths else 0,
    }

    # 问题长度分布
    length_buckets = {"<10": 0, "10-20": 0, "20-30": 0, "30-50": 0, ">50": 0}
    for length in question_lengths:
        if length < 10:
            length_buckets["<10"] += 1
        elif length < 20:
            length_buckets["10-20"] += 1
        elif length < 30:
            length_buckets["20-30"] += 1
        elif length < 50:
            length_buckets["30-50"] += 1
        else:
            length_buckets[">50"] += 1

    stats["length_distribution"] = length_buckets

    # 重复检测
    question_counter = Counter(all_questions)
    duplicates = {q: count for q, count in question_counter.items() if count > 1}
    stats["duplicate_questions"] = len(duplicates)
    stats["duplicate_rate"] = len(duplicates) / len(all_questions) if all_questions else 0

    # 质量检查
    issues = []

    # 检查 1: 成功率
    if stats["success_rate"] < 0.95:
        issues.append(f"成功率过低: {stats['success_rate']:.2%} < 95%")

    # 检查 2: 问题数量
    if stats["avg_questions_per_sample"] < 2:
        issues.append(f"平均问题数过少: {stats['avg_questions_per_sample']:.2f} < 2")
    if stats["avg_questions_per_sample"] > 4:
        issues.append(f"平均问题数过多: {stats['avg_questions_per_sample']:.2f} > 4")

    # 检查 3: 问题长度
    if stats["avg_question_length"] > 50:
        issues.append(f"平均问题长度过长: {stats['avg_question_length']:.1f} > 50 字")

    # 检查 4: 重复率
    if stats["duplicate_rate"] > 0.05:
        issues.append(f"重复率过高: {stats['duplicate_rate']:.2%} > 5%")

    # 检查 5: 过短或过长的问题
    short_questions = sum(1 for l in question_lengths if l < 5)
    long_questions = sum(1 for l in question_lengths if l > 100)
    if short_questions > 0:
        issues.append(f"发现 {short_questions} 个过短问题（<5 字）")
    if long_questions > 0:
        issues.append(f"发现 {long_questions} 个过长问题（>100 字）")

    stats["issues"] = issues
    stats["passed"] = len(issues) == 0

    return stats


def print_report(stats: Dict[str, Any]):
    """打印质量报告"""
    print("=" * 70)
    print("引导性问题质量检查报告")
    print("=" * 70)

    print("\n【基本统计】")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  成功生成: {stats['success_count']} ({stats['success_rate']:.2%})")
    print(f"  生成失败: {stats['fail_count']}")

    print("\n【引导问题统计】")
    print(f"  总问题数: {stats['total_questions']}")
    print(f"  平均每样本: {stats['avg_questions_per_sample']:.2f} 个")
    print(f"  最少: {stats['min_questions']} 个")
    print(f"  最多: {stats['max_questions']} 个")

    print("\n【问题长度统计】")
    print(f"  平均长度: {stats['avg_question_length']:.1f} 字")
    print(f"  最短: {stats['min_question_length']} 字")
    print(f"  最长: {stats['max_question_length']} 字")

    print("\n【长度分布】")
    for bucket, count in stats["length_distribution"].items():
        pct = count / stats["total_questions"] if stats["total_questions"] > 0 else 0
        print(f"  {bucket:6s}: {count:4d} ({pct:.1%})")

    print("\n【重复检测】")
    print(f"  重复问题数: {stats['duplicate_questions']}")
    print(f"  重复率: {stats['duplicate_rate']:.2%}")

    print("\n【质量检查】")
    if stats["passed"]:
        print("  ✓ 所有检查通过！")
    else:
        print(f"  ✗ 发现 {len(stats['issues'])} 个问题:")
        for i, issue in enumerate(stats["issues"], 1):
            print(f"    {i}. {issue}")

    print("\n" + "=" * 70)


def main():
    args = parse_args()

    # 读取数据
    print(f"Reading: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Total samples: {len(results)}")

    # 检查质量
    stats = check_quality(results)

    # 打印报告
    print_report(stats)

    # 保存报告（如果指定）
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "input_file": args.input_file,
            "statistics": stats,
            "passed": stats["passed"]
        }

        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()
