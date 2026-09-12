#!/usr/bin/env python3
"""
分析 teacher-gap screen/construct 作业结果。

功能：
1. 统计截断率（truncated_or_missing_eos）
2. 分类统计 image_only / external_required / ambiguous
3. 与 C2 aligned 包联结生成人工审核优先队列
4. 对比 r1 vs r2 修复效果

使用：
    python analyze_teacher_gap_results.py --run-dir <archive_dir>
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter


def load_jsonl(path: Path) -> list:
    """加载 JSONL 文件"""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_screen_output(run_dir: Path) -> dict:
    """分析 screen 作业输出"""
    result = {
        "run_dir": str(run_dir),
        "total": 0,
        "parsed_valid": 0,
        "truncated": 0,
        "invalid_json": 0,
        "image_only": 0,
        "external_required": 0,
        "ambiguous": 0,
        "candidates": [],
    }

    # 查找输出文件
    output_dir = run_dir / "raw" / "summaries"
    if not output_dir.exists():
        print(f"警告: {output_dir} 不存在", file=sys.stderr)
        return result

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        records = load_jsonl(jsonl_file)
        for rec in records:
            result["total"] += 1

            # 检查截断
            if rec.get("truncated") or rec.get("truncated_output"):
                result["truncated"] += 1
                continue

            # 检查解析
            if rec.get("parse_error") or not rec.get("parsed"):
                result["invalid_json"] += 1
                continue

            result["parsed_valid"] += 1

            # 分类
            candidate = {
                "candidate_id": rec.get("candidate_id"),
                "category": rec.get("category", "unknown"),
                "reason": rec.get("reason", ""),
            }

            category = rec.get("category", "").lower()
            if "image_only" in category or "image-only" in category:
                result["image_only"] += 1
            elif "external" in category:
                result["external_required"] += 1
            elif "ambiguous" in category:
                result["ambiguous"] += 1

            result["candidates"].append(candidate)

    return result


def analyze_construct_output(run_dir: Path) -> dict:
    """分析 construct 作业输出"""
    result = {
        "run_dir": str(run_dir),
        "total": 0,
        "constructed": 0,
        "failed": 0,
        "records": [],
    }

    output_dir = run_dir / "processed"
    if not output_dir.exists():
        print(f"警告: {output_dir} 不存在", file=sys.stderr)
        return result

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        records = load_jsonl(jsonl_file)
        for rec in records:
            result["total"] += 1

            if rec.get("constructed"):
                result["constructed"] += 1
            else:
                result["failed"] += 1

            result["records"].append({
                "candidate_id": rec.get("candidate_id"),
                "guide_count": len(rec.get("guides", [])),
                "leakage_check": rec.get("leakage_check", "unknown"),
            })

    return result


def compare_r1_r2(r1_dir: Path, r2_dir: Path) -> dict:
    """对比 r1 和 r2 的修复效果"""
    r1_result = analyze_screen_output(r1_dir)
    r2_result = analyze_screen_output(r2_dir)

    comparison = {
        "r1": {
            "total": r1_result["total"],
            "truncated": r1_result["truncated"],
            "truncation_rate": r1_result["truncated"] / max(r1_result["total"], 1),
        },
        "r2": {
            "total": r2_result["total"],
            "truncated": r2_result["truncated"],
            "truncation_rate": r2_result["truncated"] / max(r2_result["total"], 1),
        },
        "improvement": None,
    }

    if r1_result["total"] > 0 and r2_result["total"] > 0:
        r1_rate = r1_result["truncated"] / r1_result["total"]
        r2_rate = r2_result["truncated"] / r2_result["total"]
        comparison["improvement"] = {
            "absolute_reduction": r1_rate - r2_rate,
            "relative_reduction": (r1_rate - r2_rate) / r1_rate if r1_rate > 0 else None,
        }

    return comparison


def generate_review_priority(screen_result: dict, aligned_packet_path: Path = None) -> list:
    """生成人工审核优先队列"""
    priority_queue = []

    # 优先级规则：
    # 1. image_only 且无歧义 -> 高优先级
    # 2. ambiguous -> 中优先级
    # 3. external_required -> 低优先级（可能需要重构）

    for candidate in screen_result.get("candidates", []):
        category = candidate.get("category", "").lower()

        if "image_only" in category:
            priority = "high"
        elif "ambiguous" in category:
            priority = "medium"
        elif "external" in category:
            priority = "low"
        else:
            priority = "unknown"

        priority_queue.append({
            "candidate_id": candidate.get("candidate_id"),
            "priority": priority,
            "category": candidate.get("category"),
            "reason": candidate.get("reason"),
        })

    # 按优先级排序
    priority_order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    priority_queue.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return priority_queue


def main():
    parser = argparse.ArgumentParser(description="分析 teacher-gap 作业结果")
    parser.add_argument("--run-dir", type=Path, required=True, help="作业档案目录")
    parser.add_argument("--compare-r1", type=Path, help="r1 档案目录（用于对比）")
    parser.add_argument("--aligned-packet", type=Path, help="C2 aligned 包路径")
    parser.add_argument("--output", type=Path, help="输出 JSON 路径")

    args = parser.parse_args()

    # 判断是 screen 还是 construct
    run_name = args.run_dir.name.lower()
    if "screen" in run_name:
        result = analyze_screen_output(args.run_dir)
    elif "construct" in run_name:
        result = analyze_construct_output(args.run_dir)
    else:
        print("错误: 无法判断作业类型（名称中应包含 screen 或 construct）", file=sys.stderr)
        sys.exit(1)

    # 对比 r1 vs r2
    if args.compare_r1:
        comparison = compare_r1_r2(args.compare_r1, args.run_dir)
        result["comparison_with_r1"] = comparison

    # 生成审核优先队列（仅 screen）
    if "screen" in run_name:
        priority_queue = generate_review_priority(result, args.aligned_packet)
        result["review_priority_queue"] = priority_queue

    # 输出
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"结果已写入: {args.output}")
    else:
        print(output_json)

    # 打印摘要
    print("\n=== 摘要 ===")
    if "screen" in run_name:
        print(f"总记录数: {result['total']}")
        print(f"有效解析: {result['parsed_valid']}")
        print(f"截断: {result['truncated']} ({result['truncated']/max(result['total'],1)*100:.1f}%)")
        print(f"无效JSON: {result['invalid_json']}")
        print(f"\n分类统计:")
        print(f"  image_only: {result['image_only']}")
        print(f"  external_required: {result['external_required']}")
        print(f"  ambiguous: {result['ambiguous']}")
    elif "construct" in run_name:
        print(f"总记录数: {result['total']}")
        print(f"成功构建: {result['constructed']}")
        print(f"失败: {result['failed']}")


if __name__ == "__main__":
    main()
