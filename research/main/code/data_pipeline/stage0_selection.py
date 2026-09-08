"""
Stage 0 样本选择

按实验方案 §19 从 Diagnostic Test 中抽取 400-600 样本用于 Privilege Gap Screening。

覆盖要求：
- P / R / PR 三类任务
- 四大 guidance strategies
- Depth 1-4+
"""

import numpy as np
from collections import defaultdict
from typing import Optional
import json
import os


def depth_category(depth: int) -> str:
    """Return the protocol's four depth strata: D1, D2, D3, and D4+."""
    return f"D{depth}" if depth in (1, 2, 3) else "D4+"


def load_test_samples(jsonl_path: str) -> list[dict]:
    """加载 test 集样本"""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def stratify_samples(
    samples: list[dict],
    task_family_key: str = "task_family",
    strategy_key: str = "guidance_strategy",
    depth_key: str = "reasoning_depth"
) -> dict[tuple, list[dict]]:
    """
    按 (task_family, strategy, depth) 分层

    Returns:
        {(P, localization, D1): [sample, ...], ...}
    """
    strata = defaultdict(list)

    for sample in samples:
        family = sample.get(task_family_key, "P")
        strategy = sample.get(strategy_key, "localization")
        depth = sample.get(depth_key, 1)

        depth_cat = depth_category(depth)

        key = (family, strategy, depth_cat)
        strata[key].append(sample)

    return dict(strata)


def select_stage0_samples(
    test_samples: list[dict],
    target_size: int = 500,
    task_family_key: str = "task_family",
    strategy_key: str = "guidance_strategy",
    depth_key: str = "reasoning_depth",
    seed: int = 42
) -> list[dict]:
    """
    选择 Stage 0 样本（方案 §19）

    确保覆盖：
    - P / R / PR 三类任务
    - 四大 guidance strategies
    - Depth 1-4+

    Args:
        test_samples: Diagnostic Test 集样本
        target_size: 目标样本数（400-600）
        task_family_key: 任务类型字段名
        strategy_key: 策略字段名
        depth_key: 深度字段名
        seed: 随机种子

    Returns:
        选中的样本列表
    """
    np.random.seed(seed)

    # 分层
    strata = stratify_samples(test_samples, task_family_key, strategy_key, depth_key)

    # 目标：每个 stratum 尽量均匀分配
    n_strata = len(strata)
    if n_strata == 0:
        return []

    # 基础配额
    base_quota = target_size // n_strata

    selected = []
    remaining_quota = target_size

    # 第一轮：每个 stratum 取基础配额
    strata_remaining = {}
    for key, stratum_samples in strata.items():
        n_select = min(base_quota, len(stratum_samples))
        np.random.shuffle(stratum_samples)
        selected.extend(stratum_samples[:n_select])
        strata_remaining[key] = stratum_samples[n_select:]
        remaining_quota -= n_select

    # 第二轮：用剩余配额填充还有样本的 strata
    if remaining_quota > 0:
        available_keys = [k for k, v in strata_remaining.items() if len(v) > 0]
        np.random.shuffle(available_keys)

        for key in available_keys:
            if remaining_quota <= 0:
                break
            available = strata_remaining[key]
            n_add = min(len(available), remaining_quota)
            selected.extend(available[:n_add])
            remaining_quota -= n_add

    return selected


def validate_stage0_coverage(
    selected: list[dict],
    task_family_key: str = "task_family",
    strategy_key: str = "guidance_strategy",
    depth_key: str = "reasoning_depth"
) -> dict:
    """
    验证 Stage 0 覆盖度

    Returns:
        {
            "total": 总数,
            "family_coverage": {P: n, R: n, PR: n},
            "strategy_coverage": {localization: n, ...},
            "depth_coverage": {D1: n, D2: n, D3: n, D4+: n},
            "gaps": [缺失的组合]
        }
    """
    family_counts = defaultdict(int)
    strategy_counts = defaultdict(int)
    depth_counts = defaultdict(int)
    combo_coverage = set()

    for sample in selected:
        family = sample.get(task_family_key, "P")
        strategy = sample.get(strategy_key, "localization")
        depth = sample.get(depth_key, 1)
        depth_cat = depth_category(depth)

        family_counts[family] += 1
        strategy_counts[strategy] += 1
        depth_counts[depth_cat] += 1
        combo_coverage.add((family, strategy, depth_cat))

    # §19 要求各维度均被覆盖，并未要求每个 3×4×4 交叉组合都存在。
    expected_families = {"P", "R", "PR"}
    expected_strategies = {"localization", "entity_tracking", "logical_bridge", "evidence_integration"}
    expected_depths = {"D1", "D2", "D3", "D4+"}

    missing_families = expected_families - set(family_counts)
    missing_strategies = expected_strategies - set(strategy_counts)
    missing_depths = expected_depths - set(depth_counts)

    return {
        "total": len(selected),
        "family_coverage": dict(family_counts),
        "strategy_coverage": dict(strategy_counts),
        "depth_coverage": dict(depth_counts),
        "n_combinations_covered": len(combo_coverage),
        "n_combinations_expected": None,
        "gaps": {
            "task_families": sorted(missing_families),
            "guidance_strategies": sorted(missing_strategies),
            "depths": sorted(missing_depths),
        }
    }


def save_stage0_samples(
    selected: list[dict],
    output_path: str,
    prefix: str = "PI-VQA"
):
    """保存 Stage 0 样本"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in selected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(selected)} Stage 0 samples -> {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select Stage 0 screening samples")
    parser.add_argument("--test-jsonl", type=str, required=True, help="Test set JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--target-size", type=int, default=500, help="Target sample count (400-600)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Loading test samples from {args.test_jsonl}...")
    test_samples = load_test_samples(args.test_jsonl)
    print(f"Loaded {len(test_samples)} samples")

    print(f"Selecting {args.target_size} Stage 0 samples...")
    selected = select_stage0_samples(
        test_samples,
        target_size=args.target_size,
        seed=args.seed
    )

    # 验证覆盖度
    coverage = validate_stage0_coverage(selected)
    print(f"\n=== Stage 0 Coverage ===")
    print(f"Total: {coverage['total']}")
    print(f"Family: {coverage['family_coverage']}")
    print(f"Strategy: {coverage['strategy_coverage']}")
    print(f"Depth: {coverage['depth_coverage']}")
    print(f"Combinations: {coverage['n_combinations_covered']}/{coverage['n_combinations_expected']}")
    if coverage['gaps']:
        print(f"Gaps: {coverage['gaps'][:10]}{'...' if len(coverage['gaps']) > 10 else ''}")

    # 保存
    save_stage0_samples(selected, args.output)
