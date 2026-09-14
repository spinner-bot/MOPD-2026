#!/usr/bin/env python3
"""Replay the frozen B0/B2a sampling chain without reading question content."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_summary(rows: list[dict]) -> dict:
    counts = [row["source_record_count"] for row in rows]
    multi = sum(count > 1 for count in counts)
    return {
        "images": len(rows),
        "source_question_record_sum": sum(counts),
        "source_record_count_min": min(counts),
        "source_record_count_median": statistics.median(counts),
        "source_record_count_mean": sum(counts) / len(counts),
        "source_record_count_max": max(counts),
        "multi_record_images": multi,
        "multi_record_image_rate": multi / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--b2a", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--b2a-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = load_jsonl(args.pool)
    sample = load_jsonl(args.sample)
    b2a = load_jsonl(args.b2a)
    rank = lambda row: hashlib.sha256(  # noqa: E731
        f"{args.seed}\0{row['image_candidate_id']}".encode("utf-8")
    ).hexdigest()
    expected_sample = sorted(pool, key=rank)[: args.sample_size]
    expected_b2a = expected_sample[: args.b2a_size]

    assertions = {
        "pool_image_candidate_ids_unique": len({row["image_candidate_id"] for row in pool}) == len(pool),
        "sample_exactly_replays_global_hash_selection": sample == expected_sample,
        "sample_is_strictly_hash_ordered": all(
            rank(sample[index]) < rank(sample[index + 1])
            for index in range(len(sample) - 1)
        ),
        "sample_prefix32_equals_direct_global32": sample[: args.b2a_size]
        == sorted(pool, key=rank)[: args.b2a_size],
        "b2a_ids_equal_sample_prefix32": [row["image_candidate_id"] for row in b2a]
        == [row["image_candidate_id"] for row in expected_b2a],
        "b2a_source_indices_are_contiguous": [row["calibration_source_index"] for row in b2a]
        == list(range(args.b2a_size)),
        "b2a_source_questions_hidden": all(row["source_question_exposed"] is False for row in b2a),
        "b2a_machine_proposals_hidden": all(row["machine_proposal_exposed"] is False for row in b2a),
    }
    result = {
        "rules_version": "b2a-sampling-audit-v1",
        "seed": args.seed,
        "ranking_function": "sha256(seed\\0image_candidate_id)",
        "pool_sha256": sha256_file(args.pool),
        "sample_sha256": sha256_file(args.sample),
        "b2a_sha256": sha256_file(args.b2a),
        "assertions": assertions,
        "all_assertions_passed": all(assertions.values()),
        "descriptive_source_record_counts": {
            "full_unique_image_pool": count_summary(pool),
            "frozen_sample_256": count_summary(sample),
            "b2a_prefix_32": count_summary(expected_b2a),
        },
        "question_content_read": False,
        "semantic_image_review_performed": False,
        "human_verified_records": 0,
    }
    if not result["all_assertions_passed"]:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
