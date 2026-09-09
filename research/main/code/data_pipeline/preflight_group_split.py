"""Dry-run a leakage-safe split using exact and near-duplicate image groups.

This is deliberately a preflight only: it proves the grouping and reports the
blocking semantic-label requirements. It does not create a formal frozen split
until T02--T06 have supplied and approved those labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.candidates.open(encoding="utf-8") if line.strip()]
    duplicate_data = json.loads(args.duplicates.read_text(encoding="utf-8"))
    images = duplicate_data["images"]
    union_find = UnionFind(len(images))
    exact_clusters = near_clusters = 0
    for cluster in duplicate_data["duplicate_clusters"]:
        members = cluster["members"]
        for member in members[1:]:
            union_find.union(members[0], member)
        if cluster["kind"] == "exact":
            exact_clusters += 1
        elif cluster["kind"] == "near":
            near_clusters += 1
        else:
            raise ValueError(f"unknown duplicate cluster kind: {cluster['kind']}")

    image_to_index = {image["image_original"]: index for index, image in enumerate(images)}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        image_index = image_to_index.get(row["image_original"])
        # A missing inventory entry remains isolated and is reported explicitly.
        group = f"image-{union_find.find(image_index)}" if image_index is not None else f"missing-{row['image_original']}"
        groups[group].append(row)

    split_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_split: dict[str, str] = {}
    for group, items in groups.items():
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 73 else "val" if bucket < 82 else "test" if bucket < 95 else "stage0"
        group_split[group] = split
        split_counts[split] += len(items)
        for item in items:
            family_counts[split][item["task_family_candidate"]] += 1

    leakage_violations = 0
    for cluster in duplicate_data["duplicate_clusters"]:
        cluster_splits = {
            group_split[f"image-{union_find.find(member)}"]
            for member in cluster["members"]
        }
        leakage_violations += max(0, len(cluster_splits) - 1)
    report = {
        "status": "dry_run_not_formal",
        "records": len(rows),
        "groups": len(groups),
        "duplicate_clusters": {"exact": exact_clusters, "near": near_clusters},
        "missing_inventory_images": sum(row["image_original"] not in image_to_index for row in rows),
        "duplicate_cluster_cross_split_violations": leakage_violations,
        "split_counts": dict(split_counts),
        "family_counts": {split: dict(counts) for split, counts in family_counts.items()},
        "blocking_requirements": [
            "PR candidates absent",
            "guide QA and controls absent",
            "T05/T06 approval absent",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
