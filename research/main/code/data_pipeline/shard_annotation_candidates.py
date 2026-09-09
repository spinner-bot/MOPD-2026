"""Deterministically shard candidate records for independent Slurm workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def shard_id(candidate_id: str, shards: int) -> int:
    return int(hashlib.sha256(candidate_id.encode()).hexdigest(), 16) % shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--limit", type=int, default=0, help="0 retains every candidate")
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = [(args.output_dir / f"shard-{index:03d}.jsonl").open("w", encoding="utf-8") for index in range(args.shards)]
    counts = [0] * args.shards
    try:
        for line in args.input.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            if args.limit and sum(counts) >= args.limit:
                break
            index = shard_id(record["candidate_id"], args.shards)
            handles[index].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[index] += 1
    finally:
        for handle in handles:
            handle.close()
    manifest = {"input": str(args.input), "shards": args.shards, "limit": args.limit, "counts": counts}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
