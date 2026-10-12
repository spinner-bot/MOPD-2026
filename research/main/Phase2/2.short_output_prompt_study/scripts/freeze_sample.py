#!/usr/bin/env python3
"""Freeze a paired sample for the five-condition prompt study."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=261011)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    if isinstance(source, dict) and isinstance(source.get("records"), list):
        source = source["records"]
    if not isinstance(source, list):
        raise ValueError("source must be a JSON array or an object with records")
    if len(source) < args.sample_size:
        raise ValueError(
            f"insufficient source data: {len(source)} valid records, "
            f"{args.sample_size} required; repeated sampling is forbidden"
        )

    records = []
    seen = set()
    for row in source:
        qid = str(row.get("sample_id", row.get("question_id", row.get("index", ""))))
        if not qid or qid in seen:
            continue
        if not (row.get("images") or row.get("filename")) or not row.get("query") or not row.get("response"):
            continue
        seen.add(qid)
        records.append(row)

    if len(records) < args.sample_size:
        raise ValueError(
            f"insufficient valid unique records: {len(records)}, "
            f"{args.sample_size} required"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(records, args.sample_size)
    selected.sort(key=lambda row: str(row.get("question_id", row.get("index", ""))))

    payload = {
        "schema_version": "short-output-study-sample-v1",
        "sample_size": args.sample_size,
        "seed": args.seed,
        "source_file": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "sample_ids_sha256": sha256_text(
            "\n".join(str(x.get("sample_id", x.get("question_id", x.get("index", "")))) for x in selected)
        ),
        "records": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "source_records": len(source),
        "valid_unique_records": len(records),
        "sample_size": len(selected),
        "seed": args.seed,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

