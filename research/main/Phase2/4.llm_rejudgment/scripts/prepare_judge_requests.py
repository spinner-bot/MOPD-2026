#!/usr/bin/env python3
"""Freeze 10,000 answer-extraction requests from the authoritative Phase2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.work_root / "raw/input-freeze-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = out_dir / "judge_requests.jsonl"
    manifest_path = out_dir / "input_manifest.json"
    if requests_path.exists() or manifest_path.exists():
        raise SystemExit(f"frozen input already exists; create a new version directory: {out_dir}")

    sample_path = args.source_root / "raw/sample_manifest.json"
    sample_doc = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_by_id = {r["sample_id"]: r for r in sample_doc["records"]}
    expected_ids = set(sample_by_id)
    if len(expected_ids) != 2000 or sample_doc.get("sample_size") != 2000:
        raise ValueError("frozen sample manifest must contain 2,000 unique IDs")

    raw_files = {}
    observed = {}
    pending = []
    for condition in CONDITIONS:
        path = args.source_root / f"runs/60011513/outputs/{condition}.jsonl"
        raw_files[condition] = {"path": str(path.relative_to(args.source_root)), "sha256": sha256(path)}
        seen = set()
        count = 0
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record["sample_id"]
                if record.get("condition") != condition:
                    raise ValueError(f"condition mismatch at {path}:{line_no}")
                if sample_id not in sample_by_id or sample_id in seen:
                    raise ValueError(f"unknown or duplicate sample_id at {path}:{line_no}: {sample_id}")
                seen.add(sample_id)
                count += 1
                frozen = sample_by_id[sample_id]
                choices = frozen.get("choices")
                gold = str(frozen.get("response", "")).strip().upper()
                if not isinstance(choices, list) or not 2 <= len(choices) <= 4 or gold not in "ABCD"[:len(choices)]:
                    raise ValueError(f"invalid choices/gold answer for {sample_id}")
                response = str(record.get("response") or "")
                payload = {
                    "sample_id": sample_id,
                    "condition": condition,
                    "source_line": line_no,
                    "question": str(frozen.get("question") or ""),
                    "query": str(frozen.get("query") or ""),
                    "choices": [{"option": chr(65 + i), "text": str(choice)} for i, choice in enumerate(choices)],
                    "gold_option": gold,
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                    "response_chars": len(response),
                    "image_sha256": record.get("image_sha256"),
                    "usage": record.get("usage") or {},
                    "api_error": record.get("api_error"),
                }
                pending.append(payload)
        if count != 2000 or seen != expected_ids:
            raise ValueError(f"{condition}: expected the same 2,000 sample IDs; got {count}")
        observed[condition] = count

    if len(pending) != 10000 or len({(r["sample_id"], r["condition"]) for r in pending}) != 10000:
        raise ValueError("prepared records must contain exactly 10,000 unique sample-condition pairs")
    with requests_path.open("x", encoding="utf-8", newline="\n") as f:
        for row in pending:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "phase2-llm-rejudge-input-v1",
        "run_id": "60011513",
        "n_samples": 2000,
        "conditions": CONDITIONS,
        "n_judgments": len(pending),
        "gold_source": str(sample_path.relative_to(args.source_root)),
        "gold_manifest_sha256": sha256(sample_path),
        "raw_outputs": raw_files,
        "condition_counts": observed,
        "judge_requests_sha256": sha256(requests_path),
        "judge_requests_records": len(pending),
        "gold_option_sent_to_judge": False,
        "image_sent_to_judge": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_judgments": len(pending), "counts": observed, "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
