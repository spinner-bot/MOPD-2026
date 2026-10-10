"""Deterministically sample Vision-OPD JSONL outputs for human audit.

This utility never decides whether evidence or reasoning is valid. It preserves
the raw model output and creates explicit human-review fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def stable_key(record: dict, seed: str) -> str:
    uid = record.get("sample_uid") or record.get("id") or record.get("index")
    if uid is None:
        uid = json.dumps(
            {
                "query": record.get("query", ""),
                "images": record.get("images", []),
                "response": record.get("response", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    return hashlib.sha256(f"{seed}\0{uid}".encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: expected JSON object")
            record["_input_line"] = line_no
            records.append(record)
    return records


def make_review_record(record: dict, rank: int, seed: str) -> dict:
    raw = record.get("model_answer", "")
    if not isinstance(raw, str):
        raw = str(raw)
    return {
        "audit_rank": rank,
        "sample_uid": record.get("sample_uid") or record.get("id") or record.get("index"),
        "input_line": record.get("_input_line"),
        "image": (record.get("images") or [record.get("image", "")])[0],
        "query": record.get("query", ""),
        "reference_answer": record.get("response", ""),
        "raw_model_output": raw,
        "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "sampling_seed": seed,
        "human_output_class": "",
        "human_evidence_present": "",
        "human_evidence_valid": "",
        "human_reasoning_present": "",
        "human_reasoning_valid": "",
        "human_answer_visible_before_reasoning": "",
        "human_reasoning_appears_decision_relevant": "",
        "human_answer_correct": "",
        "human_reviewer": "",
        "human_reviewed_at": "",
        "human_notes": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive")

    records = load_jsonl(args.input)
    ranked = sorted(records, key=lambda item: stable_key(item, args.seed))
    selected = ranked[: min(args.n, len(ranked))]
    output_records = [
        make_review_record(record, rank, args.seed)
        for rank, record in enumerate(selected, 1)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = args.output.with_suffix(".csv")
    if output_records:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_records[0]))
            writer.writeheader()
            writer.writerows(output_records)
    summary = {
        "input": str(args.input),
        "input_records": len(records),
        "sample_n_requested": args.n,
        "sample_n_written": len(output_records),
        "seed": args.seed,
        "selection": "sha256(seed + sample_uid_or_stable_record)",
        "raw_output_preserved": True,
        "semantic_fields_filled_automatically": False,
    }
    summary_path = args.output.with_name(args.output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
