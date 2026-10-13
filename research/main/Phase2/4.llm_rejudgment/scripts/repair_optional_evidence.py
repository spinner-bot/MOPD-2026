#!/usr/bin/env python3
"""Normalize only non-verbatim optional evidence on NO_ANSWER/AMBIGUOUS rows.

The original judge JSON remains untouched in the existing JSONL attempt. A new
record is appended with the same substantive model label and empty optional
evidence, explicitly marking the deterministic normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from run_judge_shard import parse_and_validate


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--expected-repairs", type=int, default=3)
    args = parser.parse_args()

    requests = {}
    with args.input_freeze.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                requests[(row["sample_id"], row["condition"])] = row

    repairs = []
    updates = []
    for shard_path in sorted(args.judge_dir.glob("judge-shard-*.jsonl")):
        rows = read_jsonl(shard_path)
        latest = {}
        for row in rows:
            latest[(row.get("sample_id"), row.get("condition"))] = row
        for key, row in latest.items():
            if row.get("request_status") == "OK":
                continue
            if row.get("request_status") != "INVALID_JUDGE_JSON":
                raise ValueError(f"unexpected non-OK status blocks repair: {key} {row.get('request_status')}")
            req = requests.get(key)
            if not req or row.get("response_sha256") != req["response_sha256"]:
                raise ValueError(f"repair source mismatch: {key}")
            raw = row.get("judge_raw_text", "")
            candidate = parse_and_validate(raw, req["response"])
            if candidate["status"] == "ANSWERED":
                raise ValueError(f"this repair is only for non-answer statuses: {key}")
            if candidate.get("evidence_validation") != "nonverbatim_optional_evidence_discarded":
                raise ValueError(f"record does not meet the narrow repair rule: {key}")
            repaired = {
                **row,
                **candidate,
                "request_status": "OK",
                "judge_error": None,
                "validation_error": None,
                "evidence_repair": "discarded_nonverbatim_optional_evidence",
                "repaired_from_request_status": row["request_status"],
            }
            updates.append((shard_path, repaired))
            repairs.append({
                "sample_id": key[0],
                "condition": key[1],
                "shard_file": shard_path.name,
                "prior_request_status": row["request_status"],
                "prior_validation_error": row.get("validation_error", ""),
                "model_status_preserved": candidate["status"],
                "model_selected_option_preserved": candidate["selected_option"],
                "judge_raw_text_sha256": sha256_text(raw),
                "normalized_evidence": "",
                "normalization": repaired["evidence_repair"],
            })

    if len(repairs) != args.expected_repairs:
        raise ValueError(f"expected {args.expected_repairs} narrow repairs; found {len(repairs)}")

    for shard_path, row in updates:
        with shard_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    repairs_by_shard = Counter(item["shard_file"] for item in repairs)
    for shard_name, n_repairs in repairs_by_shard.items():
        manifest_path = args.judge_dir / shard_name.replace(".jsonl", ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_rows = read_jsonl(args.judge_dir / shard_name)
        if sum(row.get("request_status") == "OK" for row in output_rows) != manifest["selected_records"]:
            raise ValueError(f"post-repair success count still differs from selected records: {shard_name}")
        manifest["successful_records"] = manifest["selected_records"]
        manifest["optional_evidence_repairs"] = {
            "count": n_repairs,
            "policy": "For NO_ANSWER/AMBIGUOUS only, non-verbatim optional evidence was blanked; status, option, reason code, and original judge_raw_text are unchanged.",
            "repair_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    repair_log = {
        "schema": "optional-evidence-repair-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repair_count": len(repairs),
        "repairs": repairs,
        "interpretation": "This is deterministic evidence-field normalization, not a new model judgment. The original raw judge output and failed validation attempt remain in each shard JSONL; only optional non-verbatim evidence on non-answer statuses was removed.",
    }
    (args.judge_dir / "optional_evidence_repairs.json").write_text(
        json.dumps(repair_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"repairs": len(repairs), "by_shard": dict(repairs_by_shard)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
