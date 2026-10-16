#!/usr/bin/env python3
"""Conservatively normalize auditable model verdict/reference inconsistencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_visual_audit import parse_result, read_jsonl_records, sha256_file


CONSISTENCY_ERROR = "A definite candidate verdict requires a supported benchmark reference"
NORMALIZATION_RULE = (
    "A definite CORRECT/INCORRECT verdict is not retained when the model itself assesses "
    "the benchmark reference as CONTRADICTED or UNVERIFIABLE; conservatively recode as UNCERTAIN."
)


def normalize_records(
    source_rows: list[dict[str, Any]], expected_ids: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        request_id = row.get("request_id")
        if request_id not in expected_ids:
            raise ValueError(f"Unexpected visual-audit result ID: {request_id}")
        by_id[request_id] = row
    if set(by_id) != expected_ids:
        raise ValueError(f"Visual-audit result coverage is {len(by_id)}/{len(expected_ids)}")

    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for request_id in sorted(expected_ids):
        row = dict(by_id[request_id])
        status = row.get("request_status")
        if status == "OK":
            row["audit"] = parse_result(json.dumps(row.get("audit"), ensure_ascii=False))
            row["postprocessing"] = {"action": "none", "source_request_status": "OK"}
            counts["accepted"] += 1
        elif status == "INVALID_AUDIT_JSON":
            raw = row.get("raw_output", "")
            try:
                parsed = json.loads(raw)
                parse_result(raw)
            except ValueError as error:
                if CONSISTENCY_ERROR not in str(error):
                    raise ValueError(f"{request_id}: unsupported model-output error: {error}") from error
                if not isinstance(parsed, dict):
                    raise ValueError(f"{request_id}: inconsistent output is not a JSON object") from error
                model_verdict = parsed.get("verdict")
                reference_assessment = parsed.get("reference_assessment")
                if model_verdict not in {"CORRECT", "INCORRECT"} or reference_assessment == "SUPPORTED":
                    raise ValueError(f"{request_id}: inconsistency does not match conservative rule") from error
                parsed["verdict"] = "UNCERTAIN"
                row["audit"] = parse_result(json.dumps(parsed, ensure_ascii=False))
                row["postprocessing"] = {
                    "action": "conservative_verdict_normalization",
                    "source_request_status": status,
                    "model_reported_verdict": model_verdict,
                    "model_reported_reference_assessment": reference_assessment,
                    "rule": NORMALIZATION_RULE,
                    "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                }
                row["request_status"] = "OK"
                counts["normalized_to_uncertain"] += 1
        else:
            raise ValueError(f"{request_id}: refusing non-reviewable status {status!r}")
        output.append(row)
    return output, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    inputs_path = run_dir / "inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "run-manifest.json"
    output_path = run_dir / "results-normalized.jsonl"
    output_manifest_path = run_dir / "normalization-manifest.json"
    if output_path.exists() or output_manifest_path.exists():
        raise FileExistsError("Refusing to overwrite existing normalized visual-audit outputs")
    inputs = read_jsonl_records(inputs_path)
    input_ids = [row.get("request_id") for row in inputs]
    if len(inputs) != 300 or len(set(input_ids)) != 300 or None in input_ids:
        raise ValueError("Expected exactly 300 unique frozen visual-audit requests")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("input_sha256") != sha256_file(inputs_path):
        raise ValueError("Run manifest input hash does not match frozen inputs")
    normalized, counts = normalize_records(read_jsonl_records(results_path), set(input_ids))
    output_bytes = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                           for row in normalized).encode("utf-8")
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema": "phase2-work5-visual-audit-normalization-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalizer_commit": commit,
        "source_run_manifest_sha256": sha256_file(manifest_path),
        "source_input_sha256": sha256_file(inputs_path),
        "source_results_sha256": sha256_file(results_path),
        "normalized_results_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "request_count": len(normalized),
        "counts": dict(counts),
        "rule": NORMALIZATION_RULE,
        "raw_source_results_preserved": True,
    }
    with output_manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"PASS: normalized {len(normalized)} paired requests; counts={dict(counts)}")
    print(f"results={output_path}; manifest={output_manifest_path}")


if __name__ == "__main__":
    main()
