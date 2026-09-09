"""Validate T03 model proposals and preserve rejected records with reason codes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from guide_contract import validate_t03_proposal


def validate(row: dict) -> str | None:
    if row.get("error"):
        return "runner_error"
    proposal = row.get("proposal")
    if not isinstance(proposal, dict):
        return "invalid_json"
    if proposal.get("decision") == "reject":
        return "model_reject"
    answer = row["candidate"].get("source_answer", "")
    errors = validate_t03_proposal(proposal, str(answer))
    return ";".join(errors) if errors else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for path in sorted(Path().glob(args.input_glob)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted, rejected = [], []
    for row in rows:
        reason = validate(row)
        if reason:
            row["t03_rejection_reason"] = reason
            rejected.append(row)
        else:
            accepted.append(row)
    for path, records in ((args.accepted, accepted), (args.rejected, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    summary = {"total": len(rows), "accepted": len(accepted), "rejected": len(rejected), "rejections": dict(Counter(row["t03_rejection_reason"] for row in rejected))}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
