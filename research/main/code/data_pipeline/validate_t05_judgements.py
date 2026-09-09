"""Route independently judged T05 records to accepted, borderline, or rejects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quality_contract import eligible, validate_quality_judgement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--borderline", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    args = parser.parse_args()
    buckets = {"accepted": [], "borderline": [], "rejected": []}
    for path in sorted(Path().glob(args.input_glob)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line); judgement = row.get("judgement")
            if row.get("error") or not isinstance(judgement, dict) or validate_quality_judgement(judgement):
                row["t05_rejection_reason"] = "invalid_judgement"; buckets["rejected"].append(row)
            elif eligible(judgement): buckets["accepted"].append(row)
            elif judgement["decision"] == "borderline": buckets["borderline"].append(row)
            else: buckets["rejected"].append(row)
    for name, path in (("accepted", args.accepted), ("borderline", args.borderline), ("rejected", args.rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in buckets[name]), encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in buckets.items()}))


if __name__ == "__main__": main()
