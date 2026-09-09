"""Fail closed unless the queued T02 pilot produced its complete record count."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary.exists() else {"total": 0}
    result = {
        "expected_total": args.expected_total,
        "observed_total": summary.get("total", 0),
        "accepted": summary.get("accepted", 0),
        "rejected": summary.get("rejected", 0),
        "complete": summary.get("total", 0) == args.expected_total,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["complete"]:
        raise SystemExit("T02 pilot incomplete: do not use for throughput or quality decisions")


if __name__ == "__main__":
    main()
