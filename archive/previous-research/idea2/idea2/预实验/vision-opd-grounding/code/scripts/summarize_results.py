from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    args = p.parse_args()
    rows = []
    for path in args.paths:
        with Path(path).open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    stats = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        s = stats[r["case"]]
        s[0] += 1
        s[1] += int(bool(r.get("correct")))
        s[2] += int(bool(r.get("error")))
    print("case,n,correct,accuracy,errors")
    for case in sorted(stats):
        n, c, e = stats[case]
        acc = c / n if n else 0.0
        print(f"{case},{n},{c},{acc:.4f},{e}")


if __name__ == "__main__":
    main()
