#!/usr/bin/env python3
"""Create an interpreted derivative while preserving the native judge's raw result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_native_long_context import terminal_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    record.pop("judge_response", None)
    record["judge_terminal_decision"] = terminal_decision(record["judge_output"])
    record["interpretation_method"] = "exact final nonempty line: Yes/No; otherwise other (v1)"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "terminal_decision": record["judge_terminal_decision"]}))


if __name__ == "__main__":
    main()
