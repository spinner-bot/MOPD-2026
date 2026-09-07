from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spinner_bot.dataio import load_jsonl
from spinner_bot.experiment import build_grounding_question


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT.parent / "data" / "train.jsonl"))
    p.add_argument("--output", default=str(ROOT.parent / "data" / "grounding_prompts.jsonl"))
    args = p.parse_args()
    rows = load_jsonl(Path(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            question = row.get("extra_info", {}).get("question", "")
            grounded = build_grounding_question(question, row.get("bbox"), row.get("problem"))
            f.write(json.dumps({"index": i, "bbox": row.get("bbox"), "original_question": question, "grounding_question": grounded}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} grounding prompts to {out}")


if __name__ == "__main__":
    main()
