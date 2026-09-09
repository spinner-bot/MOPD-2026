"""Create provenance-preserving Stage 0 annotation candidates from M3VQA.

M3VQA exposes multi-hop evidence but not the multiple-choice, crop, or guide
QA fields required by the formal Stage 0 contract.  This script freezes those
source facts in an annotation queue without inventing any of the missing
labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_candidate(source: dict) -> dict:
    return {
        "candidate_id": f"m3vqa-{source['data_id']}",
        "source_dataset": "M3VQA",
        "source_data_id": source["data_id"],
        "source_image_id": source["image_id"],
        "source_image_archive": "images.zip",
        "task_family_candidate": "R" if int(source.get("question_hop", 0)) >= 2 else "P",
        "reasoning_depth_candidate": int(source.get("question_hop", 0)),
        "final_question_source": source["question"],
        "reference_answers": source["answers"],
        "evidence": source["evidence"],
        "evidence_urls": source["evidence_urls"],
        "annotation_required": [
            "four-option multiple-choice formulation",
            "verified guide question/answer chain",
            "guidance strategy",
            "primary crop or documented full-image exception",
            "length-matched frozen random questions",
            "source image extraction and checksum",
        ],
        "stage0_admission": "blocked_pending_annotation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line in source:
            if not line.strip():
                continue
            output.write(json.dumps(make_candidate(json.loads(line)), ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} M3VQA annotation candidates to {args.output}")


if __name__ == "__main__":
    main()
