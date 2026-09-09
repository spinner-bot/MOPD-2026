"""Build a provenance-preserving, deterministic P/R candidate pool.

This is intentionally a candidate artifact, not a formal Stage 0 dataset.
Records without source images remain rejected and records without a validated
guide chain/crop remain blocked for later H-model processing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def vision_candidate(record: dict) -> dict:
    return {
        "candidate_id": record["sample_id"],
        "source_dataset": record["source_dataset"],
        "source_record_index": record["source_record_index"],
        "source_image_id": record["source_image_id"],
        "image_original": record["image_original"],
        "crop_primary_source": record["crop_primary"],
        "final_question_source": record["final_question"],
        "source_options": {key: record[f"option_{key}"] for key in "ABCD"},
        "source_answer": record["final_answer"],
        "source_bbox_xyxy": record["bbox_xyxy"],
        "task_family_candidate": "P",
        "reasoning_depth_candidate": 1,
        "needs_grounded_crop": False,
        "needs_guidance_chain": True,
        "needs_random_control": True,
        "stage0_admission": "blocked_pending_annotation",
    }


def m3vqa_candidate(record: dict, image_root: Path) -> dict:
    image = image_root / "images" / record["source_image_id"]
    return {
        "candidate_id": record["candidate_id"],
        "source_dataset": record["source_dataset"],
        "source_record_id": record["source_data_id"],
        "source_image_id": record["source_image_id"],
        "image_original": str(image),
        "final_question_source": record["final_question_source"],
        "reference_answers": record["reference_answers"],
        "evidence": record["evidence"],
        "evidence_urls": record["evidence_urls"],
        "task_family_candidate": record["task_family_candidate"],
        "reasoning_depth_candidate": record["reasoning_depth_candidate"],
        "needs_grounded_crop": True,
        "needs_guidance_chain": True,
        "needs_multiple_choice": True,
        "needs_random_control": True,
        "stage0_admission": "blocked_pending_annotation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-records", required=True, type=Path)
    parser.add_argument("--m3vqa-candidates", required=True, type=Path)
    parser.add_argument("--m3vqa-image-root", required=True, type=Path)
    parser.add_argument("--accepted-output", required=True, type=Path)
    parser.add_argument("--rejected-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    accepted: list[dict] = []
    rejected: list[dict] = []
    for record in read_jsonl(args.vision_records):
        candidate = vision_candidate(record)
        if Path(candidate["image_original"]).is_file() and Path(candidate["crop_primary_source"]).is_file():
            accepted.append(candidate)
        else:
            candidate["rejection_reason"] = "missing_materialized_vision_asset"
            rejected.append(candidate)
    for record in read_jsonl(args.m3vqa_candidates):
        candidate = m3vqa_candidate(record, args.m3vqa_image_root)
        if Path(candidate["image_original"]).is_file():
            accepted.append(candidate)
        else:
            candidate["rejection_reason"] = "missing_materialized_m3vqa_asset"
            rejected.append(candidate)

    for path in (args.accepted_output, args.rejected_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.accepted_output.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.rejected_output.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_by_source": dict(Counter(row["source_dataset"] for row in accepted)),
        "accepted_by_preliminary_family": dict(Counter(row["task_family_candidate"] for row in accepted)),
        "rejections": dict(Counter(row["rejection_reason"] for row in rejected)),
        "formal_stage0_eligible": False,
        "blocking_requirements": ["validated P/R/PR labels", "guidance QA chains", "M3VQA grounded crops", "multiple-choice controls"],
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
