#!/usr/bin/env python3
"""Build a question-firewalled, one-record-per-image M3VQA pool.

The output is input for image-first affordance screening.  It deliberately
contains no source question, answer, evidence, URL, hop count, or downstream
model result.  Multiple M3VQA question records may point to the same image;
their opaque record IDs are retained only for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


RULES_VERSION = "m3vqa-image-affordance-pool-v1"
FORBIDDEN_OUTPUT_KEYS = {
    "final_question_source", "reference_answers", "reference_answer_alias_groups",
    "canonical_answer_text", "evidence", "evidence_urls", "question_hop",
    "reasoning_depth_candidate", "task_family_candidate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(source_image_id: str) -> str:
    suffix = hashlib.sha256(source_image_id.encode("utf-8")).hexdigest()[:20]
    return f"m3vqa-image-{suffix}"


def select_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Select an image-only pilot without consulting source questions/results."""

    if size < 1 or size > len(rows):
        raise ValueError(f"sample size {size} outside 1..{len(rows)}")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['image_candidate_id']}".encode("utf-8")
        ).hexdigest(),
    )[:size]


def build_pool(rows: list[dict], image_root: Path) -> tuple[list[dict], list[dict], dict]:
    image_root = image_root.resolve(strict=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    rejects = []
    for index, row in enumerate(rows):
        source_image_id = str(row.get("source_image_id", "")).strip()
        source_record_id = str(row.get("source_record_id", "")).strip()
        reasons = []
        if row.get("source_dataset") != "M3VQA":
            reasons.append("source_dataset_not_m3vqa")
        if not source_image_id or Path(source_image_id).name != source_image_id:
            reasons.append("invalid_source_image_id")
        if not source_record_id:
            reasons.append("missing_source_record_id")
        if reasons:
            rejects.append({"input_index": index, "rejection_reasons": sorted(set(reasons))})
            continue
        grouped[source_image_id].append(row)

    output = []
    for source_image_id in sorted(grouped):
        source_rows = grouped[source_image_id]
        source_record_ids = sorted({str(row["source_record_id"]).strip() for row in source_rows})
        if len(source_record_ids) != len(source_rows):
            rejects.append({
                "source_image_id": source_image_id,
                "rejection_reasons": ["duplicate_source_record_id_within_image"],
            })
            continue
        image_path = (image_root / source_image_id).resolve()
        if not image_path.is_relative_to(image_root) or not image_path.is_file():
            rejects.append({
                "source_image_id": source_image_id,
                "rejection_reasons": ["missing_or_escaping_image"],
            })
            continue
        record = {
            "image_candidate_id": stable_id(source_image_id),
            "source_dataset": "M3VQA",
            "source_image_id": source_image_id,
            "image_original": str(image_path),
            "image_sha256": sha256_file(image_path),
            "source_record_ids": source_record_ids,
            "source_record_count": len(source_record_ids),
            "selection_basis": "all_unique_m3vqa_source_images_before_question_screening",
            "source_questions_exposed_to_screen": False,
            "question_origin": "none_image_screening",
            "rules_version": RULES_VERSION,
        }
        if FORBIDDEN_OUTPUT_KEYS & record.keys():
            raise AssertionError("question firewall violated")
        output.append(record)

    ids = [row["image_candidate_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("image_candidate_id collision")
    summary = {
        "rules_version": RULES_VERSION,
        "input_question_records": len(rows),
        "unique_source_image_ids": len(grouped),
        "output_images": len(output),
        "rejected_groups_or_rows": len(rejects),
        "question_fields_exposed": False,
        "selection_uses_teacher_gap_results": False,
    }
    return output, rejects, summary


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.sample_output is not None and args.sample_size < 1:
        raise SystemExit("--sample-size must be positive when --sample-output is set")
    if args.sample_output is None and args.sample_size:
        raise SystemExit("--sample-size requires --sample-output")
    try:
        output, rejects, summary = build_pool(load_jsonl(args.input), args.image_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_jsonl(args.output, output)
    write_jsonl(args.rejects, rejects)
    if args.sample_output is not None:
        sample = select_sample(output, args.sample_size, args.seed)
        write_jsonl(args.sample_output, sample)
        summary.update({
            "sample_records": len(sample),
            "sample_seed": args.seed,
            "sample_selection": "sha256(seed\\0image_candidate_id)",
        })
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
