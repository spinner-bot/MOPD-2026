#!/usr/bin/env python3
"""Build evidence-firewalled teacher-gap inputs with correct M3VQA answers.

The legacy candidate pool copied M3VQA ``answers`` into a field described as
aliases and discarded ``answer_evals``. This tool joins candidates to the raw
source, converts the official target groups deterministically, and emits only
fields approved for blind/audit/construct input. It never copies evidence,
URLs, entity names, or post-hoc locator metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_m3vqa_teacher_gap_review_packet import reference_answer_spec
from teacher_gap_candidate_contract import validate_candidate


RULES_VERSION = "m3vqa-answer-contract-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(row)
    return rows


def source_record_id(candidate: dict) -> str:
    provenance = candidate.get("provenance")
    if isinstance(provenance, dict):
        for key in ("raw_data_id", "source_record_id"):
            value = str(provenance.get(key, "")).strip()
            if value:
                return value
    value = str(candidate.get("source_record_id", "")).strip()
    if value:
        return value
    candidate_id = str(candidate.get("candidate_id", "")).strip()
    return candidate_id.removeprefix("m3vqa-")


def upgrade(candidates: list[dict], raw_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    raw_index = {}
    duplicate_raw_ids = set()
    for raw in raw_rows:
        key = str(raw.get("data_id", "")).strip()
        if not key or key in raw_index:
            duplicate_raw_ids.add(key or "<missing>")
            continue
        raw_index[key] = raw

    upgraded = []
    rejects = []
    seen_candidates = set()
    seen_images = set()
    cardinalities = Counter()
    collapse_count = 0
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        reasons = []
        if not candidate_id:
            reasons.append("missing_candidate_id")
        elif candidate_id in seen_candidates:
            reasons.append("duplicate_candidate_id")
        seen_candidates.add(candidate_id)
        image_id = str(candidate.get("source_image_id", "")).strip()
        if not image_id:
            reasons.append("missing_source_image_id")
        elif image_id in seen_images:
            reasons.append("duplicate_source_image_id")
        seen_images.add(image_id)
        raw_id = source_record_id(candidate)
        raw = raw_index.get(raw_id)
        if raw is None:
            reasons.append("raw_join_miss")
        if raw is not None:
            if str(raw.get("image_id", "")).strip() != image_id:
                reasons.append("source_image_id_mismatch")
        if reasons:
            rejects.append({"candidate_id": candidate_id or None, "rejection_reasons": sorted(set(reasons))})
            continue
        try:
            targets, alias_groups, canonical_text = reference_answer_spec(raw)
        except ValueError as error:
            rejects.append({"candidate_id": candidate_id, "rejection_reasons": [str(error)]})
            continue
        question = str(candidate.get("final_question_source", "")).strip()
        if question != str(raw.get("question", "")).strip():
            rejects.append({"candidate_id": candidate_id, "rejection_reasons": ["question_mismatch"]})
            continue
        output = {
            "candidate_id": candidate_id,
            "source_image_id": image_id,
            "image_original": str(candidate.get("image_original", "")).strip(),
            "final_question_source": question,
            "reference_answers": targets,
            "reference_answer_alias_groups": alias_groups,
            "canonical_answer_text": canonical_text,
            "question_hop": raw.get("question_hop"),
            "answer_contract": {
                "rules_version": RULES_VERSION,
                "official_target_field": "answer_evals",
                "target_cardinality": len(alias_groups),
                "legacy_answers_count": len(raw.get("answers", [])),
                "legacy_alias_list_collapsed": len(raw.get("answers", [])) != len(alias_groups),
            },
        }
        errors = validate_candidate(output)
        if errors:
            rejects.append({"candidate_id": candidate_id, "rejection_reasons": errors})
            continue
        upgraded.append(output)
        cardinalities[len(alias_groups)] += 1
        collapse_count += output["answer_contract"]["legacy_alias_list_collapsed"]

    reason_counts = Counter(reason for row in rejects for reason in row["rejection_reasons"])
    summary = {
        "rules_version": RULES_VERSION,
        "candidate_input_records": len(candidates),
        "raw_source_records": len(raw_rows),
        "upgraded_records": len(upgraded),
        "rejected_records": len(rejects),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "target_cardinality_distribution": {
            str(key): value for key, value in sorted(cardinalities.items())
        },
        "legacy_alias_lists_collapsed": collapse_count,
        "duplicate_raw_ids": sorted(duplicate_raw_ids),
        "official_evidence_included": False,
    }
    return upgraded, rejects, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--raw-m3vqa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        candidates = load_jsonl(args.candidates)
        upgraded, rejects, summary = upgrade(candidates, load_jsonl(args.raw_m3vqa))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    write_jsonl(args.output, upgraded)
    write_jsonl(args.rejects, rejects)
    summary.update({
        "candidate_input_sha256": sha256_file(args.candidates),
        "raw_source_sha256": sha256_file(args.raw_m3vqa),
        "output_sha256": sha256_file(args.output),
        "rejects_sha256": sha256_file(args.rejects),
    })
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if rejects or len(upgraded) != len(candidates):
        raise SystemExit("answer-contract upgrade rejected one or more candidates")


if __name__ == "__main__":
    main()
