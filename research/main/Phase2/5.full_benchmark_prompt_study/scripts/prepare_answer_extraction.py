#!/usr/bin/env python3
"""Freeze text-only Work5 answer-extraction requests from verified full outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
from pathlib import Path

CONDITIONS = ("S0", "SC", "S1", "S2", "S3")
TRACKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
OPTION_LINE = re.compile(r"(?m)^\s*(?:\(([A-F])\)|([A-F])[.)])\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("sample_uid")
            if not isinstance(uid, str) or not uid or uid in rows:
                raise ValueError(f"{path}:{line_no}: missing or duplicate sample_uid {uid!r}")
            rows[uid] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--campaign", default="full/runs/prompt-registry-v1")
    parser.add_argument("--output-version", default="answer-extraction-v1")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    campaign = data_root / args.campaign
    if not (campaign / "output-verification.json").is_file():
        raise FileNotFoundError("Full five-condition output verification must pass before freezing judge input")
    judgment_root = data_root / "judgment"
    out_root = judgment_root / args.output_version
    if out_root.exists():
        raise FileExistsError(f"Frozen judgment input already exists; choose a new version: {out_root}")
    build_root = judgment_root / f".building-{args.output_version}-{os.getpid()}"
    build_root.mkdir(parents=True, exist_ok=False)

    manifest: dict = {
        "schema": "phase2-work5-answer-extraction-input-v1",
        "campaign": args.campaign,
        "conditions": list(CONDITIONS),
        "tracks": list(TRACKS),
        "model_payload_excludes": ["gold_answer", "image", "condition", "sample_uid", "category", "track"],
        "shards": {},
    }
    total = 0
    per_track_ids: dict[str, set[str]] = {}

    for track in TRACKS:
        source_path = data_root / "prepared" / track / "benchmark.json"
        source_rows = json.loads(source_path.read_text(encoding="utf-8"))
        source_by_uid = {row["sample_uid"]: row for row in source_rows}
        if len(source_by_uid) != len(source_rows):
            raise ValueError(f"{track}: prepared source contains duplicate UIDs")
        per_track_ids[track] = set(source_by_uid)
        outputs_by_condition: dict[str, dict[str, dict]] = {}

        for condition in CONDITIONS:
            output_path = campaign / "outputs" / condition / track / f"OPD-V-Qwen3.5-4B-{condition}_answer.jsonl"
            outputs = read_jsonl(output_path)
            if set(outputs) != set(source_by_uid):
                raise ValueError(f"{condition}/{track}: full output coverage mismatch")
            outputs_by_condition[condition] = outputs

        target = build_root / track
        target.mkdir(parents=True, exist_ok=True)
        for condition in CONDITIONS:
            path = target / f"{condition}.jsonl"
            count = 0
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                for uid in sorted(source_by_uid):
                    source = source_by_uid[uid]
                    output = outputs_by_condition[condition][uid]
                    if output.get("prompt_condition") != condition:
                        raise ValueError(f"{condition}/{track}/{uid}: output condition metadata mismatch")
                    query = str(source.get("query") or "")
                    answer = str(output.get("model_answer") or "")
                    question_type = str(source.get("question_type") or "").strip().casefold()
                    answer_format = "NUMERIC_SHORT_ANSWER" if track == "zoombench" and question_type == "blank" else "MCQ_OPTION"
                    labels = [left or right for left, right in OPTION_LINE.findall(query)] if answer_format == "MCQ_OPTION" else []
                    if answer_format == "MCQ_OPTION" and len(labels) < 2:
                        raise ValueError(f"{track}/{uid}: expected visible option lines in query")
                    request = {
                        "sample_uid": uid,
                        "condition": condition,
                        "track": track,
                        "answer_format": answer_format,
                        "query": query,
                        "available_option_labels_in_order": labels,
                        "gold_answer": str(source.get("response") or ""),
                        "model_response": answer,
                        "model_response_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                        "model_response_chars": len(answer),
                        "source_category": source.get("category"),
                        "source_l2_category": source.get("l2-category"),
                        "source_cycle_category": source.get("cycle_category"),
                        "source_question_type": source.get("question_type"),
                    }
                    handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
            manifest["shards"][f"{condition}/{track}"] = {
                "path": str((out_root / track / f"{condition}.jsonl").relative_to(data_root)),
                "records": count,
                "sha256": sha256_file(path),
                "source_output_sha256": sha256_file(
                    campaign / "outputs" / condition / track / f"OPD-V-Qwen3.5-4B-{condition}_answer.jsonl"
                ),
            }
            total += count
            print(f"FROZEN {condition}/{track}: {count}", flush=True)

    if total != 160810:
        raise ValueError(f"Expected 160,810 answer-extraction records, found {total}")
    manifest["total_records"] = total
    manifest_path = build_root / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_root.rename(out_root)
    manifest_path = out_root / "input-manifest.json"
    print(f"PASS ALL: {total} requests; manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
