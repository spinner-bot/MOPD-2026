#!/usr/bin/env python3
"""Build a normalized CV-Bench 2D+3D sampling pool."""

import argparse
import hashlib
import json
from pathlib import Path


def normalize_answer(value):
    value = str(value).strip().upper()
    if value.startswith("(") and ")" in value:
        value = value[1:value.index(")")]
    return value


def load_rows(path, split):
    rows = []
    excluded = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        choices = row.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"{path}:{line_no}: invalid choices")
        if len(choices) > 4:
            excluded["more_than_four_choices"] = excluded.get("more_than_four_choices", 0) + 1
            continue
        answer = normalize_answer(row.get("answer", ""))
        if answer not in "ABCD":
            excluded["answer_outside_A_to_D"] = excluded.get("answer_outside_A_to_D", 0) + 1
            continue
        filename = row.get("filename")
        if not filename:
            raise ValueError(f"{path}:{line_no}: missing filename")
        uid = f"cvbench-{split}-{row.get('source')}-{row.get('source_filename')}-{line_no}"
        rows.append({
            "sample_id": uid,
            "dataset": "CV-Bench",
            "split": split,
            "type": row.get("type"),
            "task": row.get("task"),
            "source": row.get("source"),
            "source_dataset": row.get("source_dataset"),
            "source_filename": row.get("source_filename"),
            "filename": filename,
            "query": row.get("prompt") or "",
            "question": row.get("question") or "",
            "choices": choices,
            "response": answer,
        })
    return rows, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-d", required=True, type=Path)
    parser.add_argument("--three-d", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    two_d, excluded_2d = load_rows(args.two_d, "2D")
    three_d, excluded_3d = load_rows(args.three_d, "3D")
    rows = two_d + three_d
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs")
    payload = {
        "schema_version": "cvbench-pool-v1",
        "dataset": "CV-Bench",
        "counts": {
            "total": len(rows),
            "2D": len(two_d),
            "3D": len(three_d),
        },
        "excluded": {"2D": excluded_2d, "3D": excluded_3d},
        "source_sha256": {
            "2D": hashlib.sha256(args.two_d.read_bytes()).hexdigest(),
            "3D": hashlib.sha256(args.three_d.read_bytes()).hexdigest(),
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()

