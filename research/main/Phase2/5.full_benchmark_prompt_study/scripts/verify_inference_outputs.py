#!/usr/bin/env python3
"""Gate pilot/full model outputs for complete paired coverage and exact input preservation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "prompts/prompt_registry-v1.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("sample_uid")
            if not uid or uid in rows:
                raise ValueError(f"{path}:{line_no}: missing or duplicate sample_uid {uid!r}")
            rows[uid] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--conditions", default="S0,SC,S1,S2,S3")
    parser.add_argument("--pilot", action="store_true", help="Use pilot inputs instead of full prepared inputs")
    parser.add_argument("--prompt-registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    registry_bytes = args.prompt_registry.read_bytes()
    registry_hash = sha256_bytes(registry_bytes)
    registry = json.loads(registry_bytes)
    if registry.get("schema") != "phase2-work5-prompt-registry-v1":
        raise ValueError("Unsupported prompt registry schema")

    input_root = args.data_root / ("pilot/inputs" if args.pilot else "prepared")
    tracks = sorted(p.stem for p in input_root.glob("*.json")) if args.pilot else sorted(
        p.parent.name for p in input_root.glob("*/benchmark.json")
    )
    if not tracks:
        raise ValueError(f"No input tracks found under {input_root}")
    expected = {
        track: json.loads((input_root / f"{track}.json" if args.pilot else input_root / track / "benchmark.json").read_text(encoding="utf-8"))
        for track in tracks
    }
    common_ids = {track: {row["sample_uid"] for row in rows} for track, rows in expected.items()}
    if any(len(common_ids[track]) != len(expected[track]) for track in tracks):
        raise ValueError("Duplicate prepared sample_uid in input")

    summary = {"conditions": {}, "tracks": tracks, "pilot": args.pilot}
    uid_compatibility = {"legacy_wrapped_uid_rows": 0}
    baseline = {}
    for condition in conditions:
        config = registry["conditions"][condition]
        system_text = config["system_message"]
        system_hash = None if system_text is None else sha256_bytes(system_text.encode("utf-8"))
        summary["conditions"][condition] = {}
        for track in tracks:
            path = args.run_root / "outputs" / condition / track / f"{args.model_id}-{condition}_answer.jsonl"
            outputs = read_jsonl(path)
            wanted = common_ids[track]
            input_by_id = {row["sample_uid"]: row for row in expected[track]}
            normalized_outputs = {}
            legacy_prefix = f"{track}:sample_uid:"
            for raw_uid, record in outputs.items():
                uid = raw_uid
                if uid not in wanted and uid.startswith(legacy_prefix) and uid[len(legacy_prefix):] in wanted:
                    uid = uid[len(legacy_prefix):]
                    uid_compatibility["legacy_wrapped_uid_rows"] += 1
                if uid in normalized_outputs:
                    raise ValueError(f"{condition}/{track}: duplicate UID after compatibility normalization: {uid}")
                normalized_outputs[uid] = record
            observed = set(normalized_outputs)
            if observed != wanted:
                missing = sorted(wanted - observed)[:3]
                unexpected = sorted(observed - wanted)[:3]
                raise ValueError(
                    f"{condition}/{track}: output UID set mismatch; expected={len(wanted)}, "
                    f"observed={len(observed)}, missing_examples={missing}, unexpected_examples={unexpected}"
                )
            for uid, record in normalized_outputs.items():
                if record.get("prompt_condition") != condition:
                    raise ValueError(f"{condition}/{track}/{uid}: condition metadata mismatch")
                if record.get("system_prompt_sha256") != system_hash:
                    raise ValueError(f"{condition}/{track}/{uid}: system prompt hash mismatch")
                if record.get("prompt_registry_sha256") != registry_hash:
                    raise ValueError(f"{condition}/{track}/{uid}: prompt registry hash mismatch")
                answer = record.get("model_answer")
                if not isinstance(answer, str) or not answer.strip() or answer.startswith(("[API_ERROR]", "[FUTURE_ERROR]")):
                    raise ValueError(f"{condition}/{track}/{uid}: empty or technical-error model output")
            for uid, record in normalized_outputs.items():
                source = input_by_id[uid]
                for key, value in source.items():
                    observed_value = uid if key == "sample_uid" else record.get(key)
                    if observed_value != value:
                        raise ValueError(f"{condition}/{track}/{uid}: inference changed input field {key}")
            if condition == "S0":
                baseline[track] = normalized_outputs
            else:
                for uid, record in normalized_outputs.items():
                    base = baseline[track][uid]
                    for key in ("query", "response", "images", "crop_images", "category", "l2-category", "cycle_category", "question_type", "bbox", "language"):
                        if record.get(key) != base.get(key):
                            raise ValueError(f"{condition}/{track}/{uid}: paired condition input mismatch at {key}")
            summary["conditions"][condition][track] = {"n": len(normalized_outputs), "technical_errors": 0}
            print(f"PASS {condition}/{track}: {len(normalized_outputs)} outputs", flush=True)

    summary["total_records_per_condition"] = sum(len(rows) for rows in expected.values())
    summary["total_generation_outputs"] = summary["total_records_per_condition"] * len(conditions)
    summary["prompt_registry_sha256"] = registry_hash
    summary["uid_compatibility"] = uid_compatibility
    args.run_root.mkdir(parents=True, exist_ok=True)
    result_path = args.run_root / "output-verification.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS ALL; WROTE {result_path}", flush=True)


if __name__ == "__main__":
    main()
