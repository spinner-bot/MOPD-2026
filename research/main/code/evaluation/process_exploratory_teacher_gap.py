"""Parse raw 1017 pilot inference, apply the gate, and make the frozen decision."""

from __future__ import annotations

import argparse
import glob
import json
import hashlib
import math
from pathlib import Path

from exploratory_teacher_gap import (
    CHOICES,
    decide_teacher_gap,
    decide_teacher_gap_label_rotation,
    validate_manifest_bundle,
    validate_manifest_profile,
    validate_result_records,
    validate_results_against_manifest,
)
from g_pc0 import (
    decide_g_pc0,
    decide_g_pc0_label_rotation,
    diagnose_g_pc0,
    diagnose_g_pc0_label_rotation,
    validate_g_pc0_assets,
    write_json_exclusive,
)
from teacher_gap_inputs import COMMON_SHELL, validate_generic_length


def load_jsonl(patterns: list[str]) -> list[dict]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    rows = []
    for value in paths:
        with open(value, encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def parse_constrained_answer(raw: dict) -> tuple[str | None, str | None]:
    sequences = raw.get("choice_token_ids")
    token_ids = raw.get("generated_token_ids")
    if not isinstance(sequences, dict) or set(sequences) != set(CHOICES) or not isinstance(token_ids, list):
        return None, None
    matches = []
    for choice in CHOICES:
        sequence = sequences[choice]
        if isinstance(sequence, list) and token_ids[: len(sequence)] == sequence:
            matches.append(choice)
    if len(matches) == 1:
        return matches[0], "constrained_token_trie"
    text = str(raw.get("raw_generation", "")).strip()
    if text in CHOICES:
        return text, "exact_text_fallback"
    return None, None


def enrich_raw_record(raw: dict) -> dict:
    row = dict(raw)
    parsed, method = parse_constrained_answer(raw)
    row["parsed_answer"] = parsed
    row["parse_method"] = method
    scores = raw.get("choice_log_likelihoods")
    if not isinstance(scores, dict) or set(scores) != set(CHOICES):
        return row
    values = [float(scores[choice]) for choice in CHOICES]
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    probabilities = {choice: weights[index] / total for index, choice in enumerate(CHOICES)}
    correct_choice = raw.get("correct_choice")
    row["choice_probabilities"] = probabilities
    if correct_choice in CHOICES:
        row["correct"] = parsed == correct_choice
        row["correct_log_prob"] = float(scores[correct_choice])
        incorrect = max(float(scores[choice]) for choice in CHOICES if choice != correct_choice)
        row["margin"] = float(scores[correct_choice]) - incorrect
    row["entropy"] = -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--inputs", nargs="+", required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--expected-samples", type=int, required=True)
    validate_parser.add_argument(
        "--manifest-profile",
        choices=("natural", "natural_rotation", "g_pc0", "g_pc0_rotation"),
        default="natural",
    )
    validate_parser.add_argument("--base-manifest", type=Path)
    validate_parser.add_argument("--image-root", type=Path)
    validate_parser.add_argument("--parsed-output", type=Path, required=True)
    validate_parser.add_argument("--summary", type=Path, required=True)
    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--input", type=Path, required=True)
    decide_parser.add_argument("--expected-samples", type=int, default=96)
    decide_parser.add_argument("--leakage-free", action="store_true", required=True)
    decide_parser.add_argument("--output", type=Path, required=True)
    rotation_natural_decide_parser = subparsers.add_parser("decide-natural-label-rotation")
    rotation_natural_decide_parser.add_argument("--input", type=Path, required=True)
    rotation_natural_decide_parser.add_argument("--manifest", type=Path, required=True)
    rotation_natural_decide_parser.add_argument("--base-manifest", type=Path, required=True)
    rotation_natural_decide_parser.add_argument("--expected-samples", type=int, required=True)
    rotation_natural_decide_parser.add_argument("--leakage-free", action="store_true", required=True)
    rotation_natural_decide_parser.add_argument("--output", type=Path, required=True)
    natural_rotation_manifest_parser = subparsers.add_parser("validate-natural-label-rotation-manifest")
    natural_rotation_manifest_parser.add_argument("--manifest", type=Path, required=True)
    natural_rotation_manifest_parser.add_argument("--base-manifest", type=Path, required=True)
    natural_rotation_manifest_parser.add_argument("--expected-samples", type=int, required=True)
    natural_rotation_manifest_parser.add_argument("--image-root", type=Path, required=True)
    natural_rotation_manifest_parser.add_argument("--tokenizer", type=Path, required=True)
    natural_rotation_manifest_parser.add_argument("--summary", type=Path, required=True)
    pc_decide_parser = subparsers.add_parser("decide-positive-control")
    pc_decide_parser.add_argument("--input", type=Path, required=True)
    pc_decide_parser.add_argument("--manifest", type=Path, required=True)
    pc_decide_parser.add_argument("--image-root", type=Path, required=True)
    pc_decide_parser.add_argument("--output", type=Path, required=True)
    pc_diagnose_parser = subparsers.add_parser("diagnose-positive-control")
    pc_diagnose_parser.add_argument("--input", type=Path, required=True)
    pc_diagnose_parser.add_argument("--manifest", type=Path, required=True)
    pc_diagnose_parser.add_argument("--image-root", type=Path, required=True)
    pc_diagnose_parser.add_argument("--decision", type=Path, required=True)
    pc_diagnose_parser.add_argument("--output", type=Path, required=True)
    pc_manifest_parser = subparsers.add_parser("validate-positive-control-manifest")
    pc_manifest_parser.add_argument("--manifest", type=Path, required=True)
    pc_manifest_parser.add_argument("--image-root", type=Path, required=True)
    pc_manifest_parser.add_argument("--tokenizer", type=Path, required=True)
    pc_manifest_parser.add_argument("--summary", type=Path, required=True)
    rotation_manifest_parser = subparsers.add_parser("validate-label-rotation-manifest")
    rotation_manifest_parser.add_argument("--manifest", type=Path, required=True)
    rotation_manifest_parser.add_argument("--base-manifest", type=Path, required=True)
    rotation_manifest_parser.add_argument("--image-root", type=Path, required=True)
    rotation_manifest_parser.add_argument("--tokenizer", type=Path, required=True)
    rotation_manifest_parser.add_argument("--summary", type=Path, required=True)
    rotation_decide_parser = subparsers.add_parser("decide-label-rotation")
    rotation_decide_parser.add_argument("--input", type=Path, required=True)
    rotation_decide_parser.add_argument("--manifest", type=Path, required=True)
    rotation_decide_parser.add_argument("--base-manifest", type=Path, required=True)
    rotation_decide_parser.add_argument("--image-root", type=Path, required=True)
    rotation_decide_parser.add_argument("--output", type=Path, required=True)
    rotation_diagnose_parser = subparsers.add_parser("diagnose-label-rotation")
    rotation_diagnose_parser.add_argument("--input", type=Path, required=True)
    rotation_diagnose_parser.add_argument("--manifest", type=Path, required=True)
    rotation_diagnose_parser.add_argument("--base-manifest", type=Path, required=True)
    rotation_diagnose_parser.add_argument("--image-root", type=Path, required=True)
    rotation_diagnose_parser.add_argument("--decision", type=Path, required=True)
    rotation_diagnose_parser.add_argument("--output", type=Path, required=True)
    bundle_parser = subparsers.add_parser("validate-manifests")
    bundle_parser.add_argument("--canary", type=Path, required=True)
    bundle_parser.add_argument("--dev", type=Path, required=True)
    bundle_parser.add_argument("--locked", type=Path, required=True)
    bundle_parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate-manifests":
        if args.summary.exists():
            raise SystemExit(f"refusing to overwrite manifest gate summary: {args.summary}")
        paths = {"canary": args.canary, "dev": args.dev, "locked": args.locked}
        partitions = {name: load_jsonl([str(path)]) for name, path in paths.items()}
        errors = validate_manifest_bundle(partitions)
        summary = {
            "status": "passed" if not errors else "failed",
            "partition_records": {name: len(rows) for name, rows in partitions.items()},
            "partition_sha256": {name: file_sha256(path) for name, path in paths.items()},
            "validation_errors": errors,
            "performance_results_read": False,
        }
        write_json_new(args.summary, summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if errors:
            raise SystemExit("teacher-gap manifest bundle FAILED: " + "; ".join(errors))
    elif args.command in {
        "validate-positive-control-manifest",
        "validate-label-rotation-manifest",
        "validate-natural-label-rotation-manifest",
    }:
        if args.summary.exists():
            raise SystemExit(f"refusing to overwrite manifest summary: {args.summary}")
        samples = load_jsonl([str(args.manifest)])
        is_pc_rotation = args.command == "validate-label-rotation-manifest"
        is_natural_rotation = args.command == "validate-natural-label-rotation-manifest"
        is_rotation = is_pc_rotation or is_natural_rotation
        base_samples = load_jsonl([str(args.base_manifest)]) if is_rotation else None
        base_hash = file_sha256(args.base_manifest) if is_rotation else None
        expected_n = args.expected_samples * 4 if is_natural_rotation else (64 if is_pc_rotation else 16)
        errors = validate_manifest_profile(
            samples,
            expected_n=expected_n,
            manifest_profile=(
                "natural_rotation" if is_natural_rotation else ("g_pc0_rotation" if is_pc_rotation else "g_pc0")
            ),
            base_samples=base_samples,
            base_manifest_hash=base_hash,
        )
        errors.extend(validate_g_pc0_assets(samples, args.image_root))
        length_check = None
        tokenizer_hashes = {}
        choice_token_ids = None
        try:
            from transformers import AutoTokenizer
            from run_exploratory_teacher_gap import choice_token_sequences

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            length_check = validate_generic_length(
                samples,
                lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            )
            if not length_check["passes"]:
                errors.append(f"Generic-T length contract failed: {length_check}")
            choice_token_ids = choice_token_sequences(tokenizer)
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                path = args.tokenizer / name
                if path.is_file():
                    tokenizer_hashes[name] = file_sha256(path)
        except Exception as error:
            errors.append(f"tokenizer preflight failed: {type(error).__name__}: {error}")
        summary = {
            "status": "passed" if not errors else "failed",
            "manifest_profile": (
                "natural_label_rotation_v1"
                if is_natural_rotation
                else ("g_pc0_label_rotation_diagnostic_v1" if is_pc_rotation else "g_pc0_v1")
            ),
            "manifest_records": len(samples),
            "manifest_sha256": file_sha256(args.manifest),
            "image_root": str(args.image_root.resolve()),
            "tokenizer_path": str(args.tokenizer.resolve()),
            "tokenizer_artifact_sha256": tokenizer_hashes,
            "choice_token_ids": choice_token_ids,
            "shell_sha256": hashlib.sha256(COMMON_SHELL.encode("utf-8")).hexdigest(),
            "generic_length_check": length_check,
            "validation_errors": sorted(set(errors)),
            "performance_results_read": False,
            "gpu_used": False,
        }
        write_json_new(args.summary, summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if errors:
            raise SystemExit("label-rotation manifest preflight FAILED: " + "; ".join(sorted(set(errors))))
    elif args.command == "validate":
        if args.parsed_output.exists() or args.summary.exists():
            raise SystemExit("refusing to overwrite parsed output or validation summary")
        rows = [enrich_raw_record(row) for row in load_jsonl(args.inputs)]
        samples = load_jsonl([str(args.manifest)])
        manifest_hash = file_sha256(args.manifest)
        base_samples = load_jsonl([str(args.base_manifest)]) if args.base_manifest else None
        base_hash = file_sha256(args.base_manifest) if args.base_manifest else None
        errors = validate_result_records(rows, expected_samples=args.expected_samples)
        errors.extend(validate_manifest_profile(
            samples,
            expected_n=args.expected_samples,
            manifest_profile=args.manifest_profile,
            base_samples=base_samples,
            base_manifest_hash=base_hash,
        ))
        if args.manifest_profile in {"g_pc0", "g_pc0_rotation"}:
            if args.image_root is None:
                errors.append("g_pc0 validation requires --image-root")
            else:
                errors.extend(validate_g_pc0_assets(samples, args.image_root))
        errors.extend(validate_results_against_manifest(rows, samples, manifest_hash))
        summary = {
            "expected_samples": args.expected_samples,
            "expected_records": args.expected_samples * 4,
            "observed_records": len(rows),
            "unique_sample_condition": len({(row.get("sample_id"), row.get("condition")) for row in rows}),
            "manifest_sha256": manifest_hash,
            "validation_errors": errors,
            "status": "passed" if not errors else "failed",
            "note": "No condition-level performance is emitted before the completeness gate passes.",
        }
        write_json_new(args.summary, summary)
        if errors:
            raise SystemExit("teacher-gap result gate FAILED: " + "; ".join(errors))
        write_jsonl(args.parsed_output, rows)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    elif args.command == "decide":
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite decision output: {args.output}")
        rows = load_jsonl([str(args.input)])
        decision = decide_teacher_gap(
            rows,
            leakage_free=args.leakage_free,
            expected_samples=args.expected_samples,
        )
        write_json_new(args.output, decision)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        if decision["decision"] == "INVALID":
            raise SystemExit("teacher-gap decision refused invalid records")
    elif args.command == "decide-natural-label-rotation":
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite decision output: {args.output}")
        rows = load_jsonl([str(args.input)])
        samples = load_jsonl([str(args.manifest)])
        base_samples = load_jsonl([str(args.base_manifest)])
        manifest_hash = file_sha256(args.manifest)
        base_hash = file_sha256(args.base_manifest)
        errors = validate_result_records(rows, expected_samples=args.expected_samples * 4)
        errors.extend(validate_manifest_profile(
            samples,
            expected_n=args.expected_samples * 4,
            manifest_profile="natural_rotation",
            base_samples=base_samples,
            base_manifest_hash=base_hash,
        ))
        errors.extend(validate_results_against_manifest(rows, samples, manifest_hash))
        decision = decide_teacher_gap_label_rotation(
            rows,
            leakage_free=args.leakage_free,
            expected_samples=args.expected_samples,
            validation_errors=errors,
        )
        decision["manifest_sha256"] = manifest_hash
        decision["base_manifest_sha256"] = base_hash
        write_json_new(args.output, decision)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        if decision["decision"] == "INVALID":
            raise SystemExit("natural label-rotation decision refused invalid records")
    elif args.command == "decide-positive-control":
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite decision output: {args.output}")
        rows = load_jsonl([str(args.input)])
        samples = load_jsonl([str(args.manifest)])
        manifest_hash = file_sha256(args.manifest)
        errors = validate_result_records(rows, expected_samples=16)
        errors.extend(validate_manifest_profile(samples, expected_n=16, manifest_profile="g_pc0"))
        errors.extend(validate_g_pc0_assets(samples, args.image_root))
        errors.extend(validate_results_against_manifest(rows, samples, manifest_hash))
        decision = decide_g_pc0(rows, errors)
        decision["manifest_sha256"] = manifest_hash
        write_json_exclusive(args.output, decision)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        if decision["decision"] == "INVALID":
            raise SystemExit("G-PC0 decision refused invalid records")
    elif args.command == "decide-label-rotation":
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite decision output: {args.output}")
        rows = load_jsonl([str(args.input)])
        samples = load_jsonl([str(args.manifest)])
        base_samples = load_jsonl([str(args.base_manifest)])
        manifest_hash = file_sha256(args.manifest)
        base_hash = file_sha256(args.base_manifest)
        errors = validate_result_records(rows, expected_samples=64)
        errors.extend(validate_manifest_profile(
            samples,
            expected_n=64,
            manifest_profile="g_pc0_rotation",
            base_samples=base_samples,
            base_manifest_hash=base_hash,
        ))
        errors.extend(validate_g_pc0_assets(samples, args.image_root))
        errors.extend(validate_results_against_manifest(rows, samples, manifest_hash))
        decision = decide_g_pc0_label_rotation(rows, errors)
        decision["manifest_sha256"] = manifest_hash
        decision["base_manifest_sha256"] = base_hash
        write_json_exclusive(args.output, decision)
        print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
        if decision["decision"] == "INVALID":
            raise SystemExit("G-PC0 label-rotation decision refused invalid records")
    elif args.command == "diagnose-label-rotation":
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite diagnostic output: {args.output}")
        rows = load_jsonl([str(args.input)])
        samples = load_jsonl([str(args.manifest)])
        base_samples = load_jsonl([str(args.base_manifest)])
        manifest_hash = file_sha256(args.manifest)
        base_hash = file_sha256(args.base_manifest)
        errors = validate_result_records(rows, expected_samples=64)
        errors.extend(validate_manifest_profile(
            samples,
            expected_n=64,
            manifest_profile="g_pc0_rotation",
            base_samples=base_samples,
            base_manifest_hash=base_hash,
        ))
        errors.extend(validate_g_pc0_assets(samples, args.image_root))
        errors.extend(validate_results_against_manifest(rows, samples, manifest_hash))
        decision = json.loads(args.decision.read_text(encoding="utf-8"))
        if decision.get("manifest_sha256") != manifest_hash or decision.get("base_manifest_sha256") != base_hash:
            errors.append("decision manifest binding differs from frozen manifests")
        if errors:
            raise SystemExit("label-rotation diagnosis refused invalid inputs: " + "; ".join(sorted(set(errors))))
        diagnostic = diagnose_g_pc0_label_rotation(rows, str(decision.get("decision", "")))
        if diagnostic["recomputed_decision"] != decision.get("decision"):
            raise SystemExit("posthoc diagnostic recomputation differs from frozen decision")
        diagnostic["manifest_sha256"] = manifest_hash
        diagnostic["base_manifest_sha256"] = base_hash
        write_json_exclusive(args.output, diagnostic)
        print(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True))
    elif args.command == "diagnose-positive-control":
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite diagnostic output: {args.output}")
        rows = load_jsonl([str(args.input)])
        samples = load_jsonl([str(args.manifest)])
        manifest_hash = file_sha256(args.manifest)
        errors = validate_result_records(rows, expected_samples=16)
        errors.extend(validate_manifest_profile(samples, expected_n=16, manifest_profile="g_pc0"))
        errors.extend(validate_g_pc0_assets(samples, args.image_root))
        errors.extend(validate_results_against_manifest(rows, samples, manifest_hash))
        decision = json.loads(args.decision.read_text(encoding="utf-8"))
        if decision.get("manifest_sha256") != manifest_hash:
            errors.append("decision manifest hash differs from frozen manifest")
        if errors:
            raise SystemExit("G-PC0 diagnosis refused invalid inputs: " + "; ".join(sorted(set(errors))))
        diagnostic = diagnose_g_pc0(rows, str(decision.get("decision", "")))
        diagnostic["manifest_sha256"] = manifest_hash
        write_json_exclusive(args.output, diagnostic)
        print(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
