"""Run one immutable raw-output shard for a 1016 candidate stage.

Stages are independent so raw model output and deterministic parsing remain in
separate files.  Prompts are rendered from an allow-list and never include the
official M3VQA evidence text or URLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from extract_json_response import extract_json_object
from generation_audit import collect_eos_ids, hit_token_cap
from teacher_gap_candidate_contract import (
    STAGES,
    proposal_errors,
    render_compact_stage_prompt,
    render_stage_prompt,
    stable_hash,
    validate_candidate,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def unwrap_candidate(row: dict) -> tuple[dict, dict | None]:
    if isinstance(row.get("candidate"), dict):
        candidate = row["candidate"]
    else:
        candidate = row
    audit = row.get("audit_proposal")
    return candidate, audit if isinstance(audit, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="per-rank record limit; zero means all")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--fallback-max-new-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must satisfy 0 <= rank < world-size")
    if args.max_new_tokens < 1 or args.fallback_max_new_tokens < 1:
        raise SystemExit("primary and fallback max-new-tokens must be positive")
    if not args.model.is_dir():
        raise SystemExit(f"missing model directory: {args.model}")
    current_commit = git_commit()
    if current_commit != args.code_commit:
        raise SystemExit(
            f"code commit changed after submission: expected={args.code_commit} actual={current_commit}"
        )
    image_root = args.image_root.resolve(strict=True)

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", local_files_only=True
    ).to("cuda").eval()
    eos_ids = collect_eos_ids(processor, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = 0

    def generate_attempt(image_path: Path, prompt: str, kind: str, max_new_tokens: int, candidate: dict) -> dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }]
        template = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=[template], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to("cuda")
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_tokens = generated[0, inputs.input_ids.shape[1]:]
        eos_reached = any(int(token) in eos_ids for token in new_tokens)
        truncated = hit_token_cap(new_tokens, eos_ids)
        raw = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        proposal = extract_json_object(raw)
        validation_errors = []
        if truncated:
            validation_errors.append("truncated_output")
        if proposal is None:
            validation_errors.append("invalid_json")
        else:
            validation_errors.extend(proposal_errors(args.stage, proposal, candidate))
        return {
            "attempt_kind": kind,
            "max_new_tokens": max_new_tokens,
            "prompt_hash": stable_hash(prompt),
            "chat_template_hash": stable_hash(template),
            "input_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": int(len(new_tokens)),
            "eos_reached": eos_reached,
            "truncated": truncated,
            "raw_model_output": raw,
            "validation_errors": sorted(set(validation_errors)),
            "schema_valid": not validation_errors,
        }

    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink, torch.inference_mode():
        for index, line in enumerate(source):
            if index % args.world_size != args.rank:
                continue
            if args.limit and selected >= args.limit:
                break
            selected += 1
            input_row = json.loads(line)
            candidate, audit_proposal = unwrap_candidate(input_row)
            record = {
                "candidate_id": candidate.get("candidate_id"),
                "source_image_id": candidate.get("source_image_id"),
                "stage": args.stage,
                "input_index": index,
                "rank": args.rank,
                "world_size": args.world_size,
                "candidate": candidate,
                "model_revision": args.model_revision,
                "code_commit": args.code_commit,
                "thinking_mode": args.thinking,
                "chat_template": {
                    "enable_thinking": args.thinking,
                    "add_generation_prompt": True,
                },
                "model_dtype": str(getattr(model, "dtype", "unknown")),
                "generation_config": {
                    "do_sample": False,
                    "primary_max_new_tokens": args.max_new_tokens,
                    "fallback_max_new_tokens": args.fallback_max_new_tokens,
                    "max_attempts": 2,
                    # Repetition constraints are deliberately disabled for the
                    # exact JSON schema. r2's no-repeat n-gram constraint
                    # systematically corrupted repeated field-name tokens.
                    "structured_output_repetition_constraints": "disabled",
                },
                "prompt_fields_used": (
                    ["image_original", "final_question_source"]
                    if args.stage == "blind"
                    else ["image_original", "final_question_source", "reference_answers"]
                    if args.stage == "audit"
                    else ["image_original", "final_question_source", "reference_answers", "audit_proposal.required_evidence", "audit_proposal.minimal_reasoning_graph"]
                ),
            }
            if args.stage == "construct":
                record["blind_proposal"] = input_row.get("blind_proposal")
                record["audit_proposal"] = audit_proposal
                record["screening_provenance"] = input_row.get("screening_provenance")
            try:
                candidate_errors = validate_candidate(candidate)
                if candidate_errors:
                    raise ValueError(",".join(candidate_errors))
                image_path = Path(candidate["image_original"]).resolve(strict=True)
                if not image_path.is_relative_to(image_root):
                    raise ValueError(f"image path escapes image root: {image_path}")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                with Image.open(image_path) as image:
                    image.verify()
                primary_prompt = render_stage_prompt(candidate, args.stage, audit_proposal)
                attempts = [generate_attempt(
                    image_path, primary_prompt, "primary", args.max_new_tokens, candidate
                )]
                if attempts[0]["validation_errors"]:
                    compact_prompt = render_compact_stage_prompt(candidate, args.stage, audit_proposal)
                    attempts.append(generate_attempt(
                        image_path,
                        compact_prompt,
                        "compact_fallback",
                        args.fallback_max_new_tokens,
                        candidate,
                    ))
                chosen = attempts[-1]
                record.update({
                    "image_checksum": sha256_file(image_path),
                    "attempts": attempts,
                    "attempt_count": len(attempts),
                    "chosen_attempt": len(attempts) - 1,
                    "fallback_used": len(attempts) == 2,
                    "fallback_recovered": len(attempts) == 2 and not chosen["validation_errors"],
                    "primary_failure_reasons": attempts[0]["validation_errors"],
                    **{
                        key: chosen[key]
                        for key in (
                            "prompt_hash",
                            "chat_template_hash",
                            "input_tokens",
                            "generated_tokens",
                            "eos_reached",
                            "truncated",
                            "raw_model_output",
                        )
                    },
                })
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()


if __name__ == "__main__":
    main()
