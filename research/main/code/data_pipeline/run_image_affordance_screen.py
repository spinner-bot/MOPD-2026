#!/usr/bin/env python3
"""Run one immutable raw shard of question-free image affordance proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from extract_json_response import extract_json_object
from generation_audit import collect_eos_ids, hit_token_cap
from image_affordance_contract import (
    affordance_fallback_lock,
    affordance_lock_errors,
    render_affordance_prompt,
    render_compact_affordance_prompt,
    stable_hash,
    validate_affordance_proposal,
    validate_image_candidate,
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
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--fallback-max-new-tokens", type=int, default=768)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must satisfy 0 <= rank < world-size")
    if git_commit() != args.code_commit:
        raise SystemExit("code commit changed after submission")
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

    def generate_attempt(image_path: Path, prompt: str, kind: str, max_tokens: int, lock: dict) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]}]
        template = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=[template], images=images, videos=videos, padding=True,
            return_tensors="pt",
        ).to("cuda")
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new_tokens = generated[0, inputs.input_ids.shape[1]:]
        eos_reached = any(int(token) in eos_ids for token in new_tokens)
        truncated = hit_token_cap(new_tokens, eos_ids)
        raw = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        proposal = extract_json_object(raw)
        errors = []
        if truncated:
            errors.append("truncated_output")
        if proposal is None:
            errors.append("invalid_json")
        else:
            errors.extend(validate_affordance_proposal(proposal))
            errors.extend(affordance_lock_errors(proposal, lock))
        return {
            "attempt_kind": kind,
            "max_new_tokens": max_tokens,
            "prompt_hash": stable_hash(prompt),
            "chat_template_hash": stable_hash(template),
            "input_tokens": int(inputs.input_ids.shape[1]),
            "generated_tokens": int(len(new_tokens)),
            "eos_reached": eos_reached,
            "truncated": truncated,
            "raw_model_output": raw,
            "validation_errors": sorted(set(errors)),
            "schema_valid": not errors,
            "preserved_decision_fields": lock,
        }

    selected = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink, torch.inference_mode():
        for index, line in enumerate(source):
            if index % args.world_size != args.rank:
                continue
            if args.limit and selected >= args.limit:
                break
            selected += 1
            candidate = json.loads(line)
            record = {
                "image_candidate_id": candidate.get("image_candidate_id"),
                "source_image_id": candidate.get("source_image_id"),
                "stage": "image_affordance",
                "input_index": index,
                "rank": args.rank,
                "world_size": args.world_size,
                "candidate": candidate,
                "model_revision": args.model_revision,
                "code_commit": args.code_commit,
                "thinking_mode": args.thinking,
                "chat_template": {"enable_thinking": args.thinking, "add_generation_prompt": True},
                "model_dtype": str(getattr(model, "dtype", "unknown")),
                "generation_config": {
                    "do_sample": False,
                    "primary_max_new_tokens": args.max_new_tokens,
                    "fallback_max_new_tokens": args.fallback_max_new_tokens,
                    "max_attempts": 2,
                    "structured_output_repetition_constraints": "disabled",
                },
                "prompt_fields_used": ["image_original"],
            }
            try:
                errors = validate_image_candidate(candidate)
                if errors:
                    raise ValueError(",".join(errors))
                image_path = Path(candidate["image_original"]).resolve(strict=True)
                if not image_path.is_relative_to(image_root) or not image_path.is_file():
                    raise ValueError("image path escapes image root")
                checksum = sha256_file(image_path)
                if checksum != candidate["image_sha256"]:
                    raise ValueError("image checksum mismatch")
                with Image.open(image_path) as image:
                    image.verify()
                primary_prompt = render_affordance_prompt(candidate)
                attempts = [generate_attempt(image_path, primary_prompt, "primary", args.max_new_tokens, {})]
                if attempts[0]["validation_errors"]:
                    primary = extract_json_object(attempts[0]["raw_model_output"])
                    lock = affordance_fallback_lock(primary)
                    attempts.append(generate_attempt(
                        image_path, render_compact_affordance_prompt(primary),
                        "compact_fallback", args.fallback_max_new_tokens, lock,
                    ))
                chosen = attempts[-1]
                record.update({
                    "image_checksum": checksum,
                    "attempts": attempts,
                    "attempt_count": len(attempts),
                    "chosen_attempt": len(attempts) - 1,
                    "fallback_used": len(attempts) == 2,
                    "fallback_recovered": len(attempts) == 2 and not chosen["validation_errors"],
                    "primary_failure_reasons": attempts[0]["validation_errors"],
                    **{key: chosen[key] for key in (
                        "prompt_hash", "chat_template_hash", "input_tokens",
                        "generated_tokens", "eos_reached", "truncated",
                        "raw_model_output",
                    )},
                })
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()


if __name__ == "__main__":
    main()
