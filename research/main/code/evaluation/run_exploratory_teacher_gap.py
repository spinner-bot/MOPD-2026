"""Run immutable raw inference records for the frozen 1017 four-condition pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from exploratory_teacher_gap import CHOICES, CONDITIONS, validate_locked_manifest
from teacher_gap_inputs import render_teacher_gap_input, validate_generic_length


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def frozen_schedule(samples: list[dict], manifest_hash: str) -> list[tuple[dict, str]]:
    pairs = [(sample, condition) for sample in samples for condition in CONDITIONS]
    return sorted(
        pairs,
        key=lambda pair: sha256_bytes(
            f"{manifest_hash}\0{pair[0]['sample_id']}\0{pair[1]}".encode("utf-8")
        ),
    )


def choice_token_sequences(tokenizer) -> dict[str, list[int]]:
    sequences = {choice: list(tokenizer.encode(choice, add_special_tokens=False)) for choice in CHOICES}
    if any(not sequence for sequence in sequences.values()):
        raise ValueError("choice labels must tokenize to non-empty sequences")
    tuples = list(map(tuple, sequences.values()))
    if len(set(tuples)) != len(tuples):
        raise ValueError("choice label token sequences must be distinct")
    if len({len(sequence) for sequence in tuples}) != 1:
        raise ValueError("choice label token sequences must have equal lengths")
    for left in tuples:
        for right in tuples:
            if left != right and len(left) <= len(right) and right[: len(left)] == left:
                raise ValueError("choice label token sequences must not be prefixes of one another")
    return sequences


def allowed_token_function(prefix_length: int, sequences: dict[str, list[int]], eos_ids: set[int]):
    """Build a token-trie constraint that permits only A/B/C/D followed by EOS."""

    values = list(sequences.values())

    def allowed(_batch_id, input_ids):
        suffix = [int(value) for value in input_ids[prefix_length:].tolist()]
        options = []
        for sequence in values:
            if suffix == sequence:
                options.extend(sorted(eos_ids))
            elif len(suffix) < len(sequence) and sequence[: len(suffix)] == suffix:
                options.append(sequence[len(suffix)])
        return sorted(set(options)) or sorted(eos_ids)

    return allowed


def conditional_choice_scores(model, inputs, sequences: dict[str, list[int]], torch) -> dict[str, float]:
    prefix_ids = inputs["input_ids"]
    prefix_mask = inputs["attention_mask"]
    prefix_length = int(prefix_ids.shape[1])
    scores = {}
    for choice, sequence in sequences.items():
        target = torch.tensor([sequence], device=prefix_ids.device, dtype=prefix_ids.dtype)
        combined_ids = torch.cat([prefix_ids, target], dim=1)
        combined_mask = torch.cat([prefix_mask, torch.ones_like(target)], dim=1)
        model_inputs = {
            key: value for key, value in inputs.items() if key not in {"input_ids", "attention_mask"}
        }
        model_inputs.update({"input_ids": combined_ids, "attention_mask": combined_mask})
        logits = model(**model_inputs, use_cache=False).logits
        log_probs = torch.log_softmax(logits[0, prefix_length - 1 : prefix_length - 1 + len(sequence)], dim=-1)
        positions = torch.arange(len(sequence), device=log_probs.device)
        token_ids = torch.tensor(sequence, device=log_probs.device)
        scores[choice] = float(log_probs[positions, token_ids].sum().item())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--expected-samples", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0, help="per-rank scheduled-pair limit")
    args = parser.parse_args()

    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must satisfy 0 <= rank < world-size")
    current_commit = git_commit()
    if current_commit != args.code_commit:
        raise SystemExit(
            f"code commit changed after submission: expected={args.code_commit} actual={current_commit}"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing raw output: {args.output}")
    samples = load_jsonl(args.manifest)
    manifest_errors = validate_locked_manifest(samples, expected_n=args.expected_samples)
    if manifest_errors:
        raise SystemExit("locked manifest invalid: " + "; ".join(manifest_errors))
    manifest_hash = sha256_file(args.manifest)
    image_root = args.image_root.resolve(strict=True)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", local_files_only=True
    ).to("cuda").eval()
    sequences = choice_token_sequences(processor.tokenizer)
    length_check = validate_generic_length(
        samples, lambda text: len(processor.tokenizer.encode(text, add_special_tokens=False))
    )
    if not length_check["passes"]:
        raise SystemExit(f"Generic-T length contract failed: {length_check}")
    eos_value = getattr(model.generation_config, "eos_token_id", None)
    eos_ids = {eos_value} if isinstance(eos_value, int) else set(eos_value or [])
    tokenizer_eos = getattr(processor.tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, int):
        eos_ids.add(tokenizer_eos)
    if not eos_ids:
        raise SystemExit("no EOS token ids available")
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = min(eos_ids)

    schedule = frozen_schedule(samples, manifest_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output.open("x", encoding="utf-8") as sink, torch.inference_mode():
        for schedule_index, (sample, condition) in enumerate(schedule):
            if schedule_index % args.world_size != args.rank:
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1
            rendered = render_teacher_gap_input(sample, condition)
            record = {
                "sample_id": sample["sample_id"],
                "source_image_id": sample["source_image_id"],
                "source_cluster_id": sample["source_cluster_id"],
                "condition": condition,
                "schedule_index": schedule_index,
                "rank": args.rank,
                "world_size": args.world_size,
                "manifest_hash": manifest_hash,
                "model_revision": args.model_revision,
                "processor_revision": args.model_revision,
                "code_commit": args.code_commit,
                "prompt_hash": rendered.prompt_hash,
                "shell_hash": rendered.shell_hash,
                "question_hash": sha256_bytes(sample["final_question"].encode("utf-8")),
                "options_hash": sha256_bytes(json.dumps(
                    [sample[f"option_{choice}"] for choice in CHOICES],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")),
                "correct_choice": sample["final_answer"],
                "choice_token_ids": sequences,
                "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
                "generation_config": {"do_sample": False, "constrained_to": list(CHOICES)},
            }
            try:
                image_path = Path(rendered.image_path).resolve(strict=True)
                if not image_path.is_relative_to(image_root):
                    raise ValueError(f"image path escapes image root: {image_path}")
                image_checksum = sha256_file(image_path)
                if image_checksum != sample["image_checksum"]:
                    raise ValueError("image checksum differs from locked manifest")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": rendered.prompt},
                    ],
                }]
                template = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                images, videos = process_vision_info(messages)
                inputs = processor(
                    text=[template], images=images, videos=videos, padding=True, return_tensors="pt"
                ).to("cuda")
                scores = conditional_choice_scores(model, inputs, sequences, torch)
                prefix_length = int(inputs.input_ids.shape[1])
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max(map(len, sequences.values())) + 1,
                    do_sample=False,
                    eos_token_id=sorted(eos_ids),
                    pad_token_id=pad_token_id,
                    prefix_allowed_tokens_fn=allowed_token_function(prefix_length, sequences, eos_ids),
                )
                new_tokens = generated[0, prefix_length:]
                token_list = [int(token) for token in new_tokens.tolist()]
                record.update({
                    "image_checksum": image_checksum,
                    "input_tokens": prefix_length,
                    "choice_log_likelihoods": scores,
                    "generated_token_ids": token_list,
                    "generated_tokens": len(token_list),
                    "eos_reached": any(token in eos_ids for token in token_list),
                    "truncated": not any(token in eos_ids for token in token_list),
                    "raw_generation": processor.tokenizer.decode(token_list, skip_special_tokens=True),
                    "model_dtype": str(getattr(model, "dtype", "unknown")),
                })
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()


if __name__ == "__main__":
    main()
