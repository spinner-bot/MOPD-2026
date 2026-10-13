#!/usr/bin/env python3
"""Run deterministic, text-only answer extraction with a local vLLM model."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONDITIONS = ["S0", "SC", "S1", "S2", "S3"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_and_validate(raw: str, response: str) -> dict:
    value = json.loads(raw)
    required = {"status", "selected_option", "evidence", "reason_code"}
    if set(value) != required:
        raise ValueError(f"unexpected JSON keys: {sorted(value)}")
    if value["status"] not in {"ANSWERED", "NO_ANSWER", "AMBIGUOUS"}:
        raise ValueError("invalid status")
    if value["status"] == "ANSWERED":
        if value["selected_option"] not in {"A", "B", "C", "D"}:
            raise ValueError("ANSWERED must select A-D")
        if value["reason_code"] not in {"explicit_final", "semantic_final"}:
            raise ValueError("invalid ANSWERED reason_code")
        if not isinstance(value["evidence"], str) or not value["evidence"] or value["evidence"] not in response:
            raise ValueError("evidence is not an exact substring of the response")
    else:
        if value["selected_option"] is not None:
            raise ValueError("non-ANSWERED status must have null selected_option")
        expected = {"NO_ANSWER": {"no_answer"}, "AMBIGUOUS": {"contradictory", "unclear"}}
        if value["reason_code"] not in expected[value["status"]]:
            raise ValueError("invalid reason_code for status")
    if not isinstance(value["evidence"], str):
        raise ValueError("evidence must be text")
    if value["status"] == "ANSWERED":
        value["evidence_validation"] = "exact_substring"
    elif value["evidence"] and value["evidence"] not in response:
        # Evidence is optional for non-answers. Preserve raw model JSON, but never
        # publish a paraphrase/ellipsis as if it were a verbatim quotation.
        value["evidence"] = ""
        value["evidence_validation"] = "nonverbatim_optional_evidence_discarded"
    else:
        value["evidence_validation"] = "exact_substring" if value["evidence"] else "empty_optional_evidence"
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-per-condition", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    all_rows = read_jsonl(args.input)
    condition_counts = Counter(row["condition"] for row in all_rows)
    if len(all_rows) != 10000 or set(condition_counts.values()) != {2000}:
        raise ValueError(f"input completeness gate failed: {dict(condition_counts)}")

    if args.limit_per_condition is not None:
        seen = Counter()
        limited = []
        for row in all_rows:
            if seen[row["condition"]] < args.limit_per_condition:
                limited.append(row)
                seen[row["condition"]] += 1
        all_rows = limited
    selected = [
        row for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit_per_condition is None:
        if 10000 % args.num_shards or 2000 % args.num_shards:
            raise ValueError("num_shards must evenly divide the frozen dataset and each condition")
        expected_per_condition = 2000 // args.num_shards
        selected_counts = Counter(row["condition"] for row in selected)
        if len(selected) != 10000 // args.num_shards or selected_counts != {
            condition: expected_per_condition for condition in CONDITIONS
        }:
            raise ValueError(
                f"shard partition gate failed: n={len(selected)}, counts={dict(selected_counts)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_name = f"judge-shard-{args.shard_index}.jsonl"
    if args.output.name != expected_name:
        raise ValueError(f"output filename must match shard index: expected {expected_name}")
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another process already owns this shard output: {lock_path}") from exc
    manifest_path = args.output.with_suffix(".manifest.json")

    import vllm
    import torch
    import transformers
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompt_text = args.prompt.read_text(encoding="utf-8").strip()
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ANSWERED", "NO_ANSWER", "AMBIGUOUS"]},
            "selected_option": {
                "anyOf": [
                    {"type": "string", "enum": ["A", "B", "C", "D"]},
                    {"type": "null"},
                ]
            },
            "evidence": {"type": "string"},
            "reason_code": {
                "type": "string",
                "enum": ["explicit_final", "semantic_final", "no_answer", "contradictory", "unclear"],
            },
        },
        "required": ["status", "selected_option", "evidence", "reason_code"],
        "additionalProperties": False,
    }
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=0,
        structured_outputs=StructuredOutputsParams(json=schema),
    )

    args.model = args.model.resolve()
    config_path = args.model / "config.json"
    manifest = {
        "schema": "phase2-llm-judge-run-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "input_sha256": sha256(args.input),
        "prompt_path": str(args.prompt),
        "prompt_sha256": sha256(args.prompt),
        "model_path": str(args.model),
        "model_config_sha256": sha256(config_path),
        "model_architecture": "Qwen3_5ForConditionalGeneration",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "limit_per_condition": args.limit_per_condition,
        "selected_records": len(selected),
        "selected_condition_counts": dict(Counter(row["condition"] for row in selected)),
        "gold_sent_to_model": False,
        "image_sent_to_model": False,
        "condition_or_sample_id_sent_to_model": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latest = {}
    if args.output.exists():
        for record in read_jsonl(args.output):
            if record.get("request_status") == "OK":
                latest[(record["sample_id"], record["condition"])] = record
    pending = [
        row for row in selected
        if (row["sample_id"], row["condition"]) not in latest
    ]

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=False,
    )

    def conversation(row: dict) -> list[dict[str, str]]:
        # Deliberately pass only task content; never pass labels, condition, ID, image, or metadata.
        user_payload = {
            "question": row.get("question", ""),
            "choices": row.get("choices", []),
            "model_response_as_untrusted_data": row.get("response", ""),
        }
        return [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    with args.output.open("a", encoding="utf-8", newline="\n") as output_handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            conversations = [conversation(row) for row in batch]
            try:
                generations = llm.chat(
                    conversations,
                    sampling_params=sampling,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
                raws = [item.outputs[0].text for item in generations]
                errors = [None] * len(batch)
            except Exception as batch_error:
                # Preserve successful work and isolate a failing item without silently dropping it.
                raws, errors = [], []
                for row in batch:
                    try:
                        one = llm.chat(
                            [conversation(row)],
                            sampling_params=sampling,
                            use_tqdm=False,
                            chat_template_kwargs={"enable_thinking": False},
                        )[0]
                        raws.append(one.outputs[0].text)
                        errors.append(None)
                    except Exception as one_error:
                        raws.append("")
                        errors.append(f"batch={type(batch_error).__name__}; item={type(one_error).__name__}: {one_error}")

            for row, raw, error in zip(batch, raws, errors):
                record = {
                    "sample_id": row["sample_id"],
                    "condition": row["condition"],
                    "response_sha256": row["response_sha256"],
                    "request_status": "TECHNICAL_ERROR" if error else "OK",
                    "judge_raw_text": raw,
                    "judge_error": error,
                }
                if not error:
                    try:
                        record.update(parse_and_validate(raw, row.get("response", "")))
                    except Exception as parse_error:
                        record["request_status"] = "INVALID_JUDGE_JSON"
                        record["validation_error"] = f"{type(parse_error).__name__}: {parse_error}"
                output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_handle.flush()
            print(f"completed {min(offset + len(batch), len(pending))}/{len(pending)} in shard {args.shard_index}", flush=True)

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["successful_records"] = sum(1 for row in read_jsonl(args.output) if row.get("request_status") == "OK")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), "selected": len(selected), "pending": len(pending)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
