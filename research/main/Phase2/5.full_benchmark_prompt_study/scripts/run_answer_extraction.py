#!/usr/bin/env python3
"""Text-only, structured answer extraction for one Work5 condition/track shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prepare_answer_extraction import extract_option_labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _strip_markdown_formatting(text: str) -> tuple[str, list[int]]:
    """Remove only common paired Markdown emphasis/code markers and retain offsets."""
    pattern = re.compile(r"(\*\*|__|`)(.+?)\1", flags=re.DOTALL)
    removed: set[int] = set()
    for match in pattern.finditer(text):
        marker = match.group(1)
        removed.update(range(match.start(), match.start() + len(marker)))
        removed.update(range(match.end() - len(marker), match.end()))
    offsets = [index for index in range(len(text)) if index not in removed]
    return "".join(text[index] for index in offsets), offsets


def _recover_markdown_exact_span(response: str, evidence: str) -> str | None:
    """Map a quote differing only by Markdown markers back to exact source text."""
    normalized_response, offsets = _strip_markdown_formatting(response)
    normalized_evidence, _ = _strip_markdown_formatting(evidence)
    if not normalized_evidence:
        return None
    start = normalized_response.find(normalized_evidence)
    if start < 0:
        return None
    first = offsets[start]
    last = offsets[start + len(normalized_evidence) - 1]
    return response[first : last + 1]


def validate(raw: str, row: dict) -> dict:
    value = json.loads(raw)
    required = {"status", "selected_option", "answer_text", "evidence", "reason_code"}
    if set(value) != required:
        raise ValueError(f"unexpected keys: {sorted(value)}")
    status = value["status"]
    reason = value["reason_code"]
    if status not in {"ANSWERED", "NO_ANSWER", "AMBIGUOUS"}:
        raise ValueError("invalid status")
    if not isinstance(value["answer_text"], str) or not isinstance(value["evidence"], str):
        raise ValueError("answer_text and evidence must be strings")
    response = str(row.get("model_response") or "")
    fmt = row["answer_format"]

    if status == "ANSWERED":
        if reason not in {"explicit_final", "semantic_final"}:
            raise ValueError("invalid ANSWERED reason")
        if value["evidence"] and value["evidence"] not in response:
            recovered = _recover_markdown_exact_span(response, value["evidence"])
            if recovered is not None:
                value["evidence"] = recovered
                value["evidence_validation"] = "exact_substring_markdown_normalized"
        if not value["evidence"] or value["evidence"] not in response:
            raise ValueError("ANSWERED evidence is not a nonempty exact response substring")
        if fmt == "MCQ_OPTION":
            valid_labels = set(row.get("available_option_labels_in_order") or [])
            valid_labels.update(extract_option_labels(str(row.get("query") or "")))
            if value["selected_option"] not in valid_labels or value["answer_text"]:
                raise ValueError("MCQ answer must be one listed literal label and have empty answer_text")
        else:
            if value["selected_option"] is not None:
                raise ValueError("short-answer selected_option must be null")
            if not value["answer_text"] or value["answer_text"] not in response:
                raise ValueError("short answer must be an exact substring of the response")
    else:
        expected = {"NO_ANSWER": {"no_answer"}, "AMBIGUOUS": {"contradictory", "unclear"}}
        if value["selected_option"] is not None or value["answer_text"]:
            raise ValueError("non-answered record must not contain a selected answer")
        if reason not in expected[status]:
            raise ValueError("invalid non-ANSWERED reason")
        if value["evidence"] and value["evidence"] not in response:
            value["evidence"] = ""
            value["evidence_validation"] = "nonverbatim_optional_evidence_discarded"

    if "evidence_validation" not in value:
        value["evidence_validation"] = "exact_substring" if value["evidence"] else "empty_optional_evidence"
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor parallel size must be positive")

    rows = read_jsonl(args.input)
    keys = [(row.get("sample_uid"), row.get("condition"), row.get("track")) for row in rows]
    if not rows or len(set(keys)) != len(rows) or any(not all(key) for key in keys):
        raise ValueError("input shard must contain unique nonempty sample/condition/track keys")
    if len({row["condition"] for row in rows}) != 1 or len({row["track"] for row in rows}) != 1:
        raise ValueError("each input shard must contain exactly one condition and one track")
    if any(row.get("answer_format") not in {"MCQ_OPTION", "NUMERIC_SHORT_ANSWER"} for row in rows):
        raise ValueError("unknown answer_format")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock_handle = args.output.with_suffix(args.output.suffix + ".lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another process owns output shard {args.output}") from exc

    manifest_path = args.output.with_suffix(".manifest.json")
    model_path = args.model.resolve()
    manifest = {
        "schema": "phase2-work5-answer-extraction-run-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "input_records": len(rows),
        "condition": rows[0]["condition"],
        "track": rows[0]["track"],
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": sha256(args.prompt),
        "model_path": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "answer_gold_sent_to_model": False,
        "image_sent_to_model": False,
        "condition_or_sample_uid_sent_to_model": False,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latest: dict[tuple[str, str, str], dict] = {}
    if args.output.exists():
        for record in read_jsonl(args.output):
            key = (record.get("sample_uid"), record.get("condition"), record.get("track"))
            if record.get("request_status") == "OK":
                latest[key] = record
    pending = [row for row, key in zip(rows, keys) if key not in latest]
    print(f"input={len(rows)} resumed={len(latest)} pending={len(pending)}", flush=True)
    if not pending:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_records"] = len(latest)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    import vllm
    import torch
    import transformers
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ANSWERED", "NO_ANSWER", "AMBIGUOUS"]},
            "selected_option": {"anyOf": [{"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]}, {"type": "null"}]},
            "answer_text": {"type": "string"},
            "evidence": {"type": "string"},
            "reason_code": {"type": "string", "enum": ["explicit_final", "semantic_final", "no_answer", "contradictory", "unclear"]},
        },
        "required": ["status", "selected_option", "answer_text", "evidence", "reason_code"],
        "additionalProperties": False,
    }
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=0,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    prompt_text = args.prompt.read_text(encoding="utf-8").strip()
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=False,
    )

    def conversation(row: dict) -> list[dict[str, str]]:
        payload = {
            "answer_format": row["answer_format"],
            "query": row["query"],
            "model_response_as_untrusted_data": row["model_response"],
        }
        return [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            try:
                generations = llm.chat(
                    [conversation(row) for row in batch],
                    sampling_params=sampling,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
                raw_values = [item.outputs[0].text for item in generations]
                errors = [None] * len(batch)
            except Exception as batch_error:
                raw_values, errors = [], []
                for row in batch:
                    try:
                        item = llm.chat(
                            [conversation(row)], sampling_params=sampling, use_tqdm=False,
                            chat_template_kwargs={"enable_thinking": False},
                        )[0]
                        raw_values.append(item.outputs[0].text)
                        errors.append(None)
                    except Exception as item_error:
                        raw_values.append("")
                        errors.append(f"batch={type(batch_error).__name__}; item={type(item_error).__name__}: {item_error}")

            for row, raw, error in zip(batch, raw_values, errors):
                result = {
                    "sample_uid": row["sample_uid"],
                    "condition": row["condition"],
                    "track": row["track"],
                    "model_response_sha256": row["model_response_sha256"],
                    "request_status": "TECHNICAL_ERROR" if error else "OK",
                    "judge_raw_text": raw,
                    "judge_error": error,
                }
                if not error:
                    try:
                        result.update(validate(raw, row))
                    except Exception as parse_error:
                        result["request_status"] = "INVALID_JUDGE_JSON"
                        result["validation_error"] = f"{type(parse_error).__name__}: {parse_error}"
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"completed {min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)

    final_rows = read_jsonl(args.output)
    final_by_key = {}
    for row in final_rows:
        final_by_key[(row.get("sample_uid"), row.get("condition"), row.get("track"))] = row
    statuses = Counter(row.get("request_status") for row in final_by_key.values())
    manifest.update({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "unique_final_records": len(final_by_key),
        "final_request_status_counts": dict(statuses),
        "output_sha256": sha256(args.output),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "unique": len(final_by_key), "statuses": dict(statuses)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
