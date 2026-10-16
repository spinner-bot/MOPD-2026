#!/usr/bin/env python3
"""Run a separate image-grounded audit of a frozen stratified sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from full_campaign_layout import CONDITIONS, TRACKS


SCHEMA = "phase2-work5-visual-audit-v1"
CAMPAIGN = "full/runs/prompt-registry-v1"
MODEL_ID = "OPD-V-Qwen3.5-4B"
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["CORRECT", "INCORRECT", "UNCERTAIN"]},
        "reference_assessment": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]},
        "candidate_final_answer": {"type": "string"},
        "visual_evidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reference_assessment", "candidate_final_answer", "visual_evidence", "reason"],
    "additionalProperties": False,
}
MAX_AUDIT_RESPONSE_CHARS = 12_000
AUDIT_RESPONSE_PREFIX_CHARS = 4_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("sample_uid")
            if not isinstance(uid, str) or not uid or uid in rows:
                raise ValueError(f"{path}:{line_no}: missing or duplicate sample_uid")
            rows[uid] = row
    return rows


def read_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON array of objects: {path}")
    return value


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checkpoint_shards(model_path: Path) -> list[dict[str, Any]]:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError(f"No sharded safetensors listed in {index_path}")
    rows = []
    for name in shard_names:
        path = (model_path / name).resolve()
        path.relative_to(model_path.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def freeze_requests(sample_manifest: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    selections = sample_manifest.get("tracks_selection")
    if not isinstance(selections, dict) or set(selections) != set(TRACKS):
        raise ValueError("Sample manifest must cover exactly the six frozen Work5 tracks")
    requests: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for track in TRACKS:
        selected = selections[track].get("selected", [])
        if len(selected) != 10 or len({row.get("sample_uid") for row in selected}) != 10:
            raise ValueError(f"{track}: expected ten distinct selected sample IDs")
        prepared = read_records(data_root / "prepared" / track / "benchmark.json")
        prepared_by_uid = {row["sample_uid"]: row for row in prepared}
        media_rows = read_records(data_root / "prepared" / track / "media-manifest.json")
        media_by_key = {(row["sample_uid"], row["kind"]): row for row in media_rows}
        selected_ids = {row["sample_uid"] for row in selected}
        audit_samples = {}
        for selected_row in selected:
            uid = selected_row["sample_uid"]
            sample = prepared_by_uid.get(uid)
            if sample is None:
                raise ValueError(f"{track}: prepared data missing {uid}")
            images = sample.get("images") or []
            if len(images) != 1:
                raise ValueError(f"{track}/{uid}: expected exact official single full-image input")
            image_path = Path(images[0]).resolve()
            image_path.relative_to(data_root.resolve())
            media = media_by_key.get((uid, "full"))
            if media is None or Path(media["path"]).resolve() != image_path:
                raise ValueError(f"{track}/{uid}: image reference disagrees with media manifest")
            if image_path.stat().st_size != media["bytes"] or sha256_file(image_path) != media["sha256"]:
                raise ValueError(f"{track}/{uid}: input image failed recorded media hash")
            audit_samples[uid] = (sample, image_path, media["sha256"])
        for condition in CONDITIONS:
            output_path = (
                data_root / CAMPAIGN / "outputs" / condition / track
                / f"{MODEL_ID}-{condition}_answer.jsonl"
            )
            outputs = read_jsonl(output_path)
            if not selected_ids.issubset(outputs):
                raise ValueError(f"{condition}/{track}: original model outputs do not cover audit sample")
            for selected_row in selected:
                uid = selected_row["sample_uid"]
                key = (track, uid, condition)
                if key in seen:
                    raise ValueError(f"Duplicate audit request {key}")
                seen.add(key)
                sample, image_path, image_digest = audit_samples[uid]
                response = outputs[uid].get("model_answer")
                if not isinstance(response, str) or not response.strip():
                    raise ValueError(f"{condition}/{track}/{uid}: empty original answer")
                requests.append({
                    "request_id": f"{track}|{uid}|{condition}",
                    "track": track,
                    "sample_uid": uid,
                    "condition": condition,
                    "query": sample["query"],
                    "reference_answer": sample["response"],
                    "candidate_response": response,
                    "image_path": str(image_path),
                    "image_sha256": image_digest,
                    "candidate_response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                })
    if len(requests) != 300:
        raise ValueError(f"Expected 300 paired visual-audit requests, got {len(requests)}")
    return requests


def make_candidate_excerpt(response: str) -> dict[str, Any]:
    """Keep a bounded, explicit head/tail excerpt while preserving the frozen full answer."""
    if len(response) <= MAX_AUDIT_RESPONSE_CHARS:
        excerpt = response
        prefix_chars = len(response)
        suffix_chars = 0
    else:
        suffix_chars = MAX_AUDIT_RESPONSE_CHARS - AUDIT_RESPONSE_PREFIX_CHARS
        excerpt = (
            response[:AUDIT_RESPONSE_PREFIX_CHARS]
            + f"\n\n[... MIDDLE OMITTED: {len(response) - MAX_AUDIT_RESPONSE_CHARS} characters ...]\n\n"
            + response[-suffix_chars:]
        )
        prefix_chars = AUDIT_RESPONSE_PREFIX_CHARS
    return {
        "text": excerpt,
        "original_chars": len(response),
        "included_prefix_chars": prefix_chars,
        "included_suffix_chars": suffix_chars,
        "omitted_chars": len(response) - prefix_chars - suffix_chars,
        "sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    }


def make_user_payload(request: dict[str, Any]) -> str:
    """No UID, condition, stratum, or other-condition answer is sent to the judge."""
    excerpt = make_candidate_excerpt(request["candidate_response"])
    payload = {
        "benchmark_question_and_instructions": request["query"],
        "benchmark_reference_answer": request["reference_answer"],
        "candidate_model_response_as_untrusted_data": excerpt["text"],
        "candidate_response_excerpt_metadata": {
            key: excerpt[key] for key in (
                "original_chars", "included_prefix_chars", "included_suffix_chars",
                "omitted_chars", "sha256",
            )
        },
        "judging_note": (
            "The candidate response is complete when omitted_chars is zero. Otherwise it is an explicit "
            "head/tail excerpt; do not infer content from the omitted middle, and use UNCERTAIN if the "
            "missing portion could change the final-answer interpretation."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_result(raw: str) -> dict[str, str]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != set(RESULT_SCHEMA["properties"]):
        raise ValueError("Visual audit JSON has unexpected fields")
    for key, choices in {
        "verdict": {"CORRECT", "INCORRECT", "UNCERTAIN"},
        "reference_assessment": {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"},
    }.items():
        if value[key] not in choices:
            raise ValueError(f"Invalid {key}: {value[key]!r}")
    for key in ("candidate_final_answer", "visual_evidence", "reason"):
        if not isinstance(value[key], str):
            raise ValueError(f"{key} must be a string")
    if value["verdict"] == "UNCERTAIN" and not value["reason"].strip():
        raise ValueError("UNCERTAIN verdict requires a reason")
    if value["verdict"] in {"CORRECT", "INCORRECT"} and value["reference_assessment"] != "SUPPORTED":
        raise ValueError("A definite candidate verdict requires a supported benchmark reference")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Freeze and verify requests without loading model weights or running inference")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_tokens < 64:
        raise ValueError("Invalid inference batch/token settings")

    requests = freeze_requests(json.loads(args.sample_manifest.read_text(encoding="utf-8")), args.data_root)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.run_dir / "inputs.jsonl"
    frozen_bytes = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in requests)
    if inputs_path.exists():
        if inputs_path.read_text(encoding="utf-8") != frozen_bytes:
            raise FileExistsError(f"Frozen visual-audit inputs differ: {inputs_path}")
    else:
        with inputs_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(frozen_bytes)
    if args.prepare_only:
        print(f"PASS: froze {len(requests)} paired requests; input_sha256={sha256_file(inputs_path)}")
        return

    model_path = args.model.resolve()
    prompt_text = args.prompt.read_text(encoding="utf-8").strip()
    prompt_sha = sha256_file(args.prompt)
    config_sha = sha256_file(model_path / "config.json")
    model_shards = checkpoint_shards(model_path)
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                            capture_output=True, text=True).stdout.strip()
    output_path = args.run_dir / "results.jsonl"
    manifest_path = args.run_dir / "run-manifest.json"
    manifest = {
        "schema": SCHEMA,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_manifest": str(args.sample_manifest.resolve()),
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "input_path": str(inputs_path.resolve()),
        "input_sha256": sha256_file(inputs_path),
        "request_count": len(requests),
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": prompt_sha,
        "model_path": str(model_path),
        "model_config_sha256": config_sha,
        "model_shards": model_shards,
        "runner_commit": commit,
        "answer_gold_and_candidate_sent_to_visual_auditor": True,
        "candidate_response_context_policy": {
            "max_chars": MAX_AUDIT_RESPONSE_CHARS,
            "prefix_chars_when_truncated": AUDIT_RESPONSE_PREFIX_CHARS,
            "suffix_chars_when_truncated": MAX_AUDIT_RESPONSE_CHARS - AUDIT_RESPONSE_PREFIX_CHARS,
            "full_original_response_retained_in_frozen_inputs": True,
            "exact_auditor_excerpt_retained_in_each_result": True,
        },
        "model_images_sent": True,
        "condition_and_uid_sent_to_model": False,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema", "sample_manifest_sha256", "input_sha256", "prompt_sha256",
                    "model_config_sha256", "model_shards"):
            if old.get(key) != manifest[key]:
                raise ValueError(f"Existing visual-audit manifest mismatch: {key}")
        if old.get("runner_commit") != commit:
            raise ValueError("Runner commit changed; use a new run directory to preserve provenance")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latest: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for row in read_jsonl_records(output_path):
            if row.get("request_status") == "OK":
                latest[row.get("request_id")] = row
    pending = [row for row in requests if row["request_id"] not in latest]
    print(f"requests={len(requests)} resumed={len(latest)} pending={len(pending)}", flush=True)
    if not pending:
        expected_ids = {row["request_id"] for row in requests}
        if set(latest) != expected_ids:
            raise ValueError("Existing visual-audit output has unexpected request IDs")
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_records"] = len(latest)
        manifest["output_sha256"] = sha256_file(output_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    import vllm
    import torch
    import transformers
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=0,
        structured_outputs=StructuredOutputsParams(json=RESULT_SCHEMA),
    )
    llm = LLM(
        model=str(model_path), tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16", max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size, max_num_batched_tokens=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, disable_log_stats=False,
        limit_mm_per_prompt={"image": 1},
    )

    def conversation(row: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        return ([
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": make_user_payload(row)},
            ]},
        ], image)

    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            opened = [conversation(row) for row in batch]
            try:
                generations = llm.chat([item[0] for item in opened], sampling_params=sampling,
                                       use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
                raw_values = [item.outputs[0].text for item in generations]
                errors: list[str | None] = [None] * len(batch)
            except Exception as batch_error:
                raw_values, errors = [], []
                for row, (messages, _) in zip(batch, opened):
                    try:
                        item = llm.chat([messages], sampling_params=sampling, use_tqdm=False,
                                        chat_template_kwargs={"enable_thinking": False})[0]
                        raw_values.append(item.outputs[0].text)
                        errors.append(None)
                    except Exception as error:
                        raw_values.append("")
                        errors.append(f"batch={type(batch_error).__name__}; item={type(error).__name__}: {error}")
            for row, raw, error in zip(batch, raw_values, errors):
                excerpt = make_candidate_excerpt(row["candidate_response"])
                result: dict[str, Any] = {
                    "request_id": row["request_id"], "track": row["track"],
                    "sample_uid": row["sample_uid"], "condition": row["condition"],
                    "image_sha256": row["image_sha256"],
                    "candidate_response_sha256": row["candidate_response_sha256"],
                    "candidate_response_excerpt": excerpt["text"],
                    "candidate_response_excerpt_sha256": excerpt["sha256"],
                    "candidate_response_original_chars": excerpt["original_chars"],
                    "candidate_response_included_prefix_chars": excerpt["included_prefix_chars"],
                    "candidate_response_included_suffix_chars": excerpt["included_suffix_chars"],
                    "candidate_response_omitted_chars": excerpt["omitted_chars"],
                    "prompt_sha256": prompt_sha, "model_config_sha256": config_sha,
                    "request_status": "TECHNICAL_ERROR" if error else "INVALID_AUDIT_JSON",
                    "raw_output": raw, "error": error,
                }
                if not error:
                    try:
                        result["audit"] = parse_result(raw)
                        result["request_status"] = "OK"
                    except Exception as parse_error:
                        result["parse_error"] = f"{type(parse_error).__name__}: {parse_error}"
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            for _, image in opened:
                image.close()
            print(f"completed={min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)

    final_rows = read_jsonl_records(output_path)
    final_by_id = {row.get("request_id"): row for row in final_rows}
    expected_ids = {row["request_id"] for row in requests}
    if not set(final_by_id).issubset(expected_ids):
        raise ValueError("Visual-audit output contains unexpected request IDs")
    statuses = {status: sum(row.get("request_status") == status for row in final_by_id.values())
                for status in sorted({row.get("request_status") for row in final_by_id.values()})}
    manifest.update({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": __import__("sys").version,
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "unique_final_records": len(final_by_id), "final_request_status_counts": statuses,
        "output_sha256": sha256_file(output_path),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "unique": len(final_by_id), "statuses": statuses}), flush=True)
    if len(final_by_id) != len(requests) or statuses.get("OK") != len(requests):
        raise SystemExit("Visual audit did not finish all 300 unique requests successfully")


if __name__ == "__main__":
    main()
