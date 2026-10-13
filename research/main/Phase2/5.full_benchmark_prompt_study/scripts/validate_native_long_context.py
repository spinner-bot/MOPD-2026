#!/usr/bin/env python3
"""Exercise the unmodified Vision-OPD judge prompt on one full-length pilot answer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def terminal_decision(text: str) -> str:
    final_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    return final_line if final_line.casefold() in {"yes", "no"} else "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eval_dir = args.repository / "reproduction" / "1. Vision-OPD" / "code" / "eval"
    sys.path.insert(0, str(eval_dir))
    from judge_qwenlm import PROMPT_TEMPLATE, judge_via_vllm

    request = json.loads(args.input.read_text(encoding="utf-8").splitlines()[0])
    source_rows = json.loads(args.prepared.read_text(encoding="utf-8"))
    source = next(row for row in source_rows if row["sample_uid"] == request["sample_uid"])
    response = request["model_response"]
    question = request["query"].replace("<image>", "")
    prompt = PROMPT_TEMPLATE.format(gt=source["response"], response=response, question=question)
    digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
    result = judge_via_vllm(
        [prompt], str(args.model), 2048, tensor_parallel_size=2, max_model_len=65536
    )
    if len(result) != 1 or not result[0].strip():
        raise SystemExit(f"Native judge returned no response: {result!r}")
    if digest != request["model_response_sha256"]:
        raise SystemExit("Native judge input response does not match the full pilot answer hash")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_uid": request["sample_uid"],
        "condition": request["condition"],
        "track": request["track"],
        "judge_output": result[0],
        "judge_terminal_decision": terminal_decision(result[0]),
        "input_response_chars": len(response),
        "input_response_sha256": digest,
        "prompt_chars": len(prompt),
        "tensor_parallel_size": 2,
        "max_model_len": 65536,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "judge_output": result[0], "response_chars": len(response),
                      "prompt_chars": len(prompt), "sha256_matches": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
