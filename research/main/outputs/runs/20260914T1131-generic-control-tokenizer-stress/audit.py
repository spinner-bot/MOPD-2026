"""Stress per-sample generic length matching with the frozen tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT / "research/main/code/data_pipeline"))
sys.path.insert(0, str(REPO_ROOT / "research/main/code/evaluation"))

from finalize_teacher_gap_manifests import generic_candidate_pool, generic_context_for
from teacher_gap_inputs import privileged_context


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if len(args.model_revision) != 40:
        raise SystemExit("model revision must be a full 40-character hash")

    from transformers import AutoTokenizer

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pool = generic_candidate_pool(tokenizer)
    rows = []
    for guide_count in range(1, 5):
        for question_words in (4, 8, 16, 32, 48):
            for answer_words in (1, 4, 12, 24):
                sample = {
                    "guides": [
                        {
                            "question": " ".join(["object"] * question_words),
                            "answer": " ".join(["state"] * answer_words),
                        }
                        for _ in range(guide_count)
                    ]
                }
                for condition in ("T_Q", "T_QA"):
                    generic = generic_context_for([sample], tokenizer, condition, pool)
                    target = len(tokenizer.encode(
                        privileged_context(sample, condition), add_special_tokens=False
                    ))
                    control = len(tokenizer.encode(generic, add_special_tokens=False))
                    rows.append({
                        "guide_count": guide_count,
                        "question_words": question_words,
                        "answer_words": answer_words,
                        "condition": condition,
                        "target_tokens": target,
                        "control_tokens": control,
                        "relative_difference": abs(control - target) / target,
                    })
    artifact_hashes = {
        name: sha256_file(args.model / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
        if (args.model / name).is_file()
    }
    summary = {
        "status": "passed" if all(row["relative_difference"] <= 0.10 for row in rows) else "failed",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_path": str(args.model.resolve()),
        "model_revision": args.model_revision,
        "tokenizer_artifact_sha256": artifact_hashes,
        "algorithm": "neutral-prefix-nearest-per-sample-v1",
        "candidate_pool_size": len(pool),
        "case_count": len(rows),
        "failure_count": sum(row["relative_difference"] > 0.10 for row in rows),
        "maximum_relative_difference": max(row["relative_difference"] for row in rows),
        "worst_cases": sorted(rows, key=lambda row: row["relative_difference"], reverse=True)[:10],
        "elapsed_seconds": time.monotonic() - started,
        "performance_results_read": False,
        "gpu_used": False,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit("generic tokenizer stress failed")


if __name__ == "__main__":
    main()
