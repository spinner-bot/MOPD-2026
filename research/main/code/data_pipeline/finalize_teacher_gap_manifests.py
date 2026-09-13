"""Freeze human-reviewed 8/32/96 manifests for protocol 1017.

This deterministic step never fills human fields.  It joins completed reviews
to valid construction proposals, removes duplicate-image clusters, balances
answer labels, creates a tokenizer-matched generic control, and fails closed if
the full three-way partition cannot be produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "evaluation"))

from exploratory_teacher_gap import validate_locked_manifest, validate_partition_isolation
from teacher_gap_inputs import privileged_context, validate_generic_length
from build_teacher_gap_human_review_queue import validate_constructed_row
from reasoning_graph_contract import validate_guide_graph_alignment, validate_reasoning_graph


PARTITION_SIZES = {"canary": 8, "dev": 32, "locked": 96}
CHOICES = ("A", "B", "C", "D")
GENERIC_WORDS = (
    "Carefully inspect the visible evidence and compare the relevant positions attributes relationships "
    "and states before checking the conclusion against the image. Verify each necessary observation in "
    "order and avoid relying on outside facts. Recheck plausible alternatives against what is directly "
    "shown and choose only the conclusion supported by the complete visual evidence."
).split()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def require_new_output_dir(path: Path) -> None:
    """Reject reuse before reading reviews or loading the tokenizer."""

    if path.exists():
        raise ValueError(f"output directory already exists; use a new run ID: {path}")


def source_dataset_for_candidate(candidate: dict) -> str:
    """Return explicit provenance; never infer it from a candidate ID."""

    value = str(candidate.get("source_dataset", "")).strip()
    if not value:
        raise ValueError("candidate_missing_source_dataset")
    return value


def duplicate_cluster_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images")
    clusters = data.get("duplicate_clusters")
    if not isinstance(images, list) or not isinstance(clusters, list):
        raise ValueError("invalid duplicate-cluster map")
    parent = list(range(len(images)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for cluster in clusters:
        members = cluster.get("members", [])
        if not isinstance(members, list) or any(not isinstance(value, int) or not 0 <= value < len(images) for value in members):
            raise ValueError("invalid duplicate cluster members")
        for member in members[1:]:
            union(members[0], member)
    grouped = {}
    for index, image in enumerate(images):
        image_path = str(image.get("image_original", ""))
        digest = str(image.get("sha256", ""))
        if not image_path or not digest:
            raise ValueError("cluster-map image lacks path or sha256")
        grouped.setdefault(find(index), []).append(digest)
    cluster_ids = {root: "sha256-cluster-" + min(values)[:20] for root, values in grouped.items()}
    return {str(image["image_original"]): cluster_ids[find(index)] for index, image in enumerate(images)}


def validate_human_review(review: dict) -> list[str]:
    errors = []
    required_true = (
        "human_unique_answer",
        "human_full_image_valid",
        "human_guides_ok",
        "human_options_ok",
    )
    if review.get("human_decision") != "accept":
        errors.append("human_decision_not_accept")
    if review.get("human_knowledge_label") != "image_only":
        errors.append("human_knowledge_label_not_image_only")
    if review.get("human_task_family") not in {"R", "PR"}:
        errors.append("human_task_family_not_R_or_PR")
    for key in required_true:
        if review.get(key) is not True:
            errors.append(f"{key}_not_true")
    for key in ("human_answer_leakage", "human_external_fact_injected"):
        if review.get(key) is not False:
            errors.append(f"{key}_not_false")
    if not str(review.get("human_reviewer", "")).strip():
        errors.append("missing_human_reviewer")
    reviewed_at = str(review.get("human_reviewed_at", "")).strip()
    if not reviewed_at:
        errors.append("missing_human_reviewed_at")
    else:
        try:
            parsed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
                errors.append("human_reviewed_at_requires_timezone")
        except ValueError:
            errors.append("invalid_human_reviewed_at")
    evidence = review.get("human_required_evidence")
    evidence_valid = (
        isinstance(evidence, list)
        and len(evidence) >= 2
        and all(isinstance(value, str) and value.strip() for value in evidence)
        and len({value.strip().casefold() for value in evidence}) == len(evidence)
    )
    if not evidence_valid:
        errors.append("invalid_human_required_evidence")
    depth = review.get("human_reasoning_depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 2:
        errors.append("invalid_human_reasoning_depth")
    graph = review.get("human_reasoning_graph")
    graph_errors, _ = validate_reasoning_graph(
        graph,
        declared_depth=depth,
        require_two_evidence=True,
    )
    errors.extend(f"invalid_human_reasoning_graph:{error}" for error in graph_errors)
    if evidence_valid and isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
        graph_evidence = {
            str(node.get("claim", "")).strip().casefold()
            for node in graph["nodes"]
            if isinstance(node, dict) and node.get("type") == "evidence"
        }
        if graph_evidence != {value.strip().casefold() for value in evidence}:
            errors.append("human_required_evidence_must_match_graph_evidence")

    difficult = review.get("human_difficult_visual_evidence")
    difficult_valid = (
        isinstance(difficult, list)
        and all(isinstance(value, str) and value.strip() for value in difficult)
        and len({value.strip().casefold() for value in difficult}) == len(difficult)
    )
    if not difficult_valid:
        errors.append("invalid_human_difficult_visual_evidence")
    elif evidence_valid and any(
        value.strip().casefold() not in {item.strip().casefold() for item in evidence}
        for value in difficult
    ):
        errors.append("human_difficult_evidence_not_in_required_evidence")
    elif review.get("human_task_family") == "PR" and not difficult:
        errors.append("human_PR_requires_difficult_visual_evidence")
    elif review.get("human_task_family") == "R" and difficult:
        errors.append("human_R_must_not_have_difficult_visual_evidence")
    return errors


def generic_context_for(samples: list[dict], tokenizer) -> str:
    target_lengths = sorted(len(tokenizer.encode(privileged_context(row, "T_Q"), add_special_tokens=False)) for row in samples)
    middle = len(target_lengths) // 2
    target = target_lengths[middle] if len(target_lengths) % 2 else (target_lengths[middle - 1] + target_lengths[middle]) / 2
    words = GENERIC_WORDS * 12
    candidates = [" ".join(words[:count]) for count in range(5, len(words) + 1)]
    return min(candidates, key=lambda text: abs(len(tokenizer.encode(text, add_special_tokens=False)) - target))


def arrange_options(proposal: dict, correct_label: str, candidate_id: str) -> dict[str, str]:
    correct = str(proposal["correct_answer_text"]).strip()
    def render(value) -> str:
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value).strip()
    distractors = sorted(
        [render(value) for value in proposal["distractors"]],
        key=lambda value: stable_hash(f"{candidate_id}\0{value}"),
    )
    values = iter(distractors)
    return {choice: correct if choice == correct_label else next(values) for choice in CHOICES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constructed", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--duplicate-clusters", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    try:
        require_new_output_dir(args.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    from transformers import AutoTokenizer

    constructed_rows = load_jsonl(args.constructed)
    constructed_ids = [str(row.get("candidate_id", "")).strip() for row in constructed_rows]
    if any(not value for value in constructed_ids) or len(set(constructed_ids)) != len(constructed_ids):
        raise SystemExit("constructed candidate_id values must be non-empty and unique")
    constructed = dict(zip(constructed_ids, constructed_rows))
    reviews = load_jsonl(args.human_review)
    if len({row.get("candidate_id") for row in reviews}) != len(reviews):
        raise SystemExit("human review candidate_id values must be unique")
    cluster_by_path = duplicate_cluster_map(args.duplicate_clusters)
    eligible, rejected = [], []
    for review in reviews:
        candidate_id = review.get("candidate_id")
        construction = constructed.get(candidate_id)
        reasons = validate_human_review(review)
        if construction is None:
            reasons.append("missing_construction_record")
        else:
            candidate = construction.get("candidate", {})
            proposal = construction.get("proposal", {})
            reasons.extend(validate_constructed_row(construction))
            try:
                source_dataset_for_candidate(candidate)
            except ValueError as error:
                reasons.append(str(error))
            reasons.extend(
                f"guide_graph:{error}"
                for error in validate_guide_graph_alignment(
                    proposal.get("guides"), review.get("human_reasoning_graph")
                )
            )
            image_path = str(candidate.get("image_original", ""))
            if image_path not in cluster_by_path:
                reasons.append("image_missing_from_cluster_map")
        if reasons:
            rejected.append({"candidate_id": candidate_id, "rejection_reasons": sorted(set(reasons))})
            continue
        eligible.append({"review": review, "construction": construction, "cluster_id": cluster_by_path[image_path]})

    # Keep one question per known exact/near-duplicate cluster for this pilot.
    unique_clusters = {}
    for row in sorted(eligible, key=lambda item: stable_hash(f"{args.seed}\0{item['construction']['candidate_id']}")):
        unique_clusters.setdefault(row["cluster_id"], row)
    ordered = list(unique_clusters.values())
    required = sum(PARTITION_SIZES.values())
    if len(ordered) < required:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output_dir / "finalization-rejects.jsonl", rejected)
        summary = {
            "human_reviews": len(reviews),
            "eligible_before_cluster_dedup": len(eligible),
            "eligible_unique_clusters": len(ordered),
            "required": required,
            "status": "insufficient_human_verified_unique_clusters",
        }
        write_json_exclusive(args.output_dir / "finalization-summary.json", summary)
        raise SystemExit(json.dumps(summary, ensure_ascii=False, sort_keys=True))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    partitions = {}
    offset = 0
    for name, size in PARTITION_SIZES.items():
        chosen = ordered[offset : offset + size]
        offset += size
        rotation = int(stable_hash(f"{args.seed}\0{name}")[:8], 16) % len(CHOICES)
        labels = [CHOICES[(index + rotation) % len(CHOICES)] for index in range(size)]
        rows = []
        for item, correct_label in zip(chosen, labels):
            construction, review = item["construction"], item["review"]
            candidate, proposal = construction["candidate"], construction["proposal"]
            options = arrange_options(proposal, correct_label, construction["candidate_id"])
            image_path = Path(candidate["image_original"])
            rows.append({
                "sample_id": "tg-" + construction["candidate_id"],
                "source_image_id": candidate["source_image_id"],
                "source_cluster_id": item["cluster_id"],
                "source_dataset": source_dataset_for_candidate(candidate),
                "source_record_id": construction["candidate_id"],
                "image_original": str(image_path),
                "image_checksum": sha256_file(image_path),
                "final_question": candidate["final_question_source"],
                **{f"option_{choice}": options[choice] for choice in CHOICES},
                "final_answer": correct_label,
                "guides": proposal["guides"],
                "generic_context": "PENDING_TOKEN_MATCH",
                "knowledge_dependency_label": "image_only",
                "task_family": review["human_task_family"],
                "full_image_reference_valid": True,
                "unique_answer": True,
                "human_verified": True,
                "human_reviewer": review["human_reviewer"],
                "human_reviewed_at": review["human_reviewed_at"],
                "answer_leakage": 0,
                "external_fact_injected": 0,
                "required_evidence": review["human_required_evidence"],
                "reasoning_depth": review["human_reasoning_depth"],
                "minimal_reasoning_graph": review["human_reasoning_graph"],
            })
        generic = generic_context_for(rows, tokenizer)
        for row in rows:
            row["generic_context"] = generic
        length_result = validate_generic_length(
            rows, lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        )
        if not length_result["passes"]:
            raise SystemExit(f"unable to satisfy Generic-T length contract for {name}: {length_result}")
        errors = validate_locked_manifest(rows, expected_n=size)
        if errors:
            raise SystemExit(f"{name} manifest invalid: {'; '.join(errors)}")
        partitions[name] = rows

    isolation_errors = validate_partition_isolation(partitions)
    if isolation_errors:
        raise SystemExit("partition isolation failed: " + "; ".join(isolation_errors))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in partitions.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        hashes[path.name] = sha256_file(path)
    write_jsonl(args.output_dir / "finalization-rejects.jsonl", rejected)
    summary = {
        "seed": args.seed,
        "human_reviews": len(reviews),
        "eligible_before_cluster_dedup": len(eligible),
        "eligible_unique_clusters": len(ordered),
        "partition_sizes": PARTITION_SIZES,
        "answer_labels": {name: dict(Counter(row["final_answer"] for row in rows)) for name, rows in partitions.items()},
        "hashes": hashes,
        "status": "frozen",
    }
    write_json_exclusive(args.output_dir / "finalization-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
