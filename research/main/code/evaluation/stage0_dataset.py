"""Admission checks for a frozen, formal Stage 0 dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from stage0_inputs import STAGE0_CONDITIONS, validate_stage0_sample


REQUIRED_FAMILIES = {"P", "R", "PR"}
REQUIRED_STRATEGIES = {
    "localization",
    "entity_tracking",
    "logical_bridge",
    "evidence_integration",
}
REQUIRED_DEPTHS = {1, 2, 3, 4}


def load_stage0_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {error}") from error
    return records


def validate_formal_stage0_dataset(
    records: list[dict], project_root: str | Path | None = None, minimum_size: int = 400
) -> dict:
    """Validate eligibility for the nine-condition Stage 0 screening.

    Results are structured so a Slurm entry point can archive them and refuse
    to consume a partial or visual-only candidate set.
    """
    errors: list[str] = []
    sample_ids: set[str] = set()
    families: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    depths: Counter[int] = Counter()

    if len(records) < minimum_size:
        errors.append(f"requires at least {minimum_size} samples, found {len(records)}")

    for index, sample in enumerate(records):
        prefix = f"record {index}"
        sample_id = str(sample.get("sample_id", ""))
        if sample_id in sample_ids:
            errors.append(f"{prefix}: duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        if sample.get("source_type") == "visual_only":
            errors.append(f"{prefix}: visual_only records cannot enter formal Stage 0")
        for error in validate_stage0_sample(sample, project_root):
            errors.append(f"{prefix}: {error}")
        family = str(sample.get("task_family", ""))
        strategy = str(sample.get("guidance_strategy", ""))
        depth = sample.get("reasoning_depth")
        families[family] += 1
        strategies[strategy] += 1
        if isinstance(depth, int):
            depths[min(depth, 4)] += 1
        else:
            errors.append(f"{prefix}: reasoning_depth must be an integer")

    missing_families = sorted(REQUIRED_FAMILIES - set(families))
    missing_strategies = sorted(REQUIRED_STRATEGIES - set(strategies))
    missing_depths = sorted(REQUIRED_DEPTHS - set(depths))
    if missing_families:
        errors.append(f"missing task families: {', '.join(missing_families)}")
    if missing_strategies:
        errors.append(f"missing guidance strategies: {', '.join(missing_strategies)}")
    if missing_depths:
        errors.append(f"missing reasoning depths: {', '.join(map(str, missing_depths))}")

    return {
        "eligible": not errors,
        "n_samples": len(records),
        "conditions": list(STAGE0_CONDITIONS),
        "family_counts": dict(families),
        "strategy_counts": dict(strategies),
        "depth_counts": dict(depths),
        "errors": errors,
    }
