"""Audit a candidate source before it is admitted to the formal Stage 0 set.

The audit deliberately does not manufacture guiding QA chains.  A source can
only enter the nine-condition Stage 0 runner after it supplies every field
needed by ``stage0_inputs.validate_stage0_sample``.  This prevents a
Vision-OPD perception-only record from being presented as evidence for the
text-privilege hypothesis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_STAGE0_FIELDS = (
    "sample_id",
    "source_image_id",
    "final_question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "final_answer",
    "image_original",
    "crop_primary",
    "random_questions",
    "random_questions_frozen",
)


def audit_jsonl(path: Path) -> dict:
    """Return field coverage and task annotations for a JSONL candidate set."""
    field_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    guide_counts: Counter[int] = Counter()
    total = 0

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {error}") from error
            total += 1
            for field in REQUIRED_STAGE0_FIELDS:
                if sample.get(field) not in (None, "", [], False):
                    field_counts[field] += 1
            family_counts[str(sample.get("task_family", "MISSING"))] += 1
            strategy_counts[str(sample.get("guidance_strategy", "MISSING"))] += 1
            depth_counts[str(sample.get("reasoning_depth", "MISSING"))] += 1
            guides = sum(
                1
                for index in range(1, 5)
                if sample.get(f"guide_{index}_question") and sample.get(f"guide_{index}_answer")
            )
            guide_counts[guides] += 1

    missing = {
        field: total - field_counts[field]
        for field in REQUIRED_STAGE0_FIELDS
        if field_counts[field] != total
    }
    return {
        "source": str(path),
        "n_records": total,
        "required_field_coverage": dict(field_counts),
        "missing_required_field_records": missing,
        "task_family": dict(family_counts),
        "guidance_strategy": dict(strategy_counts),
        "reasoning_depth": dict(depth_counts),
        "complete_guiding_qa_pairs_per_record": dict(guide_counts),
        "formal_stage0_eligible": total > 0 and not missing and guide_counts.get(0, 0) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Candidate JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Audit JSON output")
    args = parser.parse_args()

    result = audit_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
