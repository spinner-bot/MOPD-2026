"""Normalize Vision-OPD records for visual-privilege controls.

The resulting records are explicitly marked ``visual_only``.  They are useful
for S/V-crop/V-add checks, but cannot be admitted to formal nine-condition
Stage 0 until independently annotated guide QA chains and the required text
controls have been added.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OPTION = re.compile(r"^([A-D])\.\s*(.+)$", flags=re.MULTILINE)


def parse_question(question_with_options: str) -> tuple[str, dict[str, str]]:
    """Split Vision-OPD's answer prompt into question text and four options."""
    matches = list(OPTION.finditer(question_with_options))
    if len(matches) != 4 or {match.group(1) for match in matches} != set("ABCD"):
        raise ValueError("expected exactly four A-D options")
    question = question_with_options[: matches[0].start()].strip()
    options = {match.group(1): match.group(2).strip() for match in matches}
    if not question or any(not options[key] for key in "ABCD"):
        raise ValueError("empty question or option")
    return question, options


def materialized_path(path_in_archive: str, asset_root: Path) -> Path:
    """Archives contain flat file names after extraction into ``asset_root``."""
    return asset_root / Path(path_in_archive).name


def normalize_record(index: int, source: dict, asset_root: Path) -> dict:
    question, options = parse_question(source["extra_info"]["question"])
    original = materialized_path(source["original_images"][0], asset_root)
    crop = materialized_path(source["teacher_images"][0], asset_root)
    if not original.is_file() or not crop.is_file():
        missing = [str(path) for path in (original, crop) if not path.is_file()]
        raise FileNotFoundError(", ".join(missing))
    return {
        "sample_id": f"vision-opd-{index:06d}",
        "source_image_id": Path(source["original_images"][0]).stem,
        "source_dataset": "Vision-OPD-6K",
        "source_record_index": index,
        "source_type": "visual_only",
        "task_family": "P",
        "reasoning_depth": 1,
        "final_question": question,
        "option_A": options["A"],
        "option_B": options["B"],
        "option_C": options["C"],
        "option_D": options["D"],
        "final_answer": source["answer"],
        "image_original": str(original),
        "crop_primary": str(crop),
        "bbox_xyxy": source["bbox"],
        "stage0_admission": "blocked_missing_guidance_chain_and_controls",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            record = normalize_record(index, json.loads(line), args.asset_root)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} visual-only records to {args.output}")


if __name__ == "__main__":
    main()
