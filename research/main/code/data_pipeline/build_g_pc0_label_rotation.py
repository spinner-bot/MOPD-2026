"""Build the frozen four-way label-rotation diagnostic from exact G-PC0 v1 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "evaluation"))

from g_pc0 import (
    BASE_MANIFEST_SHA256,
    CHOICES,
    ROTATION_PROFILE,
    ROTATION_ROLE,
    label_rotation_mapping,
    validate_g_pc0_rotation_manifest,
)


ROTATION_SPEC = {
    "version": "g-pc0-label-rotation-v1",
    "base_manifest_sha256": BASE_MANIFEST_SHA256,
    "rotation_order": [0, 1, 2, 3],
    "displayed_to_semantic_rule": "semantic_index=(displayed_index+rotation) mod 4",
    "aggregation": "mean raw single-token log-likelihood by semantic option across four rotations",
    "tie_break": list(CHOICES),
}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def expand_rows(base_samples: list[dict], base_manifest_hash: str) -> list[dict]:
    if base_manifest_hash != BASE_MANIFEST_SHA256:
        raise ValueError("base manifest differs from frozen G-PC0 v1")
    rotation_hash = hashlib.sha256(canonical_json(ROTATION_SPEC)).hexdigest()
    expanded: list[dict] = []
    for base in base_samples:
        semantic_options = {choice: base[f"option_{choice}"] for choice in CHOICES}
        for rotation in range(4):
            mapping = label_rotation_mapping(rotation)
            row = deepcopy(base)
            row_id = f"{base['sample_id']}-lr{rotation}"
            row.update({
                "sample_id": row_id,
                "source_record_id": row_id,
                "manifest_profile": ROTATION_PROFILE,
                "experiment_role": ROTATION_ROLE,
                "base_sample_id": base["sample_id"],
                "base_manifest_sha256": base_manifest_hash,
                "label_rotation": rotation,
                "rotation_spec_hash": rotation_hash,
                "semantic_options": semantic_options,
                "correct_semantic_option_id": base["final_answer"],
                "displayed_label_to_semantic_option": mapping,
            })
            for displayed, semantic in mapping.items():
                row[f"option_{displayed}"] = semantic_options[semantic]
            row["final_answer"] = next(
                displayed for displayed, semantic in mapping.items() if semantic == base["final_answer"]
            )
            expanded.append(row)
    return expanded


def build_bundle(base_manifest: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    base_hash = sha256_file(base_manifest)
    base_samples = load_jsonl(base_manifest)
    rows = expand_rows(base_samples, base_hash)
    errors = validate_g_pc0_rotation_manifest(rows, base_samples, base_hash)
    if errors:
        raise ValueError("rotation manifest invalid: " + "; ".join(errors))
    output_dir.mkdir(parents=True)
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "rotation-spec.json").write_text(
        json.dumps(ROTATION_SPEC, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "manifest_profile": ROTATION_PROFILE,
        "records": len(rows),
        "base_records": len(base_samples),
        "base_manifest_sha256": base_hash,
        "manifest_sha256": sha256_file(manifest),
        "rotation_spec_sha256": hashlib.sha256(canonical_json(ROTATION_SPEC)).hexdigest(),
        "diagnostic_only": True,
        "eligible_for_natural_g0": False,
        "eligible_for_training": False,
        "performance_results_read": False,
        "gpu_used": False,
        "validation_errors": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "sha256.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in (manifest, output_dir / "rotation-spec.json", output_dir / "summary.json")),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.base_manifest, args.output_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
