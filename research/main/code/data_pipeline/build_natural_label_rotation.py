"""Expand a frozen 1017 natural manifest into all four displayed-label rotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "evaluation"))

from exploratory_teacher_gap import (
    NATURAL_ROTATION_PROFILE,
    NATURAL_ROTATION_SPEC,
    NATURAL_ROTATION_SPEC_HASH,
    expand_natural_rotation_manifest,
    validate_natural_rotation_manifest,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_bundle(base_manifest: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    base_hash = sha256_file(base_manifest)
    base_samples = load_jsonl(base_manifest)
    if len(base_samples) not in {8, 32, 96}:
        raise ValueError("natural rotation base manifest must contain 8, 32, or 96 samples")
    rows = expand_natural_rotation_manifest(base_samples, base_hash)
    errors = validate_natural_rotation_manifest(rows, base_samples, base_hash, len(rows))
    if errors:
        raise ValueError("natural rotation manifest invalid: " + "; ".join(errors))
    output_dir.mkdir(parents=True)
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    spec = output_dir / "rotation-spec.json"
    spec.write_text(
        json.dumps(NATURAL_ROTATION_SPEC, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "manifest_profile": NATURAL_ROTATION_PROFILE,
        "base_records": len(base_samples),
        "rotation_records": len(rows),
        "condition_records_when_run": len(rows) * 5,
        "statistical_units": len(base_samples),
        "base_manifest_sha256": base_hash,
        "manifest_sha256": sha256_file(manifest),
        "rotation_contract_sha256": NATURAL_ROTATION_SPEC_HASH,
        "rotation_spec_file_sha256": sha256_file(spec),
        "validation_errors": [],
        "performance_results_read": False,
        "gpu_used": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "sha256.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in (manifest, spec, summary_path)),
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
