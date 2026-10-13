#!/usr/bin/env python3
"""Select a deterministic small pilot spanning tracks, categories, and edge formats."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "WORK5_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data",
)).resolve()
PREPARED = ROOT / "prepared"
PILOT = ROOT / "pilot" / "inputs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(track: str, rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["sample_uid"])
    chosen: dict[str, dict] = {}

    def add_first(predicate):
        item = next((row for row in ordered if predicate(row)), None)
        if item is None:
            raise ValueError(f"{track}: no sample matched pilot stratum")
        chosen[item["sample_uid"]] = item

    if track == "vstar":
        for category in sorted({row.get("category") for row in rows}):
            add_first(lambda row, category=category: row.get("category") == category)
    elif track == "zoombench":
        add_first(lambda row: str(row.get("question_type") or "").casefold() == "mcq")
        add_first(lambda row: not str(row.get("question_type") or "").strip())
    elif track.startswith("hrbench-"):
        for category in ("single", "cross"):
            for cycle in range(4):
                add_first(lambda row, category=category, cycle=cycle:
                          row.get("category") == category and row.get("cycle_category") == cycle)
    elif track in {"mme-realworld", "mme-realworld-cn"}:
        for category in sorted({row.get("category") for row in rows}):
            add_first(lambda row, category=category: row.get("category") == category)
        add_first(lambda row: row.get("option_count") == 6)
    else:
        raise ValueError(f"Unsupported pilot track: {track}")
    return list(chosen.values())


def main() -> None:
    PILOT.mkdir(parents=True, exist_ok=True)
    manifest = {"selection": "lexicographically smallest stable UID per predefined stratum", "tracks": {}}
    for directory in sorted(PREPARED.iterdir()):
        if not directory.is_dir():
            continue
        source = directory / "benchmark.json"
        if not source.exists():
            raise FileNotFoundError(f"Prepared input missing: {source}")
        rows = json.loads(source.read_text(encoding="utf-8"))
        selected = choose(directory.name, rows)
        target = PILOT / f"{directory.name}.json"
        target.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["tracks"][directory.name] = {
            "source_rows": len(rows),
            "pilot_rows": len(selected),
            "selected_uids": [row["sample_uid"] for row in selected],
            "input_sha256": digest(target),
        }
        print(f"{directory.name}: {len(selected)} pilot rows / {len(rows)} prepared rows", flush=True)
    manifest["total_pilot_rows"] = sum(x["pilot_rows"] for x in manifest["tracks"].values())
    manifest["five_condition_requests"] = 5 * manifest["total_pilot_rows"]
    target = ROOT / "pilot" / "pilot-manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {target}; {manifest['five_condition_requests']} planned requests", flush=True)


if __name__ == "__main__":
    main()
