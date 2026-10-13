#!/usr/bin/env python3
"""Report unfinished native-score/extraction shards as original 0..29 array IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ("S0", "SC", "S1", "S2", "S3")
TRACKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
TRACK_COUNTS = {
    "vstar": 191,
    "zoombench": 845,
    "hrbench-4k": 800,
    "hrbench-8k": 800,
    "mme-realworld": 23609,
    "mme-realworld-cn": 5917,
}


def missing_native(
    data_root: Path,
    conditions: tuple[str, ...] = CONDITIONS,
    tracks: tuple[str, ...] = TRACKS,
) -> list[int]:
    result = []
    for ci, condition in enumerate(conditions):
        for ti, track in enumerate(tracks):
            shard_id = ci * len(tracks) + ti
            task_root = data_root / "official-scoring" / "vision-opd-native-v1" / condition / track
            model_id = f"OPD-V-Qwen3.5-4B-{condition}"
            score = task_root / "judge" / track / f"{model_id}_answer.jsonl"
            if not (task_root / "COMPLETE").is_file() or not score.is_file() or score.stat().st_size == 0:
                result.append(shard_id)
    return result


def missing_extraction(
    data_root: Path,
    conditions: tuple[str, ...] = CONDITIONS,
    tracks: tuple[str, ...] = TRACKS,
    track_counts: dict[str, int] = TRACK_COUNTS,
) -> list[int]:
    result = []
    for ci, condition in enumerate(conditions):
        for ti, track in enumerate(tracks):
            shard_id = ci * len(tracks) + ti
            path = data_root / "judgment" / "runs" / "answer-extraction-v1" / track / f"{condition}.jsonl"
            latest: dict[str, str] = {}
            try:
                with path.open(encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        uid = record.get("sample_uid")
                        if not isinstance(uid, str) or not uid:
                            raise ValueError(f"{path}:{line_no}: missing sample_uid")
                        if record.get("condition") != condition or record.get("track") != track:
                            raise ValueError(f"{path}:{line_no}: wrong condition/track")
                        latest[uid] = str(record.get("request_status"))
                valid = len(latest) == track_counts[track] and all(status == "OK" for status in latest.values())
            except (OSError, ValueError, json.JSONDecodeError, AttributeError):
                valid = False
            if not valid:
                result.append(shard_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("native", "extraction"), required=True)
    args = parser.parse_args()
    missing = missing_native(args.data_root) if args.stage == "native" else missing_extraction(args.data_root)
    print(",".join(map(str, missing)) if missing else "NONE")


if __name__ == "__main__":
    main()
