"""Canonical task-index mapping for the full five-condition, six-track campaign."""
from __future__ import annotations

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


def task_id(condition: str, track: str) -> int:
    return CONDITIONS.index(condition) * len(TRACKS) + TRACKS.index(track)


def task_coordinates(array_id: int) -> tuple[str, str]:
    if not 0 <= array_id < len(CONDITIONS) * len(TRACKS):
        raise ValueError(f"array id outside 0..29: {array_id}")
    ci, ti = divmod(array_id, len(TRACKS))
    return CONDITIONS[ci], TRACKS[ti]
