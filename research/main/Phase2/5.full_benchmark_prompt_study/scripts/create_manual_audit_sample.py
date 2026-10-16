#!/usr/bin/env python3
"""Select deterministic, paired five-condition samples for manual score audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from full_campaign_layout import CONDITIONS, TRACKS


QUOTAS = (
    ("native_extraction_disagreement", 3),
    ("native_terminal_disagreement", 2),
    ("non_answer_extraction", 2),
    ("long_tail_output", 1),
    ("three_score_agreement", 2),
)


def _rank(seed: int, track: str, uid: str) -> str:
    return hashlib.sha256(f"{seed}|{track}|{uid}".encode("utf-8")).hexdigest()


def load_candidates(path: Path) -> tuple[dict[str, dict[str, list[dict[str, str]]]], str]:
    digest = hashlib.sha256()
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with path.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["track"]][row["sample_uid"]].append(row)
    return groups, digest.hexdigest()


def _flags(rows: list[dict[str, str]]) -> set[str]:
    by_condition = {row["condition"]: row for row in rows}
    if set(by_condition) != set(CONDITIONS) or len(rows) != len(CONDITIONS):
        raise ValueError("Every sample must have exactly one row for each of the five conditions")
    flags = set()
    if any(row["native_correct"] != row["extraction_correct"] for row in rows):
        flags.add("native_extraction_disagreement")
    if any(row["native_correct"] != row["terminal_correct"] for row in rows):
        flags.add("native_terminal_disagreement")
    if any(row["extraction_status"].upper() in {"AMBIGUOUS", "NO_ANSWER"} for row in rows):
        flags.add("non_answer_extraction")
    if all(row["native_correct"] == row["terminal_correct"] == row["extraction_correct"]
           for row in rows):
        flags.add("three_score_agreement")
    return flags


def select_audit_sample(groups: dict[str, dict[str, list[dict[str, str]]]], seed: int,
                        per_track: int = 10) -> dict[str, Any]:
    if per_track < sum(quota for _, quota in QUOTAS):
        raise ValueError(f"per_track must be at least {sum(quota for _, quota in QUOTAS)}")
    if set(groups) != set(TRACKS):
        raise ValueError("Input must contain exactly the six frozen Work5 tracks")

    selected_tracks = {}
    for track in TRACKS:
        track_groups = groups[track]
        if not track_groups:
            raise ValueError(f"No sample rows for track {track}")
        flags_by_uid = {uid: _flags(rows) for uid, rows in track_groups.items()}
        max_tokens = {
            uid: max(int(row["completion_tokens"] or 0) for row in rows)
            for uid, rows in track_groups.items()
        }
        ranked_uids = sorted(track_groups, key=lambda uid: (_rank(seed, track, uid), uid))
        sorted_lengths = sorted(max_tokens.values())
        p95_index = min(len(sorted_lengths) - 1, math.ceil(0.95 * len(sorted_lengths)) - 1)
        p95 = sorted_lengths[p95_index]
        flags_by_uid = {
            uid: flags | ({"long_tail_output"} if max_tokens[uid] >= p95 else set())
            for uid, flags in flags_by_uid.items()
        }

        chosen: list[dict[str, Any]] = []
        chosen_uids: set[str] = set()
        availability = {}
        candidate_uids_by_stratum = {}
        for stratum, quota in QUOTAS:
            candidates = [uid for uid in ranked_uids if stratum in flags_by_uid[uid]]
            availability[stratum] = len(candidates)
            candidate_uids_by_stratum[stratum] = set(candidates)
            covered = sum(uid in chosen_uids for uid in candidates)
            for uid in candidates:
                if covered >= quota:
                    break
                if uid in chosen_uids:
                    continue
                chosen_uids.add(uid)
                chosen.append({"sample_uid": uid, "primary_stratum": stratum,
                               "all_flags": sorted(flags_by_uid[uid]), "max_completion_tokens": max_tokens[uid]})
                covered += 1

        for uid in ranked_uids:
            if len(chosen) >= per_track:
                break
            if uid not in chosen_uids:
                chosen_uids.add(uid)
                chosen.append({"sample_uid": uid, "primary_stratum": "deterministic_random_remainder",
                               "all_flags": sorted(flags_by_uid[uid]), "max_completion_tokens": max_tokens[uid]})
        if len(chosen) != per_track:
            raise ValueError(f"{track}: requested {per_track} samples but only {len(chosen)} unique samples available")
        selected_by_stratum = {
            stratum: sum(uid in chosen_uids for uid in candidates)
            for stratum, candidates in candidate_uids_by_stratum.items()
        }
        selected_tracks[track] = {
            "population_n": len(track_groups), "long_tail_p95_max_completion_tokens": p95,
            "strata_available": availability, "strata_selected": selected_by_stratum,
            "selected": chosen,
        }

    return {
        "schema": "phase2-work5-manual-audit-sample-v2",
        "purpose": "Qualitative, paired audit of score parsing and disagreement; not a prevalence estimator.",
        "selection_policy": "A selected sample may satisfy multiple strata; selected counts reflect all matched strata, not only the primary selection reason.",
        "seed": seed, "conditions": list(CONDITIONS), "tracks": list(TRACKS),
        "samples_per_track": per_track, "paired_condition_rows": per_track * len(TRACKS) * len(CONDITIONS),
        "quota_priority": [{"stratum": name, "quota_per_track": quota} for name, quota in QUOTAS],
        "tracks_selection": selected_tracks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--per-track", type=int, default=10)
    args = parser.parse_args()
    groups, source_sha256 = load_candidates(args.merged_csv)
    result = select_audit_sample(groups, args.seed, args.per_track)
    result["source"] = {"path": str(args.merged_csv), "sha256": source_sha256}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = result["samples_per_track"] * len(TRACKS)
    print(f"PASS: selected {total} unique sample IDs ({result['paired_condition_rows']} paired condition rows)")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
