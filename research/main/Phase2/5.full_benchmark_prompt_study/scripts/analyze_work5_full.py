#!/usr/bin/env python3
"""Compute every number the Work5 evaluation report needs, into one JSON.

Covers: per-source accuracy, pairwise Cohen's kappa, 2x2 disagreement tables,
per-benchmark / per-condition matrices, category and question-type strata,
extraction-status composition, length statistics, redundancy kappa, and the
cluster structure. Pure CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

TRACKS = ("vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn")
CONDS = ("S0", "SC", "S1", "S2", "S3")
SOURCES = (("official", "native_correct"), ("terminal", "terminal_correct"),
           ("extraction", "extraction_correct"))


def b(v) -> int:
    return 1 if str(v).strip().lower() in ("true", "1", "yes") else 0


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


def kappa(a: list[int], z: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    pa = sum(a) / n
    pz = sum(z) / n
    po = sum(1 for x, y in zip(a, z) if x == y) / n
    pe = pa * pz + (1 - pa) * (1 - pz)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def paired(rows, col, track=None):
    sel = [r for r in rows if track is None or r["track"] == track]
    a = {r["sample_uid"]: b(r[col]) for r in sel if r["condition"] == "S0"}
    z = {r["sample_uid"]: b(r[col]) for r in sel if r["condition"] == "S3"}
    keys = set(a) & set(z)
    n = len(keys)
    up = sum(1 for k in keys if z[k] > a[k])
    dn = sum(1 for k in keys if a[k] > z[k])
    d = (up - dn) / n * 100
    se = math.sqrt(up + dn) / n * 100
    return {"n": n, "delta_pp": d, "ci_low": d - 1.959964 * se, "ci_high": d + 1.959964 * se,
            "up": up, "down": dn, "same": n - up - dn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader((args.data_dir / "sample_level_merged.csv").open(encoding="utf-8-sig")))
    out: dict = {"n_rows": len(rows), "n_unique": len({(r["track"], r["condition"], r["sample_uid"]) for r in rows})}

    # --- 2. discriminator comparison ---
    acc, kappas, cross = {}, {}, {}
    for name, col in SOURCES:
        acc[name] = {}
        for c in CONDS:
            g = [r for r in rows if r["condition"] == c]
            k = sum(b(r[col]) for r in g)
            lo, hi = wilson(k, len(g))
            acc[name][c] = {"correct": k, "n": len(g), "acc": 100 * k / len(g), "lo": lo, "hi": hi}
    for i in range(len(SOURCES)):
        for j in range(i + 1, len(SOURCES)):
            a, b_ = SOURCES[i], SOURCES[j]
            key = f"{a[0]}_vs_{b_[0]}"
            kappas[key] = kappa([b(r[a[1]]) for r in rows], [b(r[b_[1]]) for r in rows])
            ca = [b(r[a[1]]) for r in rows]
            cb = [b(r[b_[1]]) for r in rows]
            cross[key] = {"both": sum(1 for x, y in zip(ca, cb) if x and y),
                          "a_only": sum(1 for x, y in zip(ca, cb) if x and not y),
                          "b_only": sum(1 for x, y in zip(ca, cb) if not x and y),
                          "neither": sum(1 for x, y in zip(ca, cb) if not x and not y)}
    out["discriminators"] = {"accuracy": acc, "kappa": kappas, "cross": cross}
    out["discriminator_by_track"] = {
        t: {name: {"official": None} for name, _ in SOURCES} for t in TRACKS
    }
    for t in TRACKS:
        g = [r for r in rows if r["track"] == t]
        for name, col in SOURCES:
            k = sum(b(r[col]) for r in g)
            out["discriminator_by_track"][t][name] = {"acc": 100 * k / len(g), "n": len(g)}

    # --- 4. overall matrix ---
    matrix = {}
    for t in TRACKS:
        for c in CONDS:
            g = [r for r in rows if r["track"] == t and r["condition"] == c]
            matrix[f"{t}|{c}"] = {
                **{name: 100 * sum(b(r[col]) for r in g) / len(g) for name, col in SOURCES},
                "n": len(g)}
    out["matrix"] = matrix

    # --- 5. paired effects per track ---
    out["effects"] = {name: {t: paired(rows, col, t) for t in TRACKS} for name, col in SOURCES}
    out["effects"]["terminal"]["ALL"] = paired(rows, "terminal_correct")
    out["effects"]["extraction"]["ALL"] = paired(rows, "extraction_correct")
    out["effects"]["official"]["ALL"] = paired(rows, "native_correct")

    # --- 6. categorical strata (terminal) ---
    for dim in ("category", "question_type", "answer_format"):
        agg = defaultdict(lambda: {"n": 0, "o": 0, "t": 0})
        for r in rows:
            k = r[dim] or "(missing)"
            agg[k]["n"] += 1
            agg[k]["o"] += b(r["native_correct"])
            agg[k]["t"] += b(r["terminal_correct"])
        out[f"strata_{dim}"] = {
            k: {"n": v["n"], "official": 100 * v["o"] / v["n"], "terminal": 100 * v["t"] / v["n"]}
            for k, v in agg.items() if v["n"] >= 200}

    # --- 8. extraction status composition ---
    comp = defaultdict(lambda: defaultdict(int))
    for r in rows:
        comp[r["condition"]][r["extraction_status"]] += 1
    out["extraction_status"] = {c: dict(v) for c, v in comp.items()}

    # --- 7. length statistics ---
    lens = {}
    for c in CONDS:
        toks = [int(r["completion_tokens"]) for r in rows if r["condition"] == c and r["completion_tokens"].isdigit()]
        toks.sort()
        n = len(toks)
        lens[c] = {"n": n, "mean": sum(toks) / n, "median": toks[n // 2],
                   "p95": toks[int(n * 0.95)], "max": toks[-1]}
    out["length"] = lens

    # --- 9. redundancy kappa between conditions ---
    wide = defaultdict(dict)
    for r in rows:
        wide[r["sample_uid"]][r["condition"]] = b(r["terminal_correct"])
    uids = [u for u, v in wide.items() if len(v) == 5]
    red = {}
    for i in range(len(CONDS)):
        for j in range(i + 1, len(CONDS)):
            a, z = CONDS[i], CONDS[j]
            red[f"{a}_vs_{z}"] = kappa([wide[u][a] for u in uids], [wide[u][z] for u in uids])
    out["redundancy_kappa"] = red

    # --- 11. cluster structure ---
    clusters = defaultdict(int)
    for r in rows:
        if r["condition"] == "S0":
            clusters[r["cluster_uid"]] += 1
    out["cluster"] = {"n_items": sum(1 for r in rows if r["condition"] == "S0"),
                      "n_clusters": len(clusters),
                      "max_size": max(clusters.values()) if clusters else 0,
                      "size_hist": dict(Counter(clusters.values()))}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
