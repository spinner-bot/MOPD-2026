#!/usr/bin/env python3
"""Reference-grade analysis of the human review against the three automated scorers.

EVIDENCE TIER: T2 (reference only). The reviewer's competence is not established,
so every output here is a cross-check, never a confirmation. Nothing in this
analysis may revise the main evaluation conclusions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

CONDS = ("S0", "SC", "S1", "S2", "S3")
SRC = (("official", "native_correct"), ("terminal", "terminal_correct"),
       ("extraction", "extraction_correct"))
HUMAN = {"Yes / 是": 1, "No / 否": 0}


def flag(v) -> int:
    return 1 if str(v).strip().lower() in ("true", "1", "yes") else 0


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p via binomial."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--human-rows", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.human_rows.read_text(encoding="utf-8"))
    csv_path = args.data_dir / "sample_level_merged.csv"
    auto = {}
    with csv_path.open(encoding="utf-8-sig") as h:
        for r in csv.DictReader(h):
            auto[(r["sample_uid"], r["condition"])] = r

    joined, missing = [], 0
    for r in rows:
        a = auto.get((r["sample_uid"], r["condition"]))
        if not a:
            missing += 1
            continue
        h = HUMAN.get(r["human_sem"])
        if h is None:            # Unclear / Pending 的行不进入二分类比较
            continue
        joined.append({**r, "human": h,
                       **{name: flag(a[col]) for name, col in SRC},
                       "tokens": int(a["completion_tokens"]) if a["completion_tokens"].isdigit() else None})

    out = {"schema": "work5-human-review-reference-analysis-v1",
           "evidence_tier": "T2-reference-only",
           "rows_total": len(rows), "rows_joined": len(joined),
           "rows_unclear_excluded": len(rows) - missing - len(joined),
           "rows_missing_auto": missing, "n_compared": len(joined)}

    # --- 1. each scorer vs human ---
    per_src = {}
    for name, _ in SRC:
        a = [r["human"] for r in joined]
        b = [r[name] for r in joined]
        both_y = sum(1 for x, y in zip(a, b) if x and y)
        h_y_a_n = sum(1 for x, y in zip(a, b) if x and not y)
        h_n_a_y = sum(1 for x, y in zip(a, b) if not x and y)
        both_n = sum(1 for x, y in zip(a, b) if not x and not y)
        n = len(joined)
        per_src[name] = {
            "agree_pct": 100 * (both_y + both_n) / n,
            "kappa": kappa(a, b),
            "sensitivity": 100 * both_y / (both_y + h_y_a_n) if (both_y + h_y_a_n) else None,
            "specificity": 100 * both_n / (both_n + h_n_a_y) if (both_n + h_n_a_y) else None,
            "precision": 100 * both_y / (both_y + h_n_a_y) if (both_y + h_n_a_y) else None,
            "human_yes_auto_no": h_y_a_n,       # 自动化漏判为错
            "human_no_auto_yes": h_n_a_y,       # 自动化误判为对
            "both_yes": both_y, "both_no": both_n,
        }
    out["scorer_vs_human"] = per_src

    # --- 2. by condition ---
    by_cond = {}
    for c in CONDS:
        sub = [r for r in joined if r["condition"] == c]
        if not sub:
            continue
        d = {"n": len(sub), "human_yes": sum(r["human"] for r in sub)}
        for name, _ in SRC:
            d[name + "_yes"] = sum(r[name] for r in sub)
            d[name + "_kappa"] = kappa([r["human"] for r in sub], [r[name] for r in sub])
        by_cond[c] = d
    out["by_condition"] = by_cond

    # --- 3. human-observed S0 vs S3 (paired) ---
    h = defaultdict(dict)
    for r in joined:
        h[r["sample_uid"]][r["condition"]] = r["human"]
    common = [u for u, v in h.items() if "S0" in v and "S3" in v]
    up = sum(1 for u in common if h[u]["S3"] > h[u]["S0"])
    dn = sum(1 for u in common if h[u]["S0"] > h[u]["S3"])
    out["human_s0_vs_s3"] = {"n_paired": len(common), "up": up, "down": dn,
                             "delta_pp": 100 * (up - dn) / len(common) if common else None,
                             "mcnemar_p": mcnemar_exact(up, dn)}

    # --- 4. by stratum ---
    strat = defaultdict(lambda: {"n": 0, "human_yes": 0})
    for r in joined:
        s = r["stratum"] or "(none)"
        strat[s]["n"] += 1
        strat[s]["human_yes"] += r["human"]
        for name, _ in SRC:
            strat[s][name + "_yes"] = strat[s].get(name + "_yes", 0) + r[name]
    out["by_stratum"] = dict(strat)

    # --- 5. judge-invalid / extraction-unfaithful detail ---
    bad_judge = [r for r in rows if r["human_judge"] in ("No / 否", "Unclear / 不确定")]
    bad_ext = [r for r in rows if r["human_ext"] in ("No / 否", "Unclear / 不确定")]
    def overview(items):
        out2 = []
        for r in items:
            a = auto.get((r["sample_uid"], r["condition"]))
            out2.append({"case": r["case"], "condition": r["condition"], "track": r["track"],
                         "human_ext": r["human_ext"], "human_sem": r["human_sem"],
                         "human_judge": r["human_judge"],
                         "auto": {k: flag(a[c]) for k, c in SRC} if a else None,
                         "note": r["note"][:120]})
        return out2
    out["judge_flagged"] = overview(bad_judge)
    out["extraction_flagged"] = overview(bad_ext)

    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"joined={len(joined)} unclear={out['rows_unclear_excluded']} missing={missing}")


if __name__ == "__main__":
    main()
