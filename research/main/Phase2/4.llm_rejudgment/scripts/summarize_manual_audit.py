#!/usr/bin/env python3
"""Summarize the stratified human audit of model-extracted answer labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False)
    if len(df) != 209 or df.audit_id.nunique() != 209:
        raise ValueError("manual audit must contain 209 unique records")
    if not df.manual_extraction_correct.astype(str).str.lower().eq("true").all():
        raise ValueError("all selected audit records must be manually reviewed")

    df["manual_extraction_correct_bool"] = df.manual_extraction_correct.astype(str).str.lower().eq("true")
    df["manual_correctness_bool"] = df.manual_correctness.astype(str).str.lower().eq("true")
    strata = []
    for (condition, stratum), group in df.groupby(["condition", "audit_stratum"], sort=False):
        statuses = group.llm_status.value_counts().to_dict()
        strata.append({
            "condition": condition,
            "stratum": stratum,
            "population_n": int(group.audit_population_n.iloc[0]),
            "sample_n": int(len(group)),
            "design_weight": float(group.audit_weight.iloc[0]),
            "extraction_errors": int((~group.manual_extraction_correct_bool).sum()),
            "observed_extraction_match_rate": float(group.manual_extraction_correct_bool.mean()),
            "status_counts": {str(key): int(value) for key, value in statuses.items()},
            "manual_correct_n": int(group.manual_correctness_bool.sum()),
            "manual_incorrect_n": int((~group.manual_correctness_bool).sum()),
        })

    nonanswer = df[df.llm_status.isin(["AMBIGUOUS", "NO_ANSWER"])]
    if len(nonanswer) != 9:
        raise ValueError("expected all nine non-answer/ambiguous records to be audited")
    result = {
        "schema": "phase2-stratified-manual-audit-v1",
        "audit_purpose": "Check whether the LLM status/option faithfully represents the original response; correctness against gold is tracked separately.",
        "n_audited": int(len(df)),
        "n_answered_audited": int((df.llm_status == "ANSWERED").sum()),
        "n_nonanswer_census_audited": int(len(nonanswer)),
        "nonanswer_status_counts": {str(k): int(v) for k, v in nonanswer.llm_status.value_counts().items()},
        "observed_extraction_errors": int((~df.manual_extraction_correct_bool).sum()),
        "observed_extraction_match_rate": float(df.manual_extraction_correct_bool.mean()),
        "weighted_extraction_match_rate_point_estimate": float(
            sum(row["population_n"] * row["observed_extraction_match_rate"] for row in strata)
            / sum(row["population_n"] for row in strata)
        ),
        "population_estimate_caution": "The sample is stratified and deliberately oversamples rule/model disagreements; the unweighted 209-row rate is descriptive only. Zero observed errors is not proof of zero population error.",
        "strata": strata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audited": len(df), "answered": result["n_answered_audited"], "nonanswer_census": len(nonanswer), "extraction_errors": result["observed_extraction_errors"], "strata": len(strata)}))


if __name__ == "__main__":
    main()
