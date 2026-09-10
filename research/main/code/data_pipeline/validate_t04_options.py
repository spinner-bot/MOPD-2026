"""Validate T04 option proposals before controls and independent T05 review."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def reason(row: dict) -> str | None:
    if row.get("error"): return "runner_error"
    proposal = row.get("proposal")
    if not isinstance(proposal, dict): return "truncated_output" if row.get("truncated") else "invalid_json"
    if proposal.get("decision") == "reject": return "model_reject"
    options = proposal.get("options")
    if proposal.get("decision") != "accept_candidate" or not isinstance(options, dict) or set(options) != set("ABCD"):
        return "invalid_option_schema"
    if not all(isinstance(value, str) and value.strip() for value in options.values()): return "empty_option"
    if len({value.strip().casefold() for value in options.values()}) != 4: return "duplicate_option"
    if proposal.get("correct_option") not in options or proposal.get("unique_answer") is not True: return "non_unique_answer"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--input-glob', required=True); p.add_argument('--accepted', type=Path, required=True); p.add_argument('--rejected', type=Path, required=True); p.add_argument('--summary', type=Path, required=True); a = p.parse_args()
    rows = [json.loads(line) for path in sorted(Path().glob(a.input_glob)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    accepted, rejected = [], []
    for row in rows:
        failure = reason(row)
        if failure: row['t04_rejection_reason'] = failure; rejected.append(row)
        else: accepted.append(row)
    for path, records in ((a.accepted, accepted), (a.rejected, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records), encoding='utf-8')
    summary = {'total':len(rows), 'accepted':len(accepted), 'rejected':len(rejected), 'rejections':dict(Counter(row['t04_rejection_reason'] for row in rejected))}
    a.summary.parent.mkdir(parents=True, exist_ok=True); a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8'); print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__': main()
