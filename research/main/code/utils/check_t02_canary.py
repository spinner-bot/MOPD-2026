"""Fail-closed canary gate: inspect parsed T02 canary records before the full run.

Runs on CPU inside the same allocation as the GPU pilot. If the fixed
configuration (thinking disabled, norm1000 bboxes, raised token cap) does not
produce mostly-parseable, mostly-accepted, mostly-untruncated output on a small
sample, the job aborts before burning the remaining GPU hours.
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-glob', required=True)
    p.add_argument('--min-total', type=int, default=64)
    p.add_argument('--min-parse-rate', type=float, default=0.8)
    p.add_argument('--min-accept-rate', type=float, default=0.4)
    p.add_argument('--max-truncation-rate', type=float, default=0.1)
    p.add_argument('--summary', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for f in sorted(glob.glob(a.input_glob)):
        rows += [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
    total = len(rows)
    parsed = sum(1 for r in rows if isinstance(r.get('proposal'), dict))
    accepted = sum(1 for r in rows if isinstance(r.get('proposal'), dict) and r['proposal'].get('decision') == 'accept_candidate')
    truncated = sum(1 for r in rows if r.get('truncated'))
    errors = sum(1 for r in rows if r.get('error'))
    summary = {
        'total': total, 'parsed': parsed, 'accepted': accepted,
        'truncated': truncated, 'runner_errors': errors,
        'parse_rate': parsed / total if total else 0.0,
        'accept_rate': accepted / total if total else 0.0,
        'truncation_rate': truncated / total if total else 0.0,
    }
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    failures = []
    if total < a.min_total: failures.append(f'canary_total={total} < {a.min_total}')
    if summary['parse_rate'] < a.min_parse_rate: failures.append(f"parse_rate={summary['parse_rate']:.3f} < {a.min_parse_rate}")
    if summary['accept_rate'] < a.min_accept_rate: failures.append(f"accept_rate={summary['accept_rate']:.3f} < {a.min_accept_rate}")
    if summary['truncation_rate'] > a.max_truncation_rate: failures.append(f"truncation_rate={summary['truncation_rate']:.3f} > {a.max_truncation_rate}")
    if errors: failures.append(f'runner_errors={errors}')
    if failures:
        raise SystemExit('T02 canary FAILED: ' + '; '.join(failures))
    print('T02 canary PASSED')


if __name__ == '__main__':
    main()
