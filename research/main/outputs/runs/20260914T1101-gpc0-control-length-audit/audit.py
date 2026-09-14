import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--tokenizer', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f'refusing to overwrite {args.output}')
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'code' / 'evaluation'))
from teacher_gap_inputs import privileged_context
from transformers import AutoTokenizer
rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line]
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
lengths = {}
for condition in ('T_Q', 'T_QA', 'T_generic'):
    values = [len(tokenizer.encode(privileged_context(row, condition), add_special_tokens=False)) for row in rows]
    lengths[condition] = {
        'min': min(values), 'median': statistics.median(values),
        'mean': sum(values) / len(values), 'max': max(values),
    }
summary = {
    'run_id': '20260914T1101-gpc0-control-length-audit',
    'status': 'complete',
    'analysis_type': 'input_only_control_length_audit',
    'records': len(rows),
    'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'token_lengths': lengths,
    'tqa_vs_generic_median_ratio': lengths['T_QA']['median'] / lengths['T_generic']['median'],
    'performance_results_read': False,
    'gpu_used': False,
}
args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(summary, sort_keys=True))
