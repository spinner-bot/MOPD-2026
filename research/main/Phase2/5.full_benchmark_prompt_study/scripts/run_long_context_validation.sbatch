#!/usr/bin/env bash
# Validate the dual-A100 65,536-context answer extractor on the longest real pilot response.
#SBATCH --job-name=w5-longctx-test
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/longctx-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/longctx-%j.err

set -euo pipefail
FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
PILOT="${DATA_ROOT}/pilot/runs/60398038/outputs"
PREPARED="${DATA_ROOT}/prepared/mme-realworld/benchmark.json"
PROMPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/answer_extraction_v1.txt"
TEST_ROOT="${DATA_ROOT}/validation/long-context-extraction-v1"
mkdir -p "${TEST_ROOT}" "${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm" "${DATA_ROOT}/logs"
export HF_HOME="${DATA_ROOT}/hf-cache"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/${SLURM_JOB_ID}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/${SLURM_JOB_ID}/torchinductor"
export TMPDIR="/tmp" PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source /etc/profile
module load cuda/12.6
cd "${FAST_REPO}"

"${PYTHON}" - "${PILOT}" "${PREPARED}" "${TEST_ROOT}/input.jsonl" <<'PY'
import glob, json, re, sys
from pathlib import Path
pilot, prepared, out = map(Path, sys.argv[1:])
source = json.loads(prepared.read_text(encoding="utf-8"))
by_uid = {row["sample_uid"]: row for row in source}
best = None
for path in glob.glob(str(pilot / "*" / "mme-realworld" / "*.jsonl")):
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        response = str(record.get("model_answer") or "")
        usage = record.get("api_usage") or {}
        token_count = int(usage.get("completion_tokens") or 0)
        uid = str(record.get("sample_uid") or "")
        source_uid = uid.split(":sample_uid:", 1)[-1]
        if source_uid not in by_uid:
            continue
        candidate = (token_count, len(response), path, record, by_uid[source_uid])
        if best is None or candidate[:2] > best[:2]:
            best = candidate
if best is None or best[0] < 30000:
    raise SystemExit("No >=30k-token pilot response matched prepared benchmark data")
tokens, chars, path, record, row = best
query = str(row.get("query") or "")
labels = [a or b for a, b in re.findall(r"(?m)^\s*(?:\(([A-F])\)|([A-F])[.)])\s+", query)]
if len(labels) < 2:
    raise SystemExit("Long-context validation example is not a parsable MCQ")
request = {
    "sample_uid": row["sample_uid"], "condition": record.get("prompt_condition") or Path(path).parts[-3],
    "track": "mme-realworld", "answer_format": "MCQ_OPTION", "query": query,
    "available_option_labels_in_order": labels, "model_response": str(record["model_answer"]),
    "model_response_sha256": __import__("hashlib").sha256(str(record["model_answer"]).encode()).hexdigest(),
}
out.write_text(json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"pilot_output": str(path), "uid": row["sample_uid"], "completion_tokens": tokens,
                  "response_chars": chars, "query_chars": len(query), "condition": request["condition"]}, ensure_ascii=False))
PY

nvidia-smi
"${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/run_answer_extraction.py" \
  --input "${TEST_ROOT}/input.jsonl" \
  --prompt "${PROMPT}" \
  --model "${MODEL}" \
  --output "${TEST_ROOT}/output.jsonl" \
  --batch-size 1 \
  --max-model-len 65536 \
  --tensor-parallel-size 2 \
  --max-tokens 384 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee "${TEST_ROOT}/run-${SLURM_JOB_ID}.log"

"${PYTHON}" - "${TEST_ROOT}/input.jsonl" "${TEST_ROOT}/output.jsonl" <<'PY'
import json, sys
source = json.loads(open(sys.argv[1], encoding="utf-8").readline())
rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
if len(rows) != 1 or rows[0].get("request_status") != "OK":
    raise SystemExit(f"Long-context request did not complete cleanly: {[(r.get('request_status'), r.get('judge_error')) for r in rows]}")
if rows[0].get("model_response_sha256") != source.get("model_response_sha256"):
    raise SystemExit("Validation output is not linked to the exact full pilot response")
print("PASS: full long response accepted; request_status=OK; response SHA-256 matches")
PY
