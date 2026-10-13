#!/usr/bin/env bash
#SBATCH --job-name=w5-pilot
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/pilot-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/pilot-%j.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL_DIR="/leonardo_scratch/large/userexternal/rqian000/models/OPD-V-Qwen3.5-4B"
MODEL_ID="OPD-V-Qwen3.5-4B"
PYTHON="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/python"
VLLM="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/vllm"
PORT="$((20000 + SLURM_JOB_ID % 30000))"
API_BASE="http://127.0.0.1:${PORT}/v1"
RUN_ROOT="${DATA_ROOT}/pilot/runs/${SLURM_JOB_ID}"

export WORK5_DATA_ROOT="${DATA_ROOT}"
export HF_HOME="${DATA_ROOT}/hf-cache"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/${SLURM_JOB_ID}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/${SLURM_JOB_ID}/torchinductor"
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/outputs" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${FAST_REPO}"
if find "${DATA_ROOT}/pilot/runs" -mindepth 2 -maxdepth 2 -type f -name COMPLETE -print -quit 2>/dev/null | grep -q .; then
  echo "A prior pilot already passed output verification; no duplicate pilot requests will be sent."
  exit 0
fi
VERIFIED_PILOT="$(find "${DATA_ROOT}/pilot/runs" -mindepth 2 -maxdepth 2 -type f -name output-verification.json -print -quit 2>/dev/null || true)"
if [[ -n "${VERIFIED_PILOT}" ]]; then
  if "${PYTHON}" - "${VERIFIED_PILOT}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary.get("pilot") is True
assert summary.get("total_records_per_condition") == 40
assert summary.get("total_generation_outputs") == 200
assert set(summary.get("conditions", {})) == {"S0", "SC", "S1", "S2", "S3"}
assert len(summary.get("tracks", [])) == 6
PY
  then
    printf 'reused_verified_pilot=%s\nend=%s\n' "${VERIFIED_PILOT}" "$(date -Is)" | tee "${RUN_ROOT}/logs/run.txt"
    touch "${RUN_ROOT}/COMPLETE"
    exit 0
  fi
fi
printf 'job_id=%s\nstart=%s\ncommit=%s\ncondition_count=5\n' "${SLURM_JOB_ID}" "$(date -Is)" "$(git rev-parse HEAD)" | tee "${RUN_ROOT}/logs/run.txt"
nvidia-smi | tee "${RUN_ROOT}/logs/gpu.txt"
sha256sum "${MODEL_DIR}/config.json" "${MODEL_DIR}/model.safetensors" | tee "${RUN_ROOT}/logs/model-sha256.txt"
INTEGRITY_SUMMARY="${DATA_ROOT}/source-audit/prepared-integrity-verification.json"
if [[ -s "${INTEGRITY_SUMMARY}" ]]; then
  "${PYTHON}" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["total_rows"] == 32162 and x["five_condition_generation_requests"] == 160810' "${INTEGRITY_SUMMARY}"
  echo "Using previously completed prepared-media integrity gate: ${INTEGRITY_SUMMARY}" | tee "${RUN_ROOT}/logs/prepared-integrity.log"
else
  WORK5_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/verify_prepared_benchmarks.py" \
    2>&1 | tee "${RUN_ROOT}/logs/prepared-integrity.log"
fi
WORK5_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/make_stratified_pilot.py" \
  2>&1 | tee "${RUN_ROOT}/logs/pilot-input-selection.log"

"${VLLM}" serve "${MODEL_DIR}" \
  --served-model-name "${MODEL_ID}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --trust-remote-code \
  --port "${PORT}" >"${RUN_ROOT}/logs/vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 120); do
  if curl --silent --fail "${API_BASE}/models" >"${RUN_ROOT}/logs/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" != 1 ]]; then
  tail -200 "${RUN_ROOT}/logs/vllm.log" || true
  exit 1
fi

for condition in S0 SC S1 S2 S3; do
  for benchmark in vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn; do
    input="${DATA_ROOT}/pilot/inputs/${benchmark}.json"
    [[ -f "${input}" ]] || { echo "Missing pilot input: ${input}"; exit 2; }
    "${PYTHON}" "${FAST_REPO}/reproduction/1. Vision-OPD/code/eval/infer.py" \
      --benchmark "${benchmark}" \
      --benchmark_json "${input}" \
      --out_dir "${RUN_ROOT}/outputs/${condition}" \
      --model_name "${MODEL_ID}-${condition}" \
      --seed 42 \
      --api_base "${API_BASE}" \
      --model_id "${MODEL_ID}" \
      --max_tokens 32768 \
      --max_retries 3 \
      --parallel_workers 8 \
      --prompt_registry "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/prompt_registry-v1.json" \
      --condition "${condition}" \
      2>&1 | tee "${RUN_ROOT}/logs/${condition}-${benchmark}.log"
  done
done

"${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/verify_inference_outputs.py" \
  --data-root "${DATA_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --model-id "${MODEL_ID}" \
  --pilot \
  --prompt-registry "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/prompt_registry-v1.json" \
  2>&1 | tee "${RUN_ROOT}/logs/output-verification.log"

printf 'end=%s\n' "$(date -Is)" | tee -a "${RUN_ROOT}/logs/run.txt"
touch "${RUN_ROOT}/COMPLETE"
