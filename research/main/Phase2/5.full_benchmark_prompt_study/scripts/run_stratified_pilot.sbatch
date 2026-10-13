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
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/outputs"
cd "${FAST_REPO}"
printf 'job_id=%s\nstart=%s\ncommit=%s\ncondition_count=5\n' "${SLURM_JOB_ID}" "$(date -Is)" "$(git rev-parse HEAD)" | tee "${RUN_ROOT}/logs/run.txt"
nvidia-smi | tee "${RUN_ROOT}/logs/gpu.txt"
sha256sum "${MODEL_DIR}/config.json" "${MODEL_DIR}/model.safetensors" | tee "${RUN_ROOT}/logs/model-sha256.txt"
WORK5_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/verify_prepared_benchmarks.py" \
  2>&1 | tee "${RUN_ROOT}/logs/prepared-integrity.log"
WORK5_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/make_stratified_pilot.py" \
  2>&1 | tee "${RUN_ROOT}/logs/pilot-input-selection.log"

"${VLLM}" serve "${MODEL_DIR}" \
  --served-model-name "${MODEL_ID}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
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

printf 'end=%s\n' "$(date -Is)" | tee -a "${RUN_ROOT}/logs/run.txt"
touch "${RUN_ROOT}/COMPLETE"
