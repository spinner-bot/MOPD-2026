#!/usr/bin/env bash
# Run as a five-element Slurm array, one independent single-GPU task per prompt condition.
# Output is written directly to the large data area and each benchmark is resumable via infer.py.
#SBATCH --job-name=w5-full
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/full-%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/full-%A_%a.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL_DIR="/leonardo_scratch/large/userexternal/rqian000/models/OPD-V-Qwen3.5-4B"
MODEL_ID="OPD-V-Qwen3.5-4B"
PYTHON="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/python"
VLLM="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/vllm"
CONDITIONS=(S0 SC S1 S2 S3)
BENCHMARKS=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
CONDITION="${CONDITIONS[${SLURM_ARRAY_TASK_ID}]}"
CAMPAIGN="${DATA_ROOT}/full/runs/prompt-registry-v1"
RUN_ROOT="${CAMPAIGN}/jobs/${CONDITION}/${SLURM_JOB_ID}"
CONDITION_OUTPUT="${CAMPAIGN}/outputs/${CONDITION}"
PORT="$((20000 + SLURM_JOB_ID % 30000))"
API_BASE="http://127.0.0.1:${PORT}/v1"
REGISTRY="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/prompt_registry-v1.json"

export WORK5_DATA_ROOT="${DATA_ROOT}"
export HF_HOME="${DATA_ROOT}/hf-cache"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/${SLURM_JOB_ID}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/${SLURM_JOB_ID}/torchinductor"
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${RUN_ROOT}/logs" "${CONDITION_OUTPUT}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${FAST_REPO}"
MARKER="${CAMPAIGN}/COMPLETE-${CONDITION}"
if [[ -f "${MARKER}" ]]; then
  echo "Condition ${CONDITION} already has a completion marker: ${MARKER}"
  exit 0
fi
[[ -s "${DATA_ROOT}/source-audit/prepared-integrity-verification.json" ]] || {
  echo "Missing successful full-media integrity verification; refusing inference."
  exit 2
}
[[ -s "${REGISTRY}" ]] || { echo "Missing frozen prompt registry: ${REGISTRY}"; exit 2; }

printf 'job_id=%s\narray_task=%s\ncondition=%s\nstart=%s\ncommit=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${CONDITION}" "$(date -Is)" "$(git rev-parse HEAD)" \
  | tee "${RUN_ROOT}/logs/run.txt"
nvidia-smi | tee "${RUN_ROOT}/logs/gpu.txt"
sha256sum "${MODEL_DIR}/config.json" "${MODEL_DIR}/model.safetensors" | tee "${RUN_ROOT}/logs/model-sha256.txt"
sha256sum "${REGISTRY}" | tee "${RUN_ROOT}/logs/prompt-registry-sha256.txt"

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

for benchmark in "${BENCHMARKS[@]}"; do
  input="${DATA_ROOT}/prepared/${benchmark}/benchmark.json"
  [[ -s "${input}" ]] || { echo "Missing prepared benchmark: ${input}"; exit 2; }
  "${PYTHON}" "${FAST_REPO}/reproduction/1. Vision-OPD/code/eval/infer.py" \
    --benchmark "${benchmark}" \
    --benchmark_json "${input}" \
    --out_dir "${CONDITION_OUTPUT}" \
    --model_name "${MODEL_ID}-${CONDITION}" \
    --seed 42 \
    --api_base "${API_BASE}" \
    --model_id "${MODEL_ID}" \
    --max_tokens 32768 \
    --max_retries 3 \
    --parallel_workers 256 \
    --prompt_registry "${REGISTRY}" \
    --condition "${CONDITION}" \
    2>&1 | tee "${RUN_ROOT}/logs/${CONDITION}-${benchmark}.log"
done

printf 'condition=%s\nend=%s\njob_id=%s\n' "${CONDITION}" "$(date -Is)" "${SLURM_JOB_ID}" \
  | tee -a "${RUN_ROOT}/logs/run.txt"
touch "${MARKER}"
