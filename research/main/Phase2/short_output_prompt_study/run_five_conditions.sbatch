#!/bin/bash
#SBATCH --job-name=phase2-short-output
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/Phase2/short_output_prompt_study/slurm-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/Phase2/short_output_prompt_study/slurm-%j.err

set -euo pipefail

REPO="/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026"
WORK="/leonardo_scratch/large/userexternal/rqian000/Phase2/short_output_prompt_study"
MODEL_ID="OPD-V-Qwen3.5-4B"
MODEL_DIR="${REPO}/research/main/models/OPD-V-Qwen3.5-4B"
PYTHON="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/python"
VLLM="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/vllm"
API_BASE="http://127.0.0.1:8000/v1"
RUN_ROOT="${WORK}/runs/${SLURM_JOB_ID}"
INPUT="${REPO}/research/main/Phase2/short_output_prompt_study/raw/inference_input.json"
IMAGE_ROOT="${WORK}/raw/images"
PROMPT_DIR="${REPO}/research/main/Phase2/short_output_prompt_study/prompts"

mkdir -p "${RUN_ROOT}/outputs" "${RUN_ROOT}/logs"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

"${VLLM}" serve "${MODEL_DIR}" \
  --served-model-name "${MODEL_ID}" --tensor-parallel-size 4 --port 8000 \
  --trust-remote-code --max-model-len 32768 --gpu-memory-utilization 0.90 >"${RUN_ROOT}/logs/vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for attempt in $(seq 1 120); do
  if curl --silent --fail "${API_BASE}/models" >/dev/null; then break; fi
  sleep 5
done
curl --silent --fail "${API_BASE}/models" >"${RUN_ROOT}/logs/models.json"

for condition in S0 SC S1 S2 S3; do
  "${PYTHON}" "${REPO}/research/main/Phase2/short_output_prompt_study/scripts/run_inference.py" \
    --input "${INPUT}" --image-root "${IMAGE_ROOT}" --prompt-dir "${PROMPT_DIR}" \
    --condition "${condition}" --model-id "${MODEL_ID}" --api-base "${API_BASE}" \
    --output "${RUN_ROOT}/outputs/${condition}.jsonl" --workers 32 \
    --enable-thinking False >"${RUN_ROOT}/logs/${condition}.log" 2>&1
done

date -Is >"${RUN_ROOT}/logs/completed-at.txt"
