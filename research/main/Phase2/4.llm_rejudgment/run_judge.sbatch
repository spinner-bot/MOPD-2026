#!/bin/bash
#SBATCH --job-name=w4-judge
#SBATCH --partition=boost_usr_prod
#SBATCH --account=aifac_f02_774
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/Phase2/llm_rejudgment/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/Phase2/llm_rejudgment/logs/%x-%j.err

set -euo pipefail
source /etc/profile
module load cuda/12.6

REPO="/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026"
WORK="${REPO}/research/main/Phase2/4.llm_rejudgment"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
export PATH="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin:${PATH}"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
LIMIT_PER_CONDITION="${LIMIT_PER_CONDITION:-}"
OUTPUT="${WORK}/raw/runs/${RUN_TAG}/judge-shard-${SHARD_INDEX}.jsonl"

mkdir -p "$(dirname "${OUTPUT}")" /leonardo_scratch/large/userexternal/rqian000/Phase2/llm_rejudgment/logs
cd "${REPO}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ARGS=(
  "${WORK}/scripts/run_judge_shard.py"
  --input "${WORK}/raw/input-freeze-v1/judge_requests.jsonl"
  --prompt "${WORK}/prompts/answer_extraction_v1.txt"
  --model "${MODEL}"
  --output "${OUTPUT}"
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
  --batch-size 8
  --max-model-len 8192
  --max-tokens 384
  --gpu-memory-utilization 0.92
)
if [[ -n "${LIMIT_PER_CONDITION}" ]]; then
  ARGS+=(--limit-per-condition "${LIMIT_PER_CONDITION}")
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} tag=${RUN_TAG} shard=${SHARD_INDEX}/${NUM_SHARDS}"
"${PYTHON}" "${ARGS[@]}"
