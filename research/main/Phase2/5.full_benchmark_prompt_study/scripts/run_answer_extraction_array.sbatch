#!/usr/bin/env bash
# Thirty independent text-only judge shards: five conditions x six tracks.
#SBATCH --job-name=w5-extract
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-29%16
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/extract-%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/extract-%A_%a.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
CONDITIONS=(S0 SC S1 S2 S3)
TRACKS=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID / 6))]}"
TRACK="${TRACKS[$((SLURM_ARRAY_TASK_ID % 6))]}"
VERSION="answer-extraction-v1"
INPUT="${DATA_ROOT}/judgment/${VERSION}/${TRACK}/${CONDITION}.jsonl"
OUTPUT="${DATA_ROOT}/judgment/runs/${VERSION}/${TRACK}/${CONDITION}.jsonl"
PROMPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/answer_extraction_v1.txt"
RUN_LOG_DIR="${DATA_ROOT}/judgment/runs/${VERSION}/${TRACK}/logs"

export HF_HOME="${DATA_ROOT}/hf-cache"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/${SLURM_JOB_ID}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/${SLURM_JOB_ID}/torchinductor"
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set +euo pipefail
source /etc/profile
set -euo pipefail
module load cuda/12.6
module load python/3.11.7
export CPATH=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/python-3.11.7-ziwh63aulhhzxksf42k5u3gnim2rbpmp/include/python3.11:${CPATH:-}
mkdir -p "$(dirname "${OUTPUT}")" "${RUN_LOG_DIR}" \
  "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${FAST_REPO}"
[[ -s "${INPUT}" ]] || { echo "Missing frozen extraction shard: ${INPUT}"; exit 2; }
[[ -s "${PROMPT}" ]] || { echo "Missing extraction prompt: ${PROMPT}"; exit 2; }

printf 'job=%s\narray_task=%s\ncondition=%s\ntrack=%s\nstart=%s\ncommit=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${CONDITION}" "${TRACK}" "$(date -Is)" "$(git rev-parse HEAD)"
nvidia-smi
# This checkpoint is sharded; identify it via its config and shard index.
[[ -s "${MODEL}/model.safetensors.index.json" ]] || { echo "Missing sharded checkpoint index: ${MODEL}/model.safetensors.index.json"; exit 2; }
sha256sum "${INPUT}" "${PROMPT}" "${MODEL}/config.json" "${MODEL}/model.safetensors.index.json"

"${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/run_answer_extraction.py" \
  --input "${INPUT}" \
  --prompt "${PROMPT}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --batch-size 8 \
  --max-model-len 65536 \
  --tensor-parallel-size 2 \
  --max-tokens 384 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee "${RUN_LOG_DIR}/${CONDITION}-${TRACK}-${SLURM_JOB_ID}.log"
