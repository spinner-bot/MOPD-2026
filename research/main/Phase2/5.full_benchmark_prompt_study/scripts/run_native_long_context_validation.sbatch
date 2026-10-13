#!/usr/bin/env bash
# Validate the unchanged Vision-OPD local judge prompt on the full longest pilot answer.
#SBATCH --job-name=w5-native-longctx
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/native-longctx-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/native-longctx-%j.err

set -euo pipefail
FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
PREPARED="${DATA_ROOT}/prepared/mme-realworld/benchmark.json"
INPUT="${DATA_ROOT}/validation/long-context-extraction-v1/input.jsonl"
OUTPUT="${DATA_ROOT}/validation/native-long-context-v1/result.json"
mkdir -p "$(dirname "${OUTPUT}")" "${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export HF_HOME="${DATA_ROOT}/hf-cache"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/${SLURM_JOB_ID}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/${SLURM_JOB_ID}/torchinductor"
export TMPDIR="/tmp" PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set +euo pipefail
source /etc/profile
set -euo pipefail
module load cuda/12.6
cd "${FAST_REPO}"
nvidia-smi

"${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/validate_native_long_context.py" \
  --repository "${FAST_REPO}" \
  --model "${MODEL}" \
  --prepared "${PREPARED}" \
  --input "${INPUT}" \
  --output "${OUTPUT}"
