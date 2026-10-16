#!/usr/bin/env bash
# Translate the manual-review packet's English fields into Chinese with the local
# Qwen model. See report 2231 for why the engine is the local LLM rather than Codex.
#SBATCH --job-name=w5-review-translate
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/review-translate-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/review-translate-%j.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
IN_DIR="${DATA_ROOT}/derived/translation/en"
OUT_DIR="${DATA_ROOT}/derived/translation/zh"
WORK="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study"

mkdir -p "${OUT_DIR}" "${DATA_ROOT}/logs"

# 环境：与 run_answer_extraction_array.sbatch 保持一致；PATH 需含 sc-vllm/bin，
# 否则 vLLM 的 Triton JIT 找不到 ninja（首次提交即因此失败）。
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

module load cuda/12.6
module load python/3.11.7
export CPATH=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/python-3.11.7-ziwh63aulhhzxksf42k5u3gnim2rbpmp/include/python3.11:${CPATH:-}
export PATH="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin:${PATH}"

cd "${WORK}"

printf 'job=%s\nstart=%s\ncommit=%s\nin=%s\nout=%s\n' \
  "${SLURM_JOB_ID}" "$(date -Is)" "$(git rev-parse HEAD)" "${IN_DIR}" "${OUT_DIR}"
sha256sum "${WORK}/scripts/translate_review_cases.py"

"${PYTHON}" scripts/translate_review_cases.py \
  --in-dir "${IN_DIR}" \
  --out-dir "${OUT_DIR}" \
  --model "${MODEL}" \
  --tensor-parallel-size 2 \
  --batch-size 16 \
  --max-model-len 32768 \
  --max-tokens 4096 \
  --first 25 \
  --last 60

printf 'end=%s\n' "$(date -Is)"
