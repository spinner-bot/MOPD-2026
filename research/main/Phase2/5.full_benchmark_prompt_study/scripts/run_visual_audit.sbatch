#!/usr/bin/env bash
# Multimodal, image-grounded adjudication for the frozen 60 x 5 stratified audit.
# This is supplementary quality-control evidence; it never overwrites official scores.
#SBATCH --job-name=w5-visual-audit
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/visual-audit-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/visual-audit-%j.err

set -euo pipefail
FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
SCRIPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/run_visual_audit.py"
SAMPLE="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/audits/manual-review-sample-v2.json"
PROMPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/visual_audit_v1.txt"
RUN_VERSION="${RUN_VERSION:-v1}"
RUN_DIR="${DATA_ROOT}/judgment/visual-audit-${RUN_VERSION}"

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
module load python/3.11.7
export CPATH="/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/python-3.11.7-ziwh63aulhhzxksf42k5u3gnim2rbpmp/include/python3.11:${CPATH:-}"
mkdir -p "${DATA_ROOT}/logs" "${RUN_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${FAST_REPO}"
printf 'job=%s\nrun_version=%s\nrun_dir=%s\nstart=%s\ncommit=%s\n' "${SLURM_JOB_ID}" "${RUN_VERSION}" "${RUN_DIR}" "$(date -Is)" "$(git rev-parse HEAD)"
[[ -s "${MODEL}/config.json" && -s "${MODEL}/model.safetensors.index.json" ]] || { echo "Model checkpoint is missing or not sharded"; exit 2; }
sha256sum "${MODEL}/config.json" "${MODEL}/model.safetensors.index.json" "${PROMPT}" "${SAMPLE}"
"${PYTHON}" "${SCRIPT}" \
  --sample-manifest "${SAMPLE}" \
  --data-root "${DATA_ROOT}" \
  --prompt "${PROMPT}" \
  --model "${MODEL}" \
  --run-dir "${RUN_DIR}" \
  --batch-size 4 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --max-tokens 384 \
  --gpu-memory-utilization 0.90
printf 'end=%s\n' "$(date -Is)"
