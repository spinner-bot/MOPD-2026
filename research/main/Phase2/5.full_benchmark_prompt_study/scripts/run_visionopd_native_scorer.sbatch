#!/usr/bin/env bash
# Reproduce the unmodified Vision-OPD judge_qwenlm.py / cal_acc.py path per condition/track.
#SBATCH --job-name=w5-native-score
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-29%16
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/native-score-%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/native-score-%A_%a.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
CODE_DIR="${FAST_REPO}/reproduction/1. Vision-OPD/code/eval"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
CONDITIONS=(S0 SC S1 S2 S3)
TRACKS=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID / 6))]}"
TRACK="${TRACKS[$((SLURM_ARRAY_TASK_ID % 6))]}"
MODEL_ID="OPD-V-Qwen3.5-4B-${CONDITION}"
CAMPAIGN="${DATA_ROOT}/full/runs/prompt-registry-v1"
TASK_ROOT="${DATA_ROOT}/official-scoring/vision-opd-native-v1/${CONDITION}/${TRACK}"
MARKER="${TASK_ROOT}/COMPLETE"
LOG_DIR="${DATA_ROOT}/official-scoring/vision-opd-native-v1/logs"

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
mkdir -p "${TASK_ROOT}" "${LOG_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${TASK_ROOT}"
if [[ -f "${MARKER}" ]]; then
  echo "Native score already complete: ${MARKER}"
  exit 0
fi
ln -sfn "${CAMPAIGN}/outputs/${CONDITION}" model_answer

printf 'job=%s\ncondition=%s\ntrack=%s\nstart=%s\ncommit=%s\n' \
  "${SLURM_JOB_ID}" "${CONDITION}" "${TRACK}" "$(date -Is)" "$(git -C "${FAST_REPO}" rev-parse HEAD)"
nvidia-smi
sha256sum "${MODEL}/config.json" "${CODE_DIR}/judge_qwenlm.py" "${CODE_DIR}/cal_acc.py"
find "${MODEL}" -maxdepth 1 -type f -name '*.safetensors' -printf '%f\t%s bytes\n' | sort

"${PYTHON}" "${CODE_DIR}/judge_qwenlm.py" \
  --benchmark "${TRACK}" \
  --model "${MODEL_ID}" \
  --judge_model_path "${MODEL}" \
  --judge_max_tokens 2048 \
  --judge_tensor_parallel_size 2 \
  --judge_max_model_len 65536 \
  2>&1 | tee "${LOG_DIR}/${CONDITION}-${TRACK}-${SLURM_JOB_ID}-judge.log"

JUDGE_JSON="${TASK_ROOT}/judge/${TRACK}/${MODEL_ID}_answer.jsonl"
[[ -s "${JUDGE_JSON}" ]] || { echo "Missing native judge output: ${JUDGE_JSON}"; exit 3; }
"${PYTHON}" "${CODE_DIR}/cal_acc.py" \
  --benchmark "${TRACK}" \
  --judge_json "${JUDGE_JSON}" \
  --benchmark_json "${DATA_ROOT}/prepared/${TRACK}/benchmark.json" \
  2>&1 | tee "${LOG_DIR}/${CONDITION}-${TRACK}-${SLURM_JOB_ID}-accuracy.log"

printf 'condition=%s\ntrack=%s\njob=%s\nend=%s\n' "${CONDITION}" "${TRACK}" "${SLURM_JOB_ID}" "$(date -Is)" \
  > "${TASK_ROOT}/run-manifest.txt"
touch "${MARKER}"
