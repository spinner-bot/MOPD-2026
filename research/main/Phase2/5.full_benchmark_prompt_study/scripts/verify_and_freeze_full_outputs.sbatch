#!/usr/bin/env bash
# Final paired-coverage gate, then atomically freeze text-only answer-extraction shards.
#SBATCH --job-name=w5-verify
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/verify-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/verify-%j.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
PYTHON="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/python"
CAMPAIGN="${DATA_ROOT}/full/runs/prompt-registry-v1"
REGISTRY="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/prompt_registry-v1.json"

export WORK5_DATA_ROOT="${DATA_ROOT}"
export TMPDIR="/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "${DATA_ROOT}/logs"
cd "${FAST_REPO}"
printf 'job=%s\nstart=%s\ncommit=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "$(git rev-parse HEAD)"

for condition in S0 SC S1 S2 S3; do
  [[ -f "${CAMPAIGN}/COMPLETE-${condition}" ]] || { echo "Missing condition completion: ${condition}"; exit 2; }
done

"${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/verify_inference_outputs.py" \
  --data-root "${DATA_ROOT}" \
  --run-root "${CAMPAIGN}" \
  --model-id "OPD-V-Qwen3.5-4B" \
  --prompt-registry "${REGISTRY}" \
  2>&1 | tee "${DATA_ROOT}/logs/full-output-verification-${SLURM_JOB_ID}.log"

"${PYTHON}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/prepare_answer_extraction.py" \
  --data-root "${DATA_ROOT}" \
  --campaign "full/runs/prompt-registry-v1" \
  --output-version "answer-extraction-v1" \
  2>&1 | tee "${DATA_ROOT}/logs/answer-extraction-input-freeze-${SLURM_JOB_ID}.log"

printf 'end=%s\n' "$(date -Is)"
touch "${CAMPAIGN}/VERIFIED_AND_JUDGE_INPUT_FROZEN"
