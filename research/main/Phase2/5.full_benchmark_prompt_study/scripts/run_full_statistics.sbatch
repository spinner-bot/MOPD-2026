#!/usr/bin/env bash
# Analyze the verified full benchmark after both scoring chains complete.
#SBATCH --job-name=w5-statistics
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/statistics-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/statistics-%j.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
PYTHON="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/python"
ANALYZER="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/analyze_full_results.py"

export PYTHONUNBUFFERED=1
mkdir -p "${DATA_ROOT}/logs" "${DATA_ROOT}/derived"
cd "${FAST_REPO}"
printf 'job=%s\nstart=%s\ncommit=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "$(git rev-parse HEAD)"
sha256sum "${ANALYZER}" \
  "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/prompts/prompt_registry-v1.json"
"${PYTHON}" -m unittest discover \
  -s "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/tests" -v
"${PYTHON}" "${ANALYZER}" --data-root "${DATA_ROOT}" --output-version full-statistics-v1 \
  2>&1 | tee "${DATA_ROOT}/derived/full-statistics-v1-run-${SLURM_JOB_ID}.log"
printf 'end=%s\n' "$(date -Is)"
