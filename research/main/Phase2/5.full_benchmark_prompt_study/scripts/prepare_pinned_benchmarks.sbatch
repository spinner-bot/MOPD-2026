#!/usr/bin/env bash
#SBATCH --job-name=w5-prepare
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/prepare-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/prepare-%j.err

set -euo pipefail

FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
PYTHON="/leonardo_scratch/large/userexternal/rqian000/envs/mopd-stage0-py311/bin/python"

export WORK5_DATA_ROOT="${DATA_ROOT}"
export TMPDIR="/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

mkdir -p "${DATA_ROOT}/logs"
cd "${FAST_REPO}"
printf 'job_id=%s\nstart=%s\ncommit=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "$(git rev-parse HEAD)"
nvidia-smi 2>/dev/null || true
"${PYTHON}" research/main/Phase2/5.full_benchmark_prompt_study/scripts/prepare_pinned_benchmarks.py
printf 'end=%s\n' "$(date -Is)"
