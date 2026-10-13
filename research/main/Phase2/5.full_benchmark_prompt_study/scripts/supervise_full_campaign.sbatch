#!/usr/bin/env bash
# Resume only unfinished prompt conditions and dispatch all downstream stages on full success.
#SBATCH --job-name=w5-campaign-manager
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/manager-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/manager-%j.err

set -euo pipefail
FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
CAMPAIGN="${DATA_ROOT}/full/runs/prompt-registry-v1"
FULL_SCRIPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/run_full_conditions.sbatch"
VERIFY_SCRIPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/verify_and_freeze_full_outputs.sbatch"
NATIVE_SCRIPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/run_visionopd_native_scorer.sbatch"
EXTRACT_SCRIPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/run_answer_extraction_array.sbatch"
STATS_SCRIPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/run_full_statistics.sbatch"
ATTEMPT="${WORK5_ATTEMPT:-0}"
MAX_ATTEMPTS=12
mkdir -p "${DATA_ROOT}/logs"
cd "${FAST_REPO}"

printf 'manager_job=%s\nattempt=%s\nstart=%s\ncommit=%s\n' \
  "${SLURM_JOB_ID}" "${ATTEMPT}" "$(date -Is)" "$(git rev-parse HEAD)"

CONDITIONS=(S0 SC S1 S2 S3)
MISSING=()
for index in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$index]}"
  if [[ ! -f "${CAMPAIGN}/COMPLETE-${condition}" ]]; then
    MISSING+=("${index}")
  fi
done

if (( ${#MISSING[@]} == 0 )); then
  printf 'All condition markers present; dispatching verification and downstream jobs.\n'
  verify_output="$(sbatch "${VERIFY_SCRIPT}")"
  verify_job="${verify_output##* }"
  native_output="$(sbatch --dependency="afterok:${verify_job}" "${NATIVE_SCRIPT}")"
  native_job="${native_output##* }"
  extract_output="$(sbatch --dependency="afterok:${verify_job}" "${EXTRACT_SCRIPT}")"
  extract_job="${extract_output##* }"
  stats_output="$(sbatch --dependency="afterok:${native_job}:${extract_job}" "${STATS_SCRIPT}")"
  printf 'verify_job=%s\nnative_job=%s\nextract_job=%s\nstats_job=%s\n' \
    "${verify_job}" "${native_job}" "${extract_job}" "${stats_output##* }"
  exit 0
fi

printf 'Missing conditions: %s\n' "${MISSING[*]}"
if (( ATTEMPT >= MAX_ATTEMPTS )); then
  echo "Maximum continuation rounds (${MAX_ATTEMPTS}) reached; refusing an infinite retry loop." >&2
  exit 3
fi

array_spec="$(IFS=,; printf '%s' "${MISSING[*]}")"
retry_output="$(sbatch --array="${array_spec}%4" "${FULL_SCRIPT}")"
retry_job="${retry_output##* }"
next_attempt=$((ATTEMPT + 1))
manager_output="$(sbatch --dependency="afterany:${retry_job}" \
  --export="ALL,WORK5_ATTEMPT=${next_attempt}" "${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/supervise_full_campaign.sbatch")"
printf 'retry_array=%s\nnext_manager=%s\nnext_attempt=%s\n' \
  "${retry_job}" "${manager_output##* }" "${next_attempt}"
