#!/usr/bin/env bash
# Resume unfinished shards across generation, verification, scoring and extraction.
#SBATCH --job-name=w5-campaign-manager
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
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
CHECKER="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/check_judgment_stage.py"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
MANAGER_SCRIPT="${FAST_REPO}/research/main/Phase2/5.full_benchmark_prompt_study/scripts/supervise_full_campaign.sbatch"
ATTEMPT="${WORK5_ATTEMPT:-0}"
MAX_ATTEMPTS=80
STAGE="${WORK5_STAGE:-campaign}"
mkdir -p "${DATA_ROOT}/logs"
cd "${FAST_REPO}"

printf 'manager_job=%s\nattempt=%s\nstart=%s\ncommit=%s\n' \
  "${SLURM_JOB_ID}" "${ATTEMPT}" "$(date -Is)" "$(git rev-parse HEAD)"

launch_manager() {
  local stage="$1" attempt="$2" dependency="$3" extra_export=""
  if [[ -n "${WORK5_VERIFY_JOB:-}" ]]; then extra_export+=",WORK5_VERIFY_JOB=${WORK5_VERIFY_JOB}"; fi
  if [[ -n "${WORK5_NATIVE_JOB:-}" ]]; then extra_export+=",WORK5_NATIVE_JOB=${WORK5_NATIVE_JOB}"; fi
  if [[ -n "${WORK5_EXTRACT_JOB:-}" ]]; then extra_export+=",WORK5_EXTRACT_JOB=${WORK5_EXTRACT_JOB}"; fi
  local output
  output="$(sbatch --dependency="afterany:${dependency}" \
    --export="ALL,WORK5_STAGE=${stage},WORK5_ATTEMPT=${attempt}${extra_export}" "${MANAGER_SCRIPT}")"
  printf '%s' "${output##* }"
}

if [[ "${STAGE}" == "verify" ]]; then
  verify_job="${WORK5_VERIFY_JOB:?missing verification job id}"
  verify_state="$(sacct -X -n -j "${verify_job}" --format=State | head -n 1 | tr -d ' +\t')"
  printf 'verification_job=%s\nverification_state=%s\n' "${verify_job}" "${verify_state}"
  if [[ "${verify_state}" == "COMPLETED" ]]; then
    native_output="$(sbatch --dependency="afterok:${verify_job}" "${NATIVE_SCRIPT}")"
    native_job="${native_output##* }"
    extract_output="$(sbatch --dependency="afterok:${verify_job}" "${EXTRACT_SCRIPT}")"
    extract_job="${extract_output##* }"
    next_manager="$(WORK5_NATIVE_JOB="${native_job}" WORK5_EXTRACT_JOB="${extract_job}" \
      launch_manager judges "${ATTEMPT}" "${native_job}:${extract_job}")"
    printf 'native_job=%s\nextract_job=%s\njudgment_manager=%s\n' \
      "${native_job}" "${extract_job}" "${next_manager}"
    exit 0
  fi
  verification_log="${DATA_ROOT}/logs/full-output-verification-${verify_job}.log"
  if [[ -f "${verification_log}" ]] && grep -q '^PASS ALL; WROTE ' "${verification_log}"; then
    echo "Output verification passed but a downstream freeze step failed; preserving all generation markers and retrying verification/freeze only." >&2
    retry_verify_output="$(sbatch "${VERIFY_SCRIPT}")"
    retry_verify_job="${retry_verify_output##* }"
    next_manager="$(WORK5_VERIFY_JOB="${retry_verify_job}" launch_manager verify "$((ATTEMPT + 1))" "${retry_verify_job}")"
    printf 'output_verification_passed=1\nretry_verify_job=%s\nverification_manager=%s\n' \
      "${retry_verify_job}" "${next_manager}"
    exit 0
  fi
  if (( ATTEMPT >= MAX_ATTEMPTS )); then
    echo "Verification failed after ${ATTEMPT} continuation rounds; preserving logs and stopping retries." >&2
    exit 3
  fi
  for condition in S0 SC S1 S2 S3; do
    for benchmark in vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn; do
      marker="${CAMPAIGN}/COMPLETE-${condition}-${benchmark}"
      if [[ -f "${marker}" ]]; then
        mv -- "${marker}" "${marker}.verification-failed-${verify_job}"
      fi
    done
    marker="${CAMPAIGN}/COMPLETE-${condition}"
    if [[ -f "${marker}" ]]; then
      mv -- "${marker}" "${marker}.verification-failed-${verify_job}"
    fi
  done
  next_manager="$(launch_manager campaign "$((ATTEMPT + 1))" "${SLURM_JOB_ID}")"
  printf 'verification_failed=%s\nnext_campaign_manager=%s\n' "${verify_job}" "${next_manager}"
  exit 0
fi

if [[ "${STAGE}" == "judges" ]]; then
  native_missing="$("${PYTHON}" "${CHECKER}" --data-root "${DATA_ROOT}" --stage native)"
  extract_missing="$("${PYTHON}" "${CHECKER}" --data-root "${DATA_ROOT}" --stage extraction)"
  printf 'native_missing=%s\nextraction_missing=%s\n' "${native_missing}" "${extract_missing}"
  if [[ "${native_missing}" == "NONE" && "${extract_missing}" == "NONE" ]]; then
    stats_output="$(sbatch "${STATS_SCRIPT}")"
    printf 'all_30_native_and_extraction_shards_verified=1\nstats_job=%s\n' "${stats_output##* }"
    exit 0
  fi
  if (( ATTEMPT >= MAX_ATTEMPTS )); then
    echo "Judgment outputs remain incomplete after ${ATTEMPT} rounds; preserving outputs and stopping retries." >&2
    exit 3
  fi
  retry_jobs=()
  if [[ "${native_missing}" != "NONE" ]]; then
    output="$(sbatch --array="${native_missing}%4" "${NATIVE_SCRIPT}")"
    retry_jobs+=("${output##* }")
  fi
  if [[ "${extract_missing}" != "NONE" ]]; then
    output="$(sbatch --array="${extract_missing}%4" "${EXTRACT_SCRIPT}")"
    retry_jobs+=("${output##* }")
  fi
  child_dependencies="$(IFS=:; printf '%s' "${retry_jobs[*]}")"
  next_manager="$(launch_manager judges "$((ATTEMPT + 1))" "${child_dependencies}")"
  printf 'retry_native_or_extract_jobs=%s\nnext_judgment_manager=%s\n' \
    "${child_dependencies}" "${next_manager}"
  exit 0
fi

if [[ "${STAGE}" != "campaign" ]]; then
  echo "Unknown supervisor stage: ${STAGE}" >&2
  exit 2
fi

CONDITIONS=(S0 SC S1 S2 S3)
BENCHMARKS=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
MISSING=()
for ci in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$ci]}"
  complete_condition=1
  for ti in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$ti]}"
    array_id=$((ci * ${#BENCHMARKS[@]} + ti))
    if [[ ! -f "${CAMPAIGN}/COMPLETE-${condition}-${benchmark}" ]]; then
      MISSING+=("${array_id}")
      complete_condition=0
    fi
  done
  if (( complete_condition )); then
    touch "${CAMPAIGN}/COMPLETE-${condition}"
  else
    rm -f -- "${CAMPAIGN}/COMPLETE-${condition}"
  fi
done

if (( ${#MISSING[@]} == 0 )); then
  printf 'All condition markers present; dispatching verification.\n'
  verify_output="$(sbatch "${VERIFY_SCRIPT}")"
  verify_job="${verify_output##* }"
  next_manager="$(WORK5_VERIFY_JOB="${verify_job}" launch_manager verify "${ATTEMPT}" "${verify_job}")"
  printf 'verify_job=%s\nverification_manager=%s\n' "${verify_job}" "${next_manager}"
  exit 0
fi

printf 'Missing conditions: %s\n' "${MISSING[*]}"
if (( ATTEMPT >= MAX_ATTEMPTS )); then
  echo "Maximum continuation rounds (${MAX_ATTEMPTS}) reached; refusing an infinite retry loop." >&2
  exit 3
fi

array_spec="$(IFS=,; printf '%s' "${MISSING[*]}")"
retry_output="$(sbatch --time=02:00:00 --array="${array_spec}%8" "${FULL_SCRIPT}")"
retry_job="${retry_output##* }"
next_attempt=$((ATTEMPT + 1))
manager_output="$(launch_manager campaign "${next_attempt}" "${retry_job}")"
printf 'retry_array=%s\nnext_manager=%s\nnext_attempt=%s\n' \
  "${retry_job}" "${manager_output}" "${next_attempt}"
