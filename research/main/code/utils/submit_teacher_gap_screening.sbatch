#!/usr/bin/env bash
# 1017 W2A: blind-solver + entailment-auditor screening with in-allocation canaries.
#SBATCH --job-name=mopd-tg-screen
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

if [[ -n "${MOPD_CHECKOUT_ROOT:-}" ]]; then
  CHECKOUT_ROOT="$(cd -- "${MOPD_CHECKOUT_ROOT}" && pwd -P)"
  cd "${CHECKOUT_ROOT}"
  ROOT="${CHECKOUT_ROOT}/research/main"
else
  ROOT="${MOPD_ROOT:?set MOPD_ROOT to research/main}"
  CHECKOUT_ROOT="$(git -C "${ROOT}" rev-parse --show-toplevel)"
fi
INPUT="${MOPD_INPUT:?set MOPD_INPUT to a candidate JSONL}"
RUNS_ROOT="${MOPD_RUNS_ROOT:-${ROOT}/outputs/runs}"
RUN="${RUNS_ROOT}/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_9B:-/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.5-9B-base}"
IMAGE_ROOT="${MOPD_IMAGE_ROOT:-/leonardo_scratch/large/userexternal/rqian000/data/m3vqa/materialized/images}"
REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EXPECTED_COMMIT="${MOPD_EXPECTED_CODE_COMMIT:?set MOPD_EXPECTED_CODE_COMMIT to the submitted git HEAD}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "MOPD_EXPECTED_CODE_COMMIT must be the full 40-character lowercase git commit" >&2; exit 2; }
MAX_NEW_TOKENS="${MOPD_MAX_NEW_TOKENS:-768}"
FALLBACK_MAX_NEW_TOKENS="${MOPD_FALLBACK_MAX_NEW_TOKENS:-512}"
THINKING="${MOPD_THINKING:-false}"
EXPECTED="$(wc -l < "${INPUT}")"
MAX_FULL_TRUNCATED="$((EXPECTED / 100))"
RUNNER="${ROOT}/code/data_pipeline/run_teacher_gap_candidate_stage.py"
PROCESS="${ROOT}/code/data_pipeline/process_teacher_gap_candidate_stage.py"
GATE="${ROOT}/code/utils/check_teacher_gap_candidate_gate.py"

[[ -x "${PY}" ]] || { echo "missing Python environment: ${PY}" >&2; exit 2; }
[[ -d "${MODEL}" ]] || { echo "missing 9B model: ${MODEL}" >&2; exit 2; }
[[ -f "${INPUT}" ]] || { echo "missing candidate input: ${INPUT}" >&2; exit 2; }
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize run archive first: ${RUN}" >&2; exit 2; }
[[ "${THINKING}" == false || "${THINKING}" == true ]] || { echo "MOPD_THINKING must be true or false" >&2; exit 2; }
ACTUAL_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || {
  echo "code commit changed after submission: expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2
  exit 2
}
mkdir -p "${RUN}"/{canary,raw,parsed,rejects,summaries,logs}
export CHECKOUT_ROOT ROOT INPUT RUN PY MODEL IMAGE_ROOT REVISION EXPECTED_COMMIT MAX_NEW_TOKENS FALLBACK_MAX_NEW_TOKENS THINKING RUNNER

run_stage() {
  local stage="$1" destination="$2" limit="$3"
  export stage destination limit
  srun --ntasks=4 --gpus-per-task=1 bash -c '
    rank="${SLURM_PROCID}"
    extra=()
    if [[ "${limit}" -gt 0 ]]; then extra+=(--limit "${limit}"); fi
    if [[ "${THINKING}" == true ]]; then extra+=(--thinking); fi
    "${PY}" "${RUNNER}" \
      --stage "${stage}" --input "${INPUT}" \
      --output "${destination}/rank-${rank}.jsonl" \
      --model "${MODEL}" --model-revision "${REVISION}" --code-commit "${EXPECTED_COMMIT}" \
      --image-root "${IMAGE_ROOT}" \
      --rank "${rank}" --world-size 4 --max-new-tokens "${MAX_NEW_TOKENS}" \
      --fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}" \
      "${extra[@]}"
  '
}

for stage in blind audit; do
  echo "[W2A] ${stage} canary"
  mkdir -p "${RUN}/canary/${stage}"
  run_stage "${stage}" "${RUN}/canary/${stage}" 2
  "${PY}" "${GATE}" --stage "${stage}" \
    --input-glob "${RUN}/canary/${stage}/rank-*.jsonl" \
    --summary "${RUN}/summaries/${stage}-canary.json" \
    --expected-total 8 --min-parse-rate 0.75 \
    --expected-world-size 4 --expected-thinking "${THINKING}" \
    --expected-primary-max-new-tokens "${MAX_NEW_TOKENS}" \
    --expected-fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}"

  echo "[W2A] ${stage} full input=${EXPECTED}"
  mkdir -p "${RUN}/raw/${stage}"
  run_stage "${stage}" "${RUN}/raw/${stage}" 0
  "${PY}" "${GATE}" --stage "${stage}" \
    --input-glob "${RUN}/raw/${stage}/rank-*.jsonl" \
    --summary "${RUN}/summaries/${stage}-completion.json" \
    --expected-total "${EXPECTED}" --min-parse-rate 0.90 \
    --expected-world-size 4 --expected-thinking "${THINKING}" \
    --expected-primary-max-new-tokens "${MAX_NEW_TOKENS}" \
    --expected-fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}" \
    --max-truncated "${MAX_FULL_TRUNCATED}"
  "${PY}" "${PROCESS}" parse --stage "${stage}" \
    --inputs "${RUN}/raw/${stage}/rank-*.jsonl" \
    --parsed-output "${RUN}/parsed/${stage}.jsonl" \
    --rejected-output "${RUN}/rejects/${stage}.jsonl" \
    --summary "${RUN}/summaries/${stage}-parse.json"
done

"${PY}" "${PROCESS}" select-construction \
  --blind "${RUN}/parsed/blind.jsonl" \
  --audit "${RUN}/parsed/audit.jsonl" \
  --selected-output "${RUN}/parsed/construction-inputs.jsonl" \
  --rejected-output "${RUN}/rejects/screening.jsonl" \
  --summary "${RUN}/summaries/screening-selection.json"

echo "[W2A] screening complete; model-selected records remain pending human review"
