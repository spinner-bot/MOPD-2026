#!/usr/bin/env bash
# 1034 branch B1: question-free image-affordance proposal pilot.
#SBATCH --job-name=mopd-affordance
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

ROOT="${MOPD_ROOT:?set MOPD_ROOT to research/main}"
INPUT="${MOPD_INPUT:?set MOPD_INPUT to image-only JSONL}"
RUN="${ROOT}/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_scratch/large/userexternal/rqian000/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_9B:-/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.5-9B-base}"
IMAGE_ROOT="${MOPD_IMAGE_ROOT:-/leonardo_scratch/large/userexternal/rqian000/data/m3vqa/materialized/images}"
REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EXPECTED_COMMIT="${MOPD_EXPECTED_CODE_COMMIT:?set fixed git commit}"
MAX_NEW_TOKENS="${MOPD_MAX_NEW_TOKENS:-1024}"
FALLBACK_MAX_NEW_TOKENS="${MOPD_FALLBACK_MAX_NEW_TOKENS:-768}"
EXPECTED="$(wc -l < "${INPUT}")"
MAX_FULL_TRUNCATED="$((EXPECTED / 100))"
RUNNER="${ROOT}/code/data_pipeline/run_image_affordance_screen.py"
PROCESS="${ROOT}/code/data_pipeline/process_image_affordance_screen.py"
GATE="${ROOT}/code/utils/check_image_affordance_gate.py"

[[ "${EXPECTED}" -gt 0 ]] || { echo "empty input" >&2; exit 2; }
[[ -x "${PY}" && -d "${MODEL}" && -d "${IMAGE_ROOT}" ]] || { echo "missing runtime asset" >&2; exit 2; }
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize run archive first" >&2; exit 2; }
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "code commit drift" >&2; exit 2; }
mkdir -p "${RUN}"/{canary,raw,parsed,rejects,summaries,logs}
export ROOT INPUT RUN PY MODEL IMAGE_ROOT REVISION EXPECTED_COMMIT MAX_NEW_TOKENS FALLBACK_MAX_NEW_TOKENS RUNNER

run_stage() {
  local destination="$1" limit="$2"
  export destination limit
  srun --ntasks=4 --gpus-per-task=1 bash -c '
    rank="${SLURM_PROCID}"
    extra=()
    if [[ "${limit}" -gt 0 ]]; then extra+=(--limit "${limit}"); fi
    "${PY}" "${RUNNER}" --input "${INPUT}" --output "${destination}/rank-${rank}.jsonl" \
      --model "${MODEL}" --model-revision "${REVISION}" --code-commit "${EXPECTED_COMMIT}" \
      --image-root "${IMAGE_ROOT}" --rank "${rank}" --world-size 4 \
      --max-new-tokens "${MAX_NEW_TOKENS}" --fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}" \
      "${extra[@]}"
  '
}

CANARY_EXPECTED="${EXPECTED}"
if [[ "${CANARY_EXPECTED}" -gt 8 ]]; then CANARY_EXPECTED=8; fi
mkdir -p "${RUN}/canary/image-affordance"
run_stage "${RUN}/canary/image-affordance" 2
"${PY}" "${GATE}" --input-glob "${RUN}/canary/image-affordance/rank-*.jsonl" \
  --summary "${RUN}/summaries/canary.json" --expected-total "${CANARY_EXPECTED}" \
  --min-parse-rate 0.75 --expected-commit "${EXPECTED_COMMIT}" \
  --expected-primary-max-new-tokens "${MAX_NEW_TOKENS}" \
  --expected-fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}"

mkdir -p "${RUN}/raw/image-affordance"
run_stage "${RUN}/raw/image-affordance" 0
"${PY}" "${GATE}" --input-glob "${RUN}/raw/image-affordance/rank-*.jsonl" \
  --summary "${RUN}/summaries/completion.json" --expected-total "${EXPECTED}" \
  --min-parse-rate 0.90 --max-truncated "${MAX_FULL_TRUNCATED}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --expected-primary-max-new-tokens "${MAX_NEW_TOKENS}" \
  --expected-fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}"
"${PY}" "${PROCESS}" --inputs "${RUN}/raw/image-affordance/rank-*.jsonl" \
  --parsed-output "${RUN}/parsed/image-affordance.jsonl" \
  --rejected-output "${RUN}/rejects/image-affordance.jsonl" \
  --summary "${RUN}/summaries/parse.json"
echo "[B1] machine affordance proposals complete; human_verified=0"
