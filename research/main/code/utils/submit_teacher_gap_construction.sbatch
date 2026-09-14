#!/usr/bin/env bash
# 1017 W2B: construct guide/options proposals for screened model candidates.
#SBATCH --job-name=mopd-tg-construct
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
INPUT="${MOPD_INPUT:?set MOPD_INPUT to construction-inputs JSONL}"
RUNS_ROOT="${MOPD_RUNS_ROOT:-${ROOT}/outputs/runs}"
RUN="${RUNS_ROOT}/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_9B:-/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.5-9B-base}"
IMAGE_ROOT="${MOPD_IMAGE_ROOT:-/leonardo_scratch/large/userexternal/rqian000/data/m3vqa/materialized/images}"
REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_REPO="Qwen/Qwen3.5-9B"
ASSET_REVISIONS="${ROOT}/outputs/runs/20260911T1740-teacher-gap-assets/outputs/revisions.tsv"
ASSET_CHECKSUMS="${ROOT}/outputs/runs/20260911T1740-teacher-gap-assets/outputs/model-files.sha256"
MODEL_ASSET_GATE="${ROOT}/code/utils/verify_model_asset.py"
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

[[ "${EXPECTED}" -gt 0 ]] || { echo "no screened candidates; construction not run" >&2; exit 3; }
[[ -x "${PY}" ]] || { echo "missing Python environment: ${PY}" >&2; exit 2; }
[[ -d "${MODEL}" ]] || { echo "missing 9B model: ${MODEL}" >&2; exit 2; }
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize run archive first: ${RUN}" >&2; exit 2; }
[[ "${THINKING}" == false || "${THINKING}" == true ]] || { echo "MOPD_THINKING must be true or false" >&2; exit 2; }
ACTUAL_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || {
  echo "code commit changed after submission: expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2
  exit 2
}
mkdir -p "${RUN}"/{canary,raw,parsed,rejects,summaries,logs}
export CHECKOUT_ROOT ROOT INPUT RUN PY MODEL IMAGE_ROOT REVISION EXPECTED_COMMIT MAX_NEW_TOKENS FALLBACK_MAX_NEW_TOKENS THINKING RUNNER

"${PY}" "${MODEL_ASSET_GATE}" --model-dir "${MODEL}" --repo "${MODEL_REPO}" \
  --revision "${REVISION}" --revisions-tsv "${ASSET_REVISIONS}" \
  --checksums "${ASSET_CHECKSUMS}" --output "${RUN}/summaries/model-asset-gate.json"

run_construct() {
  local destination="$1" limit="$2"
  export destination limit
  srun --ntasks=4 --gpus-per-task=1 bash -c '
    rank="${SLURM_PROCID}"
    extra=()
    if [[ "${limit}" -gt 0 ]]; then extra+=(--limit "${limit}"); fi
    if [[ "${THINKING}" == true ]]; then extra+=(--thinking); fi
    "${PY}" "${RUNNER}" \
      --stage construct --input "${INPUT}" \
      --output "${destination}/rank-${rank}.jsonl" \
      --model "${MODEL}" --model-revision "${REVISION}" --code-commit "${EXPECTED_COMMIT}" \
      --image-root "${IMAGE_ROOT}" \
      --rank "${rank}" --world-size 4 --max-new-tokens "${MAX_NEW_TOKENS}" \
      --fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}" \
      "${extra[@]}"
  '
}

CANARY_EXPECTED="${EXPECTED}"
if [[ "${CANARY_EXPECTED}" -gt 8 ]]; then CANARY_EXPECTED=8; fi
PER_RANK_LIMIT=2
if [[ "${EXPECTED}" -lt 4 ]]; then PER_RANK_LIMIT=1; fi

mkdir -p "${RUN}/canary/construct"
run_construct "${RUN}/canary/construct" "${PER_RANK_LIMIT}"
"${PY}" "${GATE}" --stage construct \
  --input-glob "${RUN}/canary/construct/rank-*.jsonl" \
  --summary "${RUN}/summaries/construct-canary.json" \
  --expected-total "${CANARY_EXPECTED}" --min-parse-rate 0.75 \
  --expected-world-size 4 --expected-thinking "${THINKING}" \
  --expected-primary-max-new-tokens "${MAX_NEW_TOKENS}" \
  --expected-fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}"

mkdir -p "${RUN}/raw/construct"
run_construct "${RUN}/raw/construct" 0
"${PY}" "${GATE}" --stage construct \
  --input-glob "${RUN}/raw/construct/rank-*.jsonl" \
  --summary "${RUN}/summaries/construct-completion.json" \
  --expected-total "${EXPECTED}" --min-parse-rate 0.90 \
  --expected-world-size 4 --expected-thinking "${THINKING}" \
  --expected-primary-max-new-tokens "${MAX_NEW_TOKENS}" \
  --expected-fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}" \
  --max-truncated "${MAX_FULL_TRUNCATED}"
"${PY}" "${PROCESS}" parse --stage construct \
  --inputs "${RUN}/raw/construct/rank-*.jsonl" \
  --parsed-output "${RUN}/parsed/construct.jsonl" \
  --rejected-output "${RUN}/rejects/construct.jsonl" \
  --summary "${RUN}/summaries/construct-parse.json"

echo "[W2B] construction complete; every proposal remains pending 100% human review"
