#!/usr/bin/env bash
#SBATCH --job-name=mopd-tg-audit-r5
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
ROOT="${MOPD_ROOT:?set MOPD_ROOT}"
INPUT="${MOPD_INPUT:?set MOPD_INPUT}"
RUN="${ROOT}/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PY="${MOPD_PYTHON:?set MOPD_PYTHON}"
MODEL="${MOPD_MODEL_9B:?set MOPD_MODEL_9B}"
IMAGE_ROOT="${MOPD_IMAGE_ROOT:?set MOPD_IMAGE_ROOT}"
EXPECTED_COMMIT="${MOPD_EXPECTED_CODE_COMMIT:?set MOPD_EXPECTED_CODE_COMMIT}"
REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAX_NEW_TOKENS=768
FALLBACK_MAX_NEW_TOKENS=512
ACTUAL_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || { echo "code commit drift" >&2; exit 2; }
mkdir -p "${RUN}/canary/audit" "${RUN}/summaries"
export ROOT INPUT RUN PY MODEL IMAGE_ROOT EXPECTED_COMMIT REVISION MAX_NEW_TOKENS FALLBACK_MAX_NEW_TOKENS
srun --ntasks=4 --gpus-per-task=1 bash -c '
  rank="${SLURM_PROCID}"
  "${PY}" "${ROOT}/code/data_pipeline/run_teacher_gap_candidate_stage.py" \
    --stage audit --input "${INPUT}" --output "${RUN}/canary/audit/rank-${rank}.jsonl" \
    --model "${MODEL}" --model-revision "${REVISION}" --code-commit "${EXPECTED_COMMIT}" \
    --image-root "${IMAGE_ROOT}" --rank "${rank}" --world-size 4 --limit 2 \
    --max-new-tokens "${MAX_NEW_TOKENS}" --fallback-max-new-tokens "${FALLBACK_MAX_NEW_TOKENS}"
'
"${PY}" "${ROOT}/code/utils/check_teacher_gap_candidate_gate.py" \
  --stage audit --input-glob "${RUN}/canary/audit/rank-*.jsonl" \
  --summary "${RUN}/summaries/audit-canary.json" --expected-total 8 --min-parse-rate 0.75 \
  --expected-world-size 4 --expected-thinking false \
  --expected-primary-max-new-tokens 768 --expected-fallback-max-new-tokens 512
