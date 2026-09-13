#!/usr/bin/env bash
# 1017 W3/W4: complete canary gate before the frozen 96x4 locked pilot.
#SBATCH --job-name=mopd-tg-pilot
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
CANARY_MANIFEST="${MOPD_CANARY_MANIFEST:?set 8-sample canary manifest}"
LOCKED_MANIFEST="${MOPD_LOCKED_MANIFEST:?set 96-sample locked manifest}"
RUN="${ROOT}/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_work/AIFAC_F02_774/sqs/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_4B:-/leonardo_work/AIFAC_F02_774/sqs/models/Qwen3.5-4B-base}"
IMAGE_ROOT="${MOPD_IMAGE_ROOT:-/leonardo_work/AIFAC_F02_774/sqs/data/m3vqa/materialized/images}"
REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_COMMIT="${MOPD_EXPECTED_CODE_COMMIT:?set MOPD_EXPECTED_CODE_COMMIT to the submitted git HEAD}"
RUNNER="${ROOT}/code/evaluation/run_exploratory_teacher_gap.py"
PROCESS="${ROOT}/code/evaluation/process_exploratory_teacher_gap.py"

[[ "${MOPD_LEAKAGE_FREE:-0}" == 1 ]] || { echo "MOPD_LEAKAGE_FREE=1 required after 100% human review" >&2; exit 2; }
[[ -x "${PY}" ]] || { echo "missing Python environment: ${PY}" >&2; exit 2; }
[[ -d "${MODEL}" ]] || { echo "missing base 4B model: ${MODEL}" >&2; exit 2; }
[[ -f "${CANARY_MANIFEST}" ]] || { echo "missing canary manifest" >&2; exit 2; }
[[ -f "${LOCKED_MANIFEST}" ]] || { echo "missing locked manifest" >&2; exit 2; }
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize run archive first: ${RUN}" >&2; exit 2; }
ACTUAL_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || {
  echo "code commit changed after submission: expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2
  exit 2
}
mkdir -p "${RUN}"/{raw,parsed,summaries,logs}
export ROOT RUN PY MODEL IMAGE_ROOT REVISION EXPECTED_COMMIT RUNNER

run_manifest() {
  local manifest="$1" expected="$2" destination="$3"
  export manifest expected destination
  srun --ntasks=4 --gpus-per-task=1 bash -c '
    rank="${SLURM_PROCID}"
    "${PY}" "${RUNNER}" \
      --manifest "${manifest}" --expected-samples "${expected}" \
      --output "${destination}/rank-${rank}.jsonl" \
      --model "${MODEL}" --model-revision "${REVISION}" --code-commit "${EXPECTED_COMMIT}" \
      --image-root "${IMAGE_ROOT}" \
      --rank "${rank}" --world-size 4
  '
}

echo "[W3] 8-sample canary: all four conditions"
mkdir -p "${RUN}/raw/canary"
run_manifest "${CANARY_MANIFEST}" 8 "${RUN}/raw/canary"
"${PY}" "${PROCESS}" validate \
  --inputs "${RUN}/raw/canary/rank-*.jsonl" \
  --expected-samples 8 \
  --parsed-output "${RUN}/parsed/canary.jsonl" \
  --summary "${RUN}/summaries/canary-gate.json"

echo "[W4] frozen locked pilot: 96 samples x 4 conditions"
mkdir -p "${RUN}/raw/locked"
run_manifest "${LOCKED_MANIFEST}" 96 "${RUN}/raw/locked"
"${PY}" "${PROCESS}" validate \
  --inputs "${RUN}/raw/locked/rank-*.jsonl" \
  --expected-samples 96 \
  --parsed-output "${RUN}/parsed/locked.jsonl" \
  --summary "${RUN}/summaries/locked-gate.json"

echo "[W4] gate passed; aggregate once and apply the preregistered decision"
"${PY}" "${PROCESS}" decide \
  --input "${RUN}/parsed/locked.jsonl" \
  --expected-samples 96 --leakage-free \
  --output "${RUN}/summaries/decision.json"

echo "[W4] teacher-gap pilot complete"
