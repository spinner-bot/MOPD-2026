#!/usr/bin/env bash
# 1046: one diagnostic-only 64 x 4 label-rotation run; no natural-image claim.
#SBATCH --job-name=mopd-g-pc0-lr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --chdir=/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

ROOT="${MOPD_ROOT:-research/main}"
BASE_MANIFEST="${ROOT}/outputs/runs/20260914T0916-teacher-gap-g-pc0-v1/outputs/manifest.jsonl"
BASE_SHA256="10b09d122a5781df564a63cac8449d368a2d785c3fb06ba15c3887db2433bb1c"
MANIFEST="${ROOT}/outputs/runs/20260914T0954-teacher-gap-g-pc0-label-rotation-v1/outputs/manifest.jsonl"
MANIFEST_SHA256="f485cc5f2d973d38d896a31afb1a22208251ea662da10f35c18969454ed031fc"
RUN="${ROOT}/outputs/runs/${MOPD_G_PC0_LR_RUN_ID:?set a new MOPD_G_PC0_LR_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_4B:-/leonardo_scratch/large/userexternal/rqian000/ICLR-2027-czy/models/Qwen3.5-4B-base}"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_COMMIT="${MOPD_EXPECTED_CODE_COMMIT:?set MOPD_EXPECTED_CODE_COMMIT to submitted HEAD}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "MOPD_EXPECTED_CODE_COMMIT must be the full 40-character lowercase git commit" >&2; exit 2; }
RUNNER="${ROOT}/code/evaluation/run_exploratory_teacher_gap.py"
PROCESS="${ROOT}/code/evaluation/process_exploratory_teacher_gap.py"
CODE_PATHS=(
  code/evaluation/run_exploratory_teacher_gap.py
  code/evaluation/process_exploratory_teacher_gap.py
  code/evaluation/exploratory_teacher_gap.py
  code/evaluation/teacher_gap_inputs.py
  code/evaluation/g_pc0.py
  code/data_pipeline/reasoning_graph_contract.py
  code/data_pipeline/generate_g_pc0.py
  code/data_pipeline/build_g_pc0_label_rotation.py
  code/utils/submit_g_pc0_label_rotation.sbatch
)

assert_code_state() {
  local actual_commit
  actual_commit="$(git -C "${ROOT}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
    echo "code commit changed after submission: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
    exit 2
  }
  git -C "${ROOT}" diff --quiet HEAD -- "${CODE_PATHS[@]}" || {
    echo "tracked label-rotation runtime code differs from ${EXPECTED_COMMIT}" >&2
    exit 2
  }
}

[[ -x "${PY}" ]] || { echo "missing Python environment: ${PY}" >&2; exit 2; }
[[ -d "${MODEL}" ]] || { echo "missing base 4B model: ${MODEL}" >&2; exit 2; }
[[ "$(sha256sum "${BASE_MANIFEST}" | cut -d' ' -f1)" == "${BASE_SHA256}" ]] || exit 2
[[ "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" == "${MANIFEST_SHA256}" ]] || exit 2
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize a new run archive first: ${RUN}" >&2; exit 2; }
assert_code_state
mkdir -p "${RUN}"/{raw,parsed,summaries,logs}
export ROOT RUN PY MODEL MODEL_REVISION EXPECTED_COMMIT RUNNER MANIFEST BASE_MANIFEST

"${PY}" "${PROCESS}" validate-label-rotation-manifest \
  --manifest "${MANIFEST}" --base-manifest "${BASE_MANIFEST}" \
  --image-root . --tokenizer "${MODEL}" --summary "${RUN}/summaries/manifest-preflight.json"

assert_code_state
srun --ntasks=4 --gpus-per-task=1 bash -c '
  rank="${SLURM_PROCID}"
  "${PY}" "${RUNNER}" \
    --manifest "${MANIFEST}" --base-manifest "${BASE_MANIFEST}" \
    --manifest-profile g_pc0_rotation --expected-samples 64 \
    --output "${RUN}/raw/rank-${rank}.jsonl" \
    --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
    --code-commit "${EXPECTED_COMMIT}" --image-root . \
    --rank "${rank}" --world-size 4
'

assert_code_state
"${PY}" "${PROCESS}" validate \
  --inputs "${RUN}/raw/rank-*.jsonl" --manifest "${MANIFEST}" \
  --base-manifest "${BASE_MANIFEST}" --manifest-profile g_pc0_rotation \
  --image-root . --expected-samples 64 \
  --parsed-output "${RUN}/parsed/results.jsonl" --summary "${RUN}/summaries/result-gate.json"

assert_code_state
"${PY}" "${PROCESS}" decide-label-rotation \
  --input "${RUN}/parsed/results.jsonl" --manifest "${MANIFEST}" \
  --base-manifest "${BASE_MANIFEST}" --image-root . \
  --output "${RUN}/summaries/decision.json"

echo "[G-PC0-LR] diagnostic complete; G-PC0 v1 decision remains unchanged"
