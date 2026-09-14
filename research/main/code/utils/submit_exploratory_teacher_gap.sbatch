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
DEV_MANIFEST="${MOPD_DEV_MANIFEST:?set 32-sample construction/dev manifest}"
LOCKED_MANIFEST="${MOPD_LOCKED_MANIFEST:?set 96-sample locked manifest}"
CANARY_ROTATION_MANIFEST="${MOPD_CANARY_ROTATION_MANIFEST:?set frozen 32-view canary rotation manifest}"
LOCKED_ROTATION_MANIFEST="${MOPD_LOCKED_ROTATION_MANIFEST:?set frozen 384-view locked rotation manifest}"
RUN="${ROOT}/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_4B:-/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.5-4B-base}"
IMAGE_ROOT="${MOPD_IMAGE_ROOT:?set the explicit common root containing every frozen image}"
REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_COMMIT="${MOPD_EXPECTED_CODE_COMMIT:?set MOPD_EXPECTED_CODE_COMMIT to the submitted git HEAD}"
RUNNER="${ROOT}/code/evaluation/run_exploratory_teacher_gap.py"
PROCESS="${ROOT}/code/evaluation/process_exploratory_teacher_gap.py"
CODE_PATHS=(
  code/evaluation/run_exploratory_teacher_gap.py
  code/evaluation/process_exploratory_teacher_gap.py
  code/evaluation/exploratory_teacher_gap.py
  code/evaluation/teacher_gap_inputs.py
  code/data_pipeline/reasoning_graph_contract.py
  code/data_pipeline/build_natural_label_rotation.py
  code/utils/submit_exploratory_teacher_gap.sbatch
)

assert_code_state() {
  local actual_commit
  actual_commit="$(git -C "${ROOT}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
    echo "code commit changed after submission: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
    exit 2
  }
  git -C "${ROOT}" diff --quiet HEAD -- "${CODE_PATHS[@]}" || {
    echo "tracked teacher-gap runtime code differs from ${EXPECTED_COMMIT}" >&2
    exit 2
  }
}

[[ "${MOPD_LEAKAGE_FREE:-0}" == 1 ]] || { echo "MOPD_LEAKAGE_FREE=1 required after 100% human review" >&2; exit 2; }
[[ -x "${PY}" ]] || { echo "missing Python environment: ${PY}" >&2; exit 2; }
[[ -d "${MODEL}" ]] || { echo "missing base 4B model: ${MODEL}" >&2; exit 2; }
[[ -f "${CANARY_MANIFEST}" ]] || { echo "missing canary manifest" >&2; exit 2; }
[[ -f "${DEV_MANIFEST}" ]] || { echo "missing dev manifest" >&2; exit 2; }
[[ -f "${LOCKED_MANIFEST}" ]] || { echo "missing locked manifest" >&2; exit 2; }
[[ -f "${CANARY_ROTATION_MANIFEST}" ]] || { echo "missing canary rotation manifest" >&2; exit 2; }
[[ -f "${LOCKED_ROTATION_MANIFEST}" ]] || { echo "missing locked rotation manifest" >&2; exit 2; }
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize run archive first: ${RUN}" >&2; exit 2; }
assert_code_state
mkdir -p "${RUN}"/{raw,parsed,summaries,logs}
export ROOT RUN PY MODEL IMAGE_ROOT REVISION EXPECTED_COMMIT RUNNER

echo "[W3 preflight] validate frozen 8/32/96 bundle and partition isolation"
"${PY}" "${PROCESS}" validate-manifests \
  --canary "${CANARY_MANIFEST}" \
  --dev "${DEV_MANIFEST}" \
  --locked "${LOCKED_MANIFEST}" \
  --summary "${RUN}/summaries/manifest-bundle-gate.json"

"${PY}" "${PROCESS}" validate-natural-label-rotation-manifest \
  --manifest "${CANARY_ROTATION_MANIFEST}" --base-manifest "${CANARY_MANIFEST}" \
  --expected-samples 8 --image-root "${IMAGE_ROOT}" --tokenizer "${MODEL}" \
  --summary "${RUN}/summaries/canary-rotation-manifest-gate.json"
"${PY}" "${PROCESS}" validate-natural-label-rotation-manifest \
  --manifest "${LOCKED_ROTATION_MANIFEST}" --base-manifest "${LOCKED_MANIFEST}" \
  --expected-samples 96 --image-root "${IMAGE_ROOT}" --tokenizer "${MODEL}" \
  --summary "${RUN}/summaries/locked-rotation-manifest-gate.json"

run_manifest() {
  local manifest="$1" base_manifest="$2" expected="$3" destination="$4"
  export manifest base_manifest expected destination
  srun --ntasks=4 --gpus-per-task=1 bash -c '
    rank="${SLURM_PROCID}"
    "${PY}" "${RUNNER}" \
      --manifest "${manifest}" --base-manifest "${base_manifest}" \
      --manifest-profile natural_rotation --expected-samples "${expected}" \
      --output "${destination}/rank-${rank}.jsonl" \
      --model "${MODEL}" --model-revision "${REVISION}" --code-commit "${EXPECTED_COMMIT}" \
      --image-root "${IMAGE_ROOT}" \
      --rank "${rank}" --world-size 4
  '
}

echo "[W3] 8 base samples x 4 rotations x 4 conditions"
mkdir -p "${RUN}/raw/canary"
run_manifest "${CANARY_ROTATION_MANIFEST}" "${CANARY_MANIFEST}" 32 "${RUN}/raw/canary"
assert_code_state
"${PY}" "${PROCESS}" validate \
  --inputs "${RUN}/raw/canary/rank-*.jsonl" \
  --manifest "${CANARY_ROTATION_MANIFEST}" --base-manifest "${CANARY_MANIFEST}" \
  --manifest-profile natural_rotation --expected-samples 32 \
  --parsed-output "${RUN}/parsed/canary.jsonl" \
  --summary "${RUN}/summaries/canary-gate.json"

echo "[W4] frozen locked pilot: 96 base samples x 4 rotations x 4 conditions"
assert_code_state
mkdir -p "${RUN}/raw/locked"
run_manifest "${LOCKED_ROTATION_MANIFEST}" "${LOCKED_MANIFEST}" 384 "${RUN}/raw/locked"
assert_code_state
"${PY}" "${PROCESS}" validate \
  --inputs "${RUN}/raw/locked/rank-*.jsonl" \
  --manifest "${LOCKED_ROTATION_MANIFEST}" --base-manifest "${LOCKED_MANIFEST}" \
  --manifest-profile natural_rotation --expected-samples 384 \
  --parsed-output "${RUN}/parsed/locked.jsonl" \
  --summary "${RUN}/summaries/locked-gate.json"

echo "[W4] gate passed; aggregate once and apply the preregistered decision"
assert_code_state
"${PY}" "${PROCESS}" decide-natural-label-rotation \
  --input "${RUN}/parsed/locked.jsonl" \
  --manifest "${LOCKED_ROTATION_MANIFEST}" --base-manifest "${LOCKED_MANIFEST}" \
  --expected-samples 96 --leakage-free \
  --output "${RUN}/summaries/decision.json"

echo "[W4] teacher-gap pilot complete"
