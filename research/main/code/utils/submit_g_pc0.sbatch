#!/usr/bin/env bash
# 1043 G-PC0: one immutable 16 x 4 mechanism-control run; no natural-image claim.
#SBATCH --job-name=mopd-g-pc0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

if [[ -n "${MOPD_CHECKOUT_ROOT:-}" ]]; then
  CHECKOUT_ROOT="$(cd -- "${MOPD_CHECKOUT_ROOT}" && pwd -P)"
  cd "${CHECKOUT_ROOT}"
  ROOT="${CHECKOUT_ROOT}/research/main"
else
  ROOT="${MOPD_ROOT:-research/main}"
  CHECKOUT_ROOT="$(git -C "${ROOT}" rev-parse --show-toplevel)"
fi
MANIFEST="${ROOT}/outputs/runs/20260914T0916-teacher-gap-g-pc0-v1/outputs/manifest.jsonl"
MANIFEST_SHA256="10b09d122a5781df564a63cac8449d368a2d785c3fb06ba15c3887db2433bb1c"
RUNS_ROOT="${MOPD_RUNS_ROOT:-${ROOT}/outputs/runs}"
RUN="${RUNS_ROOT}/${MOPD_G_PC0_RUN_ID:?set a new MOPD_G_PC0_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_4B:-/leonardo_scratch/large/userexternal/rqian000/ICLR-2027-czy/models/Qwen3.5-4B-base}"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_REPO="Qwen/Qwen3.5-4B"
ASSET_REVISIONS="${ROOT}/outputs/runs/20260911T1740-teacher-gap-assets/outputs/revisions.tsv"
ASSET_CHECKSUMS="${ROOT}/outputs/runs/20260911T1740-teacher-gap-assets/outputs/model-files.sha256"
MODEL_ASSET_GATE="${ROOT}/code/utils/verify_model_asset.py"
EXPECTED_COMMIT="${MOPD_EXPECTED_CODE_COMMIT:?set MOPD_EXPECTED_CODE_COMMIT to the submitted git HEAD}"
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
  code/utils/verify_model_asset.py
  outputs/runs/20260911T1740-teacher-gap-assets/outputs/revisions.tsv
  outputs/runs/20260911T1740-teacher-gap-assets/outputs/model-files.sha256
  code/utils/submit_g_pc0.sbatch
)

assert_code_state() {
  local actual_commit
  actual_commit="$(git -C "${ROOT}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
    echo "code commit changed after submission: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
    exit 2
  }
  git -C "${ROOT}" diff --quiet HEAD -- "${CODE_PATHS[@]}" || {
    echo "tracked G-PC0 runtime code differs from ${EXPECTED_COMMIT}" >&2
    exit 2
  }
}

[[ -x "${PY}" ]] || { echo "missing Python environment: ${PY}" >&2; exit 2; }
[[ -d "${MODEL}" ]] || { echo "missing base 4B model: ${MODEL}" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "missing frozen G-PC0 manifest" >&2; exit 2; }
[[ "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" == "${MANIFEST_SHA256}" ]] || {
  echo "frozen G-PC0 manifest hash mismatch" >&2
  exit 2
}
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize a new run archive first: ${RUN}" >&2; exit 2; }
assert_code_state
mkdir -p "${RUN}"/{raw,parsed,summaries,logs}
export CHECKOUT_ROOT ROOT RUN PY MODEL MODEL_REVISION EXPECTED_COMMIT RUNNER MANIFEST

"${PY}" "${MODEL_ASSET_GATE}" --model-dir "${MODEL}" --repo "${MODEL_REPO}" \
  --revision "${MODEL_REVISION}" --revisions-tsv "${ASSET_REVISIONS}" \
  --checksums "${ASSET_CHECKSUMS}" --output "${RUN}/summaries/model-asset-gate.json"

echo "[G-PC0] CPU preflight: profile, assets, tokenizer and Generic-T length"
"${PY}" "${PROCESS}" validate-positive-control-manifest \
  --manifest "${MANIFEST}" --image-root . --tokenizer "${MODEL}" \
  --summary "${RUN}/summaries/manifest-preflight.json"

echo "[G-PC0] frozen 16 samples x four conditions"
assert_code_state
srun --ntasks=4 --gpus-per-task=1 bash -c '
  rank="${SLURM_PROCID}"
  "${PY}" "${RUNNER}" \
    --manifest "${MANIFEST}" --manifest-profile g_pc0 --expected-samples 16 \
    --output "${RUN}/raw/rank-${rank}.jsonl" \
    --model "${MODEL}" --model-revision "${MODEL_REVISION}" \
    --code-commit "${EXPECTED_COMMIT}" --image-root . \
    --rank "${rank}" --world-size 4
'

echo "[G-PC0] fail-closed result gate"
assert_code_state
"${PY}" "${PROCESS}" validate \
  --inputs "${RUN}/raw/rank-*.jsonl" \
  --manifest "${MANIFEST}" --manifest-profile g_pc0 --image-root . \
  --expected-samples 16 \
  --model-asset-gate "${RUN}/summaries/model-asset-gate.json" \
  --parsed-output "${RUN}/parsed/results.jsonl" \
  --summary "${RUN}/summaries/result-gate.json"

echo "[G-PC0] apply frozen non-inferential decision rule"
assert_code_state
"${PY}" "${PROCESS}" decide-positive-control \
  --input "${RUN}/parsed/results.jsonl" --manifest "${MANIFEST}" --image-root . \
  --output "${RUN}/summaries/decision.json"

echo "[G-PC0] complete; this run cannot authorize a natural-image or training claim"
