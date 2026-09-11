#!/usr/bin/env bash
# Restore only the pinned models required by protocol 1016.
set -euo pipefail

STORAGE_ROOT="${MOPD_STORAGE_ROOT:-/leonardo_work/AIFAC_F02_774/sqs}"
RUN_DIR="${MOPD_RUN_DIR:?set MOPD_RUN_DIR to an initialized run archive}"
HF_BIN="${MOPD_HF_BIN:-${HOME}/.local/bin/hf}"

MODEL_ROOT="${STORAGE_ROOT}/models"
CACHE_ROOT="${STORAGE_ROOT}/cache/huggingface"
MODEL_4B="${MODEL_ROOT}/Qwen3.5-4B-base"
MODEL_9B="${MODEL_ROOT}/Qwen3.5-9B-base"
REV_4B="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REV_9B="c202236235762e1c871ad0ccb60c8ee5ba337b9a"

[[ -x "${HF_BIN}" ]] || { echo "missing hf CLI: ${HF_BIN}" >&2; exit 2; }
[[ -d "${STORAGE_ROOT}" ]] || { echo "missing storage root: ${STORAGE_ROOT}" >&2; exit 2; }
[[ -d "${RUN_DIR}" ]] || { echo "missing run archive: ${RUN_DIR}" >&2; exit 2; }

mkdir -p "${MODEL_ROOT}" "${CACHE_ROOT}" "${RUN_DIR}/logs" "${RUN_DIR}/outputs"
export HF_HOME="${CACHE_ROOT}"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export HF_XET_CACHE="${CACHE_ROOT}/xet"

exec > >(tee "${RUN_DIR}/logs/prepare-assets.log") 2>&1

echo "[assets] Qwen/Qwen3.5-4B revision=${REV_4B}"
"${HF_BIN}" download Qwen/Qwen3.5-4B \
  --revision "${REV_4B}" \
  --local-dir "${MODEL_4B}"

echo "[assets] Qwen/Qwen3.5-9B revision=${REV_9B}"
"${HF_BIN}" download Qwen/Qwen3.5-9B \
  --revision "${REV_9B}" \
  --local-dir "${MODEL_9B}"

{
  printf 'repo\trevision\tpath\n'
  printf 'Qwen/Qwen3.5-4B\t%s\t%s\n' "${REV_4B}" "${MODEL_4B}"
  printf 'Qwen/Qwen3.5-9B\t%s\t%s\n' "${REV_9B}" "${MODEL_9B}"
} > "${RUN_DIR}/outputs/revisions.tsv"

find "${MODEL_4B}" "${MODEL_9B}" -type f ! -path '*/.cache/*' -printf '%p\t%s\n' \
  | LC_ALL=C sort > "${RUN_DIR}/outputs/model-files.tsv"

while IFS=$'\t' read -r path size; do
  sha256sum "${path}"
done < "${RUN_DIR}/outputs/model-files.tsv" > "${RUN_DIR}/outputs/model-files.sha256"

printf 'model_4b=%s\nmodel_9b=%s\nstatus=completed\n' \
  "${MODEL_4B}" "${MODEL_9B}" > "${RUN_DIR}/outputs/summary.txt"

