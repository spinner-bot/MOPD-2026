#!/usr/bin/env bash
# Materialize the pinned M3VQA image archive on Leonardo Work.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAYLOAD_ROOT="${MOPD_PAYLOAD_ROOT:-/leonardo_work/AIFAC_F02_774/rqian000/MOPD-2026/payload}"
SOURCE_ROOT="${PAYLOAD_ROOT}/data/external/m3-vqa"
TARGET_ROOT="${PAYLOAD_ROOT}/data/processed/m3vqa/images"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)-prepare-m3vqa-assets}"
RUN_ROOT="${PROJECT_ROOT}/outputs/runs/${RUN_ID}"
ARCHIVE="${SOURCE_ROOT}/images.zip"

[[ -f "${ARCHIVE}" ]] || { echo "missing source archive: ${ARCHIVE}" >&2; exit 2; }
mkdir -p "${TARGET_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/outputs"
exec > >(tee "${RUN_ROOT}/logs/prepare.log") 2>&1

if [[ ! -f "${TARGET_ROOT}/.complete" ]]; then
  unzip -q "${ARCHIVE}" -d "${TARGET_ROOT}"
  touch "${TARGET_ROOT}/.complete"
fi

find "${TARGET_ROOT}" -type f -printf '%P\n' | LC_ALL=C sort > "${RUN_ROOT}/outputs/materialized-files.txt"
{
  echo "source_archive=${ARCHIVE}"
  echo "target_root=${TARGET_ROOT}"
  echo "run_id=${RUN_ID}"
  echo "file_count=$(wc -l < "${RUN_ROOT}/outputs/materialized-files.txt")"
  echo "status=completed"
} > "${RUN_ROOT}/summary.txt"
