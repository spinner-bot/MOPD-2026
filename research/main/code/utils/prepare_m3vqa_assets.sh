#!/usr/bin/env bash
# Materialize the pinned M3VQA image archive on the current project storage.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STORAGE_ROOT="${MOPD_STORAGE_ROOT:-/leonardo_work/AIFAC_F02_774/sqs}"
if [[ -n "${MOPD_PAYLOAD_ROOT:-}" ]]; then
  # Backward-compatible explicit override for archived runs using the old tree.
  SOURCE_ROOT="${MOPD_PAYLOAD_ROOT}/data/external/m3-vqa"
  TARGET_ROOT="${MOPD_PAYLOAD_ROOT}/data/processed/m3vqa/images"
else
  SOURCE_ROOT="${MOPD_M3VQA_SOURCE_ROOT:-${STORAGE_ROOT}/data/m3vqa/source}"
  TARGET_ROOT="${MOPD_M3VQA_IMAGE_ROOT:-${STORAGE_ROOT}/data/m3vqa/materialized}"
fi
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)-prepare-m3vqa-assets}"
RUN_ROOT="${PROJECT_ROOT}/outputs/runs/${RUN_ID}"
ARCHIVE="${SOURCE_ROOT}/images.zip"
QUESTIONS="${SOURCE_ROOT}/quesions.jsonl"

[[ -f "${ARCHIVE}" ]] || { echo "missing source archive: ${ARCHIVE}" >&2; exit 2; }
[[ -f "${QUESTIONS}" ]] || { echo "missing source questions: ${QUESTIONS}" >&2; exit 2; }
mkdir -p "${TARGET_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/outputs"
exec > >(tee "${RUN_ROOT}/logs/prepare.log") 2>&1

if [[ ! -f "${TARGET_ROOT}/.complete" ]]; then
  # -n makes an interrupted extraction restartable without rewriting good files.
  unzip -q -n "${ARCHIVE}" -d "${TARGET_ROOT}"
  expected_files="$(unzip -Z1 "${ARCHIVE}" | awk 'substr($0,length($0),1)!="/"' | wc -l)"
  actual_files="$(find "${TARGET_ROOT}" -type f ! -name '.complete' | wc -l)"
  if [[ "${actual_files}" -ne "${expected_files}" ]]; then
    echo "incomplete extraction: expected=${expected_files} actual=${actual_files}" >&2
    exit 3
  fi
  touch "${TARGET_ROOT}/.complete"
fi

find "${TARGET_ROOT}" -type f ! -name '.complete' -printf '%P\n' | LC_ALL=C sort > "${RUN_ROOT}/outputs/materialized-files.txt"
archive_sha256="$(sha256sum "${ARCHIVE}" | cut -d' ' -f1)"
questions_sha256="$(sha256sum "${QUESTIONS}" | cut -d' ' -f1)"
{
  echo "source_archive=${ARCHIVE}"
  echo "source_questions=${QUESTIONS}"
  echo "target_root=${TARGET_ROOT}"
  echo "run_id=${RUN_ID}"
  echo "archive_sha256=${archive_sha256}"
  echo "questions_sha256=${questions_sha256}"
  echo "file_count=$(wc -l < "${RUN_ROOT}/outputs/materialized-files.txt")"
  echo "status=completed"
} > "${RUN_ROOT}/summary.txt"
