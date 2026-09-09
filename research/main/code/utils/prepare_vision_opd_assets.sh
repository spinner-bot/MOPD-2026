#!/usr/bin/env bash
# Materialize the pinned Vision-OPD archives for reproducible local inference.
# The source archives stay immutable; this command only creates a derived,
# restartable asset tree on Leonardo Work.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAYLOAD_ROOT="${MOPD_PAYLOAD_ROOT:-/leonardo_work/AIFAC_F02_774/rqian000/MOPD-2026/payload}"
SOURCE_ROOT="${PAYLOAD_ROOT}/data/raw/vision-opd-6k"
TARGET_ROOT="${PAYLOAD_ROOT}/data/processed/vision-opd-6k"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)-prepare-vision-opd-assets}"
RUN_ROOT="${PROJECT_ROOT}/outputs/runs/${RUN_ID}"

for required in \
  "${SOURCE_ROOT}/original_images/original_images.tar.gz" \
  "${SOURCE_ROOT}/teacher_images/teacher_images.tar.gz" \
  "${SOURCE_ROOT}/images/images.tar.gz00"; do
  [[ -f "${required}" ]] || { echo "missing source archive: ${required}" >&2; exit 2; }
done

mkdir -p "${TARGET_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/outputs"
exec > >(tee "${RUN_ROOT}/logs/prepare.log") 2>&1

extract_once() {
  local marker="$1"
  shift
  if [[ -f "${TARGET_ROOT}/.${marker}.complete" ]]; then
    echo "already complete: ${marker}"
    return
  fi
  "$@"
  touch "${TARGET_ROOT}/.${marker}.complete"
}

extract_once original_images tar -xzf "${SOURCE_ROOT}/original_images/original_images.tar.gz" -C "${TARGET_ROOT}"
extract_once teacher_images tar -xzf "${SOURCE_ROOT}/teacher_images/teacher_images.tar.gz" -C "${TARGET_ROOT}"

if [[ ! -f "${TARGET_ROOT}/.images.complete" ]]; then
  cat "${SOURCE_ROOT}"/images/images.tar.gz0* | tar -xzf - -C "${TARGET_ROOT}"
  touch "${TARGET_ROOT}/.images.complete"
fi

find "${TARGET_ROOT}" -type f -printf '%P\n' | LC_ALL=C sort > "${RUN_ROOT}/outputs/materialized-files.txt"
{
  echo "source_root=${SOURCE_ROOT}"
  echo "target_root=${TARGET_ROOT}"
  echo "run_id=${RUN_ID}"
  echo "file_count=$(wc -l < "${RUN_ROOT}/outputs/materialized-files.txt")"
  echo "status=completed"
} > "${RUN_ROOT}/summary.txt"
