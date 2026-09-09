#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="${MOPD_WORK_ROOT:-/leonardo_work/AIFAC_F02_774/rqian000/MOPD-2026}"
PAYLOAD_ROOT="${WORK_ROOT}/payload"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-download-all-assets"
RUN_DIR="${PROJECT_ROOT}/outputs/runs/${RUN_ID}"
LOG="${RUN_DIR}/logs/download.log"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/outputs" "${PAYLOAD_ROOT}/models" "${PAYLOAD_ROOT}/data/external"
printf 'run_id=%s\nproject_root=%s\npayload_root=%s\n' "${RUN_ID}" "${PROJECT_ROOT}" "${PAYLOAD_ROOT}" > "${RUN_DIR}/manifest.txt"

download_model() {
  local repo="$1" dest="$2" rev
  rev="$(hf models info "${repo}" --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')"
  printf '\nMODEL %s revision=%s\n' "${repo}" "${rev}" | tee -a "${LOG}"
  hf download "${repo}" --revision "${rev}" --local-dir "${dest}" 2>&1 | tee -a "${LOG}"
  printf '%s\t%s\t%s\n' "${repo}" "${rev}" "${dest}" >> "${RUN_DIR}/revisions.tsv"
}
download_dataset() {
  local repo="$1" dest="$2" rev
  rev="$(hf datasets info "${repo}" --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')"
  printf '\nDATASET %s revision=%s\n' "${repo}" "${rev}" | tee -a "${LOG}"
  hf download "${repo}" --repo-type dataset --revision "${rev}" --local-dir "${dest}" 2>&1 | tee -a "${LOG}"
  printf '%s\t%s\t%s\n' "${repo}" "${rev}" "${dest}" >> "${RUN_DIR}/revisions.tsv"
}
download_model Qwen/Qwen3.5-4B "${PAYLOAD_ROOT}/models/Qwen3.5-4B"
download_model Qwen/Qwen3.5-9B "${PAYLOAD_ROOT}/models/Qwen3.5-9B"
download_dataset yuanqianhao/Vision-OPD-6K "${PAYLOAD_ROOT}/data/raw/vision-opd-6k"
download_dataset inclusionAI/ZoomBench "${PAYLOAD_ROOT}/data/external/zoombench"
download_dataset DreamMr/HR-Bench "${PAYLOAD_ROOT}/data/external/hr-bench"
download_dataset craigwu/vstar_bench "${PAYLOAD_ROOT}/data/external/vstar-bench"
download_dataset Lin-Chen/MMStar "${PAYLOAD_ROOT}/data/external/mmstar"
download_dataset nyu-visionx/CV-Bench "${PAYLOAD_ROOT}/data/external/cv-bench"
download_dataset lmms-lab-eval/MMVP "${PAYLOAD_ROOT}/data/external/mmvp"
download_dataset lmms-lab-encoder/POPE "${PAYLOAD_ROOT}/data/external/pope"
download_dataset Mini-o3/VisualProbe_Easy "${PAYLOAD_ROOT}/data/external/visualprobe/easy"
download_dataset Mini-o3/VisualProbe_Medium "${PAYLOAD_ROOT}/data/external/visualprobe/medium"
download_dataset Mini-o3/VisualProbe_Hard "${PAYLOAD_ROOT}/data/external/visualprobe/hard"
download_dataset Mini-o3/VisualProbe_train "${PAYLOAD_ROOT}/data/external/visualprobe/train"
download_dataset CASIA-IVA-Lab/M3VQA "${PAYLOAD_ROOT}/data/external/m3-vqa"
download_dataset yifanzhang114/MME-RealWorld "${PAYLOAD_ROOT}/data/external/mme-realworld"
if [ ! -d "${PAYLOAD_ROOT}/data/external/vistahop/.git" ]; then git clone https://github.com/ahang0712/vistahop.git "${PAYLOAD_ROOT}/data/external/vistahop" 2>&1 | tee -a "${LOG}"; fi
git -C "${PAYLOAD_ROOT}/data/external/vistahop" rev-parse HEAD | tee "${RUN_DIR}/vistahop_commit.txt"
ln -sfn "${PAYLOAD_ROOT}/models" "${PROJECT_ROOT}/models"
ln -sfn "${PAYLOAD_ROOT}/data/external" "${PROJECT_ROOT}/data/external"
find "${PAYLOAD_ROOT}" -type f -print0 | xargs -0 sha256sum > "${RUN_DIR}/outputs/payload.sha256"
printf 'status=completed\n' >> "${RUN_DIR}/manifest.txt"
