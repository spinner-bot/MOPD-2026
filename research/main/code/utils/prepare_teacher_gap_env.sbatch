#!/usr/bin/env bash
# Restore the pinned Stage-0 Python environment on current WORK storage.
#SBATCH --job-name=mopd-tg-env
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

ROOT="${MOPD_ROOT:?set MOPD_ROOT to research/main}"
RUN="${ROOT}/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
STORAGE_ROOT="${MOPD_STORAGE_ROOT:-/leonardo_scratch/large/userexternal/rqian000}"
ENV_ROOT="${MOPD_ENV_ROOT:-/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311}"
BASE_PYTHON="${MOPD_BASE_PYTHON:-/leonardo_work/AIFAC_F02_774/rqian000/miniconda3/bin/python}"
LOCK="${ROOT}/requirements-stage0.lock"
WHEELHOUSE="${MOPD_WHEELHOUSE:-${STORAGE_ROOT}/cache/wheelhouse-stage0-py311}"

[[ -x "${BASE_PYTHON}" ]] || { echo "missing base Python: ${BASE_PYTHON}" >&2; exit 2; }
[[ -f "${LOCK}" ]] || { echo "missing lock file: ${LOCK}" >&2; exit 2; }
[[ -d "${WHEELHOUSE}" ]] || { echo "missing offline wheelhouse: ${WHEELHOUSE}" >&2; exit 2; }
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize run archive first: ${RUN}" >&2; exit 2; }
mkdir -p "${STORAGE_ROOT}/envs" "${STORAGE_ROOT}/cache/pip" "${RUN}/logs" "${RUN}/outputs"
export PIP_CACHE_DIR="${STORAGE_ROOT}/cache/pip"
export TMPDIR="${STORAGE_ROOT}/cache/pip/tmp"
mkdir -p "${TMPDIR}"
exec > >(tee "${RUN}/logs/environment.log") 2>&1

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --copies "${ENV_ROOT}"
fi
"${ENV_ROOT}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" --requirement "${LOCK}"
"${ENV_ROOT}/bin/python" -m pip check
"${ENV_ROOT}/bin/python" -m pip freeze --all | LC_ALL=C sort > "${RUN}/outputs/pip-freeze.txt"
"${ENV_ROOT}/bin/python" - <<'PY' > "${RUN}/outputs/import-check.txt"
import platform
import torch
import transformers
import qwen_vl_utils
import PIL

print("python=" + platform.python_version())
print("torch=" + torch.__version__)
print("transformers=" + transformers.__version__)
print("qwen_vl_utils=" + getattr(qwen_vl_utils, "__version__", "unknown"))
print("pillow=" + PIL.__version__)
print("cuda_build=" + str(torch.version.cuda))
print("cuda_available=" + str(torch.cuda.is_available()))
PY
printf 'environment=%s\nlock=%s\nstatus=completed\n' "${ENV_ROOT}" "${LOCK}" > "${RUN}/outputs/summary.txt"
