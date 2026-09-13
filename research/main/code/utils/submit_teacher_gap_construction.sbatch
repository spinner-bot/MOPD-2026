#!/usr/bin/env bash
# 1017 W2B: construct guide/options proposals for screened model candidates.
#SBATCH --job-name=mopd-tg-construct
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
INPUT="${MOPD_INPUT:?set MOPD_INPUT to construction-inputs JSONL}"
RUN="${ROOT}/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PY="${MOPD_PYTHON:-/leonardo_work/AIFAC_F02_774/sqs/envs/mopd-stage0-py311/bin/python}"
MODEL="${MOPD_MODEL_9B:-/leonardo_work/AIFAC_F02_774/sqs/models/Qwen3.5-9B-base}"
IMAGE_ROOT="${MOPD_IMAGE_ROOT:-/leonardo_work/AIFAC_F02_774/sqs/data/m3vqa/materialized/images}"
REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAX_NEW_TOKENS="${MOPD_MAX_NEW_TOKENS:-1536}"
EXPECTED="$(wc -l < "${INPUT}")"
RUNNER="${ROOT}/code/data_pipeline/run_teacher_gap_candidate_stage.py"
PROCESS="${ROOT}/code/data_pipeline/process_teacher_gap_candidate_stage.py"
GATE="${ROOT}/code/utils/check_teacher_gap_candidate_gate.py"

[[ "${EXPECTED}" -gt 0 ]] || { echo "no screened candidates; construction not run" >&2; exit 3; }
[[ -x "${PY}" ]] || { echo "missing Python environment: ${PY}" >&2; exit 2; }
[[ -d "${MODEL}" ]] || { echo "missing 9B model: ${MODEL}" >&2; exit 2; }
[[ -f "${RUN}/manifest.json" ]] || { echo "initialize run archive first: ${RUN}" >&2; exit 2; }
mkdir -p "${RUN}"/{canary,raw,parsed,rejects,summaries,logs}
export ROOT INPUT RUN PY MODEL IMAGE_ROOT REVISION MAX_NEW_TOKENS RUNNER

run_construct() {
  local destination="$1" limit="$2"
  export destination limit
  srun --ntasks=4 --gpus-per-task=1 bash -c '
    rank="${SLURM_PROCID}"
    extra=()
    if [[ "${limit}" -gt 0 ]]; then extra+=(--limit "${limit}"); fi
    "${PY}" "${RUNNER}" \
      --stage construct --input "${INPUT}" \
      --output "${destination}/rank-${rank}.jsonl" \
      --model "${MODEL}" --model-revision "${REVISION}" --image-root "${IMAGE_ROOT}" \
      --rank "${rank}" --world-size 4 --max-new-tokens "${MAX_NEW_TOKENS}" \
      "${extra[@]}"
  '
}

CANARY_EXPECTED="${EXPECTED}"
if [[ "${CANARY_EXPECTED}" -gt 8 ]]; then CANARY_EXPECTED=8; fi
PER_RANK_LIMIT=2
if [[ "${EXPECTED}" -lt 4 ]]; then PER_RANK_LIMIT=1; fi

mkdir -p "${RUN}/canary/construct"
run_construct "${RUN}/canary/construct" "${PER_RANK_LIMIT}"
"${PY}" "${GATE}" --stage construct \
  --input-glob "${RUN}/canary/construct/rank-*.jsonl" \
  --summary "${RUN}/summaries/construct-canary.json" \
  --expected-total "${CANARY_EXPECTED}" --min-parse-rate 0.75

mkdir -p "${RUN}/raw/construct"
run_construct "${RUN}/raw/construct" 0
"${PY}" "${GATE}" --stage construct \
  --input-glob "${RUN}/raw/construct/rank-*.jsonl" \
  --summary "${RUN}/summaries/construct-completion.json" \
  --expected-total "${EXPECTED}" --min-parse-rate 0.90
"${PY}" "${PROCESS}" parse --stage construct \
  --inputs "${RUN}/raw/construct/rank-*.jsonl" \
  --parsed-output "${RUN}/parsed/construct.jsonl" \
  --rejected-output "${RUN}/rejects/construct.jsonl" \
  --summary "${RUN}/summaries/construct-parse.json"

echo "[W2B] construction complete; every proposal remains pending 100% human review"
