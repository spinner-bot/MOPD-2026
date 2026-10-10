#!/usr/bin/env bash
#SBATCH --job-name=mopd-vopd-audit
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=mopd-vopd-audit-%j.out
#SBATCH --error=mopd-vopd-audit-%j.err

set -euo pipefail

REPO="/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026"
RUN_ROOT="${REPO}/research/main/Phase2/vision_opd_output_audit/runs/${SLURM_JOB_ID}-hrbench4k"
EVAL_DIR="${REPO}/reproduction/1. Vision-OPD/code/eval"
MODEL_DIR="${REPO}/research/main/models/OPD-V-Qwen3.5-4B"
PYTHON="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/python"
VLLM="/leonardo_work/AIFAC_F02_774/chentian/envs/VisAlloc/bin/vllm"
SOURCE_JSON="${REPO}/research/main/data/external/hr_bench_4k.json"

mkdir -p "${RUN_ROOT}/inputs" "${RUN_ROOT}/outputs" "${RUN_ROOT}/logs"
cd "${REPO}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

"${PYTHON}" - "${SOURCE_JSON}" "${RUN_ROOT}/inputs/hrbench4k_sample12.json" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
items = json.loads(source.read_text(encoding="utf-8"))
ranked = []
for item in items:
    uid = str(item.get("question_id", item.get("index", "")))
    score = hashlib.sha256(("phase2-vopd-audit-261010:" + uid).encode()).hexdigest()
    record = dict(item)
    images = []
    for image in item.get("images", []):
        p = pathlib.Path(image)
        fallback = pathlib.Path("/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026") / "research/main/data/external/HRBench_4k_images" / p.name
        images.append(str(p if p.exists() else fallback))
    record["images"] = images
    ranked.append((score, record))
ranked.sort(key=lambda x: x[0])
sample = [record for _, record in ranked[:12]]
target.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sample) + "\n", encoding="utf-8")
print(json.dumps({"source_count": len(items), "sample_count": len(sample), "question_ids": [x.get("question_id") for x in sample]}, ensure_ascii=False))
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${VLLM}" serve "${MODEL_DIR}" \
  --served-model-name OPD-V-Qwen3.5-4B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code \
  >"${RUN_ROOT}/logs/vllm-server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >"${RUN_ROOT}/logs/models.json" 2>/dev/null; then
    break
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8000/v1/models >"${RUN_ROOT}/logs/models-final.json"

for thinking in True False; do
  out="${RUN_ROOT}/outputs/thinking-${thinking}"
  mkdir -p "${out}"
  (
    cd "${EVAL_DIR}"
    "${PYTHON}" infer.py \
      --benchmark hrbench-4k \
      --benchmark_json "${RUN_ROOT}/inputs/hrbench4k_sample12.json" \
      --out_dir "${out}" \
      --model_name "OPD-V-Qwen3.5-4B-thinking-${thinking}" \
      --seed 42 \
      --api_base http://127.0.0.1:8000/v1/ \
      --model_id OPD-V-Qwen3.5-4B \
      --max_tokens 2048 \
      --max_retries 2 \
      --parallel_workers 12 \
      --enable_thinking "${thinking}"
  ) 2>&1 | tee "${RUN_ROOT}/logs/infer-thinking-${thinking}.log"
done

date -Is >"${RUN_ROOT}/logs/completed-at.txt"
echo "Completed output audit run: ${RUN_ROOT}"
