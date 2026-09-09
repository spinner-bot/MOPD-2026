#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
MODEL=${MODEL:-model/Qwen3.5-4B}
DATASET_JSONL=${DATASET_JSONL:-data/vision_opd_6k/train.jsonl}
OUT_DIR=${OUT_DIR:-outputs/$(date +%Y%m%d-%H%M%S)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
mkdir -p "$OUT_DIR"
IFS=',' read -ra GPU_IDS <<< "$GPUS"
SHARDS=${#GPU_IDS[@]}
if [[ "$SHARDS" -lt 1 ]]; then
  echo "No GPUs selected" >&2
  exit 2
fi
for i in "${!GPU_IDS[@]}"; do
  gpu=${GPU_IDS[$i]}
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_inference.py \
    --dataset-jsonl "$DATASET_JSONL" \
    --model "$MODEL" \
    --output "$OUT_DIR/shard_${i}.jsonl" \
    --shards "$SHARDS" \
    --shard-id "$i" \
    --limit "$LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --resume > "$OUT_DIR/shard_${i}.log" 2>&1 &
  echo $! > "$OUT_DIR/shard_${i}.pid"
done
wait
python scripts/summarize_results.py "$OUT_DIR"/shard_*.jsonl > "$OUT_DIR/summary.csv"
cat "$OUT_DIR/summary.csv"
