#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs

wait_size() {
  local path="$1" expected="$2"
  while true; do
    size=$(stat -c '%s' "$path" 2>/dev/null || printf 0)
    if [[ "$size" -eq "$expected" ]]; then return 0; fi
    sleep 30
  done
}

# The foreground download process owns images.tar.gz00. Wait for all archives,
# then assemble the split student archive and extract all three image views.
wait_size data/vision_opd_6k/images/images.tar.gz00 5368709120

download() {
  local path="$1" url="$2" expected="$3"
  if [[ -f "$path" ]] && [[ "$(stat -c '%s' "$path")" -eq "$expected" ]]; then return 0; fi
  curl -L --fail --retry 10 --retry-delay 5 --connect-timeout 20 --max-time 14400 \
    -A 'git-lfs/3.4.0' -o "$path" "$url"
  wait_size "$path" "$expected"
}

download data/vision_opd_6k/images/images.tar.gz05 \
  'https://hf-mirror.com/datasets/yuanqianhao/Vision-OPD-6K/resolve/main/images/images.tar.gz05?download=true' \
  1496961022
download data/vision_opd_6k/original_images/original_images.tar.gz \
  'https://hf-mirror.com/datasets/yuanqianhao/Vision-OPD-6K/resolve/main/original_images/original_images.tar.gz?download=true' \
  6249850743

cat data/vision_opd_6k/images/images.tar.gz0{0,1,2,3,4,5} > data/vision_opd_6k/images.tar.gz
mkdir -p data/vision_opd_6k/images/extracted data/vision_opd_6k/original_images/extracted
tar -xzf data/vision_opd_6k/images.tar.gz -C data/vision_opd_6k/images/extracted --no-same-owner
tar -xzf data/vision_opd_6k/original_images/original_images.tar.gz -C data/vision_opd_6k/original_images/extracted --no-same-owner

source .venv/bin/activate
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from spinner_bot.dataio import load_jsonl, pick_paths
rows = load_jsonl(Path('data/vision_opd_6k/train.jsonl'))
assert len(rows) == 6241
for row in rows:
    for p in pick_paths(row).values():
        assert p is not None and p.is_file(), p
print('image_integrity_ok', len(rows))
PY

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/run_inference.py \
  --limit 2 --output outputs/smoke_after_download.jsonl

GPUS=3,7 LIMIT=0 OUT_DIR=outputs/full_$(date +%Y%m%d-%H%M%S) \
  bash scripts/run_all_gpus.sh
