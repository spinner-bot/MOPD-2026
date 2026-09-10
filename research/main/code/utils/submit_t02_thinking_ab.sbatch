#!/usr/bin/env bash
# T02 thinking-mode A/B: rerun the same 96 canary candidates (first 24 per
# pilot shard) with thinking ENABLED and a 3072-token budget, 4 GPUs.
# Purpose: measure the reject->accept flip rate against the non-thinking v2
# canary, i.e. whether non-thinking rejections are under-deliberated. This
# informs the P-layer design decision (crop-relative questions), not the
# production config, which stays non-thinking until decided otherwise.
#SBATCH --job-name=mopd-t02-ab-think
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT="${MOPD_ROOT:?set MOPD_ROOT}"; PAYLOAD="${MOPD_PAYLOAD_ROOT:?set MOPD_PAYLOAD_ROOT}"
ENV="$PAYLOAD/../envs/stage0-py311/bin/python"
RUN="$ROOT/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PILOT="$PAYLOAD/data/processed/candidate-v0.1/02_candidates/t02-visual-structure-pilot"
mkdir -p "$RUN/ab-thinking" "$RUN/logs"
export ROOT PAYLOAD ENV RUN PILOT

echo "[t02-ab] thinking-mode rerun of 96 canary candidates (4 GPUs, 3072 token budget)"
srun --ntasks=4 --gpus-per-task=1 bash -c '
  SHARD=$(printf "%03d" "$SLURM_PROCID")
  "$ENV" "$ROOT/code/data_pipeline/run_t02_visual_structure.py" --input "$PILOT/shard-${SHARD}.jsonl" --output "$RUN/ab-thinking/shard-${SHARD}.jsonl" --model "$PAYLOAD/models/Qwen3.5-9B" --limit 24 --max-new-tokens 3072 --thinking
'
echo "[t02-ab] done"
