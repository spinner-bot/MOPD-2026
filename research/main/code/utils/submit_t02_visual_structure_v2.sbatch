#!/usr/bin/env bash
# T02 pilot v2: in-allocation canary first, then the full 1,200-candidate run.
# The canary gate is CPU-only; on failure the job aborts before wasting the
# remaining GPU hours of the allocation.
#SBATCH --job-name=mopd-t02-visual-v2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT="${MOPD_ROOT:?set MOPD_ROOT}"; PAYLOAD="${MOPD_PAYLOAD_ROOT:?set MOPD_PAYLOAD_ROOT}"
ENV="$PAYLOAD/../envs/stage0-py311/bin/python"
RUN="$ROOT/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
PILOT="$PAYLOAD/data/processed/candidate-v0.1/02_candidates/t02-visual-structure-pilot"
mkdir -p "$RUN/canary" "$RUN/outputs" "$RUN/summaries" "$RUN/logs"
export ROOT PAYLOAD ENV RUN PILOT

echo "[t02-v2] stage 1/3: canary (24 candidates per shard, 4 GPUs)"
srun --ntasks=4 --gpus-per-task=1 bash -c '
  SHARD=$(printf "%03d" "$SLURM_PROCID")
  "$ENV" "$ROOT/code/data_pipeline/run_t02_visual_structure.py" --input "$PILOT/shard-${SHARD}.jsonl" --output "$RUN/canary/shard-${SHARD}.jsonl" --model "$PAYLOAD/models/Qwen3.5-9B" --limit 24 --max-new-tokens 1024
'

echo "[t02-v2] stage 2/3: canary gate (CPU, fail-closed)"
"$ENV" "$ROOT/code/utils/check_t02_canary.py" --input-glob "$RUN/canary/shard-*.jsonl" --summary "$RUN/summaries/canary.json"

echo "[t02-v2] stage 3/3: full pilot run (all 1,200 candidates, 4 GPUs)"
srun --ntasks=4 --gpus-per-task=1 bash -c '
  SHARD=$(printf "%03d" "$SLURM_PROCID")
  "$ENV" "$ROOT/code/data_pipeline/run_t02_visual_structure.py" --input "$PILOT/shard-${SHARD}.jsonl" --output "$RUN/outputs/shard-${SHARD}.jsonl" --model "$PAYLOAD/models/Qwen3.5-9B" --max-new-tokens 1024
'
echo "[t02-v2] full run finished"
