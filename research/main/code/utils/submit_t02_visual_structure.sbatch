#!/usr/bin/env bash
#SBATCH --job-name=mopd-t02-visual
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
SHARD=$(printf '%03d' "$SLURM_PROCID")
RUN="$ROOT/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
mkdir -p "$RUN/outputs"
"$ENV" "$ROOT/code/data_pipeline/run_t02_visual_structure.py" --input "$PAYLOAD/data/processed/candidate-v0.1/02_candidates/t02-visual-structure-pilot/shard-${SHARD}.jsonl" --output "$RUN/outputs/shard-${SHARD}.jsonl" --model "$PAYLOAD/models/Qwen3.5-9B"
