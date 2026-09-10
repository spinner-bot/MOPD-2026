#!/usr/bin/env bash
# Submit with --dependency=afterany:<T02 job id>.  This is a real post-run
# validation task, not a resource placeholder; it uses no GPU.
#SBATCH --job-name=mopd-t02-validate
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="${MOPD_ROOT:?set MOPD_ROOT}"
PAYLOAD="${MOPD_PAYLOAD_ROOT:?set MOPD_PAYLOAD_ROOT}"
RUN="$ROOT/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
ENV="$PAYLOAD/../envs/stage0-py311/bin/python"
TAG="${MOPD_T02_TAG:-pilot}"
cd "$RUN"
"$ENV" "$ROOT/code/data_pipeline/validate_t02_proposals.py" \
  --input-glob 'outputs/shard-*.jsonl' \
  --accepted "$PAYLOAD/data/processed/candidate-v0.1/03_verified/t02-${TAG}-accepted.jsonl" \
  --rejected "$PAYLOAD/data/processed/candidate-v0.1/rejects/t02-${TAG}.jsonl" \
  --summary "$RUN/summary.json"
