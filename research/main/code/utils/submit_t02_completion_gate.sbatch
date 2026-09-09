#!/usr/bin/env bash
# CPU-only fail-closed completeness gate after the T02 validator.
#SBATCH --job-name=mopd-t02-complete
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
set -euo pipefail
ROOT="${MOPD_ROOT:?set MOPD_ROOT}"; PAYLOAD="${MOPD_PAYLOAD_ROOT:?set MOPD_PAYLOAD_ROOT}"
RUN="$ROOT/outputs/runs/${MOPD_RUN_ID:?set MOPD_RUN_ID}"
"$PAYLOAD/../envs/stage0-py311/bin/python" "$ROOT/code/utils/check_t02_pilot_completion.py" --summary "$RUN/summary.json" --expected-total 1200 --output "$RUN/t02-completion.json"
