#!/usr/bin/env bash
# Submit only after the separate T02 pilot has passed its archived validation.
# One allocation is retained across the four semantic stages; each stage has
# its own immutable directory under the supplied pipeline run archive.
#SBATCH --job-name=mopd-preprocess-h
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT="${MOPD_ROOT:?}"; PAYLOAD="${MOPD_PAYLOAD_ROOT:?}"; RUN="$ROOT/outputs/runs/${MOPD_RUN_ID:?}"
ENV="$PAYLOAD/../envs/stage0-py311/bin/python"; MODEL="$PAYLOAD/models/Qwen3.5-9B"
INPUT="$PAYLOAD/data/processed/candidate-v0.1/02_candidates/source-normalized.jsonl"
mkdir -p "$RUN"/{inputs,outputs,rejects,summaries}
stage() { local name="$1" runner="$2" input_dir="$3"; local output_dir="$4"; mkdir -p "$output_dir"; srun --ntasks=4 --gpus-per-task=1 bash -c 'i=$(printf "%03d" "$SLURM_PROCID"); "$ENV" "$ROOT/code/data_pipeline/'"$runner"'" --input "'"$input_dir"'/shard-${i}.jsonl" --output "'"$output_dir"'/shard-${i}.jsonl" --model "$MODEL"'; }
"$ENV" "$ROOT/code/data_pipeline/shard_annotation_candidates.py" --input "$INPUT" --output-dir "$RUN/inputs/t02" --shards 4
stage t02 run_t02_visual_structure.py "$RUN/inputs/t02" "$RUN/outputs/t02"
(cd "$RUN" && "$ENV" "$ROOT/code/data_pipeline/validate_t02_proposals.py" --input-glob 'outputs/t02/*.jsonl' --accepted outputs/t02-accepted.jsonl --rejected rejects/t02.jsonl --summary summaries/t02.json)
"$ENV" "$ROOT/code/data_pipeline/shard_annotation_candidates.py" --input "$RUN/outputs/t02-accepted.jsonl" --output-dir "$RUN/inputs/t03" --shards 4
stage t03 run_t03_guidance.py "$RUN/inputs/t03" "$RUN/outputs/t03"
(cd "$RUN" && "$ENV" "$ROOT/code/data_pipeline/validate_t03_proposals.py" --input-glob 'outputs/t03/*.jsonl' --accepted outputs/t03-accepted.jsonl --rejected rejects/t03.jsonl --summary summaries/t03.json)
"$ENV" "$ROOT/code/data_pipeline/shard_annotation_candidates.py" --input "$RUN/outputs/t03-accepted.jsonl" --output-dir "$RUN/inputs/t04" --shards 4
stage t04 run_t04_options.py "$RUN/inputs/t04" "$RUN/outputs/t04"
(cd "$RUN" && "$ENV" "$ROOT/code/data_pipeline/validate_t04_options.py" --input-glob 'outputs/t04/*.jsonl' --accepted outputs/t04-accepted.jsonl --rejected rejects/t04.jsonl --summary summaries/t04.json)
"$ENV" "$ROOT/code/data_pipeline/materialize_t04_controls.py" --input "$RUN/outputs/t04-accepted.jsonl" --accepted "$RUN/outputs/t04-controlled.jsonl" --rejected "$RUN/rejects/t04-controls.jsonl"
"$ENV" "$ROOT/code/data_pipeline/shard_annotation_candidates.py" --input "$RUN/outputs/t04-controlled.jsonl" --output-dir "$RUN/inputs/t05" --shards 4
stage t05 run_t05_quality.py "$RUN/inputs/t05" "$RUN/outputs/t05"
(cd "$RUN" && "$ENV" "$ROOT/code/data_pipeline/validate_t05_judgements.py" --input-glob 'outputs/t05/*.jsonl' --accepted outputs/t05-accepted.jsonl --borderline outputs/t05-borderline.jsonl --rejected rejects/t05.jsonl)
