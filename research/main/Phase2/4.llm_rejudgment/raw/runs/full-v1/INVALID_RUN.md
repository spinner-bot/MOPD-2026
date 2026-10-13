# INVALID RUN — DO NOT USE

The `full-v1` outputs are retained only as a record of a failed submission and must not be merged or analyzed.

Four Slurm jobs were intended to use shard indices 0, 1, 2, and 3. A quoting/variable-expansion error in the shell submission loop caused all four to receive the default `SHARD_INDEX=0`, while each logged `NUM_SHARDS=4`. Every process therefore selected the same quarter of the frozen dataset and wrote to `judge-shard-0.jsonl`. The first process had already produced records; later processes resumed against that shared path and concurrently appended duplicated/repeated work. The shared output is not an authoritative, complete, uniquely keyed judgment set.

Jobs: `60274811`, `60275186`, `60275188`, and `60275190`. They were cancelled after the common shard index and output path were confirmed. Their logs, manifest, and partial output are preserved alongside this marker for auditability. No `full-v1` record will enter Work2/Work3 analysis.

The runner has been hardened with shard-count balance checks, an output-filename/index match, and a nonblocking per-output process lock. The replacement run uses a new tag (`full-v2`) and four explicitly specified shard indices; each job's startup log and manifest must be checked before accepting output.
