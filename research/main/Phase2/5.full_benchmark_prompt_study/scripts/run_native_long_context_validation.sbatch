#!/usr/bin/env bash
# Validate the unchanged Vision-OPD local judge prompt on the full longest pilot answer.
#SBATCH --job-name=w5-native-longctx
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/native-longctx-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data/logs/native-longctx-%j.err

set -euo pipefail
FAST_REPO="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5"
DATA_ROOT="/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data"
MODEL="/leonardo_scratch/large/userexternal/rqian000/models/Qwen3.8-27B"
PYTHON="/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-vllm/bin/python"
PREPARED="${DATA_ROOT}/prepared/mme-realworld/benchmark.json"
INPUT="${DATA_ROOT}/validation/long-context-extraction-v1/input.jsonl"
OUTPUT="${DATA_ROOT}/validation/native-long-context-v1/result.json"
mkdir -p "$(dirname "${OUTPUT}")" "${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export HF_HOME="${DATA_ROOT}/hf-cache"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/${SLURM_JOB_ID}/vllm"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/${SLURM_JOB_ID}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/${SLURM_JOB_ID}/torchinductor"
export TMPDIR="/tmp" PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set +euo pipefail
source /etc/profile
set -euo pipefail
module load cuda/12.6
cd "${FAST_REPO}"
nvidia-smi

"${PYTHON}" - "${FAST_REPO}" "${MODEL}" "${PREPARED}" "${INPUT}" "${OUTPUT}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
repo, model, prepared, input_path, output_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / "reproduction" / "1. Vision-OPD" / "code" / "eval"))
from judge_qwenlm import PROMPT_TEMPLATE, judge_via_vllm

request = json.loads(input_path.read_text(encoding="utf-8").splitlines()[0])
source_rows = json.loads(prepared.read_text(encoding="utf-8"))
source = next(row for row in source_rows if row["sample_uid"] == request["sample_uid"])
response = request["model_response"]
question = request["query"].replace("<image>", "")
prompt = PROMPT_TEMPLATE.format(gt=source["response"], response=response, question=question)
digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
result = judge_via_vllm(
    [prompt], str(model), 2048, tensor_parallel_size=2, max_model_len=65536
)
if len(result) != 1 or not result[0].strip():
    raise SystemExit(f"Native judge returned no response: {result!r}")
if digest != request["model_response_sha256"]:
    raise SystemExit("Native judge input response does not match the full pilot answer hash")
payload = {
    "sample_uid": request["sample_uid"], "condition": request["condition"],
    "track": request["track"], "judge_output": result[0],
    "judge_response": "Yes" if re.search(r"\byes\b", result[0], re.I) else "No/other",
    "input_response_chars": len(response), "input_response_sha256": digest,
    "prompt_chars": len(prompt), "tensor_parallel_size": 2, "max_model_len": 65536,
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "judge_output": result[0], "response_chars": len(response),
                  "prompt_chars": len(prompt), "sha256_matches": True}, ensure_ascii=False))
PY
