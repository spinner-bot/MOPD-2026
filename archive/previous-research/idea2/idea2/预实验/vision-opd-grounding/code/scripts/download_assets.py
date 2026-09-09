from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from huggingface_hub import snapshot_download
from spinner_bot.dataio import data_root, maybe_extract_archives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--dataset", default="yuanqianhao/Vision-OPD-6K")
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--skip-dataset", action="store_true")
    args = p.parse_args()

    if not args.skip_dataset:
        snapshot_download(
            repo_id=args.dataset,
            repo_type="dataset",
            local_dir=str(data_root()),
            local_dir_use_symlinks=False,
            allow_patterns=["README.md", "train.jsonl", "images.tar.gz*", "teacher_images.tar.gz", "original_images.tar.gz"],
        )
        maybe_extract_archives()

    if not args.skip_model:
        snapshot_download(repo_id=args.model, repo_type="model", local_dir=str(ROOT / "model" / "Qwen3.5-4B"))


if __name__ == "__main__":
    main()
