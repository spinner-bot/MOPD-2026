from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spinner_bot.dataio import ensure_dataset, images_available, load_jsonl, maybe_extract_archives, pick_paths  # noqa: E402
from spinner_bot.experiment import CASES, PromptCase, build_grounding_separator, build_prompt, build_separator_question  # noqa: E402
from spinner_bot.modeling import load_model, infer  # noqa: E402


def normalize_answer(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else text.strip()[:1].upper()


def image_list(paths: dict[str, Path | None]) -> list[Image.Image]:
    selected = [paths["original_image"]] if paths["original_image"] is not None else []
    missing = [str(p) for p in selected if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required image(s): " + ", ".join(missing))
    return [Image.open(p).convert("RGB") for p in selected]


def run(args: argparse.Namespace) -> None:
    if args.dataset_jsonl:
        dataset_jsonl = Path(args.dataset_jsonl)
    else:
        dataset_jsonl = ensure_dataset()
        maybe_extract_archives()
    records = load_jsonl(dataset_jsonl)
    if not images_available():
        raise RuntimeError(
            "Vision-OPD image archives are not present in spinner-bot/data/vision_opd_6k. "
            "Download images.tar.gz*, teacher_images.tar.gz, and original_images.tar.gz first."
        )
    if args.limit:
        records = records[: args.limit]
    if args.shards > 1:
        records = [r for i, r in enumerate(records) if i % args.shards == args.shard_id]

    cases = [CASES[name.strip()] for name in args.cases.split(",") if name.strip()]
    model, processor = load_model(args.model, args.device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[int, str]] = set()
    if out.exists() and args.resume:
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add((row["global_index"], row["case"]))
                except Exception:
                    pass

    with out.open("a", encoding="utf-8") as f:
        for local_i, rec in enumerate(tqdm(records, desc=f"shard {args.shard_id}")):
            global_i = args.shard_id + local_i * args.shards if args.shards > 1 else local_i
            question = rec.get("extra_info", {}).get("question") or rec.get("problem", "").replace("<image>", "").strip()
            separator = build_separator_question(question, rec.get("problem"))
            grounding = build_grounding_separator(question, rec.get("bbox"), rec.get("problem"))
            paths = pick_paths(rec)
            for case in cases:
                key = (global_i, case.name)
                if key in done:
                    continue
                prompt = build_prompt(question, separator, grounding, case)
                try:
                    imgs = image_list(paths)
                    pred_text = infer(model, processor, prompt, imgs, args.max_new_tokens)
                    pred = normalize_answer(pred_text)
                    err = None
                except Exception as e:
                    pred_text = ""
                    pred = ""
                    err = repr(e)
                answer = str(rec.get("answer", rec.get("extra_info", {}).get("answer", ""))).strip().upper()
                row = {
                    "global_index": global_i,
                    "case": case.name,
                    "answer": answer,
                    "prediction": pred,
                    "raw_prediction": pred_text,
                    "correct": pred == answer,
                    "bbox": rec.get("bbox"),
                    "images": {k: str(v) if v else None for k, v in paths.items()},
                    "error": err,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-jsonl", default="")
    p.add_argument("--model", default=str(ROOT / "model" / "Qwen3.5-4B"))
    p.add_argument("--output", default="outputs/opd_vl_results.jsonl")
    p.add_argument("--cases", default="orig,separator,grounding,all_text")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
