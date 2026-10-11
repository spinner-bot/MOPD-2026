#!/usr/bin/env python3
"""Run one prompt condition while preserving raw model outputs."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


PROMPT_FILES = {
    "SC": "SC-neutral.txt",
    "S1": "S1-concise.txt",
    "S2": "S2-option-letter.txt",
    "S3": "S3-single-character.txt",
}
_write_lock = threading.Lock()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_mime(path: Path, data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def load_system_prompt(prompt_dir: Path, condition: str):
    if condition == "S0":
        return None
    path = prompt_dir / PROMPT_FILES[condition]
    return path.read_text(encoding="utf-8").strip()


def call_one(client, record, condition, system_prompt, image_root, args):
    image_path = image_root / record["image_archive_path"]
    image_bytes = image_path.read_bytes()
    mime = image_mime(image_path, image_bytes)
    image_url = "data:" + mime + ";base64," + base64.b64encode(image_bytes).decode("ascii")
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": record["query"]},
            ],
        }
    )
    result = {
        "sample_id": record["sample_id"],
        "condition": condition,
        "model_id": args.model_id,
        "system_prompt": system_prompt,
        "query": record["query"],
        "split": record.get("split"),
        "task": record.get("task"),
        "source": record.get("source"),
        "image_archive_path": record["image_archive_path"],
        "image_sha256": record["image_sha256"],
        "messages": [
            *([{"role": "system", "content": system_prompt}] if system_prompt is not None else []),
            {
                "role": "user",
                "content": [
                    {"type": "image_ref", "path": record["image_archive_path"]},
                    {"type": "text", "text": record["query"]},
                ],
            },
        ],
        "response": None,
        "model_answer": None,
        "api_error": None,
    }
    for attempt in range(args.max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=args.model_id,
                messages=messages,
                temperature=0,
                max_tokens=args.max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": args.enable_thinking == "True"
                    }
                },
            )
            text = completion.choices[0].message.content or ""
            result["response"] = text
            result["model_answer"] = text
            result["usage"] = completion.usage.model_dump() if completion.usage else None
            return result
        except Exception as exc:  # preserve failures as data for audit
            result["api_error"] = repr(exc)
            if attempt < args.max_retries:
                time.sleep(2 ** attempt)
    result["model_answer"] = "[API_ERROR]"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=["S0", "SC", "S1", "S2", "S3"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--enable-thinking", choices=["False", "True"], default="False")
    args = parser.parse_args()

    payload = read_json(args.input)
    records = payload["records"]
    done = {}
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done[row["sample_id"]] = row
    todo = [row for row in records if row["sample_id"] not in done]
    system_prompt = load_system_prompt(args.prompt_dir, args.condition)
    client = OpenAI(api_key="EMPTY", base_url=args.api_base, timeout=3600)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(call_one, client, row, args.condition, system_prompt, args.image_root, args): row
            for row in todo
        }
        with args.output.open("a", encoding="utf-8") as handle:
            for future in as_completed(futures):
                result = future.result()
                with _write_lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
    print(json.dumps({"condition": args.condition, "total": len(records), "resumed": len(done), "written": len(todo)}))


if __name__ == "__main__":
    main()
