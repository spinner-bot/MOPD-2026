#!/usr/bin/env python3
"""Reproduce Vision-OPD eval inputs from pinned parquet files, retaining audit metadata."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(os.environ.get(
    "WORK5_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data",
)).resolve()
DATA = ROOT / "datasets"
OUT = ROOT / "prepared"
IMAGE_EXT = ".jpg"  # matches the official Vision-OPD prepare_data.py naming convention


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_bytes(value, label: str) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ValueError(f"Missing image bytes: {label}")
    payload = bytes(value)
    with Image.open(BytesIO(payload)) as image:
        image.verify()
    return payload


def base64_image(value, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing base64 image: {label}")
    # Match the official prep script's non-strict base64 decoding behavior.
    payload = base64.b64decode(value)
    with Image.open(BytesIO(payload)) as image:
        image.verify()
    return payload


def save_image(track: str, uid: str, kind: str, payload: bytes, media_rows: list) -> str:
    short_id = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:24]
    directory = OUT / track / "images"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{short_id}-{kind}{IMAGE_EXT}"
    digest = sha256(payload)
    if not path.exists() or path.stat().st_size != len(payload) or sha256(path.read_bytes()) != digest:
        tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    media_rows.append({"sample_uid": uid, "kind": kind, "path": str(path), "bytes": len(payload), "sha256": digest})
    return str(path)


def batches(path: Path, columns: list[str], batch_size: int = 8):
    reader = pq.ParquetFile(path)
    yield from reader.iter_batches(batch_size=batch_size, columns=columns)


def files_for(track: str) -> list[Path]:
    if track in {"hrbench-4k", "hrbench-8k"}:
        name = "hr_bench_4k.parquet" if track == "hrbench-4k" else "hr_bench_8k.parquet"
        return sorted((DATA / track).rglob(name))
    return sorted((DATA / track).rglob("*.parquet"))


def official_mme_query(question: str, options: list[str], lang: str) -> str:
    if not options:
        return question
    if lang == "cn":
        return (
            question + " 选项如下所示:\n" + "\n".join(options)
            + "\n根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n最佳答案为： "
        )
    return (
        question + " The choices are listed below:\n" + "\n".join(options)
        + "\nSelect the best answer to the above multiple-choice question based on the image. "
        + "Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is: "
    )


def convert_row(track: str, row: dict, source_file: Path, source_row: int, media_rows: list) -> dict:
    if track == "vstar":
        source_id = row.get("question_id")
        uid = f"{track}:question_id:{source_id}"
        raw_query = (row.get("text") or "").strip()
        post = "\nAnswer with the option's letter from the given choices directly."
        query = raw_query if raw_query.endswith(post.strip()) else raw_query + post
        img = image_bytes(row.get("image"), uid)
        response = (row.get("label") or "").strip().upper()
        meta = {"question_id": source_id, "category": row.get("category") or "unknown"}
        images = [save_image(track, uid, "full", img, media_rows)]
        crop_images = []
    elif track == "zoombench":
        source_id = row.get("id")
        uid = f"{track}:id:{source_id}"
        query = (row.get("query") or "").strip()
        response = (row.get("response") or "").strip()
        meta = {
            "id": source_id,
            "question_type": row.get("question_type"),
            "bbox": row.get("bbox"),
        }
        img = image_bytes(row.get("image"), uid)
        images = [save_image(track, uid, "full", img, media_rows)]
        crop_obj = row.get("crop_image")
        crop_payload = crop_obj.get("bytes") if isinstance(crop_obj, dict) else None
        crop_images = [save_image(track, uid, "crop", bytes(crop_payload), media_rows)] if crop_payload else []
    elif track.startswith("hrbench-"):
        source_id = int(row.get("index"))
        uid = f"{track}:index:{source_id}"
        question = (row.get("question") or "").strip()
        options = [f"({letter}) {(row.get(letter) or '').strip()}" for letter in "ABCD" if (row.get(letter) or "").strip()]
        query = question + " Select from the following choices.\n" + "\n".join(options) if options else question
        response = (row.get("answer") or "").strip()
        meta = {
            "index": source_id,
            "question_id": source_id,
            "category": (row.get("category") or "").strip(),
            "cycle_category": row.get("cycle_category"),
        }
        img = base64_image(row.get("image"), uid)
        images = [save_image(track, uid, "full", img, media_rows)]
        crop_images = []
    elif track in {"mme-realworld", "mme-realworld-cn"}:
        source_id = int(row.get("index"))
        uid = f"{track}:index:{source_id}"
        question = (row.get("question") or "").strip()
        options = [str(x).strip() for x in (row.get("multi-choice options") or []) if str(x).strip()]
        lang = "cn" if track.endswith("-cn") else "en"
        query = official_mme_query(question, options, lang)
        response = (row.get("answer") or "").strip()
        meta = {
            "index": source_id,
            "question_id": source_id,
            "category": row.get("category"),
            "l2-category": row.get("l2-category"),
            "language": lang,
            "option_count": len(options),
        }
        img = base64_image(row.get("bytes"), uid)
        images = [save_image(track, uid, "full", img, media_rows)]
        crop_images = []
    else:
        raise ValueError(f"Unsupported track: {track}")
    if source_id is None or not str(source_id).strip():
        raise ValueError(f"Missing source ID: {track} row {source_row}")
    return {
        **meta,
        "sample_uid": uid,
        "source_file": str(source_file.relative_to(ROOT)),
        "source_row": source_row,
        "images": images,
        "crop_images": crop_images,
        "query": query,
        "response": response,
    }


def source_columns(track: str, path: Path) -> list[str]:
    names = pq.read_schema(path).names
    wanted = {
        "vstar": ["image", "text", "label", "question_id", "category"],
        "zoombench": ["id", "query", "response", "image", "crop_image", "question_type", "bbox"],
        "hrbench-4k": ["index", "question", "answer", "A", "B", "C", "D", "category", "cycle_category", "image"],
        "hrbench-8k": ["index", "question", "answer", "A", "B", "C", "D", "category", "cycle_category", "image"],
        "mme-realworld": ["bytes", "index", "question", "multi-choice options", "answer", "category", "l2-category"],
        "mme-realworld-cn": ["bytes", "index", "question", "multi-choice options", "answer", "category", "l2-category"],
    }[track]
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(f"{track} {path} lacks expected fields: {missing}")
    return wanted


def main() -> None:
    expected = json.loads((ROOT / "source-audit" / "row-counts.json").read_text(encoding="utf-8"))
    summary = {}
    for track, expected_rows in expected.items():
        rows, seen = [], set()
        media_rows = []
        query_hash = hashlib.sha256()
        category_counts: dict[str, Counter] = {}
        response_counts = Counter()
        for src in files_for(track):
            columns = source_columns(track, src)
            source_index = 0
            for batch in batches(src, columns):
                for raw in batch.to_pylist():
                    record = convert_row(track, raw, src, source_index, media_rows)
                    source_index += 1
                    uid = record["sample_uid"]
                    if uid in seen:
                        raise ValueError(f"Duplicate prepared UID: {uid}")
                    seen.add(uid)
                    query_hash.update(uid.encode("utf-8"))
                    query_hash.update(b"\0")
                    query_hash.update(record["query"].encode("utf-8"))
                    query_hash.update(b"\n")
                    for field in ("category", "l2-category", "cycle_category", "question_type"):
                        if field in record:
                            category_counts.setdefault(field, Counter())[str(record[field] if record[field] is not None else "<NULL>")] += 1
                    response_counts[record["response"]] += 1
                    rows.append(record)
        if len(rows) != expected_rows:
            raise ValueError(f"{track}: prepared {len(rows)} rows, expected {expected_rows}")
        destination = OUT / track
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "benchmark.json"
        temp = output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temp, output)
        media_manifest = destination / "media-manifest.json"
        media_manifest.write_text(json.dumps(media_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[track] = {
            "rows": len(rows),
            "unique_sample_uid": len(seen),
            "query_uid_sha256": query_hash.hexdigest(),
            "benchmark_json": str(output),
            "benchmark_json_sha256": sha256(output.read_bytes()),
            "media_files": len(media_rows),
            "media_bytes": sum(item["bytes"] for item in media_rows),
            "media_manifest": str(media_manifest),
            "category_counts": {k: dict(v) for k, v in category_counts.items()},
            "reference_answer_counts": dict(response_counts),
        }
        print(f"{track}: prepared={len(rows)} uid={len(seen)} images={len(media_rows)}", flush=True)
    audit = ROOT / "source-audit" / "prepared-input-audit.json"
    audit.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {audit}", flush=True)


if __name__ == "__main__":
    main()
