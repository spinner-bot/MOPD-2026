from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    p = root() / ".cache"
    p.mkdir(exist_ok=True)
    return p


def data_root() -> Path:
    p = root() / "data" / "vision_opd_6k"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _image_dir(folder: str) -> Path:
    direct = data_root() / folder
    extracted = direct / "extracted"
    return extracted if extracted.is_dir() and any(extracted.iterdir()) else direct


def dataset_manifest() -> Path:
    return data_root() / "train.jsonl"


def ensure_dataset() -> Path:
    if dataset_manifest().exists():
        return dataset_manifest()
    local_dir = snapshot_download(
        repo_id="yuanqianhao/Vision-OPD-6K",
        repo_type="dataset",
        local_dir=str(data_root()),
        local_dir_use_symlinks=False,
        allow_patterns=["train.jsonl", "images.tar.gz*", "teacher_images.tar.gz", "original_images.tar.gz", "README.md"],
    )
    return Path(local_dir) / "train.jsonl"


def images_available() -> bool:
    return all(_image_dir(folder).is_dir() and any(_image_dir(folder).iterdir())
               for folder in ("images", "teacher_images", "original_images"))


def maybe_extract_archives() -> None:
    for folder in ["images", "teacher_images", "original_images"]:
        target = data_root() / folder
        target.mkdir(exist_ok=True)
        if any(p.is_file() and not p.name.endswith(".tar.gz") for p in target.rglob("*")):
            continue
        archives = sorted(data_root().glob(f"{folder}.tar.gz*"))
        if not archives:
            continue
        if len(archives) == 1 and archives[0].suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(archives[0], "r:gz") as tf:
                tf.extractall(target, filter="data")
        else:
            merged = data_root() / f"{folder}.tar.gz"
            if not merged.exists():
                with merged.open("wb") as out:
                    for part in archives:
                        out.write(part.read_bytes())
            with tarfile.open(merged, "r:gz") as tf:
                tf.extractall(target, filter="data")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(folder: str, rel: str) -> Path:
    # JSONL paths include their top-level folder; avoid duplicating it.
    rel_path = Path(rel)
    if rel_path.parts and rel_path.parts[0] == folder:
        rel_path = Path(*rel_path.parts[1:])
    return _image_dir(folder) / rel_path


def pick_paths(rec: dict[str, Any]) -> dict[str, Path | None]:
    image = rec.get("images", [None])[0]
    teacher = rec.get("teacher_images", [None])[0]
    original = rec.get("original_images", [None])[0]
    return {
        "bbox_image": resolve_path("images", image) if image else None,
        "teacher_image": resolve_path("teacher_images", teacher) if teacher else None,
        "original_image": resolve_path("original_images", original) if original else None,
    }
