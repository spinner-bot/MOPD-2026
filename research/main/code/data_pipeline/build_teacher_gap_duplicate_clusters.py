"""Build complete exact/dHash clusters for the bounded teacher-gap review pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    return sum(
        (pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8)
        for x in range(8)
    )


def unwrap(row: dict) -> dict:
    return row["candidate"] if isinstance(row.get("candidate"), dict) else row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dhash-distance", type=int, default=5)
    args = parser.parse_args()

    by_path = {}
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            candidate = unwrap(json.loads(line))
            path = str(candidate.get("image_original", ""))
            candidate_id = str(candidate.get("candidate_id", ""))
            if not path or not candidate_id:
                raise SystemExit("every row requires candidate_id and image_original")
            by_path.setdefault(path, []).append(candidate_id)
    images = []
    for value in sorted(by_path):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append({
            "image_original": value,
            "candidate_ids": sorted(by_path[value]),
            "sha256": sha256_file(path),
            "dhash64": f"{dhash64(path):016x}",
        })

    parent = list(range(len(images)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(images)):
        left_hash = int(images[left]["dhash64"], 16)
        for right in range(left + 1, len(images)):
            exact = images[left]["sha256"] == images[right]["sha256"]
            near = (left_hash ^ int(images[right]["dhash64"], 16)).bit_count() <= args.max_dhash_distance
            if exact or near:
                union(left, right)
    components = {}
    for index in range(len(images)):
        components.setdefault(find(index), []).append(index)
    clusters = []
    for members in components.values():
        if len(members) < 2:
            continue
        exact_count = len({images[index]["sha256"] for index in members}) == 1
        clusters.append({"kind": "exact" if exact_count else "near", "members": members})
    output = {
        "method": "sha256 exact plus exhaustive 64-bit dHash Hamming components",
        "max_dhash_distance": args.max_dhash_distance,
        "n_unique_image_paths": len(images),
        "images": images,
        "duplicate_clusters": clusters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "clusters": len(clusters)}, sort_keys=True))


if __name__ == "__main__":
    main()

