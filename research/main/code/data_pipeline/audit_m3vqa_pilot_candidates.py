"""C1 audit: M3VQA field inventory, external-knowledge clues, deterministic shortlist.

Assigned by 1014-夜间协同规划 §4.2 (task C1, owner: claude code (local)).

Pure-CPU, stdlib-only, no semantic final judgements and no model calls.
Produces, for the M3VQA subset of a source-normalized candidate pool:

1. field inventory: key presence / missing rates / nested structure;
2. distributions: question_hop, answer form, question length, evidence text
   and URL counts, URL domains (deterministic external-knowledge CLUES only);
3. a fixed-seed 128-candidate shortlist, stratified by hop (>=2 prioritized),
   excluding records with missing image/question/answer and duplicate
   source_image_id. The shortlist is screening INPUT for human/model review;
   it is NOT an image-only label and NOT visual-R ground truth;
4. a deterministic representative sample (12-20 records) for the report.

Determinism contract: identical inputs + identical arguments => byte-identical
outputs (no timestamps, sorted keys, fixed seed). Verify by rerunning and
comparing SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_SEED = 20260911
SHORTLIST_N = 128
WIKIMEDIA_MARKERS = ("wikipedia", "wikimedia", "wikidata")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def field_shape(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        inner = Counter(field_shape(item) for item in value)
        return "list[" + "|".join(f"{k}x{v}" for k, v in sorted(inner.items())) + "]"
    if isinstance(value, dict):
        return "dict{" + ",".join(sorted(value.keys())) + "}"
    return type(value).__name__


def url_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return "<unparseable>"
    return netloc or "<relative-or-malformed>"


def quantiles(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n, "min": ordered[0], "p25": ordered[n // 4],
        "median": ordered[n // 2], "p75": ordered[3 * n // 4], "max": ordered[-1],
    }


def allocate_strata(strata_pools: dict[int, list[dict]], total: int) -> dict[int, int]:
    """Proportional allocation with largest remainder; deficits redistributed
    deterministically to strata with surplus capacity, in ascending hop order."""
    eligible = sum(len(pool) for pool in strata_pools.values())
    if eligible <= total:
        return {hop: len(pool) for hop, pool in strata_pools.items()}
    hops = sorted(strata_pools)
    raw = {hop: len(strata_pools[hop]) * total / eligible for hop in hops}
    alloc = {hop: int(raw[hop]) for hop in hops}
    remainder = total - sum(alloc.values())
    for hop in sorted(hops, key=lambda h: (-(raw[h] - int(raw[h])), h)):
        if remainder <= 0:
            break
        alloc[hop] += 1
        remainder -= 1
    for hop in hops:  # capacity fix, ascending
        overflow = alloc[hop] - len(strata_pools[hop])
        if overflow > 0:
            alloc[hop] = len(strata_pools[hop])
            for other in hops:
                if overflow <= 0:
                    break
                room = len(strata_pools[other]) - alloc[other]
                give = min(room, overflow)
                alloc[other] += give
                overflow -= give
    return alloc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="source-normalized.jsonl (full candidate pool)")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for audit outputs (created if absent)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="fixed sampling seed (default: %(default)s)")
    parser.add_argument("--n", type=int, default=SHORTLIST_N, help="shortlist size (default: %(default)s)")
    parser.add_argument("--representatives", type=int, default=16, help="representative record count, 12-20 (default: %(default)s)")
    parser.add_argument("--skip-image-check", action="store_true", help="do not stat image files (use when auditing off-host)")
    args = parser.parse_args()
    if not 12 <= args.representatives <= 20:
        raise SystemExit("--representatives must be within 12..20")

    rows, total_pool = [], 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_pool += 1
            record = json.loads(line)
            if record.get("source_dataset") == "M3VQA":
                rows.append(record)
    if not rows:
        raise SystemExit("no M3VQA records found in input")

    # ---- 1. field inventory -------------------------------------------------
    keys = sorted({key for row in rows for key in row})
    presence, missing, shapes = {}, {}, {}
    for key in keys:
        have = [row for row in rows if key in row]
        presence[key] = len(have)
        missing[key] = sum(1 for row in have if is_missing(row.get(key)))
        shapes[key] = dict(Counter(field_shape(row.get(key)) for row in have).most_common())

    # ---- 2. distributions and deterministic external-knowledge clues --------
    hops = Counter(int(row.get("reasoning_depth_candidate", 0)) for row in rows)
    question_lengths = [len(str(row.get("final_question_source", ""))) for row in rows]
    answer_counts, answer_lengths = [], []
    for row in rows:
        answers = row.get("reference_answers")
        if isinstance(answers, list):
            answer_counts.append(len(answers))
            answer_lengths.extend(len(str(a)) for a in answers)
        else:
            answer_counts.append(0 if is_missing(answers) else 1)
            answer_lengths.append(len(str(answers or "")))
    evidence_counts, url_counts, domains, wikimedia_rows = [], [], Counter(), 0
    evidence_shapes = Counter()
    for row in rows:
        evidence = row.get("evidence")
        texts = evidence if isinstance(evidence, list) else [evidence] if not is_missing(evidence) else []
        evidence_shapes.update(field_shape(item) for item in texts)
        evidence_counts.append(len(texts))
        urls = row.get("evidence_urls")
        url_list = urls if isinstance(urls, list) else [urls] if not is_missing(urls) else []
        url_counts.append(len(url_list))
        row_domains = {url_domain(str(u)) for u in url_list}
        domains.update(row_domains)
        if any(any(marker in d for marker in WIKIMEDIA_MARKERS) for d in row_domains):
            wikimedia_rows += 1

    # ---- 3. exclusions and stratified shortlist ------------------------------
    exclusion_reasons = Counter()
    eligible, seen_images = [], set()
    for row in sorted(rows, key=lambda r: str(r.get("candidate_id"))):
        if is_missing(row.get("final_question_source")):
            exclusion_reasons["missing_question"] += 1
            continue
        if is_missing(row.get("reference_answers")):
            exclusion_reasons["missing_answer"] += 1
            continue
        if not args.skip_image_check:
            image = row.get("image_original")
            if is_missing(image) or not Path(str(image)).is_file():
                exclusion_reasons["missing_image_file"] += 1
                continue
        image_id = row.get("source_image_id")
        if image_id in seen_images:
            exclusion_reasons["duplicate_source_image_id"] += 1
            continue
        seen_images.add(image_id)
        eligible.append(row)
    strata_pools: dict[int, list[dict]] = {}
    for row in eligible:
        hop = int(row.get("reasoning_depth_candidate", 0))
        if hop >= 2:
            strata_pools.setdefault(hop, []).append(row)
    low_hop_pool = [row for row in eligible if int(row.get("reasoning_depth_candidate", 0)) < 2]
    allocation = allocate_strata(strata_pools, args.n)
    rng = random.Random(args.seed)
    shortlist = []
    for hop in sorted(allocation):
        pool = sorted(strata_pools[hop], key=lambda r: str(r.get("candidate_id")))
        chosen = sorted(rng.sample(pool, allocation[hop]), key=lambda r: str(r.get("candidate_id")))
        for rank, row in enumerate(chosen):
            shortlist.append({
                "candidate_id": row.get("candidate_id"),
                "source_image_id": row.get("source_image_id"),
                "question_hop": hop,
                "stratum_rank": rank,
                "final_question_source": row.get("final_question_source"),
                "reference_answers": row.get("reference_answers"),
                "n_evidence_texts": len(row.get("evidence") or []),
                "n_evidence_urls": len(row.get("evidence_urls") or []),
                "image_original": row.get("image_original"),
                "shortlist_note": "screening input only; NOT image-only, NOT visual-R ground truth (1014 §4.2 C1)",
            })
    if len(shortlist) < args.n and low_hop_pool:
        fill = sorted(low_hop_pool, key=lambda r: str(r.get("candidate_id")))
        for row in fill[: args.n - len(shortlist)]:
            shortlist.append({
                "candidate_id": row.get("candidate_id"),
                "source_image_id": row.get("source_image_id"),
                "question_hop": int(row.get("reasoning_depth_candidate", 0)),
                "stratum_rank": -1,
                "final_question_source": row.get("final_question_source"),
                "reference_answers": row.get("reference_answers"),
                "n_evidence_texts": len(row.get("evidence") or []),
                "n_evidence_urls": len(row.get("evidence_urls") or []),
                "image_original": row.get("image_original"),
                "shortlist_note": "hop<2 filler; screening input only (1014 §4.2 C1)",
            })

    # ---- 4. deterministic representatives -----------------------------------
    per_hop = max(1, args.representatives // max(1, len(allocation)))
    representatives = []
    for hop in sorted(allocation):
        members = [item for item in shortlist if item["question_hop"] == hop]
        representatives.extend(members[:per_hop])
    representatives = representatives[: args.representatives]
    rep_ids = {item["candidate_id"] for item in representatives}
    representative_rows = []
    for item in representatives:
        row = next(r for r in rows if r.get("candidate_id") == item["candidate_id"])
        evidence = row.get("evidence") or []
        urls = row.get("evidence_urls") or []
        representative_rows.append({
            "candidate_id": item["candidate_id"],
            "question_hop": item["question_hop"],
            "final_question_source": row.get("final_question_source"),
            "reference_answers": row.get("reference_answers"),
            "evidence_texts": [str(e)[:300] for e in (evidence if isinstance(evidence, list) else [evidence])][:4],
            "evidence_url_domains": sorted({url_domain(str(u)) for u in (urls if isinstance(urls, list) else [urls])}),
            "in_shortlist": item["candidate_id"] in rep_ids,
        })

    # ---- 5. write deterministic outputs --------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "audit": "C1-m3vqa-field-and-shortlist",
        "spec": "1014 §4.2 C1; deterministic; no semantic final judgement",
        "input": str(args.input),
        "seed": args.seed,
        "shortlist_n_requested": args.n,
        "pool_total_records": total_pool,
        "m3vqa_records": len(rows),
        "field_presence": presence,
        "field_missing_when_present": missing,
        "field_shapes": shapes,
        "question_hop_distribution": {str(k): v for k, v in sorted(hops.items())},
        "question_length_chars": quantiles(question_lengths),
        "answers_per_record": dict(Counter(answer_counts)),
        "answer_length_chars": quantiles(answer_lengths),
        "evidence_text_shapes": dict(evidence_shapes),
        "evidence_texts_per_record": quantiles(evidence_counts),
        "evidence_urls_per_record": dict(Counter(url_counts)),
        "top_url_domains": dict(domains.most_common(15)),
        "records_with_wikimedia_url": wikimedia_rows,
        "eligible_after_exclusions": len(eligible),
        "exclusion_reasons": dict(exclusion_reasons),
        "eligible_hop_distribution": {str(k): len(v) for k, v in sorted(strata_pools.items())},
        "shortlist_allocation_by_hop": {str(k): v for k, v in sorted(allocation.items())},
        "shortlist_size": len(shortlist),
        "external_knowledge_clue_assessment": {
            "deterministic_clues_available": [
                "evidence_urls non-empty + domain (wikimedia marker => knowledge-lookup provenance clue)",
                "evidence text count/length (multi-evidence retrieval pattern)",
                "question_hop (retrieval hops != visual reasoning depth; weak clue only)",
            ],
            "not_determinable_from_fields": [
                "whether the QUESTION itself requires external knowledge to answer",
                "whether the answer is uniquely verifiable from the image",
                "whether the minimal evidence path contains non-omittable visual relation steps",
            ],
        },
    }
    outputs = {
        "field_audit.json": json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "candidate-shortlist-128.jsonl": "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in shortlist),
        "representatives.jsonl": "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in representative_rows),
    }
    manifest = {"seed": args.seed, "files": {}}
    for name, content in sorted(outputs.items()):
        path = args.output_dir / name
        path.write_text(content, encoding="utf-8")
        manifest["files"][name] = {"sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(), "records": content.count("\n")}
    manifest_path = args.output_dir / "manifest.sha256.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- 6. acceptance self-checks -------------------------------------------
    ids = [item["candidate_id"] for item in shortlist]
    image_ids = [item["source_image_id"] for item in shortlist]
    checks = {
        "shortlist_count": len(shortlist),
        "unique_candidate_ids": len(set(ids)) == len(ids),
        "unique_source_image_ids": len(set(image_ids)) == len(image_ids),
        "representative_count": len(representative_rows),
    }
    print(json.dumps({"checks": checks, "manifest": manifest["files"]}, ensure_ascii=False, indent=2))
    if not (checks["unique_candidate_ids"] and checks["unique_source_image_ids"]):
        raise SystemExit("ACCEPTANCE FAILED: duplicate ids in shortlist")


if __name__ == "__main__":
    main()
