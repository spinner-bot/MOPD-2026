"""C2: M3VQA raw-field enrichment + human review packet generator.

Assigned by 1018 §5 (task C2, owner: claude code (local)).

Pure-CPU, stdlib-only, deterministic. No GPU/LLM calls and no semantic
conclusions: structural pre-screening handles ONLY deterministic defects
(corrupt lines, missing fields, duplicate source images, path traversal,
raw-join misses). Keyword-based `image_only` judgements are forbidden
(1018 §5.3.4).

Outputs (byte-identical across reruns with identical inputs/arguments;
LF newlines on every platform; sorted JSON keys; no timestamps):

1. enriched-tier-<n>.jsonl for each requested tier (default 128/512/1024):
   - tier 1 (128) == the fixed C1 shortlist, enriched with raw fields;
   - expansion tiers reuse seed=20260911, hop>=2 strata, unique
     source_image_id, and C1-style proportional allocation; nesting
     (128 subset-of 512 subset-of 1024) is guaranteed by per-stratum
     fixed permutations plus monotonic allocation clamping;
2. review-tier-<n>.csv / review-tier-<n>.jsonl: human review templates.
   EVERY human_* field is emitted empty; fabricating human review is
   forbidden (1018 §5.3.3);
3. structural-rejects.jsonl: deterministic exclusions with reason codes;
4. packet-summary.json + manifest.sha256.json: counts, hop distributions,
   SHA-256 of every input/output, nesting validation.

Evidence-text firewall (1017 §3.2): official M3VQA evidence TEXT never
enters any packet field (runtime self-check fails closed on leakage).
Locator metadata (urls, domains, section titles/ids, image ids, entity
names) lives ONLY inside `posthoc_analysis_only`, which must not be fed
to the blind solver, the entailment auditor, or any teacher condition.

Python >= 3.11 (Leonardo miniconda) / verified on 3.13 (local Windows).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_m3vqa_pilot_candidates import (  # noqa: E402
    allocate_strata,
    is_missing,
    sha256_file,
    url_domain,
)

RULES_VERSION = "c2-v3-answer-cardinality"
DEFAULT_SEED = 20260911
DEFAULT_TIERS = "128,512,1024"

HUMAN_FIELDS = (
    "human_knowledge_label",
    "human_task_family",
    "human_unique_answer",
    "human_full_image_valid",
    "human_required_evidence",
    "human_reasoning_depth",
    "human_reasoning_graph",
    "human_guides_ok",
    "human_options_ok",
    "human_answer_leakage",
    "human_external_fact_injected",
    "human_decision",
    "human_reviewer",
    "human_reviewed_at",
    "human_notes",
)
POSTHOC_WARNING = (
    "locator metadata only; NOT for blind solver / entailment auditor / "
    "teacher input (1017 §3.2); official evidence TEXT is excluded from this packet"
)
PACKET_NOTE = (
    "screening input only; NOT image-only, NOT visual-R ground truth (1018 §5.3 C2)"
)
RAW_SCREEN_FIELDS = ("question_type", "entity_num", "answer_evals")
RAW_POSTHOC_FIELDS = (
    "img_entity_names",
    "evidence_img_ids",
    "evidence_section_ids",
    "evidence_section_titles",
    "evidence_url_titles",
)
LEAK_CHECK_MIN_LEN = 20  # evidence strings >= this length must not appear in outputs


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def write_bytes(path: Path, content: str) -> None:
    """Platform-independent LF bytes (write_text would emit CRLF on Windows)."""
    path.write_bytes(content.encode("utf-8"))


def cid(row: dict) -> str:
    return str(row.get("candidate_id"))


def flatten_evidence(evidence) -> list[str]:
    """Flatten list[list[str]] / list[str] / str evidence into plain strings."""
    texts: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, list):
                texts.extend(str(x) for x in item)
            elif item is not None:
                texts.append(str(item))
    elif not is_missing(evidence):
        texts.append(str(evidence))
    return texts


def normalize_answer(value) -> str:
    return " ".join(str(value).casefold().split())


def reference_answer_spec(raw: dict) -> tuple[list[str], list[list[str]], str]:
    """Convert M3VQA's answer fields without confusing items and aliases.

    ``answer_evals`` is the official list of required answer groups. Usually
    it aligns one-to-one with ``answers``. In 20 source rows it contains one
    group whose members include several legacy ``answers`` values; those are
    aliases for one target and are deterministically collapsed to the first
    canonical value.
    """

    answers = raw.get("answers")
    groups = raw.get("answer_evals")
    if not isinstance(answers, list) or not answers or any(is_missing(v) for v in answers):
        raise ValueError("invalid_answers")
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(group, list) or not group or any(is_missing(v) for v in group) for group in groups)
    ):
        raise ValueError("invalid_answer_evals")
    canonical = [str(value).strip() for value in answers]
    aliases = [[str(value).strip() for value in group] for group in groups]
    if len(canonical) == len(aliases):
        targets = canonical
    elif len(aliases) == 1:
        normalized_aliases = {normalize_answer(value) for value in aliases[0]}
        targets = [next(
            (value for value in canonical if normalize_answer(value) in normalized_aliases),
            canonical[0],
        )]
    else:
        raise ValueError("unsupported_answer_cardinality_mismatch")
    for target, group in zip(targets, aliases):
        if normalize_answer(target) not in {normalize_answer(value) for value in group}:
            raise ValueError("canonical_answer_missing_from_alias_group")
    canonical_text = (
        targets[0]
        if len(targets) == 1
        else json.dumps(targets, ensure_ascii=False, separators=(",", ":"))
    )
    return targets, aliases, canonical_text


def load_jsonl(path: Path):
    """Yield (line_number, record_or_None, raw_line)."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line), line
            except json.JSONDecodeError:
                yield number, None, line


def image_is_safe(image_str: str, image_root: Path | None) -> str | None:
    """Deterministic path checks only. Returns reason code or None."""
    p = Path(image_str)
    if p.is_absolute():
        if ".." in p.parts:
            return "path_traversal"
        if image_root is not None:
            try:
                resolved = p.resolve()
                root = image_root.resolve()
                if root not in resolved.parents and resolved != root:
                    return "path_traversal"
            except OSError:
                return "path_traversal"
        return None
    parts = p.parts
    if ".." in parts:
        return "path_traversal"
    if image_root is not None:
        try:
            resolved = (image_root / p).resolve()
            root = image_root.resolve()
            if root not in resolved.parents and resolved != root:
                return "path_traversal"
        except OSError:
            return "path_traversal"
    return None


def stratum_order(pool: list[dict], seed: int, hop: int) -> list[dict]:
    """Fixed per-stratum permutation; depends only on (pool, seed, hop)."""
    ordered = sorted(pool, key=cid)
    rng = random.Random(seed + 1000 * int(hop))
    rng.shuffle(ordered)
    return ordered


def ordered_with_prev_first(perm: list[dict], prev_ids: set[str]) -> list[dict]:
    in_prev = [row for row in perm if cid(row) in prev_ids]
    rest = [row for row in perm if cid(row) not in prev_ids]
    return in_prev + rest


def select_expansion(strata: dict[int, list[dict]], seed: int, tier_n: int,
                     prev_ids: set[str]) -> tuple[list[dict], dict[int, int]]:
    """Nested expansion: per-stratum prefixes over permutations whose order
    places every previously selected id first. Monotonic clamping guarantees
    the previous tier is a subset; trims are deterministic (descending hop)."""
    alloc = allocate_strata(strata, tier_n)
    hops = sorted(strata)
    prev_by_hop = {hop: sum(1 for r in strata[hop] if cid(r) in prev_ids) for hop in hops}
    for hop in hops:
        alloc[hop] = max(alloc[hop], prev_by_hop[hop])
    excess = sum(alloc.values()) - tier_n
    for hop in sorted(hops, reverse=True):
        if excess <= 0:
            break
        room = alloc[hop] - prev_by_hop[hop]
        cut = min(room, excess)
        alloc[hop] -= cut
        excess -= cut
    if excess > 0:
        raise SystemExit(f"expansion infeasible for tier {tier_n}: cannot shrink below previous tier")
    selected: list[dict] = []
    for hop in hops:
        order = ordered_with_prev_first(stratum_order(strata[hop], seed, hop), prev_ids)
        selected.extend(order[: alloc[hop]])
    return selected, alloc


def enriched_record(norm: dict, raw: dict, tier: int, origin: str, seed: int) -> dict:
    urls = norm.get("evidence_urls") or []
    if not isinstance(urls, list):
        urls = [urls]
    targets, alias_groups, canonical_text = reference_answer_spec(raw)
    return {
        "candidate_id": norm.get("candidate_id"),
        "source_image_id": norm.get("source_image_id"),
        # W2A compatibility (teacher_gap_candidate_contract.validate_candidate):
        # flat top-level duplicates so enriched tiers are drop-in --input for
        # run_teacher_gap_candidate_stage.py; the renderer is allow-list based.
        "image_original": norm.get("image_original"),
        "final_question_source": norm.get("final_question_source"),
        "reference_answers": targets,
        "reference_answer_alias_groups": alias_groups,
        "canonical_answer_text": canonical_text,
        "question_hop": int(norm.get("reasoning_depth_candidate", 0)),
        "tier": tier,
        "tier_origin": origin,
        "screening_input": {
            "final_question_source": norm.get("final_question_source"),
            "reference_answers": targets,
            "reference_answer_alias_groups": alias_groups,
            "canonical_answer_text": canonical_text,
            "image_original": norm.get("image_original"),
            "task_family_candidate": norm.get("task_family_candidate"),
            "n_evidence_texts": len(flatten_evidence(norm.get("evidence"))),
            "n_evidence_urls": len(urls),
            **{field: raw.get(field) for field in RAW_SCREEN_FIELDS},
        },
        "posthoc_analysis_only": {
            "_warning": POSTHOC_WARNING,
            "evidence_urls": [str(u) for u in urls],
            "evidence_url_domains": sorted({url_domain(str(u)) for u in urls}),
            **{field: raw.get(field) for field in RAW_POSTHOC_FIELDS},
        },
        "provenance": {
            "raw_data_id": raw.get("data_id"),
            "source_record_id": norm.get("source_record_id"),
            "seed": seed,
            "rules_version": RULES_VERSION,
            "note": PACKET_NOTE,
        },
    }


def review_row(record: dict) -> dict:
    screen = record["screening_input"]
    row = {
        "candidate_id": record["candidate_id"],
        "tier": record["tier"],
        "question_hop": record["question_hop"],
        "tier_origin": record["tier_origin"],
        "image_path": screen["image_original"],
        "question": screen["final_question_source"],
        "reference_answers_json": dumps(screen["reference_answers"]),
        "answer_evals_json": dumps(screen.get("answer_evals")),
        "blind_context": "I,Q only (blind solver); reference_answers are auditor-only (1017 §3.2)",
    }
    row.update({field: "" for field in HUMAN_FIELDS})
    return row


REVIEW_COLUMNS = (
    "candidate_id", "tier", "question_hop", "tier_origin", "image_path",
    "question", "reference_answers_json", "answer_evals_json", "blind_context",
    *HUMAN_FIELDS,
)


def review_csv(records: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(REVIEW_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(review_row(record))
    return "\ufeff" + buffer.getvalue()  # BOM: Excel-friendly for human reviewers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True,
                        help="M3VQA raw source jsonl (quesions.jsonl)")
    parser.add_argument("--normalized", type=Path, required=True,
                        help="source-normalized M3VQA pool jsonl")
    parser.add_argument("--shortlist", type=Path, required=True,
                        help="C1 candidate-shortlist-128.jsonl (tier-1 definition)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tiers", default=DEFAULT_TIERS,
                        help="comma-separated ascending tier sizes; first must equal shortlist size")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="if given, verify image existence and containment (root-verified mode)")
    parser.add_argument("--aligned", type=Path, default=None,
                        help="optional frozen external expansion shortlist (e.g. Codex archive "
                             "20260911T1910-m3vqa-expanded-1024/outputs/candidate-shortlist-1024.jsonl); "
                             "emits an additional aligned enrichment+review packet over exactly these ids")
    args = parser.parse_args(argv)

    tiers = sorted({int(t) for t in str(args.tiers).split(",") if t.strip()})
    if len(tiers) < 2:
        raise SystemExit("--tiers needs at least two ascending tiers")

    # ---- load raw index ------------------------------------------------------
    raw_index: dict[str, dict] = {}
    raw_records = 0
    raw_duplicate_ids: list[str] = []
    for _, record, _line in load_jsonl(args.raw):
        if record is None:
            continue
        raw_records += 1
        data_id = str(record.get("data_id"))
        if data_id in raw_index:
            raw_duplicate_ids.append(data_id)
            continue  # keep first; duplicate logged in summary
        raw_index[data_id] = record

    # ---- structural pre-screen of normalized pool ----------------------------
    rejects: list[dict] = []
    eligible: dict[str, tuple[dict, dict]] = {}
    seen_images: set[str] = set()
    norm_total = 0
    image_mode = "root-verified" if args.image_root is not None else "lexical-only"

    def reject(key: str, reason: str, detail: str) -> None:
        rejects.append({"key": key, "reason_code": reason, "detail": detail, "image_check_mode": image_mode})

    parsed_norm: list[dict] = []
    for line_no, record, raw_line in load_jsonl(args.normalized):
        if record is None:
            reject(f"line:{line_no}", "corrupt_json_line", raw_line.strip()[:80])
            continue
        if record.get("source_dataset") != "M3VQA":
            continue
        norm_total += 1
        parsed_norm.append(record)
    # C1-faithful rule: screen in candidate_id order so duplicate source images
    # keep the smallest candidate_id deterministically, regardless of file order
    for record in sorted(parsed_norm, key=cid):
        key = cid(record)
        if is_missing(record.get("final_question_source")):
            reject(key, "missing_question", "")
            continue
        if is_missing(record.get("reference_answers")):
            reject(key, "missing_answer", "")
            continue
        image = record.get("image_original")
        if is_missing(image):
            reject(key, "missing_image_field", "")
            continue
        traversal = image_is_safe(str(image), args.image_root)
        if traversal:
            reject(key, traversal, str(image)[:120])
            continue
        if args.image_root is not None and not Path(str(image)).is_file():
            reject(key, "missing_image_file", str(image)[:120])
            continue
        record_id = record.get("source_record_id")
        if is_missing(record_id):
            record_id = key.removeprefix("m3vqa-")
        raw_rec = raw_index.get(str(record_id))
        if raw_rec is None:
            reject(key, "raw_join_miss", f"source_record_id={record_id}")
            continue
        image_id = record.get("source_image_id")
        if image_id in seen_images:
            reject(key, "duplicate_source_image_id", str(image_id))
            continue
        seen_images.add(image_id)
        eligible[key] = (record, raw_rec)

    if not eligible:
        raise SystemExit("no eligible M3VQA records after structural screening")

    # ---- tier 1: the fixed C1 shortlist --------------------------------------
    shortlist_ids: list[str] = []
    for _, record, _line in load_jsonl(args.shortlist):
        if record is not None:
            shortlist_ids.append(cid(record))
    if tiers[0] != len(shortlist_ids):
        raise SystemExit(
            f"first tier ({tiers[0]}) must equal shortlist size ({len(shortlist_ids)})")
    missing_from_eligible = [i for i in shortlist_ids if i not in eligible]
    if missing_from_eligible:
        raise SystemExit(
            f"fail-closed: {len(missing_from_eligible)} shortlist ids absent from "
            f"eligible pool (e.g. {missing_from_eligible[:5]})")

    strata: dict[int, list[dict]] = {}
    low_hop: list[dict] = []
    for key, (norm, _raw) in sorted(eligible.items()):
        hop = int(norm.get("reasoning_depth_candidate", 0))
        if hop >= 2:
            strata.setdefault(hop, []).append(norm)
        else:
            low_hop.append(norm)

    tier_selections: dict[int, list[dict]] = {}
    tier_meta: dict[int, dict] = {}
    tier_selections[tiers[0]] = [eligible[i][0] for i in shortlist_ids]
    tier_meta[tiers[0]] = {
        "origin_counts": {"c1_shortlist": len(shortlist_ids)},
        "expansion_alloc_by_hop": None,
        "filler_ids": [],
    }
    prev_ids: set[str] = set(shortlist_ids)

    for tier in tiers[1:]:
        # carry previous-tier hop<2 filler rows (absent from strata) to keep nesting
        carried_lowhop = [r for r in tier_selections[sorted(tier_selections)[-1]]
                          if int(r.get("reasoning_depth_candidate", 0)) < 2]
        selected, alloc = select_expansion(strata, args.seed, tier - len(carried_lowhop), prev_ids)
        selected = carried_lowhop + selected
        filler: list[str] = []
        if len(selected) < tier and low_hop:
            chosen = {cid(r) for r in selected}
            for row in sorted(low_hop, key=cid):
                if len(selected) >= tier:
                    break
                if cid(row) not in chosen:
                    selected.append(row)
                    chosen.add(cid(row))
                    filler.append(cid(row))
        tier_selections[tier] = selected
        tier_meta[tier] = {
            "origin_counts": {
                "carried_from_previous": sum(1 for r in selected if cid(r) in prev_ids),
                "expansion": sum(1 for r in selected if cid(r) not in prev_ids and r not in low_hop),
                "filler_hop_lt2": len(filler),
            },
            "expansion_alloc_by_hop": {str(k): v for k, v in sorted(alloc.items())},
            "filler_ids": filler,
        }
        prev_ids = {cid(r) for r in selected}

    # ---- build outputs --------------------------------------------------------
    outputs: dict[str, str] = {}
    tier_summary: dict[str, dict] = {}
    prev_set: set[str] = set()
    nesting: dict[str, bool] = {}
    leak_failures: list[str] = []

    for tier in tiers:
        rows = sorted(tier_selections[tier],
                      key=lambda r: (int(r.get("reasoning_depth_candidate", 0)), cid(r)))
        records = []
        for norm in rows:
            key = cid(norm)
            _n, raw_rec = eligible[key]
            origin = "c1_shortlist" if key in set(shortlist_ids) else (
                "filler_hop_lt2" if key in set(tier_meta[tier]["filler_ids"]) else "expansion")
            record = enriched_record(norm, raw_rec, tier, origin, args.seed)
            # runtime evidence-text firewall (fail closed)
            dump = dumps(record)
            for text in flatten_evidence(norm.get("evidence")):
                if len(text) >= LEAK_CHECK_MIN_LEN and text in dump:
                    leak_failures.append(key)
                    break
            records.append(record)
        enriched_name = f"enriched-tier-{tier}.jsonl"
        csv_name = f"review-tier-{tier}.csv"
        jsonl_name = f"review-tier-{tier}.jsonl"
        outputs[enriched_name] = "".join(dumps(r) + "\n" for r in records)
        outputs[csv_name] = review_csv(records)
        outputs[jsonl_name] = "".join(dumps(review_row(r)) + "\n" for r in records)
        ids_now = {cid(r) for r in rows}
        if prev_set:
            nesting[f"{sorted(t for t in tiers if t < tier)[-1]}_in_{tier}"] = prev_set.issubset(ids_now)
        prev_set = ids_now
        tier_summary[str(tier)] = {
            "count": len(records),
            "hop_distribution": {str(k): v for k, v in sorted(
                Counter(int(r.get("reasoning_depth_candidate", 0)) for r in rows).items())},
            **tier_meta[tier],
            "sha256_enriched": hashlib.sha256(outputs[enriched_name].encode("utf-8")).hexdigest(),
            "sha256_review_csv": hashlib.sha256(outputs[csv_name].encode("utf-8")).hexdigest(),
            "sha256_review_jsonl": hashlib.sha256(outputs[jsonl_name].encode("utf-8")).hexdigest(),
        }
    # ---- optional aligned packet over a frozen external expansion -------------
    aligned_summary = None
    if args.aligned is not None:
        aligned_ids: list[str] = []
        for _, record, _line in load_jsonl(args.aligned):
            if record is not None:
                aligned_ids.append(cid(record))
        aligned_missing = [i for i in aligned_ids if i not in eligible]
        if aligned_missing:
            raise SystemExit(
                f"fail-closed: {len(aligned_missing)} aligned ids absent from eligible pool "
                f"(e.g. {aligned_missing[:5]})")
        c1_set = set(shortlist_ids)
        aligned_rows = sorted((eligible[i][0] for i in aligned_ids),
                              key=lambda r: (int(r.get("reasoning_depth_candidate", 0)), cid(r)))
        aligned_records = []
        for norm in aligned_rows:
            key = cid(norm)
            _n, raw_rec = eligible[key]
            record = enriched_record(norm, raw_rec, f"aligned-{len(aligned_ids)}",
                                     "codex_frozen_expansion", args.seed)
            record["in_c1_shortlist"] = key in c1_set
            record["provenance"]["aligned_source"] = str(args.aligned)
            dump = dumps(record)
            for text in flatten_evidence(norm.get("evidence")):
                if len(text) >= LEAK_CHECK_MIN_LEN and text in dump:
                    leak_failures.append(key)
                    break
            aligned_records.append(record)
        a_enriched = f"enriched-aligned-{len(aligned_ids)}.jsonl"
        a_csv = f"review-aligned-{len(aligned_ids)}.csv"
        a_jsonl = f"review-aligned-{len(aligned_ids)}.jsonl"
        outputs[a_enriched] = "".join(dumps(r) + "\n" for r in aligned_records)
        outputs[a_csv] = review_csv(aligned_records)
        outputs[a_jsonl] = "".join(dumps(review_row(r)) + "\n" for r in aligned_records)
        aligned_summary = {
            "source": str(args.aligned),
            "source_sha256": sha256_file(args.aligned),
            "count": len(aligned_records),
            "overlap_with_c1_shortlist": sum(1 for r in aligned_records if r["in_c1_shortlist"]),
            "new_vs_c1_shortlist": sum(1 for r in aligned_records if not r["in_c1_shortlist"]),
            "hop_distribution": {str(k): v for k, v in sorted(Counter(
                r["question_hop"] for r in aligned_records).items())},
            "sha256_enriched": hashlib.sha256(outputs[a_enriched].encode("utf-8")).hexdigest(),
            "sha256_review_csv": hashlib.sha256(outputs[a_csv].encode("utf-8")).hexdigest(),
            "sha256_review_jsonl": hashlib.sha256(outputs[a_jsonl].encode("utf-8")).hexdigest(),
        }

    if leak_failures:
        raise SystemExit(f"LEAK CHECK FAILED: evidence text in packet for {leak_failures[:5]}")
    if any(not v for v in nesting.values()):
        raise SystemExit(f"NESTING CHECK FAILED: {nesting}")

    rejects_sorted = sorted(rejects, key=lambda r: (r["reason_code"], str(r["key"])))
    outputs["structural-rejects.jsonl"] = "".join(dumps(r) + "\n" for r in rejects_sorted)

    summary = {
        "packet": "C2-m3vqa-teacher-gap-review-packet",
        "spec": "1018 §5.3 C2; deterministic; no semantic judgement; evidence TEXT excluded",
        "rules_version": RULES_VERSION,
        "seed": args.seed,
        "tiers_requested": tiers,
        "inputs": {
            "raw": {"path": str(args.raw), "sha256": sha256_file(args.raw), "records": raw_records},
            "normalized": {"path": str(args.normalized), "sha256": sha256_file(args.normalized),
                           "m3vqa_records": norm_total},
            "shortlist": {"path": str(args.shortlist), "sha256": sha256_file(args.shortlist),
                          "ids": len(shortlist_ids)},
        },
        "raw_duplicate_data_ids": {"count": len(raw_duplicate_ids), "kept": "first occurrence"},
        "image_check": {"mode": image_mode, "root": str(args.image_root) if args.image_root else None},
        "structural_rejects": dict(Counter(r["reason_code"] for r in rejects_sorted)),
        "eligible_after_screen": len(eligible),
        "eligible_hop_distribution": {str(k): len(v) for k, v in sorted(strata.items())},
        "eligible_low_hop_pool": len(low_hop),
        "tiers": tier_summary,
        "aligned_packet": aligned_summary,
        "nesting_verified": nesting,
        "evidence_text_leak_check": "passed",
        "human_fields": list(HUMAN_FIELDS),
        "human_fields_state": "all empty by construction; fabrication forbidden (1018 §5.3.3)",
    }
    outputs["packet-summary.json"] = json.dumps(summary, ensure_ascii=False, indent=2,
                                                sort_keys=True) + "\n"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_manifest = {}
    for name in sorted(outputs):
        path = args.output_dir / name
        write_bytes(path, outputs[name])
        content_bytes = outputs[name].encode("utf-8")
        file_manifest[name] = {
            "sha256": hashlib.sha256(content_bytes).hexdigest(),
            "bytes": len(content_bytes),
            "records": content_bytes.count(b"\n"),
        }
    manifest = {"rules_version": RULES_VERSION, "seed": args.seed, "files": file_manifest}
    write_bytes(args.output_dir / "manifest.sha256.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    checks = {
        "tiers": {t: tier_summary[str(t)]["count"] for t in tiers},
        "nesting": nesting,
        "rejects": summary["structural_rejects"],
        "leak_check": "passed",
    }
    print(json.dumps({"checks": checks, "files": file_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
