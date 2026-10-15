#!/usr/bin/env python3
"""Build a self-contained paired review packet from a frozen audit sample."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from full_campaign_layout import CONDITIONS, TRACKS
from verify_analysis_sources import verify_analysis_sources


CAMPAIGN = Path("full/runs/prompt-registry-v1")
NATIVE_ROOT = Path("official-scoring/vision-opd-native-v1")
EXTRACTION_ROOT = Path("judgment/runs/answer-extraction-v1")
EMBED_MAX_EDGE = 1920
EMBED_JPEG_QUALITY = 90


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_selected(path: Path, wanted: set[str], allow_retries: bool = False) -> dict[str, Any]:
    found: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("sample_uid")
            if uid not in wanted:
                continue
            if allow_retries:
                found.setdefault(uid, []).append(row)
            else:
                if uid in found:
                    raise ValueError(f"{path}: duplicate selected sample {uid}")
                found[uid] = row
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{path}: missing selected samples: {sorted(missing)[:5]}")
    return found


def terminal_decision(text: str) -> str:
    final_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    return final_line.casefold() if final_line.casefold() in {"yes", "no"} else "other"


def encode_review_image(path: Path) -> tuple[str, str, tuple[int, int], tuple[int, int]]:
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    with Image.open(BytesIO(original)) as source:
        original_size = source.size
        if "A" in source.getbands():
            rgba = source.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = source.convert("RGB")
        image.thumbnail((EMBED_MAX_EDGE, EMBED_MAX_EDGE), Image.Resampling.LANCZOS)
        embedded_size = image.size
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=EMBED_JPEG_QUALITY, optimize=True, progressive=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return uri, digest, original_size, embedded_size


def render_case(case: dict[str, Any]) -> str:
    esc = lambda value: html.escape("" if value is None else str(value), quote=True)
    image = case["image_data_uri"]
    image_html = (f'<img class="sample-image" onclick="window.open(this.src, \'_blank\', \'noopener\')" '
                  f'alt="Benchmark image for case {esc(case["case_index"])}" src="{image}">')
    condition_cards = []
    for item in case["conditions"]:
        parsed = item["extraction"]
        extraction_answer = parsed.get("selected_option") or parsed.get("answer_text") or "(no extracted answer)"
        attempts_html = ""
        if len(item["extraction_attempts"]) > 1:
            attempts_html = "<details><summary>Earlier extraction attempts / 早期抽取尝试</summary>" + "".join(
                f'<pre>{esc(attempt.get("judge_raw_text", ""))}</pre>'
                for attempt in item["extraction_attempts"][:-1]
            ) + "</details>"
        condition_cards.append(f'''<article class="condition">
<h4>{esc(item["condition"])} <span class="muted">· {esc(item["track"])}</span></h4>
<div class="scores"><span>Official exact-label / 官方精确标签：<b>{"Correct / 判对" if item["native_correct"] else "Not correct / 未判对"}</b></span>
<span>Terminal diagnostic / 终行诊断：<b>{esc(item["terminal_decision"])}</b> ({"Correct / 判对" if item["terminal_correct"] else "Not correct / 未判对"})</span>
<span>Extraction / 抽取：<b>{esc(parsed.get("status"))}</b> · {"Correct / 判对" if item["extraction_correct"] else "Not correct / 未判对"}</span>
<span>Native judge source / 原生裁判来源：{esc(item.get("native_judge_source"))}</span></div>
<p><b>Extracted answer / 抽取答案：</b> {esc(extraction_answer)}<br><b>Evidence / 证据：</b> {esc(parsed.get("evidence", ""))}</p>
<details><summary>Model response / 模型回答 ({esc(item["response_chars"])} chars, {esc(item.get("completion_tokens"))} tokens)</summary><pre>{esc(item["model_answer"])}</pre></details>
<details><summary>Full native judge output / 原生裁判完整输出 ({esc(item["native_judge_format"])})</summary><pre>{esc(item["native_judge_raw"])}</pre></details>
{attempts_html}
<div class="review"><b>Manual review / 人工复核</b>
<label>Extraction faithful? / 抽取忠实？ <select data-review="extraction_faithful"><option>Pending / 待审</option><option>Yes / 是</option><option>No / 否</option><option>Unclear / 不确定</option></select></label>
<label>Answer semantically correct? / 答案语义正确？ <select data-review="semantic_correct"><option>Pending / 待审</option><option>Yes / 是</option><option>No / 否</option><option>Unclear / 不确定</option></select></label>
<label>Judge rationale/label valid? / 裁判理由与标签有效？ <select data-review="judge_valid"><option>Pending / 待审</option><option>Yes / 是</option><option>No / 否</option><option>Unclear / 不确定</option></select></label>
<textarea data-review="note" placeholder="Evidence, uncertainty, or correction / 证据、不确定性或更正意见"></textarea></div>
</article>''')
    flags = ", ".join(case["all_flags"])
    return f'''<section class="case" id="case-{case["case_index"]}">
<h3>Case {case["case_index"]} <span class="muted">{esc(case["track"])} · selected stratum: {esc(case["primary_stratum"])}</span></h3>
<p class="flags">Matched audit flags / 命中审计层：{esc(flags)} · UID: {esc(case["sample_uid"])}</p>
<div class="question-grid">{image_html}<div><h4>Original benchmark input / 原始题目输入</h4><pre>{esc(case["query"])}</pre>
<p><b>Reference answer / 参考答案：</b> {esc(case["gold_answer"])}</p>
<p class="review-prompt">Review: Is the extracted answer faithful to the model response? Does each judge result follow the recorded scoring contract? / 请核查抽取答案是否忠实于模型回答，以及裁判结果是否符合记录的评分口径。</p></div></div>
<div class="conditions">{"".join(condition_cards)}</div></section>'''


def build_cases(sample_manifest: dict, data_root: Path, analysis_dir: Path) -> tuple[list[dict], list[dict]]:
    sample_rows = {}
    merged_path = analysis_dir / "sample_level_merged.csv"
    selected_by_track = {
        track: entry["selected"] for track, entry in sample_manifest["tracks_selection"].items()
    }
    for track, selected in selected_by_track.items():
        wanted = {item["sample_uid"] for item in selected}
        prepared = read_json(data_root / "prepared" / track / "benchmark.json")
        rows = {row["sample_uid"]: row for row in prepared if row.get("sample_uid") in wanted}
        if set(rows) != wanted:
            raise ValueError(f"{track}: prepared input does not cover the selected sample IDs")
        for uid in wanted:
            sample_rows[(track, uid)] = {"sample": next(x for x in selected if x["sample_uid"] == uid),
                                         "prepared": rows[uid], "conditions": {}}

    with merged_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["track"], row["sample_uid"])
            if key in sample_rows:
                sample_rows[key]["conditions"][row["condition"]] = row

    for track, selected in selected_by_track.items():
        wanted = {item["sample_uid"] for item in selected}
        for condition in CONDITIONS:
            tag = f"OPD-V-Qwen3.5-4B-{condition}_answer.jsonl"
            inference_path = data_root / CAMPAIGN / "outputs" / condition / track / tag
            inference = read_jsonl_selected(inference_path, wanted)
            native_path = (data_root / NATIVE_ROOT / condition / track / "judge" / track / tag)
            native_rows = read_json(native_path)
            native = {}
            for row in native_rows:
                uid = row.get("sample_uid")
                if uid in wanted:
                    if uid in native:
                        raise ValueError(f"{native_path}: duplicate selected sample {uid}")
                    native[uid] = row
            if set(native) != wanted:
                raise ValueError(f"{track}/{condition}: native judge outputs do not cover selected IDs")
            extraction_path = data_root / EXTRACTION_ROOT / track / f"{condition}.jsonl"
            attempts = read_jsonl_selected(extraction_path, wanted, allow_retries=True)
            for uid in wanted:
                merged = sample_rows[(track, uid)]["conditions"].get(condition)
                if merged is None:
                    raise ValueError(f"{track}/{condition}/{uid}: missing merged audit row")
                valid_attempts = [x for x in attempts[uid] if x.get("request_status") == "OK"]
                if not valid_attempts:
                    raise ValueError(f"{track}/{condition}/{uid}: no successful extraction attempt")
                output = inference[uid]
                judge = native[uid]
                sample_rows[(track, uid)]["conditions"][condition] = {
                    **merged, "condition": condition, "track": track,
                    "model_answer": output.get("model_answer", ""),
                    "native_judge_raw": judge.get("judge", ""),
                    "extraction": valid_attempts[-1], "extraction_attempts": attempts[uid],
                    "native_correct": int(merged["native_correct"]),
                    "terminal_correct": int(merged["terminal_correct"]),
                    "extraction_correct": int(merged["extraction_correct"]),
                    "terminal_decision": merged["native_judge_decision"],
                }

    cases = []
    used_sources = {}
    for index, (track, uid) in enumerate(
            ((track, item["sample_uid"]) for track in TRACKS for item in selected_by_track[track]), 1):
        data = sample_rows[(track, uid)]
        prepared = data["prepared"]
        if set(data["conditions"]) != set(CONDITIONS):
            raise ValueError(f"{track}/{uid}: incomplete five-condition coverage")
        images = prepared.get("images") or []
        if not images:
            raise ValueError(f"{track}/{uid}: no model-input image reference")
        image_path = Path(images[0]).resolve()
        image_path.relative_to(data_root.resolve())
        image_data_uri, image_digest, original_size, embedded_size = encode_review_image(image_path)
        ordered_conditions = [data["conditions"][condition] for condition in CONDITIONS]
        cases.append({
            "case_index": index, "track": track, "sample_uid": uid,
            "primary_stratum": data["sample"]["primary_stratum"],
            "all_flags": data["sample"]["all_flags"], "query": prepared.get("query", ""),
            "gold_answer": prepared.get("response", ""), "image_data_uri": image_data_uri,
            "image_original_sha256": image_digest, "image_original_size": original_size,
            "image_embedded_size": embedded_size,
            "conditions": ordered_conditions,
        })
        used_sources[str(image_path)] = image_digest
    return cases, [{"path": path, "sha256": digest} for path, digest in sorted(used_sources.items())]


def render_document(cases: list[dict], sample_manifest: dict, sources: list[dict], verification: dict) -> str:
    audit = {
        "sample_schema": sample_manifest["schema"], "seed": sample_manifest["seed"],
        "samples": len(cases), "condition_views": len(cases) * len(CONDITIONS),
        "sample_manifest_source": sample_manifest["source"], "verified_v4_source_count": verification["verified_count"],
        "embedded_image_policy": {"format": "JPEG", "max_edge_px": EMBED_MAX_EDGE,
                                  "quality": EMBED_JPEG_QUALITY, "original_bytes_hashed": True},
        "embedded_image_sha256": sources,
    }
    audit_json = html.escape(json.dumps(audit, ensure_ascii=False, indent=2))
    parts = [f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Work5 manual score audit / 工作5评分人工审阅</title><style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;color:#20242a;background:#fff;line-height:1.55;margin:0 auto;padding:28px;max-width:1180px}}
h1{{font-size:1.8rem}}h2{{margin-top:2.5rem;border-bottom:1px solid #d7dce2;padding-bottom:.35rem}}h3{{margin:0 0 .4rem}}
.muted{{color:#747d88;font-size:.88em;font-weight:400}}.notice{{background:#f4f7fa;border-left:4px solid #667d94;padding:14px 18px}}
.case{{border:1px solid #d7dce2;border-radius:10px;padding:18px;margin:24px 0;scroll-margin-top:20px}}
.question-grid{{display:grid;grid-template-columns:minmax(260px,40%) 1fr;gap:22px;align-items:start}}
.sample-image{{width:100%;height:auto;max-height:480px;object-fit:contain;border:1px solid #ddd;background:#fff;cursor:zoom-in}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;padding:12px;border-radius:6px;font: .9rem/1.5 ui-monospace,Consolas,monospace;max-height:480px;overflow:auto}}
.conditions{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}}.condition{{border:1px solid #e0e4e8;border-radius:8px;padding:12px}}
.condition h4{{margin:.1rem 0 .5rem}}.scores{{display:grid;gap:3px;font-size:.9rem}}.flags,.review-prompt{{color:#56616e;font-size:.9rem}}
details{{margin:.6rem 0}}summary{{cursor:pointer;color:#315b7d}}.review{{display:grid;gap:8px;border-top:1px solid #e0e4e8;margin-top:12px;padding-top:10px}}
.review label{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.review textarea{{min-height:54px;width:98%;font:inherit}}
@media(max-width:780px){{body{{padding:14px}}.question-grid,.conditions{{grid-template-columns:1fr}}}}
</style></head><body><h1>Work5: Manual score audit packet / 工作5：评分人工审阅样本</h1>
<p>This is a deterministic, risk-stratified qualitative audit sample—not a population accuracy estimate. Each case shows the original benchmark input and all five paired condition outputs. / 本材料是确定性风险分层定性审阅样本，不用于估计总体准确率。每个案例同时展示原评测输入及五种条件下的配对输出。</p>
<div class="notice"><b>Scoring views / 评分视角：</b>Official exact-label follows Vision-OPD <code>cal_acc.py</code>; terminal diagnostic reads only the final exact Yes/No line; text extraction transcribes the model response with verbatim evidence. These views are not interchangeable. / 官方精确标签遵循 Vision-OPD <code>cal_acc.py</code>；终行诊断只读取末尾精确Yes/No；文本抽取根据逐字证据转录模型回答。三者不可互换。<br>
<b>Review task / 审阅任务：</b>Check transcription fidelity, benchmark answer interpretation and native judge rationale; record uncertainty rather than inferring. / 核查答案转录、基准答案理解及原生裁判理由；不确定时应标记不确定，不作推断。</div>
<p>Cases: {len(cases)} · paired condition views: {len(cases) * len(CONDITIONS)} · tracks: six · seed: {sample_manifest["seed"]}</p>
<p><button onclick="downloadReview()">Download review decisions / 下载复核记录</button> <span class="muted">Answers are saved in this browser until exported. / 答案暂存在本浏览器，需导出保存。</span></p>
<nav><b>Tracks / 评测集：</b>''']
    parts.append(" · ".join(
        f'<a href="#track-{html.escape(track)}">{html.escape(track)}</a>' for track in TRACKS
    ))
    parts.append("</nav><details><summary>Audit provenance / 审计来源</summary><pre>" + audit_json + "</pre></details>")
    for track in TRACKS:
        parts.append(f'<h2 id="track-{html.escape(track)}">{html.escape(track)}</h2>')
        parts.extend(render_case(case) for case in cases if case["track"] == track)
    parts.append('''<script>
const storageKey = "work5-manual-audit-v1";
document.querySelectorAll(".condition").forEach((card, index) => {
  card.id = "review-" + index;
  card.querySelectorAll("[data-review]").forEach(field => {
    const key = card.id + ":" + field.dataset.review;
    const saved = localStorage.getItem(storageKey + ":" + key);
    if (saved !== null) field.value = saved;
    field.addEventListener("input", () => localStorage.setItem(storageKey + ":" + key, field.value));
    field.addEventListener("change", () => localStorage.setItem(storageKey + ":" + key, field.value));
  });
});
function downloadReview() {
  const rows = [...document.querySelectorAll(".condition")].map(card => ({
    case_title: card.closest(".case").querySelector("h3").innerText,
    condition: card.querySelector("h4").innerText,
    decisions: Object.fromEntries([...card.querySelectorAll("[data-review]")].map(field => [field.dataset.review, field.value]))
  }));
  const blob = new Blob([JSON.stringify({schema:"work5-manual-review-decisions-v1", rows}, null, 2)], {type:"application/json"});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "work5-manual-review-decisions-v1.json"; link.click(); URL.revokeObjectURL(link.href);
}
</script></body></html>''')
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample_manifest = read_json(args.sample_manifest)
    if sample_manifest.get("schema") != "phase2-work5-manual-audit-sample-v2":
        raise ValueError("Expected the corrected v2 manual-audit sample manifest")
    if sample_manifest.get("conditions") != list(CONDITIONS) or sample_manifest.get("tracks") != list(TRACKS):
        raise ValueError("Sample manifest does not match the frozen Work5 conditions/tracks")
    verification = verify_analysis_sources(args.analysis_dir)
    digest = hashlib.sha256()
    with (args.analysis_dir / "sample_level_merged.csv").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    merged_sha = digest.hexdigest()
    if merged_sha != sample_manifest.get("source", {}).get("sha256"):
        raise ValueError("Sample manifest source CSV hash no longer matches v4")
    cases, images = build_cases(sample_manifest, args.data_root, args.analysis_dir)
    if len(cases) != 60:
        raise ValueError(f"Expected 60 manual-review cases, got {len(cases)}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit packet: {args.output}")
    document = render_document(cases, sample_manifest, images, verification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"PASS: wrote self-contained audit packet for {len(cases)} cases / {len(cases) * 5} condition views")
    print(f"embedded_images={len(images)} max_edge={EMBED_MAX_EDGE}px quality={EMBED_JPEG_QUALITY} "
          f"output_bytes={args.output.stat().st_size} output={args.output}")


if __name__ == "__main__":
    main()
