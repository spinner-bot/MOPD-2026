#!/usr/bin/env python3
"""Loopback-only, zero-suggestion UI for frozen B2a blind human review."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from build_image_affordance_blind_calibration import (  # noqa: E402
    DECISIONS,
    EDITABLE_REVIEW_FIELDS,
    FAMILIES,
    FORBIDDEN_KEYS,
    HUMAN_FIELDS,
    load_jsonl,
    validate_completed_review,
)
from finalize_image_affordance_blind_review import (  # noqa: E402
    EXPECTED_PACKET_SHA256,
    EXPECTED_RECORDS,
    file_sha256,
    finalize,
    require_new_outputs,
    validate_original_packet,
    write_json_exclusive,
    write_jsonl_exclusive,
)


DEFAULT_INPUT = Path(
    "research/main/outputs/runs/20260914T0328-image-affordance-b2a-blind-calibration/"
    "outputs/blind-review-32.jsonl"
)
DEFAULT_IMAGE_ROOT = Path(
    "/leonardo_scratch/large/userexternal/rqian000/ICLR-2027-czy/data/"
    "m3vqa/materialized/images"
)
ARRAY_FIELDS = {
    "human_evidence_descriptions",
    "human_reasoning_steps",
    "human_counterfactual_required_evidence",
    "human_difficult_visual_evidence",
}
BOOLEAN_FIELDS = {
    "human_image_usable",
    "human_external_knowledge_needed",
    "human_verified",
}
TEXT_FIELDS = {
    "human_target_question",
    "human_reviewer",
    "human_reviewed_at",
    "human_notes",
}
ALLOWED_STATUSES = {
    "pending_blind_human_affordance_review",
    "human_blind_affordance_review_complete",
}
EXPECTED_PACKET_KEYS = {
    "calibration_record_id",
    "calibration_source_index",
    "image_candidate_id",
    "source_image_id",
    "source_dataset",
    "image_path",
    "image_sha256",
    "review_phase",
    "machine_proposal_exposed",
    "source_question_exposed",
    "review_status",
    "human_verified",
    *HUMAN_FIELDS,
}


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_loopback_host(host: str) -> None:
    if host != "127.0.0.1":
        raise ValueError("review server host must be exactly 127.0.0.1")


def validate_edit_updates(updates: Any) -> dict[str, Any]:
    if not isinstance(updates, dict):
        raise ValueError("updates must be a JSON object")
    unknown = sorted(set(updates) - EDITABLE_REVIEW_FIELDS)
    if unknown:
        raise ValueError("immutable or unknown fields in update: " + ",".join(unknown))
    normalized = {}
    for key, value in updates.items():
        if key in BOOLEAN_FIELDS:
            if value != "" and type(value) is not bool:
                raise ValueError(f"{key} must be unset, true, or false")
        elif key in ARRAY_FIELDS:
            if value != "" and not (
                isinstance(value, list)
                and all(isinstance(item, str) and item.strip() for item in value)
            ):
                raise ValueError(f"{key} must be unset or an array of non-empty strings")
        elif key == "human_canonical_answer":
            if value != "" and not (
                isinstance(value, str)
                or isinstance(value, list)
                and value
                and all(isinstance(item, str) and item.strip() for item in value)
            ):
                raise ValueError("human_canonical_answer must be unset, a string, or a string array")
        elif key == "human_affordance_decision":
            if value != "" and value not in DECISIONS:
                raise ValueError("invalid human_affordance_decision")
        elif key == "human_affordance_family":
            if value != "" and value not in FAMILIES:
                raise ValueError("invalid human_affordance_family")
        elif key == "review_status":
            if value not in ALLOWED_STATUSES:
                raise ValueError("invalid review_status")
        elif key in TEXT_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
        else:
            raise ValueError(f"unhandled editable field: {key}")
        normalized[key] = value
    return normalized


class ReviewStore:
    """Validated packet, typed draft persistence, and final export."""

    def __init__(
        self,
        *,
        input_path: Path,
        image_root: Path,
        draft_path: Path,
        export_prefix: Path,
        expected_input_sha256: str = EXPECTED_PACKET_SHA256,
        expected_records: int = EXPECTED_RECORDS,
    ) -> None:
        self.input_path = input_path.resolve(strict=True)
        self.image_root = image_root.resolve(strict=True)
        self.draft_path = draft_path.resolve(strict=False)
        self.export_prefix = export_prefix.resolve(strict=False)
        self._lock = threading.Lock()
        if self.draft_path == self.input_path:
            raise ValueError("draft path must not overwrite frozen input")

        originals = load_jsonl(self.input_path)
        self.input_sha256 = validate_original_packet(
            self.input_path,
            originals,
            expected_sha256=expected_input_sha256,
            expected_records=expected_records,
        )
        self.originals = originals
        self.expected_records = expected_records
        self._original_by_id: dict[str, dict] = {}
        self._images: dict[str, Path] = {}
        self._image_meta: dict[str, dict] = {}
        self._validate_originals()

        if self.draft_path.exists():
            self.rows = load_jsonl(self.draft_path)
            self._validate_draft(self.rows)
        else:
            self.rows = copy.deepcopy(self.originals)
            for row in self.rows:
                row["human_verified"] = ""
        self._index_by_id = {
            row["calibration_record_id"]: index for index, row in enumerate(self.rows)
        }

    def _validate_originals(self) -> None:
        for row in self.originals:
            record_id = str(row.get("calibration_record_id", "")).strip()
            if not record_id or record_id in self._original_by_id:
                raise ValueError("frozen packet has missing or duplicate calibration_record_id")
            if set(row) != EXPECTED_PACKET_KEYS:
                unexpected = sorted(set(row) - EXPECTED_PACKET_KEYS)
                missing = sorted(EXPECTED_PACKET_KEYS - set(row))
                raise ValueError(
                    f"frozen packet schema differs for {record_id}: "
                    f"unexpected={unexpected}, missing={missing}"
                )
            if FORBIDDEN_KEYS & row.keys():
                raise ValueError(f"frozen packet violates review firewall: {record_id}")
            if row.get("machine_proposal_exposed") is not False:
                raise ValueError(f"machine proposal exposure is not false: {record_id}")
            if row.get("source_question_exposed") is not False:
                raise ValueError(f"source question exposure is not false: {record_id}")
            if row.get("human_verified") is not False:
                raise ValueError(f"frozen packet is already human verified: {record_id}")
            if any(row.get(field) != "" for field in HUMAN_FIELDS):
                raise ValueError(f"frozen packet contains prefilled human fields: {record_id}")

            image_path = Path(str(row.get("image_path", ""))).resolve(strict=True)
            if not image_path.is_relative_to(self.image_root):
                raise ValueError(f"image path escapes configured root: {record_id}")
            if sha256_stream(image_path) != row.get("image_sha256"):
                raise ValueError(f"image SHA-256 mismatch: {record_id}")
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format
                mode = image.mode
            self._original_by_id[record_id] = row
            self._images[record_id] = image_path
            self._image_meta[record_id] = {
                "width": width,
                "height": height,
                "format": image_format,
                "mode": mode,
            }

    def _validate_draft(self, rows: list[dict]) -> None:
        by_id = {}
        for row in rows:
            record_id = str(row.get("calibration_record_id", "")).strip()
            if not record_id or record_id in by_id:
                raise ValueError("draft has missing or duplicate calibration_record_id")
            original = self._original_by_id.get(record_id)
            if original is None:
                raise ValueError(f"draft contains unknown calibration_record_id: {record_id}")
            if set(row) != set(original):
                unexpected = sorted(set(row) - set(original))
                missing = sorted(set(original) - set(row))
                raise ValueError(
                    f"draft schema differs for {record_id}: "
                    f"unexpected={unexpected}, missing={missing}"
                )
            immutable = (set(original) | set(row)) - EDITABLE_REVIEW_FIELDS
            changed = [key for key in sorted(immutable) if original.get(key) != row.get(key)]
            if changed:
                raise ValueError(f"draft changes immutable field {changed[0]}: {record_id}")
            if FORBIDDEN_KEYS & row.keys():
                raise ValueError(f"draft violates review firewall: {record_id}")
            missing_editable = sorted(EDITABLE_REVIEW_FIELDS - set(row))
            if missing_editable:
                raise ValueError(f"draft lacks editable field {missing_editable[0]}: {record_id}")
            validate_edit_updates({key: row.get(key) for key in EDITABLE_REVIEW_FIELDS if key in row})
            by_id[record_id] = row
        if set(by_id) != set(self._original_by_id):
            raise ValueError("draft must contain every frozen calibration record exactly once")

    def record(self, index: int) -> dict:
        if not 0 <= index < len(self.rows):
            raise ValueError("record index outside packet")
        row = self.rows[index]
        record_id = row["calibration_record_id"]
        visible = {
            "review_item_id": self.review_item_id(index),
            "review_status": row["review_status"],
            "human_verified": row["human_verified"],
            **{field: row[field] for field in HUMAN_FIELDS},
        }
        return {
            "index": index,
            "total": len(self.rows),
            "row": visible,
            "image_url": f"/image/{index}",
            "image": self._image_meta[record_id],
            "validation_errors": validate_completed_review(
                self._original_by_id[record_id], row
            ),
        }

    def review_item_id(self, index: int) -> str:
        if not 0 <= index < len(self.rows):
            raise ValueError("record index outside packet")
        return f"review-item-{index:04d}"

    def review_index(self) -> list[dict[str, Any]]:
        return [
            {"index": index, "review_item_id": self.review_item_id(index)}
            for index in range(len(self.rows))
        ]

    def save(self, record_id: str, updates: Any) -> dict:
        with self._lock:
            if record_id not in self._index_by_id:
                raise ValueError("unknown calibration_record_id")
            index = self._index_by_id[record_id]
            return self._save_at_index(index, updates)

    def save_index(self, index: int, updates: Any) -> dict:
        with self._lock:
            if not 0 <= index < len(self.rows):
                raise ValueError("record index outside packet")
            return self._save_at_index(index, updates)

    def _save_at_index(self, index: int, updates: Any) -> dict:
        changes = validate_edit_updates(updates)
        candidate = dict(self.rows[index])
        candidate.update(changes)
        self.rows[index] = candidate
        self._atomic_save_draft()
        return self.record(index)

    def _atomic_save_draft(self) -> None:
        if self.draft_path == self.input_path:
            raise ValueError("draft path must not overwrite frozen input")
        self.draft_path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=self.draft_path.name + ".tmp-",
            dir=self.draft_path.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                for row in self.rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.draft_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def finalize_export(self) -> dict:
        with self._lock:
            accepted, rejected, summary = finalize(self.originals, self.rows)
            if rejected or len(accepted) != self.expected_records:
                return {
                    "exported": False,
                    "summary": summary,
                    "rejected": rejected,
                }
            paths = {
                "accepted": Path(str(self.export_prefix) + ".accepted.jsonl"),
                "rejected": Path(str(self.export_prefix) + ".rejected.jsonl"),
                "summary": Path(str(self.export_prefix) + ".summary.json"),
            }
            require_new_outputs(list(paths.values()))
            summary["original_packet_sha256"] = self.input_sha256
            summary["reviewed_packet_sha256"] = file_sha256(self.draft_path)
            write_jsonl_exclusive(paths["accepted"], accepted)
            write_jsonl_exclusive(paths["rejected"], rejected)
            write_json_exclusive(paths["summary"], summary)
            return {
                "exported": True,
                "summary": summary,
                "paths": {key: str(value) for key, value in paths.items()},
            }

    def image_payload(self, record_id: str) -> tuple[bytes, str]:
        path = self._images.get(record_id)
        if path is None:
            raise ValueError("unknown image record")
        payload = path.read_bytes()
        expected_sha256 = self._original_by_id[record_id]["image_sha256"]
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("image SHA-256 changed after startup")
        suffix = path.suffix.lower()
        content_type = "image/png" if suffix == ".png" else "image/jpeg"
        return payload, content_type

    def image_payload_at(self, index: int) -> tuple[bytes, str]:
        if not 0 <= index < len(self.rows):
            raise ValueError("record index outside packet")
        return self.image_payload(self.rows[index]["calibration_record_id"])


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B2a 图像可供性盲审</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f0;color:#20221f}header{padding:12px 18px;background:#233128;color:#fff;position:sticky;top:0;z-index:2}.grid{display:grid;grid-template-columns:minmax(360px,1fr) minmax(430px,1fr);gap:16px;padding:16px}.card{background:#fff;border:1px solid #c8cdc7;border-radius:8px;padding:14px}.viewer{height:72vh;overflow:auto;background:#222;display:flex;align-items:center;justify-content:center}.viewer img.fit{max-width:100%;max-height:100%;width:auto;height:auto}.viewer img.native{max-width:none;max-height:none;width:auto;height:auto}label{display:block;font-weight:650;margin-top:10px}input,select,textarea,button{font:inherit}input,select,textarea{box-sizing:border-box;width:100%;padding:7px;margin-top:3px}textarea{min-height:66px}button{padding:7px 12px;margin:4px}.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}.muted{color:#606860;font-size:.9rem}.errors{white-space:pre-wrap;color:#8c1d18;background:#fff2f0;padding:8px}.ok{color:#176b32}.array-state{max-width:120px}@media(max-width:900px){.grid{grid-template-columns:1fr}.viewer{height:55vh}}
</style></head><body>
<header><div class="row"><button id="prev">上一条</button><select id="jump"></select><button id="next">下一条</button><span id="progress"></span><button id="save">保存草稿</button><button id="finalize">最终导出</button></div></header>
<main class="grid"><section class="card"><div class="row"><button id="fit">适合窗口</button><button id="native">100% 原生像素</button></div><div id="imageInfo" class="muted"></div><div class="viewer"><img id="image" class="fit" alt="完整审核图片"></div><div id="scale" class="muted"></div></section>
<section class="card"><div id="identity" class="muted"></div>
<label>图片是否可用<select id="human_image_usable"><option value="">未选择</option><option value="true">true</option><option value="false">false</option></select></label>
<label>可供性决定<select id="human_affordance_decision"><option value="">未选择</option><option>candidate</option><option>reject</option><option>ambiguous</option></select></label>
<label>目标问题<textarea id="human_target_question"></textarea></label>
<label>完整答案类型<select id="answer_type"><option value="">未选择</option><option value="string">string</option><option value="array">array[string]（每行一项）</option></select></label><textarea id="human_canonical_answer"></textarea>
<label>任务家族<select id="human_affordance_family"><option value="">未选择</option><option>R</option><option>PR</option></select></label>
<div id="arrays"></div>
<label>需要外部知识<select id="human_external_knowledge_needed"><option value="">未选择</option><option value="true">true</option><option value="false">false</option></select></label>
<label>审核者（人工填写）<input id="human_reviewer"></label><label>审核时间（人工填写，带时区 ISO 8601）<input id="human_reviewed_at"></label><label>备注<textarea id="human_notes"></textarea></label>
<label>审核状态<select id="review_status"><option>pending_blind_human_affordance_review</option><option>human_blind_affordance_review_complete</option></select></label>
<label>人工签名 human_verified<select id="human_verified"><option value="">未选择</option><option value="true">true</option><option value="false">false</option></select></label>
<div id="message"></div><div id="errors" class="errors"></div></section></main>
<script>
const arrayFields=[['human_evidence_descriptions','直接视觉证据'],['human_reasoning_steps','推理步骤'],['human_counterfactual_required_evidence','反事实必要证据'],['human_difficult_visual_evidence','困难视觉证据']];let current=0,total=0;
function el(id){return document.getElementById(id)}function tri(v){return v===''?'':String(v)}function lines(v){return Array.isArray(v)?v.join('\n'):''}
for(const [id,label] of arrayFields){el('arrays').insertAdjacentHTML('beforeend',`<label>${label}<span class="row"><select class="array-state" id="${id}_state"><option value="">未设置</option><option value="array">JSON array</option></select><textarea id="${id}"></textarea></span></label>`)}
async function load(i){const r=await fetch('/api/record?index='+i),d=await r.json();if(!r.ok){throw Error(d.error)}current=d.index;total=d.total;el('jump').selectedIndex=current;const x=d.row;el('progress').textContent=`${current+1}/${total}`;el('identity').textContent=x.review_item_id;el('image').src=d.image_url+'?v='+Date.now();el('imageInfo').textContent=`原生 ${d.image.width}×${d.image.height} · ${d.image.format} · ${d.image.mode}`;for(const id of ['human_image_usable','human_external_knowledge_needed','human_verified'])el(id).value=tri(x[id]);for(const id of ['human_affordance_decision','human_target_question','human_affordance_family','human_reviewer','human_reviewed_at','human_notes','review_status'])el(id).value=x[id];const a=x.human_canonical_answer;if(Array.isArray(a)){el('answer_type').value='array';el('human_canonical_answer').value=lines(a)}else if(a!==''){el('answer_type').value='string';el('human_canonical_answer').value=a}else{el('answer_type').value='';el('human_canonical_answer').value=''}for(const [id] of arrayFields){el(id+'_state').value=Array.isArray(x[id])?'array':'';el(id).value=lines(x[id])}showErrors(d.validation_errors);updateScale()}
function parseTri(id){const v=el(id).value;return v===''?'':v==='true'}function parseLines(id){if(el(id+'_state').value==='')return '';return el(id).value.split('\n').map(x=>x.trim()).filter(Boolean)}
function updates(){const t=el('answer_type').value,raw=el('human_canonical_answer').value;const ans=t===''?'':t==='string'?raw:raw.split('\n').map(x=>x.trim()).filter(Boolean);const u={human_image_usable:parseTri('human_image_usable'),human_affordance_decision:el('human_affordance_decision').value,human_target_question:el('human_target_question').value,human_canonical_answer:ans,human_affordance_family:el('human_affordance_family').value,human_external_knowledge_needed:parseTri('human_external_knowledge_needed'),human_reviewer:el('human_reviewer').value,human_reviewed_at:el('human_reviewed_at').value,human_notes:el('human_notes').value,review_status:el('review_status').value,human_verified:parseTri('human_verified')};for(const [id] of arrayFields)u[id]=parseLines(id);return u}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.error||JSON.stringify(d));return d}
async function save(){try{const d=await post('/api/save',{index:current,updates:updates()});el('message').textContent='草稿已原子保存';el('message').className='ok';showErrors(d.validation_errors)}catch(e){el('message').textContent=e.message;el('message').className='errors'}}
function showErrors(xs){el('errors').textContent=xs.length?'当前 finalizer 错误：\n'+xs.join('\n'):'当前记录通过 finalizer'}function updateScale(){const im=el('image');if(!im.naturalWidth)return;el('scale').textContent=`当前显示约 ${Math.round(im.clientWidth/im.naturalWidth*100)}%`}
el('prev').onclick=()=>load(Math.max(0,current-1));el('next').onclick=()=>load(Math.min(total-1,current+1));el('jump').onchange=()=>load(el('jump').selectedIndex);el('save').onclick=save;el('fit').onclick=()=>{el('image').className='fit';updateScale()};el('native').onclick=()=>{el('image').className='native';updateScale()};el('image').onload=updateScale;window.onresize=updateScale;
el('finalize').onclick=async()=>{try{await save();const d=await post('/api/finalize',{});if(!d.exported){el('message').textContent=`未完成：${d.summary.valid_signed_reviews}/${d.summary.original_records} valid`;el('message').className='errors'}else{el('message').textContent='第一阶段完成：'+JSON.stringify(d.paths);el('message').className='ok'}}catch(e){el('message').textContent=e.message;el('message').className='errors'}};
(async()=>{const r=await fetch('/api/index'),d=await r.json();for(const item of d.items){const o=document.createElement('option');o.value=item.index;o.textContent=item.review_item_id;el('jump').appendChild(o)}await load(0)})().catch(e=>{el('message').textContent=e.message;el('message').className='errors'});
</script></body></html>"""


def handler_for(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "B2aBlindReview/1"

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self'")
            self.end_headers()

        def _json(self, value: Any, status: int = 200) -> None:
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def _error(self, error: Exception, status: int = 400) -> None:
            self._json({"error": f"{type(error).__name__}: {error}"}, status)

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/":
                    payload = HTML_PAGE.encode("utf-8")
                    self._headers(200, "text/html; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif parsed.path == "/api/index":
                    self._json({"items": store.review_index()})
                elif parsed.path == "/api/record":
                    query = urllib.parse.parse_qs(parsed.query)
                    self._json(store.record(int(query.get("index", ["0"])[0])))
                elif parsed.path.startswith("/image/"):
                    index_text = urllib.parse.unquote(parsed.path[len("/image/") :])
                    payload, content_type = store.image_payload_at(int(index_text))
                    self._headers(200, content_type, len(payload))
                    self.wfile.write(payload)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as error:
                self._error(error)

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("POST requires application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 1024 * 1024:
                    raise ValueError("request body outside 0..1MiB")
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/save":
                    self._json(store.save_index(int(body.get("index", -1)), body.get("updates")))
                elif self.path == "/api/finalize":
                    self._json(store.finalize_export())
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as error:
                self._error(error)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--export-prefix", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        validate_loopback_host(args.host)
        store = ReviewStore(
            input_path=args.input,
            image_root=args.image_root,
            draft_path=args.draft,
            export_prefix=args.export_prefix,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    server = ThreadingHTTPServer((args.host, args.port), handler_for(store))
    print(f"B2a blind review: http://{args.host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
