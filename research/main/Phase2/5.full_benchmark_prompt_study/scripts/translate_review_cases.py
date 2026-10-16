#!/usr/bin/env python3
"""Translate the English review-packet fields into Chinese with a local LLM.

Consumes the per-case files written by extract_review_english.py and emits the
Chinese counterpart in the layout expected by add_manual_review_bilingual_layer.py
(one case-NNN.json per case; long model responses go to a side file).

Translation standard follows report/translation-work/TRANSLATION_PROTOCOL.md:
verbatim coverage, no omission/abridgement, chunk only at paragraph boundaries,
preserve negations, option letters, numbers and repetition.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SYSTEM_PROMPT = """你是学术论文数据集的逐字翻译器。把用户给出的英文**完整直译**为中文。

规则（全部必须遵守）：
1. 逐字完整翻译，不得跳过、摘要、删节或只译部分段落；重复内容照常翻译。
2. 保留原文的命题、限定、否定、指代、推测语气、矛盾、自我修正、重复。
3. 不得替作者消除矛盾、纠错、补足论证，或改写成更合理的说法。
4. 选项字母（A）（B）…、数字、公式、变量名、代码标识符原样保留。
5. 保持段落边界与列表顺序；Markdown 标记（**加粗**、# 标题、- 列表）保留。
6. 只输出译文本身，不要任何解释、前言、引用或"以下是翻译"之类的话。
7. 若输入是 JSON 字符串，保持 JSON 结构，只翻译其中的字符串值。"""

CHUNK_CHARS = 3500
SHORT_FIELD_LIMIT = 120        # 短字段直接内联
RESPONSE_FILE_LIMIT = 2000     # 模型回答超过此长度则外置为文件


def split_chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split the text into chunks such that "".join(chunks) == text exactly.

    Chunks carry their own trailing separator, so reassembly is plain
    concatenation and nothing can be lost, duplicated or re-ordered.
    Boundaries fall on paragraph breaks wherever possible; only a paragraph
    larger than ``limit`` is cut mid-text.
    """
    if len(text) <= limit:
        return [text]
    units = [p + "\n\n" for p in text.split("\n\n")]
    if units:
        units[-1] = units[-1][:-2]          # 末段无尾分隔符
    chunks: list[str] = []
    buf = ""
    for unit in units:
        if buf and len(buf) + len(unit) > limit:
            chunks.append(buf)
            buf = ""
        buf += unit
        while len(buf) > limit:             # 单段超长：硬切
            chunks.append(buf[:limit])
            buf = buf[limit:]
    if buf:
        chunks.append(buf)
    return chunks


def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=60)
    args = ap.parse_args()

    # ---- 1. 收集全部待译片段 ------------------------------------------------
    jobs: list[dict] = []          # 每个 job 对应一次模型调用
    plan: list[dict] = []          # 每例的重组计划
    for path in sorted(args.in_dir.glob("en-case-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        idx = case["case_index"]
        if not (args.first <= idx <= args.last):
            continue
        entry = {"case_index": idx, "case_title": case["case_title"],
                 "reference_answer": case["reference_answer"], "fields": {}, "conditions": []}
        for field, value in (("stratum", case["stratum"]), ("flags", case["flags"]),
                             ("question", case["question"])):
            chunks = split_chunks(value)
            entry["fields"][field] = len(jobs)
            for c in chunks:
                jobs.append({"case": idx, "text": c})
        for cond in case["conditions"]:
            ce = {"condition": cond["condition"], "chunks": {}}
            for field in ("scores", "extraction_evidence", "model_response", "judge_output"):
                chunks = split_chunks(cond[field])
                ce["chunks"][field] = (len(jobs), len(chunks))
                for c in chunks:
                    jobs.append({"case": idx, "text": c})
            entry["conditions"].append(ce)
        plan.append(entry)

    print(f"待译 {len(plan)} 例，共 {len(jobs)} 个片段")

    # ---- 2. 本地模型批量翻译 ------------------------------------------------
    from vllm import LLM, SamplingParams

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=0)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              dtype="bfloat16", max_model_len=args.max_model_len,
              max_num_seqs=args.batch_size, max_num_batched_tokens=32768,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)

    results: list[str] = [""] * len(jobs)
    for off in range(0, len(jobs), args.batch_size):
        batch = list(range(off, min(off + args.batch_size, len(jobs))))
        convs = [[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": jobs[i]["text"]}] for i in batch]
        try:
            gens = llm.chat(convs, sampling_params=sampling, use_tqdm=False,
                            chat_template_kwargs={"enable_thinking": False})
            for i, g in zip(batch, gens):
                results[i] = clean(g.outputs[0].text)
        except Exception as exc:                       # 单条回退
            print(f"  批次 {off} 失败({exc}); 逐条重试")
            for i in batch:
                try:
                    g = llm.chat(convs[batch.index(i)], sampling_params=sampling,
                                 use_tqdm=False, chat_template_kwargs={"enable_thinking": False})[0]
                    results[i] = clean(g.outputs[0].text)
                except Exception as exc2:
                    raise SystemExit(f"片段 {i} 翻译失败: {exc2}")
        print(f"  进度 {min(off + args.batch_size, len(jobs))}/{len(jobs)}")

    # ---- 3. 重组为 case-NNN.json -------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for entry in plan:
        idx = entry["case_index"]
        out = {"case_index": idx,
               "stratum": results[entry["fields"]["stratum"]],
               "flags": results[entry["fields"]["flags"]],
               "reference_answer": entry["reference_answer"],
               "question": results[entry["fields"]["question"]],
               "conditions": []}
        for ce in entry["conditions"]:
            cond = {"condition": ce["condition"]}
            for field, (start, n) in ce["chunks"].items():
                zh = "".join(results[start:start + n])
                if field == "model_response" and len(zh) > RESPONSE_FILE_LIMIT:
                    name = f"case-{idx:03d}-{ce['condition']}-model-response-zh.txt"
                    (args.out_dir / name).write_text(zh, encoding="utf-8")
                    cond["model_response_file"] = name
                else:
                    cond[field] = zh
            out["conditions"].append(cond)
        (args.out_dir / f"case-{idx:03d}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  写出 case-{idx:03d}.json")

    print(f"\n完成：{len(plan)} 例 → {args.out_dir}")


if __name__ == "__main__":
    main()
