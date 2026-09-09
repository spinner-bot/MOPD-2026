"""Run independent local Qwen3.5-9B quality judgements for completed candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT = """Act as an independent QA auditor. Return only JSON.
Question: {question}
Options: {options}
Guides: {guides}
Image evidence regions: {regions}
Do not reproduce reasoning. Judge the rendered sample independently and return
{{"answerability":"low|medium|high","logical_dependency":"low|medium|high","relevance":"low|medium|high","ambiguity":"low|medium|high","visual_grounding":"low|medium|high","answer_leakage":0,"decision":"accept|borderline|reject","reasons":["brief"]}}.
Use answer_leakage 1 only when a guide reveals the final answer or correct option."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype="auto", local_files_only=True).to("cuda").eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink, torch.inference_mode():
        for line in source:
            item = json.loads(line); candidate = item["candidate"]; proposal = item["proposal"]
            record = {"candidate_id": candidate["candidate_id"], "candidate": candidate, "t03": item}
            try:
                text = PROMPT.format(question=candidate["final_question_source"], options=json.dumps(candidate.get("source_options", {}), ensure_ascii=False), guides=json.dumps(proposal["guides"], ensure_ascii=False), regions=json.dumps(item["t02"]["proposal"]["evidence_regions"], ensure_ascii=False))
                messages = [{"role": "user", "content": [{"type": "image", "image": candidate["image_original"]}, {"type": "text", "text": text}]}]
                template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = process_vision_info(messages); inputs = processor(text=[template], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
                tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                raw = processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                record["raw_model_output"] = raw
                try: record["judgement"] = json.loads(raw)
                except json.JSONDecodeError: record["parse_error"] = "non_json_model_output"
            except Exception as error: record["error"] = f"{type(error).__name__}: {error}"
            sink.write(json.dumps(record, ensure_ascii=False) + "\n"); sink.flush()


if __name__ == "__main__":
    main()
