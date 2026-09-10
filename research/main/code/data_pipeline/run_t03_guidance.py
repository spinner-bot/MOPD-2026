"""Run one local Qwen3.5-9B shard for T03 graph and guide proposals.

The input is the accepted JSONL emitted by ``validate_t02_proposals.py``.
Raw model text is retained beside parsed JSON; this script deliberately does
not accept, edit, or silently repair a proposal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_json_response import extract_json_object
from generation_audit import collect_eos_ids, hit_token_cap


PROMPT = """You are preparing an auditable multimodal QA dataset. Answer with exactly one valid JSON object; output no text before or after it.
Final question: {question}
Options: {options}
Correct option label: {answer}
Visual evidence proposed by a separate stage: {regions}

First determine whether the image supplies a real, non-trivial reasoning path to the final answer. If it does not, return {{"decision":"reject","rejection_reason":"brief reason"}}.
Otherwise return {{"decision":"accept_candidate","task_family":"P|R|PR","guidance_strategy":"localization|entity_tracking|logical_bridge|evidence_integration","required_evidence":["E1"],"intermediate_states":["H1"],"guides":[{{"question":"...","answer":"...","supports_next":"H1 or final"}}],"final_answer_leaked_by_guide":false,"reasoning_depth":1,"rejection_reason":""}}.
Rules: guides must have 1-4 complete QA pairs; each guide must be answerable from the image/crop or a preceding state; do not state, paraphrase, or identify the correct option or its answer in any guide; reject uncertainty or ambiguity."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", local_files_only=True
    ).to("cuda").eval()
    eos_ids = collect_eos_ids(processor, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink, torch.inference_mode():
        for line in source:
            t02 = json.loads(line)
            candidate = t02["candidate"]
            record = {"candidate_id": candidate["candidate_id"], "candidate": candidate, "t02": t02}
            try:
                prompt = PROMPT.format(
                    question=candidate["final_question_source"],
                    options=json.dumps(candidate.get("source_options", {}), ensure_ascii=False),
                    answer=candidate.get("source_answer", candidate.get("reference_answers", "")),
                    regions=json.dumps(t02["proposal"]["evidence_regions"], ensure_ascii=False),
                )
                messages = [{"role": "user", "content": [{"type": "image", "image": candidate["image_original"]}, {"type": "text", "text": prompt}]}]
                template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                images, videos = process_vision_info(messages)
                inputs = processor(text=[template], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                record["truncated"] = hit_token_cap(generated[0, inputs.input_ids.shape[1]:], eos_ids)
                raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                record["raw_model_output"] = raw
                proposal = extract_json_object(raw)
                if proposal is not None:
                    record["proposal"] = proposal
                    record["parse_method"] = "deterministic_last_json_object"
                else:
                    record["parse_error"] = "no_json_object_found"
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()


if __name__ == "__main__":
    main()
