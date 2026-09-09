"""Run local Qwen3.5-9B option uniqueness review and free-answer MC proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT = """Return only valid JSON for an auditable visual QA item.
Question: {question}
Existing options (empty means free answer): {options}
Reference answer: {answer}
Evidence: {evidence}
If options exist, assess whether exactly one is correct. If they do not, provide exactly four options with one correct and three plausible evidence-excludable distractors. Return {{"decision":"accept_candidate|reject","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_option":"A|B|C|D","unique_answer":true,"reason":"brief"}}. Reject ambiguity; do not invent facts absent from evidence."""

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--model', type=Path, required=True); p.add_argument('--max-new-tokens', type=int, default=512); a = p.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype='auto', local_files_only=True).to('cuda').eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.input.open(encoding='utf-8') as source, a.output.open('w', encoding='utf-8') as sink, torch.inference_mode():
        for line in source:
            t03 = json.loads(line); c = t03['candidate']; record = {'candidate_id': c['candidate_id'], 'candidate': c, 't03': t03}
            try:
                text = PROMPT.format(question=c['final_question_source'], options=json.dumps(c.get('source_options', {}), ensure_ascii=False), answer=c.get('source_answer', c.get('reference_answers', '')), evidence=json.dumps(t03['proposal']['required_evidence'], ensure_ascii=False))
                messages = [{'role':'user','content':[{'type':'image','image':c['image_original']},{'type':'text','text':text}]}]
                template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True); images, videos = process_vision_info(messages)
                inputs = processor(text=[template], images=images, videos=videos, padding=True, return_tensors='pt').to('cuda'); generated = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
                raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]; record['raw_model_output'] = raw
                try: record['proposal'] = json.loads(raw)
                except json.JSONDecodeError: record['parse_error'] = 'non_json_model_output'
            except Exception as error: record['error'] = f'{type(error).__name__}: {error}'
            sink.write(json.dumps(record, ensure_ascii=False)+'\n'); sink.flush()

if __name__ == '__main__': main()
