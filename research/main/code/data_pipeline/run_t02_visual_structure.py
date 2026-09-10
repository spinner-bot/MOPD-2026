"""Run one Qwen3.5-9B shard for T02 visual-structure proposals.

v2 fixes (after pilot 20260909T failed the gate with 0/1200 accepted):
- thinking mode disabled via chat-template kwarg (v1: 82% of outputs were
  truncated inside the reasoning chain before any JSON was emitted);
- max_new_tokens raised 512 -> 1024 (v1: JSON-only responses still hit the cap);
- prompt now requests the model's native [0,1000] normalized bbox space and
  bbox_geometry deterministically rescales to pixels (v1: model emitted
  norm1000 despite a pixel-coordinate instruction, silently corrupting every
  bbox that passed validation);
- explicit EOS-based truncation flag for auditable rejection;
- --limit for the in-allocation canary stage.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from bbox_geometry import normalize_proposal_geometry
from extract_json_response import extract_json_object

PROMPT = """You are preparing an auditable multimodal QA dataset. Inspect the image and answer with exactly one JSON object. Output no text before or after the JSON object.
Final question: {question}
Existing source crop (may be null): {crop}
Return {{"decision":"accept_candidate" or "reject","evidence_regions":[{{"bbox_xyxy":[x1,y1,x2,y2],"role":"required_evidence","why":"brief"}}],"crop_primary_index":0,"visual_dependency":"low|medium|high","visual_clarity":"low|medium|high","reason":"brief"}}.
Report every bbox_xyxy in normalized coordinates: integers in [0,1000], where x is scaled by image width and y by image height, with x1<x2 and y1<y2. Reject if the image cannot ground the final question. Do not infer facts absent from the image."""


def _eos_ids(processor, model) -> set[int]:
    ids = set()
    tok_eos = getattr(processor.tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, int):
        ids.add(tok_eos)
    elif isinstance(tok_eos, (list, tuple)):
        ids.update(x for x in tok_eos if isinstance(x, int))
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        v = getattr(gen, "eos_token_id", None)
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(x for x in v if isinstance(x, int))
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--limit', type=int, default=0, help='process only the first N candidates (0 = all)')
    a = p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype='auto', local_files_only=True).to('cuda').eval()
    eos_ids = _eos_ids(processor, model)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with a.input.open(encoding='utf-8') as source, a.output.open('w', encoding='utf-8') as sink, torch.inference_mode():
        for line in source:
            if a.limit and done >= a.limit:
                break
            done += 1
            c = json.loads(line); image_path = Path(c['image_original']); record = {'candidate_id': c['candidate_id'], 'candidate': c}
            try:
                with Image.open(image_path) as im: width, height = im.size
                text = PROMPT.format(question=c['final_question_source'], crop=c.get('crop_primary_source'))
                messages = [{'role': 'user', 'content': [{'type': 'image', 'image': str(image_path)}, {'type': 'text', 'text': text}]}]
                template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                images, videos = process_vision_info(messages)
                inputs = processor(text=[template], images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
                generated = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
                new_tokens = generated[0, inputs.input_ids.shape[1]:]
                record['truncated'] = bool(len(new_tokens)) and not any(int(t) in eos_ids for t in new_tokens)
                raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                record.update({'image_width': width, 'image_height': height, 'raw_model_output': raw})
                proposal = extract_json_object(raw)
                if proposal is not None:
                    record['proposal'] = normalize_proposal_geometry(proposal, width, height)
                    record['parse_method'] = 'deterministic_last_json_object'
                else:
                    record['parse_error'] = 'no_json_object_found'
            except Exception as e:
                record['error'] = f'{type(e).__name__}: {e}'
            sink.write(json.dumps(record, ensure_ascii=False) + '\n'); sink.flush()


if __name__ == '__main__': main()
