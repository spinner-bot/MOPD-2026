"""Run one Qwen3.5-9B shard for T02 visual-structure proposals."""
from __future__ import annotations

import argparse, json
from pathlib import Path

PROMPT = """You are preparing an auditable multimodal QA dataset. Inspect the image and answer only valid JSON.
Final question: {question}
Existing source crop (may be null): {crop}
Return {{"decision":"accept_candidate" or "reject","evidence_regions":[{{"bbox_xyxy":[x1,y1,x2,y2],"role":"required_evidence","why":"brief"}}],"crop_primary_index":0,"visual_dependency":"low|medium|high","visual_clarity":"low|medium|high","reason":"brief"}}.
Use pixel coordinates. Reject if the image cannot ground the final question. Do not infer facts absent from the image."""

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--max-new-tokens',type=int,default=512); a=p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(a.model,torch_dtype='auto',local_files_only=True).to('cuda').eval()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.input.open(encoding='utf-8') as source,a.output.open('w',encoding='utf-8') as sink,torch.inference_mode():
        for line in source:
            c=json.loads(line); image_path=Path(c['image_original']); record={'candidate_id':c['candidate_id'],'candidate':c}
            try:
                with Image.open(image_path) as im: width,height=im.size
                text=PROMPT.format(question=c['final_question_source'],crop=c.get('crop_primary_source'))
                messages=[{'role':'user','content':[{'type':'image','image':str(image_path)},{'type':'text','text':text}]}]
                template=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                images,videos=process_vision_info(messages)
                inputs=processor(text=[template],images=images,videos=videos,padding=True,return_tensors='pt').to('cuda')
                generated=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False)
                generated=generated[:,inputs.input_ids.shape[1]:]
                raw=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                record.update({'image_width':width,'image_height':height,'raw_model_output':raw})
                try: record['proposal']=json.loads(raw)
                except json.JSONDecodeError: record['parse_error']='non_json_model_output'
            except Exception as e: record['error']=f'{type(e).__name__}: {e}'
            sink.write(json.dumps(record,ensure_ascii=False)+'\n'); sink.flush()
if __name__=='__main__': main()
