"""Deterministically validate T02 visual-structure proposals before T03."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

LEVELS={'low','medium','high'}
def validate(row):
    if row.get('error'): return 'runner_error'
    p=row.get('proposal')
    if not isinstance(p,dict): return 'invalid_json'
    if p.get('decision') not in {'accept_candidate','reject'}: return 'invalid_decision'
    if p['decision']=='reject': return 'model_reject'
    regions=p.get('evidence_regions')
    if not isinstance(regions,list) or not regions: return 'missing_regions'
    w,h=row.get('image_width',0),row.get('image_height',0)
    for region in regions:
        box=region.get('bbox_xyxy') if isinstance(region,dict) else None
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(x,int) for x in box): return 'invalid_bbox'
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=w and 0<=y1<y2<=h): return 'bbox_out_of_bounds'
    if not isinstance(p.get('crop_primary_index'),int) or not 0<=p['crop_primary_index']<len(regions): return 'invalid_crop_index'
    if p.get('visual_dependency') not in LEVELS or p.get('visual_clarity') not in LEVELS: return 'invalid_visual_labels'
    return None
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-glob',required=True);p.add_argument('--accepted',type=Path,required=True);p.add_argument('--rejected',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
 rows=[]
 for f in sorted(Path().glob(a.input_glob)):
  rows.extend(json.loads(x) for x in f.read_text(encoding='utf-8').splitlines() if x.strip())
 accepted=[]; rejected=[]
 for row in rows:
  reason=validate(row)
  if reason: row['t02_rejection_reason']=reason; rejected.append(row)
  else: accepted.append(row)
 for path,items in ((a.accepted,accepted),(a.rejected,rejected)):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
 summary={'total':len(rows),'accepted':len(accepted),'rejected':len(rejected),'rejections':dict(Counter(x['t02_rejection_reason'] for x in rejected))}
 a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
