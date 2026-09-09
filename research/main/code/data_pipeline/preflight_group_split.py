"""Dry-run a leakage-safe split using image and duplicate-cluster groups."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--duplicates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=[json.loads(x) for x in a.candidates.open(encoding='utf-8') if x.strip()]; d=json.loads(a.duplicates.read_text(encoding='utf-8'))
 image_to_group={r['image_original']:r['sha256'] for r in d['images']}
 groups=defaultdict(list)
 for r in rows: groups[image_to_group.get(r['image_original'],r['image_original'])].append(r)
 splits=Counter();families=defaultdict(Counter)
 for group,items in groups.items():
  h=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
  split='train' if h<73 else 'val' if h<82 else 'test' if h<95 else 'stage0'
  splits[split]+=len(items)
  for x in items: families[split][x['task_family_candidate']]+=1
 report={'status':'dry_run_not_formal','records':len(rows),'groups':len(groups),'split_counts':dict(splits),'family_counts':{k:dict(v) for k,v in families.items()},'blocking_requirements':['PR candidates absent','guide QA and controls absent','T05/T06 approval absent']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
