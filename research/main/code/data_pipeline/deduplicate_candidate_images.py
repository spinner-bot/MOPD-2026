"""Build exact and near-duplicate image clusters from candidate records."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from PIL import Image

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def dhash(path):
 with Image.open(path) as im:
  im=im.convert('L').resize((9,8)); px=list(im.getdata())
 return sum((px[y*9+x]>px[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 images={}
 for line in a.input.open(encoding='utf-8'):
  r=json.loads(line); images.setdefault(r['image_original'],[]).append(r['candidate_id'])
 rows=[]
 for image,ids in images.items():
  path=Path(image); rows.append({'image_original':image,'candidate_ids':ids,'sha256':sha256(path),'dhash64':f'{dhash(path):016x}'})
 exact=defaultdict(list)
 for i,r in enumerate(rows): exact[r['sha256']].append(i)
 clusters=[{'kind':'exact','members':v} for v in exact.values() if len(v)>1]
 buckets=defaultdict(list)
 for i,r in enumerate(rows): buckets[int(r['dhash64'][:4],16)].append(i)
 for ids in buckets.values():
  for pos,i in enumerate(ids):
   for j in ids[pos+1:]:
    if rows[i]['sha256']!=rows[j]['sha256'] and (int(rows[i]['dhash64'],16)^int(rows[j]['dhash64'],16)).bit_count()<=5: clusters.append({'kind':'near','members':[i,j]})
 out={'n_unique_image_paths':len(rows),'images':rows,'duplicate_clusters':clusters}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'images':len(rows),'clusters':len(clusters)}))
if __name__=='__main__': main()
