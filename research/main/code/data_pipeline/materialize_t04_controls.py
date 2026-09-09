"""Materialize deterministic T-random/T-shuffle/T-generic controls after T04."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from freeze_text_controls import freeze_controls

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,required=True); p.add_argument('--accepted',type=Path,required=True); p.add_argument('--rejected',type=Path,required=True); a=p.parse_args()
    rows=[json.loads(line) for line in a.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_image=defaultdict(list)
    for row in rows:
        by_image[row['candidate']['image_original']].extend(g['question'] for g in row['t03']['proposal']['guides'])
    accepted=[]; rejected=[]
    for row in rows:
        proposal=row['t03']['proposal']; sample={'sample_id':row['candidate_id']}
        for index, guide in enumerate(proposal['guides'], 1): sample[f'guide_{index}_question']=guide['question']; sample[f'guide_{index}_answer']=guide['answer']
        own={guide['question'] for guide in proposal['guides']}; pool=[question for question in by_image[row['candidate']['image_original']] if question not in own]
        try: row['controls']=freeze_controls(sample, pool); accepted.append(row)
        except ValueError as error: row['t04_control_rejection_reason']=str(error); rejected.append(row)
    for path, records in ((a.accepted,accepted),(a.rejected,rejected)):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records),encoding='utf-8')
    print(json.dumps({'accepted':len(accepted),'rejected':len(rejected)},ensure_ascii=False))
if __name__=='__main__': main()
