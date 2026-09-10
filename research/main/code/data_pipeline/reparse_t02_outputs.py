"""Create parsed T02 copies from immutable raw model-output records."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from extract_json_response import extract_json_object

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input-glob',required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    for source in sorted(Path().glob(a.input_glob)):
        output=a.output_dir/source.name
        with source.open(encoding='utf-8') as inp, output.open('w',encoding='utf-8') as out:
            for line in inp:
                row=json.loads(line)
                if 'raw_model_output' in row:
                    proposal=extract_json_object(row['raw_model_output'])
                    if proposal is not None: row['proposal']=proposal; row['parse_method']='deterministic_last_json_object'
                    else: row['parse_error']='no_json_object_found'
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
