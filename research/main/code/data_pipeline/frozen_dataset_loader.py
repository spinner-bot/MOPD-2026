"""Read only a manifest-declared frozen JSONL dataset."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def sha256(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def load_frozen(manifest_path: str|Path, split: str) -> list[dict]:
 mpath=Path(manifest_path); manifest=json.loads(mpath.read_text(encoding='utf-8'))
 if manifest.get('status')!='frozen': raise ValueError('manifest is not frozen')
 entry=manifest.get('splits',{}).get(split)
 if not entry: raise KeyError(f'unknown split: {split}')
 data=Path(entry['path'])
 if sha256(data)!=entry['sha256']: raise ValueError(f'checksum mismatch: {data}')
 return [json.loads(line) for line in data.open(encoding='utf-8') if line.strip()]
