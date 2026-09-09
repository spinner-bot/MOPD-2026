"""Freeze T04 random, shuffled, and generic controls deterministically."""
from __future__ import annotations
import hashlib, random

GENERIC='Inspect the image carefully and use only relevant visual evidence before selecting one option.'

def _guides(sample):
    return [(sample[f'guide_{i}_question'],sample[f'guide_{i}_answer']) for i in range(1,5) if sample.get(f'guide_{i}_question') and sample.get(f'guide_{i}_answer')]
def freeze_controls(sample: dict, unrelated_questions: list[str]) -> dict:
    guides=_guides(sample)
    if not guides: raise ValueError('guides required')
    if len(unrelated_questions)<len(guides): raise ValueError('insufficient unrelated questions')
    seed=int(hashlib.sha256(sample['sample_id'].encode()).hexdigest()[:16],16)
    rng=random.Random(seed); pool=[q.strip() for q in unrelated_questions if q.strip() and q.strip() not in {x[0] for x in guides}]
    if len(pool)<len(guides): raise ValueError('unrelated questions overlap guides')
    rng.shuffle(pool); shuffled=guides[:];rng.shuffle(shuffled)
    return {'random_questions':'\n'.join(pool[:len(guides)]),'random_questions_frozen':True,'random_control_seed':seed,'shuffled_guides':[{'question':q,'answer':a} for q,a in shuffled],'generic_prompt':GENERIC}
