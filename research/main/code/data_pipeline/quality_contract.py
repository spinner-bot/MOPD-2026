"""Deterministic acceptance gate for H-T05 quality judgements."""
from __future__ import annotations

SCORES={'low':0,'medium':1,'high':2}
def validate_quality_judgement(j: dict) -> list[str]:
    errors=[]
    for key in ('answerability','logical_dependency','relevance','ambiguity','visual_grounding'):
        if j.get(key) not in SCORES: errors.append(f'invalid {key}')
    if j.get('answer_leakage') not in {0,1}: errors.append('invalid answer_leakage')
    if j.get('decision') not in {'accept','borderline','reject'}: errors.append('invalid decision')
    return errors
def eligible(j: dict) -> bool:
    return not validate_quality_judgement(j) and j['answerability']=='high' and j['relevance']=='high' and j['visual_grounding']=='high' and j['logical_dependency'] in {'medium','high'} and j['answer_leakage']==0 and j['ambiguity']=='low' and j['decision']=='accept'
