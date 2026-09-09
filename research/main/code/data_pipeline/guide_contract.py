"""Deterministic contract for H-T03 latent graphs and guide QA proposals."""
from __future__ import annotations

FAMILIES={'P','R','PR'}
STRATEGIES={'localization','entity_tracking','logical_bridge','evidence_integration'}

def validate_t03_proposal(proposal: dict, final_answer_text: str='') -> list[str]:
    errors=[]
    if proposal.get('decision') not in {'accept_candidate','reject'}: return ['invalid decision']
    if proposal['decision']=='reject': return errors
    if proposal.get('task_family') not in FAMILIES: errors.append('invalid task_family')
    if proposal.get('guidance_strategy') not in STRATEGIES: errors.append('invalid guidance_strategy')
    if not isinstance(proposal.get('reasoning_depth'),int) or not 1<=proposal['reasoning_depth']<=5: errors.append('invalid reasoning_depth')
    evidence=proposal.get('required_evidence'); states=proposal.get('intermediate_states'); guides=proposal.get('guides')
    if not isinstance(evidence,list) or not evidence or not all(isinstance(x,str) and x.strip() for x in evidence): errors.append('missing required_evidence')
    if not isinstance(states,list) or not states: errors.append('missing intermediate_states')
    if not isinstance(guides,list) or not 1<=len(guides)<=4: errors.append('guides must contain 1-4 items'); return errors
    answer=final_answer_text.strip().casefold()
    for i,guide in enumerate(guides,1):
        if not isinstance(guide,dict) or not str(guide.get('question','')).strip() or not str(guide.get('answer','')).strip(): errors.append(f'guide {i} incomplete'); continue
        if answer and answer in str(guide['answer']).casefold(): errors.append(f'guide {i} leaks final answer')
    if proposal.get('final_answer_leaked_by_guide') is not False: errors.append('missing or false leakage declaration')
    return errors
