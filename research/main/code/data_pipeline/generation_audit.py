"""Shared generation-audit helpers for the T0x local Qwen3.5-9B runners.

After the T02 v1 pilot failure (82% of outputs truncated inside the thinking
chain), every runner must:
1. disable thinking mode via the chat-template kwarg ``enable_thinking=False``;
2. record whether generation hit max_new_tokens without emitting EOS, so the
   validators can reject with an auditable ``truncated_output`` reason instead
   of a generic parse error.
"""
from __future__ import annotations


def collect_eos_ids(processor, model) -> set[int]:
    """Gather every EOS token id that terminates greedy generation."""
    ids: set[int] = set()
    tok_eos = getattr(processor.tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, int):
        ids.add(tok_eos)
    elif isinstance(tok_eos, (list, tuple)):
        ids.update(x for x in tok_eos if isinstance(x, int))
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        v = getattr(gen, "eos_token_id", None)
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(x for x in v if isinstance(x, int))
    return ids


def hit_token_cap(new_token_ids, eos_ids) -> bool:
    """True when generation produced tokens but never emitted an EOS."""
    tokens = list(new_token_ids)
    return bool(tokens) and not any(int(t) in eos_ids for t in tokens)
