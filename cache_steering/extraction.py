"""Helpers to extract key-value steering tensors via Mean-of-Differences.

Positive prompts demonstrate explicit chain-of-thought reasoning; negative
prompts give only the final answer. Both share the same few-shot ICL structure
and query, differing only in the presence of reasoning. We read the cached
keys/values at the final prompt token at every layer and average the
(positive - negative) differences across contrastive pairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

SYSTEM_MESSAGE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _answer_line(gold: str) -> str:
    return f"The final answer is \\boxed{{{gold}}}."


def build_contrastive_messages(
    icl_examples: List[Dict],
    query_question: str,
    *,
    with_reasoning: bool,
) -> List[Dict]:
    """Build a chat-formatted few-shot prompt.

    Each ICL example becomes a (user question, assistant answer) turn. In the
    positive variant the assistant turn contains the full CoT solution; in the
    negative variant it contains only the final answer line.
    """
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for ex in icl_examples:
        messages.append({"role": "user", "content": f"Question: {ex['question'].strip()}"})
        if with_reasoning:
            assistant = ex["solution_cot"].strip()
        else:
            assistant = _answer_line(ex["gold"])
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": f"Question: {query_question.strip()}"})
    return messages


@torch.no_grad()
def last_token_kv(model, input_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Forward a single (batch=1) prompt and return per-layer last-token K and V.

    Returns two lists of tensors, each [H_kv, D_h] (float32, on CPU).
    """
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = out.past_key_values

    try:
        from transformers.cache_utils import Cache
    except ImportError:  # pragma: no cover
        Cache = None

    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    if Cache is not None and isinstance(past, Cache):
        n_layers = len(past.key_cache)
        for l in range(n_layers):
            keys.append(past.key_cache[l][0, :, -1, :].float().cpu())
            vals.append(past.value_cache[l][0, :, -1, :].float().cpu())
    else:
        for layer in past:
            keys.append(layer[0][0, :, -1, :].float().cpu())
            vals.append(layer[1][0, :, -1, :].float().cpu())
    return keys, vals


def mean_of_differences(
    pos_minus_neg_k: List[torch.Tensor],
    pos_minus_neg_v: List[torch.Tensor],
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack accumulated per-layer sums and divide by count -> [L, H, D]."""
    if count <= 0:
        raise ValueError("No contrastive pairs were accumulated.")
    s_k = torch.stack(pos_minus_neg_k, dim=0) / count
    s_v = torch.stack(pos_minus_neg_v, dim=0) / count
    return s_k, s_v
