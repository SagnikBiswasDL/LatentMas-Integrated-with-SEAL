"""Thought boundary classification adapted from VITA-Group/SEAL hidden_analysis.py."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


CHECK_WORDS = [
    "verify",
    "make sure",
    "hold on",
    "think again",
    "'s correct",
    "'s incorrect",
    "let me check",
    "seems right",
]
CHECK_PREFIX = ["wait"]
SWITCH_WORDS = [
    "think differenly",
    "another way",
    "another approach",
    "another method",
    "another solution",
    "another strategy",
    "another technique",
]
SWITCH_PREFIX = ["alternatively"]


def paragraph_split_token_ids(tokenizer) -> List[int]:
    vocab = tokenizer.get_vocab()
    return [vocab[token] for token in vocab.keys() if "ĊĊ" in token]


def generate_index(
    text: str,
    tokenizer,
    split_ids: Sequence[int],
    *,
    think_only: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if think_only:
        think_begin_ids = tokenizer.encode("", add_special_tokens=False)
        think_end_ids = tokenizer.encode("", add_special_tokens=False)
        if not think_begin_ids or think_begin_ids[0] not in tokens:
            return [], [], []
        start = tokens.index(think_begin_ids[0]) + 1
        if think_end_ids and think_end_ids[0] in tokens[start:]:
            end = tokens.index(think_end_ids[0], start)
        else:
            end = len(tokens)
        think_tokens = tokens[start:end]
        offset = start
    else:
        think_tokens = tokens
        offset = 0

    split_set = set(split_ids)
    boundaries = [i for i, t in enumerate(think_tokens) if t in split_set] + [len(think_tokens)]
    step_index: List[int] = []
    check_index: List[int] = []
    switch_index: List[int] = []

    for i in range(len(boundaries) - 1):
        step_index.append(boundaries[i] + offset)
        step = think_tokens[boundaries[i] + 1 : boundaries[i + 1]]
        step_text = tokenizer.decode(step).strip(" \n").lower()
        if any(step_text.startswith(p.lower()) for p in CHECK_PREFIX) or any(
            w in step_text for w in CHECK_WORDS
        ):
            check_index.append(i)
        elif any(step_text.startswith(p.lower()) for p in SWITCH_PREFIX) or any(
            w in step_text for w in SWITCH_WORDS
        ):
            switch_index.append(i)

    return step_index, check_index, switch_index


def classify_step_indices(
    text: str,
    tokenizer,
    split_ids: Sequence[int],
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    step_index, check_index, switch_index = generate_index(
        text, tokenizer, split_ids, think_only=False
    )
    if not step_index:
        return (
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
        )

    step_index_t = torch.tensor(step_index, dtype=torch.long)
    check_index_t = torch.tensor(check_index, dtype=torch.long)
    switch_index_t = torch.tensor(switch_index, dtype=torch.long)
    all_idx = torch.arange(len(step_index))
    mask = torch.ones(len(step_index), dtype=torch.bool)
    if check_index:
        mask[check_index_t] = False
    if switch_index:
        mask[switch_index_t] = False
    other_index_t = all_idx[mask]
    return step_index_t, check_index_t, switch_index_t, other_index_t
