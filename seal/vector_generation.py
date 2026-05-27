"""Steering vector generation adapted from VITA-Group/SEAL."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import torch


def compute_steering_vector(
    execution_hiddens: torch.Tensor,
    reflection_hiddens: torch.Tensor,
    transition_hiddens: torch.Tensor,
) -> torch.Tensor:
    """S = mean(reflection ∪ transition) - mean(execution)."""
    pos_parts = []
    if reflection_hiddens.numel():
        pos_parts.append(reflection_hiddens)
    if transition_hiddens.numel():
        pos_parts.append(transition_hiddens)
    if not pos_parts:
        raise ValueError("No reflection/transition hidden states collected.")
    if execution_hiddens.numel() == 0:
        raise ValueError("No execution hidden states collected.")

    positive = torch.cat(pos_parts, dim=0).mean(dim=0)
    negative = execution_hiddens.mean(dim=0)
    return positive - negative


def save_vector(vector: torch.Tensor, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(vector.cpu(), path)
    return path


def stack_layer_hiddens(entries: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors: List[torch.Tensor] = [t for t in entries if t.numel()]
    if not tensors:
        return torch.empty(0)
    return torch.cat(tensors, dim=0)
