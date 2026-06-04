"""Version-robust access to a model's KV cache.

Across transformers releases the cache object has changed shape:
  - older DynamicCache: `.key_cache` / `.value_cache` (lists of tensors)
  - newer DynamicCache: `.layers[i].keys` / `.layers[i].values`
  - legacy tuple cache:  `past[i] == (key, value)`

All accessors below return references to the *real* cache tensors so that
in-place edits (cache steering) persist into generation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def layer_kv_list(past) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return [(key, value), ...] per layer, referencing the live cache tensors."""
    if past is None:
        return []

    # Older DynamicCache API.
    key_cache = getattr(past, "key_cache", None)
    value_cache = getattr(past, "value_cache", None)
    if key_cache is not None and value_cache is not None and len(key_cache):
        return [(key_cache[i], value_cache[i]) for i in range(len(key_cache))]

    # Newer DynamicCache API (.layers[i].keys / .values).
    layers = getattr(past, "layers", None)
    if layers is not None:
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            out.append((k, v))
        if out:
            return out

    # Fall back to the legacy-cache view (still returns tensor references).
    if hasattr(past, "to_legacy_cache"):
        return [(layer[0], layer[1]) for layer in past.to_legacy_cache()]

    # Already a legacy tuple/list cache.
    return [(layer[0], layer[1]) for layer in past]


def num_layers(past) -> int:
    return len(layer_kv_list(past))
