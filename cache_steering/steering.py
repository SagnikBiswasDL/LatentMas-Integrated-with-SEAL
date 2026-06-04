"""One-shot KV-cache steering applied to the LatentMAS shared latent cache.

Reference: Belitsky et al., "KV Cache Steering for Controlling Frozen LLMs"
(arXiv:2507.08799). For each layer l we modify the cached keys/values at a set
of target positions:

    K*_l = K_l + c_k * S^k_l
    V*_l = V_l + c_v * S^v_l

S^k_l, S^v_l have shape [H_kv, D_h] (per attention head), and are added in-place
to the populated cache once, before the Judger generates. Generation then
proceeds normally over the modified cache.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:  # pragma: no cover
    Cache = None


class CacheSteering:
    def __init__(
        self,
        *,
        vector_path: str,
        c_k: float = 0.0,
        c_v: float = 4.0,
        positions: str = "last_n",
        last_n: int = 40,
        layers: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        ckpt = torch.load(vector_path, map_location="cpu", weights_only=True)
        # Stored as [L, H_kv, D_h] float tensors.
        self.S_k: torch.Tensor = ckpt["k"]
        self.S_v: torch.Tensor = ckpt["v"]
        self.c_k = float(c_k)
        self.c_v = float(c_v)
        if positions not in {"last", "last_n", "all"}:
            raise ValueError(f"Unknown positions mode: {positions}")
        self.positions = positions
        self.last_n = int(last_n)
        self.layers = list(layers) if layers is not None else None
        self.device = device

    # --- cache accessors (support both DynamicCache and legacy tuples) ---
    @staticmethod
    def _is_cache_obj(past) -> bool:
        return Cache is not None and isinstance(past, Cache)

    def _num_layers(self, past) -> int:
        if self._is_cache_obj(past):
            return len(past.key_cache)
        return len(past)

    def _layer_kv(self, past, layer_idx: int):
        if self._is_cache_obj(past):
            return past.key_cache[layer_idx], past.value_cache[layer_idx]
        layer = past[layer_idx]
        return layer[0], layer[1]

    def _target_positions(self, seq_len: int) -> List[int]:
        if seq_len <= 0:
            return []
        if self.positions == "last":
            return [seq_len - 1]
        if self.positions == "last_n":
            n = min(max(self.last_n, 1), seq_len)
            return list(range(seq_len - n, seq_len))
        return list(range(seq_len))  # "all"

    @torch.no_grad()
    def apply(self, past):
        """Modify the KV cache in place and return it."""
        if past is None or (self.c_k == 0.0 and self.c_v == 0.0):
            return past

        n_layers = self._num_layers(past)
        layer_ids = self.layers if self.layers is not None else range(n_layers)

        for l in layer_ids:
            if l < 0 or l >= n_layers:
                continue
            k, v = self._layer_kv(past, l)
            if k is None or not torch.is_tensor(k) or k.numel() == 0:
                continue

            seq_len = k.shape[-2]
            pos = self._target_positions(seq_len)
            if not pos:
                continue

            if self.c_k != 0.0:
                sk = self.S_k[l].to(device=k.device, dtype=k.dtype)  # [H, D]
                k[:, :, pos, :] += self.c_k * sk[None, :, None, :]
            if self.c_v != 0.0:
                sv = self.S_v[l].to(device=v.device, dtype=v.dtype)  # [H, D]
                v[:, :, pos, :] += self.c_v * sv[None, :, None, :]

        return past
