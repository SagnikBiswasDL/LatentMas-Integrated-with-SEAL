#!/usr/bin/env python3
"""Synthetic-cache smoke test for CacheSteering (no model download required).

Verifies that the one-shot intervention adds exactly c_k*S^k / c_v*S^v to the
targeted trailing positions and leaves all other positions untouched.

Run on the pod after `pip install torch`:
    python scripts/smoke_test_cache_steer.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cache_steering.steering import CacheSteering

L, B, H, S, D = 4, 2, 3, 10, 8
C_K, C_V, LAST_N = 0.5, 4.0, 3


class _FakeLayer:
    """Mimics newer transformers DynamicLayer (.keys / .values)."""

    def __init__(self, k, v):
        self.keys = k
        self.values = v


class _FakeDynamicCache:
    """Mimics newer transformers DynamicCache (.layers[i].keys/.values, no key_cache)."""

    def __init__(self, kv):
        self.layers = [_FakeLayer(k, v) for k, v in kv]


def _check(past_obj, key, val, s_k, s_v, label) -> bool:
    target = list(range(S - LAST_N, S))
    ok = True
    layers = past_obj.layers if hasattr(past_obj, "layers") else past_obj
    for l in range(L):
        got_k = layers[l].keys if hasattr(past_obj, "layers") else past_obj[l][0]
        got_v = layers[l].values if hasattr(past_obj, "layers") else past_obj[l][1]
        exp_k = key[l].clone()
        exp_v = val[l].clone()
        exp_k[:, :, target, :] += C_K * s_k[l][None, :, None, :]
        exp_v[:, :, target, :] += C_V * s_v[l][None, :, None, :]
        if not torch.allclose(got_k, exp_k, atol=1e-5):
            ok = False
            print(f"[FAIL:{label}] layer {l} keys mismatch")
        if not torch.allclose(got_v, exp_v, atol=1e-5):
            ok = False
            print(f"[FAIL:{label}] layer {l} values mismatch")
    return ok


def main() -> None:
    torch.manual_seed(0)
    key = [torch.randn(B, H, S, D) for _ in range(L)]
    val = [torch.randn(B, H, S, D) for _ in range(L)]
    past = tuple((key[l].clone(), val[l].clone()) for l in range(L))

    s_k = torch.randn(L, H, D)
    s_v = torch.randn(L, H, D)
    tmp = os.path.join(tempfile.gettempdir(), "smoke_cache_vec.pt")
    torch.save({"k": s_k, "v": s_v}, tmp)

    cs = CacheSteering(vector_path=tmp, c_k=C_K, c_v=C_V, positions="last_n", last_n=LAST_N)

    # Case 1: legacy tuple cache.
    cs.apply(past)
    # Case 2: newer DynamicCache-style object (.layers[i].keys/.values).
    fake = _FakeDynamicCache([(key[l].clone(), val[l].clone()) for l in range(L)])
    cs.apply(fake)

    ok = _check(past, key, val, s_k, s_v, "legacy")
    ok = _check(fake, key, val, s_k, s_v, "dynamic") and ok

    # Non-target positions must be untouched (checked on the legacy cache).
    target = list(range(S - LAST_N, S))
    untouched = [p for p in range(S) if p not in target]
    for l in range(L):
        if not torch.allclose(past[l][0][:, :, untouched, :], key[l][:, :, untouched, :]):
            ok = False
            print(f"[FAIL] layer {l} non-target keys were modified")

    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
