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
    cs.apply(past)

    target = list(range(S - LAST_N, S))
    ok = True
    for l in range(L):
        exp_k = key[l].clone()
        exp_v = val[l].clone()
        exp_k[:, :, target, :] += C_K * s_k[l][None, :, None, :]
        exp_v[:, :, target, :] += C_V * s_v[l][None, :, None, :]
        if not torch.allclose(past[l][0], exp_k, atol=1e-5):
            ok = False
            print(f"[FAIL] layer {l} keys mismatch")
        if not torch.allclose(past[l][1], exp_v, atol=1e-5):
            ok = False
            print(f"[FAIL] layer {l} values mismatch")
        # untouched positions must be identical to the original
        untouched = [p for p in range(S) if p not in target]
        if not torch.allclose(past[l][0][:, :, untouched, :], key[l][:, :, untouched, :]):
            ok = False
            print(f"[FAIL] layer {l} non-target keys were modified")

    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
