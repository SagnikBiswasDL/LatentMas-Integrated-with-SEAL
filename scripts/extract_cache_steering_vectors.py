#!/usr/bin/env python3
"""Extract key-value cache-steering tensors for Qwen3 from GSM8K train.

Follows Belitsky et al. (arXiv:2507.08799): build contrastive prompt pairs
(positive = few-shot CoT, negative = few-shot answer-only), read the cached
keys/values at the final prompt token per layer, and aggregate the
(positive - negative) differences with Mean-of-Differences.

Output is a dict {"k": [L, H_kv, D_h], "v": [L, H_kv, D_h], "meta": {...}}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_steering.extraction import (
    build_contrastive_messages,
    last_token_kv,
    mean_of_differences,
)
from data import load_gsm8k
from utils import auto_device, set_seed


def _to_pool(split: str = "train", limit: int = 4000) -> List[Dict]:
    pool: List[Dict] = []
    for i, item in enumerate(load_gsm8k(split=split)):
        if i >= limit:
            break
        # GSM8K solution text includes the CoT and a trailing "#### <ans>".
        pool.append(
            {
                "question": item["question"],
                "solution_cot": item["solution"],
                "gold": item["gold"],
            }
        )
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_pairs", type=int, default=200, help="Number of contrastive pairs.")
    parser.add_argument("--n_icl", type=int, default=5, help="Few-shot examples per prompt.")
    parser.add_argument("--max_pool", type=int, default=4000, help="GSM8K train examples to sample from.")
    parser.add_argument("--output", type=str, default="artifacts/cache_steer_vectors/qwen3-14b/gsm8k_kv.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    pool = _to_pool(split="train", limit=args.max_pool)
    if len(pool) < args.n_icl + 1:
        raise RuntimeError("Not enough GSM8K train examples for the requested ICL size.")

    rng = random.Random(args.seed)
    sum_k: List[torch.Tensor] = []
    sum_v: List[torch.Tensor] = []
    count = 0

    for _ in tqdm(range(args.n_pairs), desc="contrastive pairs"):
        sample = rng.sample(pool, args.n_icl + 1)
        icl, query = sample[:-1], sample[-1]

        pos_msgs = build_contrastive_messages(icl, query["question"], with_reasoning=True)
        neg_msgs = build_contrastive_messages(icl, query["question"], with_reasoning=False)

        pos_text = tokenizer.apply_chat_template(pos_msgs, tokenize=False, add_generation_prompt=True)
        neg_text = tokenizer.apply_chat_template(neg_msgs, tokenize=False, add_generation_prompt=True)

        pos_ids = tokenizer(pos_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        neg_ids = tokenizer(neg_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

        pk, pv = last_token_kv(model, pos_ids)
        nk, nv = last_token_kv(model, neg_ids)

        if not sum_k:
            sum_k = [pk[l] - nk[l] for l in range(len(pk))]
            sum_v = [pv[l] - nv[l] for l in range(len(pv))]
        else:
            for l in range(len(pk)):
                sum_k[l] += pk[l] - nk[l]
                sum_v[l] += pv[l] - nv[l]
        count += 1

    s_k, s_v = mean_of_differences(sum_k, sum_v, count)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "k": s_k,
            "v": s_v,
            "meta": {
                "model": args.model_name,
                "task": "gsm8k",
                "n_pairs": count,
                "n_icl": args.n_icl,
                "layers": int(s_k.shape[0]),
                "kv_heads": int(s_k.shape[1]),
                "head_dim": int(s_k.shape[2]),
                "method": "mean_of_differences@last_token (arXiv:2507.08799)",
            },
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "vector_path": args.output,
                "n_pairs": count,
                "n_icl": args.n_icl,
                "shape": list(s_k.shape),
                "k_norm_mean": float(s_k.norm(dim=-1).mean()),
                "v_norm_mean": float(s_v.norm(dim=-1).mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
