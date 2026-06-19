#!/usr/bin/env python3
"""Extract a *native* success-minus-failure steering direction for LatentMAS.

Unlike the prior SEAL/cache-steering extractors (which read activations from
vanilla *text* runs), this runs the real LatentMAS pipeline and captures the
Judger's own decode-time hidden states at a target layer, then contrasts runs
that produced a CORRECT final answer against runs that produced a WRONG one:

    S = mean(hidden | correct) - mean(hidden | wrong)

The vector lives in the same layer/space the SEAL hook injects into, so it can
be applied directly via `--use_seal --seal_mode text --seal_layer <L>`.

Two position variants are saved:
  * `all`   - mean over every generated Judger token (whole answer)
  * `lastk` - mean over the last K generated tokens (the finalization region)

Also saves a norm-matched RANDOM control vector for placebo runs.

Outputs (default dir artifacts/native_vectors/qwen3-14b/):
  success_layer{L}_all.pt      bare tensor [D]
  success_layer{L}_lastk.pt    bare tensor [D]
  random_layer{L}.pt           bare tensor [D] (||.|| matched to success_all)
  native_extraction.json       metadata (counts, norms, config)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from tqdm import tqdm

from data import load_gsm8k
from methods import default_agents
from models import ModelWrapper
from prompts import build_agent_message_sequential_latent_mas
from seal.capture import HiddenCapture
from utils import auto_device, extract_gsm8k_answer, normalize_answer, set_seed


def build_model_args(a: argparse.Namespace) -> argparse.Namespace:
    """Namespace mirroring run.py defaults that ModelWrapper / latent loop read."""
    return argparse.Namespace(
        method="latent_mas",
        model_name=a.model_name,
        task="gsm8k",
        prompt="sequential",
        device=a.device,
        device2="cuda:1",
        split="train",
        max_new_tokens=a.max_new_tokens,
        latent_steps=a.latent_steps,
        temperature=a.temperature,
        top_p=a.top_p,
        generate_bs=1,
        think=True,
        latent_space_realign=True,
        seed=a.seed,
        use_vllm=False,
        enable_prefix_caching=False,
        use_second_HF_model=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        use_seal=False,
        use_cache_steer=False,
    )


@torch.no_grad()
def run_one(
    model: ModelWrapper, capture: HiddenCapture, question: str, args
) -> Tuple[Optional[torch.Tensor], str]:
    """Run the full LatentMAS pipeline for one question; return ([T, D] states, judger_text)."""
    margs = model.args
    past_kv = None
    capture.active = False

    for agent in default_agents():
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=margs
        )
        prompts, _, _, _ = model.prepare_chat_batch([messages], add_generation_prompt=True)
        wrapped = [f"{p}<think>" for p in prompts] if margs.think else prompts
        enc = model.tokenizer(wrapped, return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)

        if agent.role != "judger":
            past_kv = model.generate_latent_batch(
                ids, attention_mask=mask, latent_steps=margs.latent_steps, past_key_values=past_kv
            )
        else:
            past_for_decoding = past_kv if margs.latent_steps > 0 else None
            capture.reset()
            capture.active = True
            generated, _ = model.generate_text_batch(
                ids,
                mask,
                max_new_tokens=margs.max_new_tokens,
                temperature=margs.temperature,
                top_p=margs.top_p,
                past_key_values=past_for_decoding,
            )
            capture.active = False
            text = generated[0].strip()

    steps = capture.stacked()  # [T, 1, D]
    if steps.numel() == 0:
        return None, text
    return steps[:, 0, :], text  # [T, D]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--n_samples", type=int, default=500, help="GSM8K-train problems to run")
    parser.add_argument("--latent_steps", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--last_k", type=int, default=8, help="Tokens for the finalization-region variant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts/native_vectors/qwen3-14b")
    # Optional sharding so two GPUs can split the train set (run twice, then merge separately).
    parser.add_argument("--start", type=int, default=0, help="Index into train set to start at")
    args = parser.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)

    model = ModelWrapper(args.model_name, device, use_vllm=False, args=build_model_args(args))
    capture = HiddenCapture(args.layer)
    capture.register(model.model)

    pool: List[Dict] = []
    for i, item in enumerate(load_gsm8k(split="train")):
        if i < args.start:
            continue
        pool.append(item)
        if len(pool) >= args.n_samples:
            break

    correct_all, wrong_all = [], []
    correct_lastk, wrong_lastk = [], []
    n_correct = n_wrong = n_empty = 0

    for item in tqdm(pool, desc="native extraction"):
        steps, text = run_one(model, capture, item["question"], args)
        if steps is None or steps.shape[0] == 0:
            n_empty += 1
            continue

        vec_all = steps.mean(dim=0)
        k = min(args.last_k, steps.shape[0])
        vec_lastk = steps[-k:].mean(dim=0)

        pred = normalize_answer(extract_gsm8k_answer(text))
        gold = item.get("gold", "")
        ok = bool(pred and gold and pred == gold)

        if ok:
            n_correct += 1
            correct_all.append(vec_all)
            correct_lastk.append(vec_lastk)
        else:
            n_wrong += 1
            wrong_all.append(vec_all)
            wrong_lastk.append(vec_lastk)

    if not correct_all or not wrong_all:
        raise RuntimeError(
            f"Need both correct and wrong runs; got correct={len(correct_all)}, wrong={len(wrong_all)}."
        )

    def diff(c: List[torch.Tensor], w: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(c, 0).mean(0) - torch.stack(w, 0).mean(0)

    s_all = diff(correct_all, wrong_all)
    s_lastk = diff(correct_lastk, wrong_lastk)

    # Norm-matched random control (Gaussian rescaled to ||s_all||).
    g = torch.randn_like(s_all)
    s_random = g * (s_all.norm() / g.norm().clamp_min(1e-8))

    os.makedirs(args.out_dir, exist_ok=True)
    p_all = os.path.join(args.out_dir, f"success_layer{args.layer}_all.pt")
    p_lastk = os.path.join(args.out_dir, f"success_layer{args.layer}_lastk.pt")
    p_rand = os.path.join(args.out_dir, f"random_layer{args.layer}.pt")
    torch.save(s_all, p_all)
    torch.save(s_lastk, p_lastk)
    torch.save(s_random, p_rand)

    # Reference scale: typical magnitude of a captured layer activation.
    ref_norm = float(torch.stack(correct_all + wrong_all, 0).norm(dim=-1).mean())
    meta = {
        "model": args.model_name,
        "task": "gsm8k",
        "layer": args.layer,
        "n_samples": len(pool),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_empty": n_empty,
        "last_k": args.last_k,
        "latent_steps": args.latent_steps,
        "dim": int(s_all.shape[0]),
        "s_all_norm": float(s_all.norm()),
        "s_lastk_norm": float(s_lastk.norm()),
        "ref_activation_norm": ref_norm,
        "vectors": {"all": p_all, "lastk": p_lastk, "random": p_rand},
        "note": "S = mean(correct) - mean(wrong) of layer-L Judger decode hidden states.",
    }
    with open(os.path.join(args.out_dir, "native_extraction.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
