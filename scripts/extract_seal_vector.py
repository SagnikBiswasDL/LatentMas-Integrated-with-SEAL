#!/usr/bin/env python3
"""Extract a SEAL steering vector for Qwen3 from GSM8K CoT traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_gsm8k
from prompts import build_agent_messages_single_agent
from seal.thought_classifier import classify_step_indices, paragraph_split_token_ids
from seal.vector_generation import compute_steering_vector, save_vector
from utils import auto_device, extract_gold, extract_gsm8k_answer, normalize_answer, set_seed


def render_prompt(tokenizer, question: str) -> str:
    messages = build_agent_messages_single_agent(question=question, args=argparse.Namespace(task="gsm8k"))
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_trace(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = encoded["input_ids"].shape[1]
    return tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def collect_hidden_at_steps(model, tokenizer, full_text: str, step_index: torch.LongTensor, layer: int):
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    out = model(**encoded, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[layer][0]
    if step_index.numel() == 0:
        return torch.empty(0, hidden.shape[-1])
    return hidden[step_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--n_per_class", type=int, default=100, help="Target traces per correct/incorrect bucket")
    parser.add_argument("--max_scan", type=int, default=400, help="Max train examples to scan")
    parser.add_argument("--output", type=str, default="artifacts/seal_vectors/qwen3-14b/layer_28_steervec.pt")
    parser.add_argument("--cache_jsonl", type=str, default="artifacts/seal_vectors/traces.jsonl")
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
    ).to(device)
    model.eval()

    split_ids = paragraph_split_token_ids(tokenizer)
    traces: List[Dict] = []
    execution_vecs: List[torch.Tensor] = []
    reflection_vecs: List[torch.Tensor] = []
    transition_vecs: List[torch.Tensor] = []

    correct_target = args.n_per_class
    incorrect_target = args.n_per_class

    for idx, item in enumerate(tqdm(load_gsm8k(split="train"), total=args.max_scan)):
        if idx >= args.max_scan:
            break
        if len(execution_vecs) >= correct_target and len(reflection_vecs) + len(transition_vecs) >= incorrect_target:
            break

        prompt = render_prompt(tokenizer, item["question"])
        response = generate_trace(model, tokenizer, prompt, args.max_new_tokens)
        pred = normalize_answer(extract_gsm8k_answer(response))
        gold = normalize_answer(extract_gold(item["solution"]))
        is_correct = pred is not None and gold is not None and pred == gold

        full_text = prompt + response
        step_index, check_index, switch_index, other_index = classify_step_indices(
            full_text, tokenizer, split_ids
        )
        if step_index.numel() == 0:
            continue

        step_hidden = collect_hidden_at_steps(model, tokenizer, full_text, step_index, args.layer)
        traces.append(
            {
                "question": item["question"],
                "correct": is_correct,
                "prediction": pred,
                "gold": gold,
                "response": response,
            }
        )

        if is_correct and len(execution_vecs) < correct_target and other_index.numel():
            execution_vecs.append(step_hidden[other_index])
        if not is_correct:
            if check_index.numel():
                reflection_vecs.append(step_hidden[check_index])
            if switch_index.numel():
                transition_vecs.append(step_hidden[switch_index])
            elif other_index.numel() and len(reflection_vecs) + len(transition_vecs) < incorrect_target:
                reflection_vecs.append(step_hidden[other_index])

    os.makedirs(os.path.dirname(args.cache_jsonl), exist_ok=True)
    with open(args.cache_jsonl, "w") as f:
        for row in traces:
            f.write(json.dumps(row) + "\n")

    execution = torch.cat(execution_vecs, dim=0) if execution_vecs else torch.empty(0)
    reflection = torch.cat(reflection_vecs, dim=0) if reflection_vecs else torch.empty(0)
    transition = torch.cat(transition_vecs, dim=0) if transition_vecs else torch.empty(0)

    if execution.numel() == 0 or (reflection.numel() == 0 and transition.numel() == 0):
        raise RuntimeError(
            f"Insufficient classified hidden states: execution={execution.shape[0]}, "
            f"reflection={reflection.shape[0]}, transition={transition.shape[0]}"
        )

    vector = compute_steering_vector(execution, reflection, transition)
    save_vector(vector, args.output)
    print(
        json.dumps(
            {
                "vector_path": args.output,
                "layer": args.layer,
                "execution_steps": int(execution.shape[0]),
                "reflection_steps": int(reflection.shape[0]),
                "transition_steps": int(transition.shape[0]),
                "traces": len(traces),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
