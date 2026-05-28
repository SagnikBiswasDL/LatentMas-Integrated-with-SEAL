# GSM8K Pilot: LatentMAS vs LatentMAS + SEAL

**Date:** 2026-05-28  
**Repo:** [LatentMas-Integrated-with-SEAL](https://github.com/SagnikBiswasDL/LatentMas-Integrated-with-SEAL)  
**Hardware:** 1× NVIDIA H200 (RunPod)

## Hypothesis

Steering latent multi-agent reasoning with a SEAL execution–reflection vector (applied during Planner/Critic/Refiner latent steps) improves GSM8K accuracy when LatentMAS passes latent KV to a Judger for final decoding.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-14B |
| Task | GSM8K test split |
| Samples | 100 |
| Seed | 42 |
| Method | `latent_mas`, sequential prompt |
| Latent steps | 40 per non-Judger agent |
| Agents | Planner → Critic → Refiner → Judger |
| Backend | HuggingFace, 1 GPU (no vLLM) |
| Batch size | 4 |
| Temperature / top_p | 0.6 / 0.95 |
| Max new tokens (Judger) | 2048 |
| SEAL layer | 28 |
| SEAL coefficient | 1.0 |
| SEAL mode | `latent` (hooks active on latent forwards only) |

### SEAL vector extraction

- Source: GSM8K train CoT traces, Qwen3-14B
- Traces used: 146 (early stop; scanned ≤300)
- Steps classified: 716 execution, 473 reflection, 7 transition
- Artifact: `artifacts/seal_vectors/qwen3-14b/layer_28_steervec.pt` (on RunPod; not committed)

## Results

| Run | Accuracy | Correct | Wall time | sec/sample |
|-----|----------|---------|-----------|------------|
| LatentMAS (baseline) | **62.0%** | 62/100 | 1255 s (~21 min) | 12.55 |
| LatentMAS + SEAL | **63.0%** | 63/100 | 1116 s (~19 min) | 11.16 |
| **Δ (SEAL − baseline)** | **+1.0 pp** | **+1** | **−139 s** | −1.39 |

Structured metrics: [`results/pilot/summary.json`](../results/pilot/summary.json)

## Interpretation

- **Integration works end-to-end:** vector extraction, hooks in latent forward pass, paired eval completed.
- **Accuracy gain is negligible:** +1 question on n=100 is within sampling noise (~±3–5 pp expected).
- **No evidence of harm:** baseline was not degraded.
- **Speed:** SEAL run was ~11% faster; may reflect less repetitive Judger output or run variance—not a primary claim.

## Limitations

1. Small n (100); not statistically powered for +1 pp effects.
2. SEAL applied in **latent mode only**—Judger text generation was not steered.
3. HF single-GPU path vs LatentMAS paper’s vLLM dual-GPU setup.
4. Answer parser sensitive to malformed `\boxed{}` (some correct reasoning scored wrong).
5. SEAL vector from text CoT, applied after latent-space realignment—possible domain mismatch.
6. Single layer/coef; no hyperparameter sweep.
7. Full per-question logs were tee’d on RunPod; only summary JSON is archived in git.

## Why SEAL might not help much (hypotheses for follow-up)

1. **Steering scope:** Final answer is produced by Judger without SEAL in `latent` mode.
2. **Weak signal at one layer:** One hook at L28 vs 40 latent steps × 3 agents.
3. **Vector / space mismatch:** CoT text activations vs realigned latent embeddings.
4. **Task fit:** GSM8K may be less sensitive to execution/reflection calibration than other benchmarks.
5. **Under-tuned vector:** Partial extraction (146 traces), fixed coef=1.0.

## Recommended next steps

- [ ] Scale to n=500 or full GSM8K test; save per-question predictions.
- [ ] Try `--seal_mode both` (latent + Judger decoding).
- [ ] Sweep `seal_coef` and `seal_layer`.
- [ ] Fix `\boxed{}` answer extraction for fairer absolute accuracy.
- [ ] 2× GPU + vLLM to match LatentMAS paper conditions.

## Reproduce

```bash
tmux new -s pilot
cd LatentMas-Integrated-with-SEAL
bash scripts/run_pilot.sh
# Ctrl+B, D to detach
```

Requires CUDA GPU, ~2 GPU-hours for eval (vector extraction skipped if artifact exists).
