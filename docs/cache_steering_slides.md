# Slides: Cache Steering for Latent Multi-Agent Reasoning

Hand to a slide-building agent. Each `##` = one slide. Numbers come from
`docs/cache_steering_report.md` and (after the run) `results/cache_steer/*.json`.
Format: 16:9, clean academic style. ~12 slides + appendix.

---

## Slide 1 — Title
- **Title:** Cache Steering for Latent Multi-Agent Reasoning
- **Subtitle:** A one-shot KV-cache intervention for LatentMAS
- Model: Qwen3-14B · Task: GSM8K · Date: 2026-06-04
- Speaker note: framed as a *better-matched* successor to our SEAL pilot.

## Slide 2 — Recap: where we left off (SEAL pilot)
- LatentMAS = agents talk through a shared **KV cache**, only the Judger writes text.
- We tried SEAL (activation steering on latent steps): **62% → 63%, within noise.**
- Why it under-delivered: Judger left unsteered; activation steering is fragile;
  steering direction ambiguous.
- Visual: prior 62/63 bar with a "within noise" amber callout.

## Slide 3 — Key idea
- **LatentMAS already communicates via the KV cache.**
- **Cache steering** edits that exact cache, **once**, before the Judger decodes.
- One sentence: *steer the working memory the Judger reads from.*
- Visual: pipeline with a star on the KV cache hand-off to the Judger.

## Slide 4 — What is cache steering? (background)
- Belitsky et al. 2025 (arXiv:2507.08799), ICLR 2026 submission.
- One-shot edit per layer: `K* = K + c_k·S^k`, `V* = V + c_v·S^v`.
- Vectors from contrastive pairs (CoT vs answer-only), **Mean-of-Differences** at
  the last prompt token.
- vs activation steering: no compounding across layers → **stable + ~free at runtime.**
- Visual: Figure-1-style "activation steering (every step) vs cache steering (once)".

## Slide 5 — Method: extraction
- Positive = few-shot **CoT** (GSM8K-train human solutions); Negative = answer-only.
- Read per-layer last-token K/V; average (pos − neg) over N pairs → `S^k, S^v` `[40,8,128]`.
- GSM8K defaults: `c_k=0` (value-only), `c_v≈1–3`, ~200 pairs, 5-shot.
- Visual: contrastive pair box → per-layer K/V deltas → saved vector.

## Slide 6 — Method: where we inject it
- Between cache accumulation (Planner→Critic→Refiner) and Judger decode.
- Positions: `last_n` (trailing latent message), all layers.
- HF backend only (vLLM doesn't expose the cache).
- Visual: annotated `methods/latent_mas.py` flow with the `cache_steer.apply()` point.

## Slide 7 — Why this should beat SEAL (table)
- Target: the actual inter-agent channel (not just latent activations).
- Timing: once (not every step). Reaches the Judger directly.
- Stability: robust to `c_v` (paper); SEAL is fragile.
- Visual: the comparison table from report §2.

## Slide 8 — Experiment design
- GSM8K n=200, seed 42, Qwen3-14B, HF, batch size 8.
- A/B + sweep: baseline vs cache steering `c_v ∈ {1,3,6}` (+ optional SEAL ref).
- One command: `bash scripts/run_cache_steer_pilot.sh`.
- Visual: condition cards.

## Slide 9 — Results: accuracy (HERO)
- Drop in the table from report §6.1; Δ vs baseline highlighted.
- Callout: report a proportion test, not just the point delta.
- Visual: bar chart baseline vs best `c_v`, SEAL point overlaid.

## Slide 10 — Results: stability + structure
- `c_v` curve (expected flat) vs SEAL's single fragile point.
- Token count + execution-thought ratio before/after (tests "more explicit reasoning").
- Visual: stability line + a 1–2 example before/after Judger trace.

## Slide 11 — Per-role execution/reflection ratio (mentor Q2)
- Classify each agent's thoughts: execution vs reflection vs transition.
- **Caveat:** only the Judger emits text in LatentMAS → true per-role needs a
  TextMAS run.
- Why it matters: tells us *where* to steer (e.g. Critic = reflection-heavy).
- Visual: bar chart of execution ratio per role (Planner/Critic/Refiner/Judger).

## Slide 12 — Conclusion + future work
- Cache steering is the right-shaped intervention for a KV-based MAS; integration
  done + verified.
- Future: `c_k>0` and finer sweeps, more extraction pairs, **MATH-train vectors**
  (highest upside), larger batch, RISER-style adaptive/per-role steering.
- One-liner: *"We steer the latent memory, once, where it actually matters."*

## Appendix
- A1 — Reproduce (`smoke_test_cache_steer.py`, `run_cache_steer_pilot.sh`).
- A2 — File index (report §4 table).
- A3 — References (report §11).
