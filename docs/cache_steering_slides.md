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

## Slide 7 — Hypothesis: why it *should* beat SEAL (table)
- Target: the actual inter-agent channel (not just latent activations).
- Timing: once (not every step). Reaches the Judger directly.
- Stability: robust to `c_v` (per paper); SEAL is fragile.
- Visual: the comparison table from report §2. (Frame as hypothesis — tested next.)

## Slide 8 — Experiment design
- GSM8K n=200, seed 42, Qwen3-14B, HF, batch size 8.
- A/B + sweep: baseline vs cache steering `c_v ∈ {1,3,6}`, `c_k=0`,
  `positions=last_n=40`, all 40 layers.
- Vectors: 200 GSM8K-train contrastive pairs, 5-shot, Mean-of-Differences.
- One command: `bash scripts/run_cache_steer_pilot.sh`.

## Slide 9 — Results: accuracy (HERO — negative)
- **Steering monotonically HURT accuracy:**
  - Baseline (c_v=0): **59.0%** (118/200)
  - c_v=1: 56.5% (−2.5 pp) · c_v=3: 49.5% (−9.5 pp) · c_v=6: 44.0% (−15.0 pp)
- **Stats:** c_v=1 within noise (p=0.61); c_v=6 significant (p≈0.002).
  Dose-response is near-perfectly linear: **−2.55 pp per unit c_v, R²=0.98**.
- Latency flat at c_v=1; the edit itself is ~free.
- Visual: bar chart sloping *down* left→right; baseline highlighted.
- Speaker note: a *clean linear* collapse = real effect (oversteering), not a bug.

## Slide 10 — Diagnosis: we overdosed
- Paper edits **one** token/layer; we edited **40 positions × 40 layers**
  ≈ **40× the per-position dose** at the same `c_v`.
- Compounding: vectors from **text** K/V applied to **latent** (realigned)
  positions → direction may not even mean "more reasoning" in latent space.
- Monotonic collapse with `c_v` is the textbook oversteering signature.
- Visual: "paper dose (1 token) vs our dose (40×40)" schematic.

## Slide 11 — The decisive next experiment
- **Paper-faithful dosing:** `positions=last` (single token), small `c_v ∈ {0.5,1,2}`.
- Recovers ≥ baseline → idea holds, overdosing was the whole story.
- Still degrades → vectors **don't transfer** text→latent (clean finding about
  LatentMAS's latent space).
- Plus: McNemar on per-question flips; per-role execution/reflection ratio (mentor
  Q2 — needs a TextMAS run since only the Judger emits text).
- One command, ~1.5 GPU-hr, already wired (`OUT=… POSITIONS=last CV_SWEEP="0.5 1 2"`).

## Slide 12 — Conclusion + future work
- **Cache steering is correctly integrated and verified — but the naive
  full-latent config oversteers and hurts GSM8K (59% → 44%).**
- A useful negative result: failure mode is specific (overdosing + text→latent
  mismatch) and directly testable.
- Next: paper-faithful single-token sweep; then more pairs, **MATH-train vectors**
  (highest upside), RISER-style per-role/adaptive steering.
- One-liner: *"Right object, wrong dose — and now we know the dose."*

## Appendix
- A1 — Reproduce (`smoke_test_cache_steer.py`, `run_cache_steer_pilot.sh`).
- A2 — File index (report §4 table).
- A3 — References (report §11).
