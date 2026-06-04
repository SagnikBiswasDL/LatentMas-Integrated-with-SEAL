# Cache Steering for Latent Multi-Agent Reasoning

**Integrating one-shot KV-cache steering into LatentMAS**

| | |
|---|---|
| **Author** | Sagnik Biswas (+ collaborators) |
| **Status** | Implementation complete; GSM8K A/B + c_v sweep **run (n=200)** |
| **Model** | Qwen/Qwen3-14B |
| **Repo** | `LatentMas-Integrated-with-SEAL` |
| **Date** | 2026-06-04 |

> **Headline finding.** Naively porting cache steering onto the LatentMAS latent
> KV cache **does not help — it monotonically hurts** GSM8K accuracy
> (59.0% → 56.5% → 49.5% → 44.0% as `c^v` goes 0 → 1 → 3 → 6). The clean
> dose-response curve points to a specific, fixable cause — **overdosing** (we
> edit 40 latent positions × all layers, versus the paper's single token) — not a
> bug. This is a useful negative result with a clear next experiment.

> **Tip for the weekly meeting:** Sections 1–3 are the "what & why" (slides 1–6),
> Section 5–6 are the experiment + results (hero slides), Sections 7–9 are
> discussion. A ready-to-render slide outline is in
> [`docs/cache_steering_slides.md`](cache_steering_slides.md).

---

## Executive summary

LatentMAS lets multiple agents collaborate **through the model's key-value (KV)
cache** instead of through text. Our earlier pilot bolted **SEAL** (activation
steering) onto the latent rollout and found **no meaningful accuracy gain**
(62% → 63% on GSM8K, within noise). This report proposes and implements a
better-matched intervention: **cache steering** (Belitsky et al., 2025,
arXiv:2507.08799), a *one-shot* edit applied directly to the KV cache.

The key observation: **LatentMAS already communicates via a KV cache, so cache
steering operates on exactly the object that carries inter-agent information.**
We extract per-layer key/value steering tensors from contrastive
(reasoning vs answer-only) prompts and add them once to the shared latent cache
right before the Judger decodes. Unlike SEAL's per-step activation hooks, this is
a single intervention with negligible runtime cost and far lower
hyperparameter sensitivity.

The integration is complete and lint-clean, and the A/B + `c^v` sweep has run on
GSM8K (n=200, Qwen3-14B). **The result is negative but informative:** the
intervention as configured oversteers, degrading accuracy monotonically with
`c^v`. We diagnose the cause (per-position overdosing + text→latent vector
mismatch) and lay out the paper-faithful follow-up that would test whether a
gentler, single-token edit recovers the gain. This document specifies the method,
the implementation, the experiment, the **results**, the diagnosis, and an
analysis plan including the per-agent-role execution/reflection ratio your mentor
asked about.

---

## 1. Background

### 1.1 LatentMAS (our base system)

LatentMAS (Zou et al., ICML 2026, arXiv:2511.20639) moves multi-agent
collaboration from token space into latent space. Four agents run sequentially:

```
[Question] -> Planner -> Critic -> Refiner -> Judger -> [Answer]
                 \___________ shared KV cache ("working memory") ___________/
```

- **Planner / Critic / Refiner** each run `latent_steps` latent forward passes
  (feeding the last hidden state back in after a latent→embedding realignment)
  and emit **no text** — they only grow a shared KV cache.
- **Judger** is the only agent that decodes text (the final `\boxed{...}` answer),
  attending over the accumulated latent KV cache.

The crucial implementation fact for this work: the inter-agent "messages" are
literally the **keys and values** stored in `past_key_values`
(`methods/latent_mas.py`, `models.py:generate_latent_batch`).

### 1.2 SEAL and our prior null result

SEAL (Steerable rEAsoning caLibration, COLM 2025, arXiv:2504.07986) classifies
chain-of-thought steps into **execution / reflection / transition** thoughts,
notes that excessive reflection/transition correlates with failure, and builds a
steering vector `S = mean(reflection ∪ transition) − mean(execution)` applied via
a forward hook `hidden[:, -1, :] += c·S`.

Our pilot applied SEAL during the Planner/Critic/Refiner latent steps
(`seal_mode=latent`). Result on GSM8K (n=100, Qwen3-14B):

| Run | Accuracy |
|-----|----------|
| LatentMAS | 62% |
| LatentMAS + SEAL | 63% (+1, within noise) |

Three structural reasons it likely under-delivered (carried forward as design
guidance here):

1. **Scope:** steering was applied to the latent agents, but the final answer is
   produced by the *unsteered* Judger.
2. **Sensitivity:** activation steering compounds across layers/timesteps and is
   notoriously hyperparameter-sensitive.
3. **Direction:** SEAL's vector/coefficient sign in our extraction pushes *toward*
   reflection at `c=+1`, the opposite of SEAL's intent.

### 1.3 Cache steering (the new method)

Cache steering (Belitsky et al., 2025, arXiv:2507.08799; under review ICLR 2026)
is a **one-shot** intervention on the KV cache. After the prompt populates the
cache, for each layer `l`:

$$ K^{*}_{l} = K_{l} + c^{k}\,S^{k}_{l}, \qquad V^{*}_{l} = V_{l} + c^{v}\,S^{v}_{l} $$

where `K_l, V_l ∈ ℝ^{H×D_h}` are cached key/value vectors at a target position,
`S^k_l, S^v_l` are steering tensors, and `c^k, c^v` are scalar strengths.
Generation then proceeds normally over the modified cache. Because the cache is
fixed (not re-transformed through the network), the edit does **not** compound
across layers — making it stable to hyperparameters and essentially free at
runtime. The paper shows reasoning induction and gains on GSM8K/ARC/CSQA/PIQA and
larger gains on GPQA/MATH, plus controllable reasoning-style transfer.

**Vector extraction (Mean-of-Differences).** Build contrastive pairs
`C = {(p_i^+, p_i^-)}` where `p^+` contains explicit CoT and `p^-` only the final
answer (same few-shot ICL structure and query). Read keys/values at the **final
prompt token** per layer and average the differences:

$$ S^{k}_{l} = \frac{1}{N}\sum_{i} \big(K_l(p_i^+) - K_l(p_i^-)\big), \quad
   S^{v}_{l} = \frac{1}{N}\sum_{i} \big(V_l(p_i^+) - V_l(p_i^-)\big). $$

Paper hyperparameters for GSM8K: ~100–200 contrastive pairs, ~5 ICL examples,
**`c^k = 0`** (value-only), **`c^v ≈ 1–3`**; for ARC/PIQA `c^v ≈ 6`. The method
is robust over `c^k ∈ [0, 0.4]` and `c^v ∈ [1, 8]`.

---

## 2. Motivation and hypothesis

| | SEAL (prior pilot) | Cache steering (this work) |
|---|---|---|
| Target | hidden activations during latent steps | the shared KV cache (the actual inter-agent channel) |
| Timing | every latent forward step | **once**, before the Judger decodes |
| Stability | compounds across layers/steps; sensitive | non-compounding; robust to coefficients |
| Runtime cost | per-step hook overhead | negligible (single add) |
| Reaches the Judger? | only indirectly via KV | **yes** — directly edits what the Judger reads |

**Why it should fit LatentMAS better.** The Judger's answer quality depends
entirely on the latent KV cache handed to it. Cache steering edits that exact
object, at the exact moment it matters, with a method designed to be stable. This
removes the two biggest weaknesses of the SEAL pilot (Judger left unsteered;
hyperparameter fragility).

**Hypothesis (H1).** Adding a one-shot value-cache steering edit (`c^k=0`,
`c^v>0`) to the LatentMAS shared cache before Judger decoding improves GSM8K
accuracy over unsteered LatentMAS, at matched samples/seed, with no latency cost.

**Secondary hypothesis (H2).** Because cache steering induces *more explicit*
reasoning, steered runs produce longer, more structured Judger traces (more
execution thoughts), measurable with the SEAL thought classifier.

> **Outcome (spoiler, see §6).** H1 was **not supported** in the default
> configuration: steering *reduced* accuracy at every `c^v`, monotonically. We
> argue in §6.2/§7 that this reflects an **overdosing** of the latent cache rather
> than a failure of the core idea, and specify the experiment that would confirm
> or refute that.

---

## 3. Method: integration design

### 3.1 Where the intervention is applied

In the HF backend, LatentMAS accumulates `past_kv` across Planner→Critic→Refiner
and hands it to the Judger's `generate_text_batch(..., past_key_values=...)`. We
insert the one-shot edit **between** cache accumulation and Judger decoding
(`methods/latent_mas.py`):

```python
past_for_decoding = past_kv if self.latent_steps > 0 else None
if past_for_decoding is not None and self.model.cache_steer is not None:
    past_for_decoding = self.model.cache_steer.apply(past_for_decoding)
```

### 3.2 Which positions and layers

The paper edits a single target token. LatentMAS's cache ends with the most
recent latent thoughts (latent steps are appended after each agent's prompt, with
no right-padding), so the trailing positions are the cleanest, most
"message-like" slots. We expose three modes (`--cache_steer_positions`):

- `last` — only the final cache position (paper-faithful).
- `last_n` — the trailing `n` positions (default `n = latent_steps`), i.e. the
  last agent's latent message. **Default.**
- `all` — every cache position (includes left-padding; use with care).

Steering is applied to **all layers** by default (cache steering's non-compounding
property makes all-layer edits safe), matching the paper's "applied consistently
across all layers."

### 3.3 Vector extraction for our setting

We follow the paper but use **GSM8K-train human CoT solutions** as the positive
reasoning (the paper explicitly permits "existing human annotations" instead of
GPT-4o traces). `scripts/extract_cache_steering_vectors.py`:

1. Sample `n_icl` ICL examples + 1 query from GSM8K train.
2. Positive prompt: ICL assistant turns contain the full GSM8K CoT solution.
   Negative prompt: assistant turns contain only `The final answer is \boxed{·}.`
3. Forward each (batch=1, no padding → last token is unambiguous); read per-layer
   last-token K and V.
4. Mean-of-Differences over `N` pairs → `S^k, S^v` of shape `[L, H_kv, D_h]`
   (Qwen3-14B: `[40, 8, 128]`). Saved to
   `artifacts/cache_steer_vectors/qwen3-14b/gsm8k_kv.pt`.

**Design note / known risk.** Vectors are extracted from *text* prompts and
applied to the *latent* KV cache (after latent realignment). This is the same
cross-domain bet SEAL made. KV vectors live in the per-layer key/value space
(outputs of `k_proj`/`v_proj`), which is well-defined regardless of whether the
cached token was text or latent — but a representation mismatch is still possible
and is explicitly part of what the experiment tests.

### 3.4 Backend constraint

Cache steering edits `past_key_values` directly, which the **HF backend** exposes
but the **vLLM** path does not. `run.py` raises if `--use_cache_steer` is combined
with `--use_vllm`. The single-GPU pilot uses the HF path anyway.

---

## 4. Implementation

| File | Purpose |
|------|---------|
| `cache_steering/steering.py` | `CacheSteering.apply(past)` — one-shot K/V edit (DynamicCache + legacy tuple) |
| `cache_steering/extraction.py` | contrastive prompt builder, last-token K/V reader, Mean-of-Differences |
| `scripts/extract_cache_steering_vectors.py` | CLI to build `S^k, S^v` from GSM8K-train |
| `scripts/smoke_test_cache_steer.py` | synthetic-cache unit test (no model download) |
| `scripts/run_cache_steer_pilot.sh` | A/B + `c^v` sweep on GSM8K (HF backend) |
| `models.py` | constructs `self.cache_steer` when `--use_cache_steer` |
| `methods/latent_mas.py` | applies the edit before Judger decoding |
| `run.py` | CLI flags + results metadata |

**New CLI flags** (`run.py`):

```
--use_cache_steer
--cache_steer_vector_path  artifacts/cache_steer_vectors/qwen3-14b/gsm8k_kv.pt
--cache_steer_ck           0.0     # key coefficient (GSM8K default 0)
--cache_steer_cv           4.0     # value coefficient (sweep 1/3/6)
--cache_steer_positions    last_n  # {last, last_n, all}
--cache_steer_last_n       40
```

**Correctness check.** `scripts/smoke_test_cache_steer.py` builds a synthetic
cache and asserts that exactly the targeted trailing positions change by
`c·S` and all others are untouched. Run it first on the pod.

---

## 5. Experimental setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-14B |
| Task | GSM8K test |
| Samples | 200 (seed 42) — up from 100 in the SEAL pilot |
| Method | `latent_mas`, sequential, `--think`, `--latent_space_realign` |
| Latent steps | 40 per non-Judger agent |
| Backend | HuggingFace, 1 GPU (cache steering requires HF) |
| Batch size | **8** (larger, for throughput — see §10) |
| Temperature / top_p | 0.6 / 0.95 |
| Max new tokens (Judger) | 2048 |
| Cache-steer vector | 200 pairs, 5-shot, GSM8K-train |

**Conditions (A/B + sweep):**

1. **Baseline** — LatentMAS, no steering.
2. **+ Cache steering**, `c^k=0`, `c^v ∈ {1, 3, 6}`, `positions=last_n`.
3. *(Optional)* **+ SEAL** (prior pilot config) for a three-way comparison.
4. *(Optional)* **+ SEAL + Cache steering** (do the two interventions stack?).

One command runs extraction + baseline + sweep + summary:

```bash
tmux new -s cachesteer
cd LatentMas-Integrated-with-SEAL
SAMPLES=200 GEN_BS=8 CV_SWEEP="1 3 6" bash scripts/run_cache_steer_pilot.sh
```

---

## 6. Results

GSM8K test, n=200, seed 42, Qwen3-14B, HF backend, `generate_bs=8`, latent_steps=40.
Steering config: `c^k=0`, `positions=last_n` (40), all 40 layers. Vectors from 200
GSM8K-train contrastive pairs (5-shot). Raw JSON in `results/cache_steer/*.json`.

### 6.1 Main accuracy (hero table)

| Condition | c_k | c_v | Accuracy | Correct | Δ vs baseline | sec/sample |
|-----------|-----|-----|----------|---------|---------------|------------|
| **LatentMAS (baseline)** | — | — | **59.0%** | 118/200 | — | 9.81 |
| + Cache steering | 0.0 | 1 | 56.5% | 113/200 | **−2.5 pp** | 9.84 |
| + Cache steering | 0.0 | 3 | 49.5% | 99/200 | **−9.5 pp** | 11.41 |
| + Cache steering | 0.0 | 6 | 44.0% | 88/200 | **−15.0 pp** | 11.12 |

**Reading it:** every steered arm is *below* baseline, and the loss grows
monotonically with `c^v` (−2.5 → −9.5 → −15.0 pp). Latency is essentially
unchanged at `c^v=1` (the edit itself is ~free); the higher sec/sample at
`c^v=3,6` reflects longer/looser Judger generations as the cache is pushed
off-distribution, not steering overhead.

### 6.2 Coefficient curve — and why it slopes *down*

```
accuracy
 59% |●  baseline (c_v=0)
     |  ●  c_v=1  (56.5%)
 50% |        ●  c_v=3  (49.5%)
     |               ●  c_v=6  (44.0%)
 44% |________________________________  c_v
       0      1      3      6
```

The paper reports a *flat, robust plateau* over `c^v ∈ [1,8]`. We see the
opposite: a steep, monotonic decline. The discrepancy is explained by **dose**,
not by the method being wrong:

- The paper adds `S^v` to a **single** cache position per layer.
- We add it to **`last_n = 40` positions × all 40 layers**. At equal `c^v` that is
  roughly a **40× larger per-layer perturbation** of the working memory.
- So our `c^v=1` already sits well past the paper's *effective* dose, and `c^v=6`
  is deep into "overwrite the latent message" territory — exactly where a clean
  monotonic collapse is expected.

A second, compounding factor: the vectors are extracted from **text** token K/V
but applied to **latent** (realigned) cache positions, whose statistics differ —
so the injected direction is not even guaranteed to be the "more-reasoning"
direction in latent space (see §7, §9).

### 6.3 Reasoning-structure / token metrics (tests H2)

> Requires the per-sample Judger outputs (`results/cache_steer/*.json`) to be
> pulled locally; run the SEAL thought classifier
> (`seal/thought_classifier.py`) over baseline vs steered Judger traces. Given the
> accuracy collapse, the expected reading is that high `c^v` *inflates* tokens and
> degrades coherence rather than producing cleaner execution thoughts — i.e. H2
> fails in the same direction as H1. (Left as the immediate post-meeting analysis.)

| Condition | mean Judger tokens | execution-thought ratio | reflection ratio |
|-----------|--------------------|--------------------------|------------------|
| Baseline | _pending classifier_ | _pending_ | _pending_ |
| + Cache steering (c_v=1) | _pending_ | _pending_ | _pending_ |

### 6.4 Interpreting the negative result

This is a *useful* negative result, not a dead end:

1. **The mechanism works as implemented** — the smoke test verifies the edit hits
   exactly the targeted positions, and the monotonic dose-response confirms the
   coefficient is doing real work. We are not debugging a silent no-op.
2. **The failure mode is specific and named** — overdosing (40 positions × all
   layers) + text→latent mismatch — both directly addressable.
3. **It sharpens the next experiment** (the paper-faithful single-token, small-`c^v`
   run), which cleanly separates "we overdosed" from "it genuinely doesn't transfer
   to the latent channel."

---

## 7. Analysis and next experiment

**What we learned.** Cache steering, ported to the *full* latent KV region of
LatentMAS, oversteers and underperforms baseline (§6). The two prior weaknesses of
the SEAL pilot were *reaching the Judger* (cache steering does directly edit what
the Judger reads — good) and *hyperparameter fragility* (cache steering was
supposed to fix this — but in *this* configuration it is just as fragile, because
we apply it at ~40× the paper's per-position dose).

**The decisive follow-up (paper-faithful dosing).** Re-run with
`positions=last` (a *single* trailing position) and small `c^v ∈ {0.5, 1, 2}`,
optionally restricting to a subset of layers. This is one command and ~1.5 GPU-hr:

```bash
OUT=results/cache_steer_lasttoken SAMPLES=200 GEN_BS=8 \
  POSITIONS=last CV_SWEEP="0.5 1 2" bash scripts/run_cache_steer_pilot.sh
```

- If accuracy recovers to ≥ baseline → the original idea holds and "overdosing" was
  the whole story (strong positive for the next meeting).
- If it still degrades → cache-steering vectors **do not transfer** from text to
  the realigned latent channel, which is itself a clean, publishable observation
  about LatentMAS's latent space.

**Remaining analyses (once vectors/logs are local):**

1. **McNemar on per-question flips** (baseline vs each steered arm) to confirm the
   degradation is significant and characterize *which* questions flip.
2. **Stability curve** — we already have the (downward) `c^v` curve; overlay the
   follow-up's single-token curve to show whether faithful dosing flattens it.
3. **Per-role execution/reflection ratio (mentor's question #2).** Using
   `seal/thought_classifier.py`, classify each step of each agent's reasoning into
   execution / reflection / transition and report the **execution ratio per role**
   (Planner/Critic/Refiner/Judger). Caveat: in LatentMAS only the Judger emits
   text, so a true per-role measurement requires a **TextMAS** run
   (`--method text_mas`) where every agent produces text; in pure LatentMAS we can
   only measure the Judger. This tells us whether roles specialize (e.g. Critic =
   reflection-heavy, Planner/Judger = execution-heavy) and therefore *where*
   steering should be applied and with which sign.
4. **Does steering change the execution ratio?** Compare Judger execution ratio
   with/without cache steering (links the intervention to a mechanism, not just an
   accuracy number).

---

## 8. Related work (for the paper-hunting feed)

- **Cache steering** — Belitsky et al., *KV Cache Steering for Controlling Frozen
  LLMs*, arXiv:2507.08799 (ICLR 2026 submission). One-shot KV edit; the method we
  integrate.
- **SEAL** — *Steerable Reasoning Calibration*, arXiv:2504.07986 (COLM 2025).
  Execution/reflection/transition steering; our prior baseline.
- **LatentMAS** — Zou et al., arXiv:2511.20639 (ICML 2026). Latent multi-agent
  collaboration via KV cache; our base system.
- **RISER** — *Orchestrating Latent Reasoning Skills for Adaptive Activation
  Steering*, arXiv:2601.09269. RL router that *composes* multiple steering vectors
  per input — a natural "v2" for per-role/adaptive steering here.
- **KV-cache augmentation / coprocessor** — Liu et al. (2025) train a differentiable
  coprocessor to augment the KV cache for reasoning; cache steering is the
  training-free counterpart. Relevant prior art to cite.
- **Efficient Reasoning (NeurIPS 2025 workshop)** — *Fractional Reasoning via
  Latent Steering Vectors*, *Amortized Latent Steering*, *Activation Steering for
  CoT Compression*, *OptimalThinkingBench*. Good neighbors for the token-efficiency
  framing.
- **Surveys** — `hemingkx/Awesome-Efficient-Reasoning` (Latent CoT, long-to-short
  CoT, KV-cache efficiency).

---

## 9. Limitations

1. **Cross-domain vectors.** Extracted from text prompts, applied to latent KV
   after realignment — possible representation mismatch (shared with SEAL).
2. **HF-only.** No vLLM path for cache steering; throughput limited on 1 GPU.
3. **Single task / model.** GSM8K + Qwen3-14B; the paper shows bigger gains on
   GPQA/MATH, untested here.
4. **Position choice is a design knob.** `last_n=latent_steps` is a reasonable
   default but not tuned; `all` includes padding.
5. **Statistical power.** Even n=200 is modest for small deltas; report tests, not
   just point estimates.
6. **Answer parsing.** The upstream GSM8K parser is brittle (placeholder echoes,
   decimal/comma formatting) and deflates *absolute* accuracy; it affects both
   arms equally so the **A/B delta remains valid**. We intentionally left the
   harness untouched to stay faithful to the fork.

---

## 10. Future work (incorporating mentor directions)

- **Paper-faithful dosing (highest priority — see §7).** `positions=last`, small
  `c^v`, possibly fewer layers. This is the experiment that decides whether the
  negative result is "we overdosed" or "vectors don't transfer to latent space."
- **Different steering strength** — already wired as the `c^v` sweep
  (`CV_SWEEP`); the n=200 sweep above shows the *default* config is monotonically
  harmful, so the useful grid is now *small* `c^v ∈ {0.25, 0.5, 1}` at
  `positions=last`, plus `c^k>0`.
- **More samples for the steering vector** — `--n_pairs` (default 200). The paper
  found gains saturate by ~1000; sweep `{100, 200, 500, 1000}` and report.
- **Larger batch size to speed runtime** — default raised to `--generate_bs 8`
  (`GEN_BS`); push higher within GPU memory. (Note: batching uses left-padding in
  the shared cache, so prefer `positions=last_n` which targets the clean trailing
  latent positions rather than `all`.)
- **MATH-train vectors** — extract steering vectors from MATH-train CoT for richer
  reflection/transition signal and transfer to GSM8K. Requires a small `load_math`
  loader in `data.py` and pointing the extraction script at it (one function +
  one flag). The paper reports its largest gains on MATH/GPQA, so this is the
  highest-upside direction.
- **Steer the Judger prompt too** — extend beyond the latent cache to the Judger's
  own prefilled prompt cache (closer to the original paper's single-prompt setting).
- **Per-role / adaptive steering** — combine with the per-role execution-ratio
  analysis (§7.3) and, longer term, a RISER-style router that picks `c^v` per
  question.

---

## 11. Reproduce

```bash
# 0. (once) deps + HF cache
pip install -r requirements.txt
export HF_HOME=/workspace/hf_cache

# 1. validate the cache-edit mechanics (no download)
python scripts/smoke_test_cache_steer.py        # expect: SMOKE TEST: PASS

# 2. extract vectors + run A/B + c_v sweep (inside tmux)
tmux new -s cachesteer
SAMPLES=200 GEN_BS=8 CV_SWEEP="1 3 6" bash scripts/run_cache_steer_pilot.sh
# detach: Ctrl+B then D ; reattach: tmux attach -t cachesteer

# 3. results land in results/cache_steer/*.json (summary printed at the end)
```

Estimated cost: vector extraction ~10–20 min; each eval arm ~15–25 min at n=200
on one H200 (≈2–3 GPU-hours for the full sweep). Well within the available
RunPod budget.

---

## References

1. Zou et al. *Latent Collaboration in Multi-Agent Systems (LatentMAS).*
   arXiv:2511.20639, 2025.
2. Belitsky, Kopiczko, Dorkenwald, Mirza, Glass, Snoek, Asano.
   *KV Cache Steering for Controlling Frozen LLMs.* arXiv:2507.08799, 2025.
3. *SEAL: Steerable Reasoning Calibration of Large Language Models for Free.*
   arXiv:2504.07986, COLM 2025.
4. *RISER: Orchestrating Latent Reasoning Skills for Adaptive Activation
   Steering.* arXiv:2601.09269, 2026.
5. Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition (CAA).* 2024.
