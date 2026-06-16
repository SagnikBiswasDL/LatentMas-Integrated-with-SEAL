# Steering Vectors for LatentMAS — Research Brief

**For:** a **research/strategy agent**. Your job is to survey the steering-vector
literature, weigh it against the system and constraints documented here, and
**return ONE concrete, fully-specified steering strategy** (plus a short fallback)
that an implementing engineer will build and run on GPU. **You do not run
experiments or write code** — you produce a precise, buildable spec. See §0 for
the exact output contract; it is the only thing you must deliver.

**Division of labor:**
- **You (research agent):** pick the best-fit steering strategy from theory +
  this brief; specify it down to vectors, hyperparameters, code hook points, and
  success criteria (§0).
- **Implementing engineer (separate agent):** builds your spec against the
  intervention surfaces (§3) and knobs (§4) documented here, and runs it on
  GPU infra the user provides.
- **User:** provides GPU infrastructure on demand.

**Status of repo at time of writing:** two steering methods integrated and
evaluated; both gave **non-positive** results (SEAL ≈ flat at +1/100; cache
steering monotonically *hurts*). This document gives you everything needed to
choose the next, better-informed strategy without re-discovering the codebase.

**How to use this doc:** §0 is your deliverable (read first and last). §1 is the
goal. §2–§4 are ground truth about the system, where you can intervene, and the
knobs that exist. §5 is what's been tried. §6 lists confirmed bugs / suspicious
choices (cheap wins live here). §7 inventories available data. §8–§9 are the
literature map + a menu of candidate experiments to draw from. §10–§12 are
measurement, constraints, and the hypotheses your strategy should target.

---

## 0. Your deliverable: the steering strategy spec (OUTPUT CONTRACT)

Return a single primary strategy and one fallback, each as a self-contained spec
the engineer can implement without further research. Use exactly these fields:

1. **Name & one-line thesis** — what you're steering and why it should beat
   unsteered LatentMAS (or cleanly answer a §12 hypothesis).
2. **Intervention surface** — which of §3's surfaces (A–F) it edits, and the
   precise edit (e.g. "add `c·S` to value cache at the single last latent
   position, layers 16–32"). If it needs a new surface, describe the exact code
   change point in `methods/latent_mas.py` / `models.py`.
3. **Steering-vector source & extraction recipe** — exactly how to build the
   vector(s): data source (text CoT? native LatentMAS latent activations?
   correct−incorrect contrast?), prompt construction, which layer/positions to
   read, aggregation (mean-of-differences, probe, etc.), and output tensor shape
   (recall Qwen3-14B = 40 layers, hidden 5120, 8 KV heads, head_dim 128). State
   whether an existing script (`extract_seal_vector.py` / `extract_cache_steering_vectors.py`)
   can be reused or modified, or a new extractor is needed.
4. **Hyperparameters + sweep grid** — concrete values and ranges (coefficients,
   layers, positions, sign), with a *default* point and a small ordered sweep.
   Keep the grid GPU-budget-aware (each GSM8K arm ≈ n × ~10–15 s on one H200).
5. **Required code changes** — bullet the edits the engineer must make
   (file + function + what to add). Prefer reusing existing knobs (§4); call out
   anything not yet wired (e.g. §6.3 Judger steering).
6. **Mechanism / prediction** — what measurable signal should move if the
   hypothesis is right (accuracy delta sign+size, thought-ratio shift, token
   counts, cache-norm change), so the engineer knows what "working" looks like.
7. **Success / kill criteria** — the paired statistical bar (test, `n`, minimum
   detectable effect) that decides keep-vs-discard, and what result would falsify
   the strategy.
8. **Why this over the alternatives** — 2–3 sentences ranking it against the
   other candidates in §8/§9, grounded in the failure modes in §5–§6.

Then a **fallback strategy** in the same format (ideally targeting a *different*
hypothesis in §12, so the two runs together are maximally informative).

Keep it implementable and specific. Do **not** return a literature survey, a
multi-phase research program, or "try several and see" — return the one strategy
you would bet the next GPU session on, fully specified.

---

## 1. The goal (the decision the experiments will reach)

The strategy you spec should move us toward answering:

> **Does steering the LatentMAS pipeline produce a meaningful, reproducible
> accuracy change on reasoning tasks — and if not, *why* (mechanistically)?**

A "good" eventual outcome is *either*:
- a steering configuration that beats unsteered LatentMAS by a margin that
  survives a paired significance test at adequate `n`, **or**
- a crisp, falsifiable mechanistic explanation for the null/negative result
  (e.g. "text-derived steering vectors do not transfer to LatentMAS's realigned
  latent KV channel," demonstrated with a controlled experiment).

Both are publishable. Avoid the trap of a third outcome: "we ran more sweeps and
it's still noisy." Your spec must be designed to *discriminate between named
hypotheses* (§12), not just chase a number.

---

## 2. The system under study: LatentMAS (precise mechanics)

Four agents run **sequentially**, sharing one growing KV cache (`past_kv`):

```
[Question] → Planner → Critic → Refiner → Judger → [boxed answer]
                 \________ shared past_key_values ("working memory") ________/
```

- **Planner / Critic / Refiner** emit **no text**. Each runs `latent_steps`
  (=40 in pilots) *latent forward passes*: take the last hidden state, project it
  back to embedding space ("latent realignment"), feed it in as the next
  `inputs_embeds`, and append the resulting K/V to the shared cache.
- **Judger** is the *only* agent that decodes text — it generates the final
  `\boxed{...}` answer, attending over the entire accumulated latent KV cache.

Agent definitions: `methods/__init__.py` (`Planner, Critic, Refiner, Judger`).
Per-role prompts: `prompts.py` (`build_agent_message_sequential_latent_mas`).

### 2.1 The latent loop (read this carefully — it defines the "latent space")

```324:400:models.py
    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
```

The realignment step is the crux of "where latent thoughts live":

```259:268:models.py
    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)
```

Key facts:
- The realign matrix is a least-squares map from the **output (lm_head)**
  embedding space to the **input** embedding space; the result is renormalized to
  the **mean input-embedding norm**. (Built in `_build_latent_realign_matrix`.)
- With `--latent_space_realign` **off**, the matrix is the identity but the
  **renorm still happens**. Both pilots ran with realign **on**.
- This realigned, renormed vector is the carrier of inter-agent "messages." Any
  steering vector you inject either modifies these vectors (SEAL) or the K/V they
  produce (cache steering). **The statistics of this space differ from ordinary
  text-token hidden states — this is the prime suspect for transfer failure.**

### 2.2 The Judger decode

The Judger calls `generate_text_batch(..., past_key_values=past_for_decoding)`
(`models.py:270`). The cache handed in is the accumulated latent KV. **Whatever
reaches the Judger is the only thing that affects the final answer.**

---

## 3. Where you can intervene (the steering "surfaces")

This is the most important section. LatentMAS exposes several *distinct*
intervention points; the two attempts so far hit only two of them. Treat this as
the design space.

| # | Surface | What it edits | When | Reaches Judger answer? | Wired today? |
|---|---------|---------------|------|------------------------|--------------|
| A | **Latent-forward activations** (SEAL) | `hidden[:, -1, :]` at one layer during P/C/R latent steps | every latent step | only indirectly (via KV) | ✅ `seal_mode=latent` |
| B | **Shared KV cache, one-shot** (cache steering) | cached K/V at chosen positions, all layers, once | right before Judger decodes | ✅ directly | ✅ `--use_cache_steer` |
| C | **Judger decode activations** | `hidden[:, -1, :]` during Judger *text* generation | every decoded token | ✅ directly | ⚠️ **dead code** (see §6.3) |
| D | **Judger prompt prefill cache** | K/V of the Judger's own prompt tokens (not just latent KV) | before/at decode | ✅ directly | ❌ not implemented |
| E | **Per-role / selective latent steering** | only some agents (e.g. Critic only), or some layers | per agent | indirectly | ❌ not implemented (single global hook) |
| F | **Realignment itself** | the realign matrix / target norm | every latent step | indirectly | ❌ (would be a method change) |

Note surfaces **C and D are the ones most likely to matter** (they touch the
answer path directly) and are exactly the ones *not* yet tested. The prior null
results steered surfaces that only reach the answer second-hand.

---

## 4. The control knobs that already exist (CLI)

All in `run.py`. No code changes needed to sweep these.

**SEAL (surface A):**
```
--use_seal
--seal_vector_path artifacts/seal_vectors/qwen3-14b/layer_28_steervec.pt
--seal_layer 28          # index into hidden_states (Qwen3-14B has 40 layers)
--seal_coef 1.0          # ⚠️ sign is suspect — see §6.1
--seal_mode latent       # {latent, text, both} — text/both currently no-op, §6.3
```

**Cache steering (surface B):**
```
--use_cache_steer        # HF backend only; errors if combined with --use_vllm
--cache_steer_vector_path artifacts/cache_steer_vectors/qwen3-14b/gsm8k_kv.pt
--cache_steer_ck 0.0     # key coefficient (paper GSM8K default 0)
--cache_steer_cv 4.0     # value coefficient
--cache_steer_positions last_n   # {last, last_n, all}
--cache_steer_last_n 40  # trailing positions when last_n
```

**Pipeline:** `--latent_steps` (0–80), `--latent_space_realign`, `--think`,
`--generate_bs`, `--temperature/--top_p`, `--max_samples`, `--seed`,
`--task {gsm8k,aime2024,aime2025,gpqa,arc_easy,arc_challenge,mbppplus,humanevalplus,medqa}`.

Model: only `Qwen/Qwen3-4B` and `Qwen/Qwen3-14B` are whitelisted in `run.py`.
Qwen3-14B: **40 layers, hidden 5120, 8 KV heads, head_dim 128** (so cache-steer
vectors are `[40, 8, 128]`; SEAL vectors are `[5120]`).

---

## 5. What has been tried (exact configs + results)

### 5.1 SEAL — activation steering on latent forwards (surface A)

- **Method:** `S = mean(reflection ∪ transition) − mean(execution)` hidden
  states; forward hook adds `coef·S` to `hidden[:, -1, :]` at layer 28.
  Implementation: `seal/hooks.py`, registered in `models.py` (`SealSteering`).
- **Extraction:** `scripts/extract_seal_vector.py` — generate GSM8K-train CoT,
  classify each step into execution/reflection/transition by keyword
  (`seal/thought_classifier.py`), collect layer-28 hidden states.
- **Vector stats** (`results/pilot/seal_vector_extraction.json`): 146 traces,
  716 execution / 473 reflection / 7 transition steps, layer 28.
- **Result** (GSM8K, n=100, seed 42, Qwen3-14B, HF):

| Run | Accuracy | sec/sample |
|-----|----------|-----------|
| LatentMAS | 62.0% (62/100) | 12.55 |
| + SEAL (layer 28, coef 1.0, mode latent) | 63.0% (63/100) | 11.16 |

→ **+1/100, within noise.** Full report: `docs/pilot_report.md`. Raw per-question
traces: `results/pilot/latent_mas.json`, `latent_mas_seal.json` (these ARE in
git — see §7).

### 5.2 Cache steering — one-shot KV edit before Judger (surface B)

- **Method:** `K* = K + c_k·S^k`, `V* = V + c_v·S^v` on the shared cache, applied
  once before Judger decode. Implementation: `cache_steering/steering.py`,
  applied in `methods/latent_mas.py:170-175`.
- **Extraction:** `scripts/extract_cache_steering_vectors.py` —
  Mean-of-Differences of last-token K/V between contrastive prompts (CoT vs
  answer-only), 200 pairs, 5-shot GSM8K-train.
- **Result** (GSM8K, n=200, seed 42, Qwen3-14B, HF, positions=last_n=40, all layers):

| c_v | Accuracy | Δ vs baseline | p (2-prop z) |
|-----|----------|---------------|--------------|
| 0 (baseline) | 59.0% | — | — |
| 1 | 56.5% | −2.5 pp | 0.61 (n.s.) |
| 3 | 49.5% | −9.5 pp | 0.055 |
| 6 | 44.0% | −15.0 pp | **0.0024** |

→ **Monotonic degradation**, near-linear (`acc ≈ 0.586 − 0.0255·c_v`, R²=0.98).
Full report + stats: `docs/cache_steering_report.md`.

**Diagnosis on record:** *overdosing* — the paper edits 1 token; this edits 40
positions × 40 layers (~40× per-layer perturbation) — compounded by a text→latent
representation mismatch. The decisive untested follow-up is `positions=last` with
small `c_v ∈ {0.5, 1, 2}`.

---

## 6. Confirmed bugs / suspicious choices (cheap, high-value to test first)

These are concrete issues I found reading the code. Several are near-free
experiments that could flip the conclusion. **Verify each before trusting prior
results.**

### 6.1 SEAL coefficient sign is probably inverted ⚠️
SEAL's *intent* is to **reduce** over-reflection by steering toward execution.
The vector here is `positive(reflection∪transition) − negative(execution)`, and
the pilot **adds** it with `coef = +1.0` — i.e. it steers **toward** reflection,
the opposite of SEAL's design. (`docs/cache_steering_report.md` §1.2 flags this
too.) **Test `--seal_coef -1.0` (and a sweep like {−2,−1,−0.5,+0.5,+1}).** This
alone may explain the flat SEAL result.

### 6.2 SEAL extraction confounds thought-type with correctness ⚠️
In `scripts/extract_seal_vector.py`, **execution** vectors are collected only
from **correct** traces, while **reflection/transition** vectors come only from
**incorrect** traces (and `other` steps of incorrect traces are dumped into
`reflection_vecs` as a fallback). So `S` partly encodes *correct−incorrect*, not
purely *execution−reflection*. This pollutes the steering direction. A cleaner
re-extraction (thought-type within the same traces) is worth doing.

### 6.3 Judger text steering (`--seal_mode text`/`both`) is dead code ⚠️⚠️
`SealSteering.set_text_mask(...)` exists (`seal/hooks.py:69`) but is **never
called** during Judger generation — `methods/latent_mas.py` only calls
`_with_latent_seal()` / `_without_seal()`. So `text_active` stays `False` and
`--seal_mode text`/`both` silently do nothing. This means **surface C has never
actually been tested**, despite being the most direct path to the answer. Wiring
`set_text_mask` around the Judger's `generate_text_batch` call is a small, high-
value change (the presentation skeleton lists it as a TODO too).

### 6.4 Cache-steer default dose is far past the paper's regime
`--cache_steer_positions last_n --cache_steer_last_n 40` × all 40 layers is the
documented overdose. The paper-faithful `positions=last` run was never executed.

### 6.5 Text→latent representation mismatch (shared by both methods)
Both vectors are extracted from **text-token** activations/KV but applied to
**realigned latent** positions (§2.1). Whether these spaces are even comparable
is *untested* and is arguably the single most important scientific question here
(see §10, H-MISMATCH). A direct diagnostic: extract steering vectors from the
*latent* activations themselves (not text CoT).

---

## 7. Data inventory (what you have vs. what's gone)

**Available locally (in git):**
- `results/pilot/latent_mas.json` (645 KB) and `latent_mas_seal.json` (613 KB) —
  **full per-question stdout traces** for the SEAL pilot (baseline + SEAL). These
  contain each problem's agent prompts and the Judger's final text/`\boxed{}`.
  → Mineable now (no GPU) for: paired **flip analysis / McNemar test**, which
  questions SEAL helped vs hurt, and parser-failure cases. *Note:* latent agents
  emit empty text, so only **Judger** output is present — per-role thought
  analysis on pure LatentMAS is limited to the Judger.
- `results/pilot/{summary,baseline_summary,seal_summary,seal_vector_extraction}.json`
  — headline metrics + extraction metadata.

**Lost / not committed (need regeneration on GPU):**
- **All steering vector `.pt` files** (SEAL `layer_28_steervec.pt` and
  cache-steer `gsm8k_kv.pt`) — were on the now-dead RunPod pod. Re-extract via the
  scripts before any run (the pilot/sweep shells auto-extract if missing).
- **Cache-steering per-sample traces** (`results/cache_steer/*.json`) — torn down
  with the pod. Only the accuracy table in the report survives. The paired
  McNemar / token-structure (H2) analysis for cache steering **cannot** be done
  without a re-run. **Transfer logs off-box immediately next time.**

**Datasets:** GSM8K loads from HF (`openai/gsm8k`); `data/medqa.json` is the only
local dataset file.

---

## 8. Literature map for steering — and fit to LatentMAS

Use this to choose method #3 (and beyond). Fit = how well it matches LatentMAS's
latent-KV channel and the constraints in §11. References already cited in
`docs/cache_steering_report.md` §8.

| Method | Idea | Fit to LatentMAS | Notes / where to apply |
|--------|------|------------------|------------------------|
| **CAA** (Rimsky et al. 2024) | contrastive activation addition at a layer | medium | classic baseline for surface C (Judger) with a *clean* contrastive set |
| **SEAL** (arXiv:2504.07986) | execution/reflection calibration | tried (A); revisit with §6.1/§6.2 fixes | also try on surface C |
| **Cache steering** (arXiv:2507.08799) | one-shot K/V edit | tried (B); revisit paper-faithful dose §6.4 | natural fit — edits the actual channel |
| **RISER** (arXiv:2601.09269) | RL router that *composes* steering vectors per input | high upside, high effort | "v2": per-role/adaptive `c_v`; needs the per-role analysis first |
| **Fractional / Amortized latent steering** (NeurIPS'25 workshop) | tunable-strength latent steering | high conceptual fit (LatentMAS *is* latent) | could steer realigned latent vecs directly (surface F-ish) |
| **KV-cache coprocessor** (Liu et al. 2025) | *trained* cache augmentation | out of scope (training-free is the project ethos) | cite as the trained counterpart |
| **ITI / contrastive-probe steering** | probe-derived directions | medium | useful for *finding* a transferable latent direction (§10) |

**Strong recommendation for vector source:** because the prime suspect is
text→latent mismatch (§6.5), prioritize methods/extractions that build the
steering direction **from LatentMAS's own latent activations** rather than from
text CoT. E.g. contrast latent trajectories of *correct vs incorrect* LatentMAS
runs at a layer, then steer surface C/A with that. This directly tests whether a
*native* direction works even if the *text-derived* one doesn't.

---

## 9. Candidate experiments to draw your strategy from (menu, not assignment)

These are concrete, already-feasible interventions the engineer can run; use them
as raw material when writing your §0 spec (your strategy may be one of these,
a refinement, or something better from §8). Ordered by (value × cheapness).
Commands are illustrative of what the engineer will run. Costs are ~H200 ballpark
from prior runs (~10–15 s/sample at n; vector extraction 10–20 min).

**E0 — Free, no GPU: mine the existing SEAL traces.**
Parse `results/pilot/latent_mas.json` vs `latent_mas_seal.json`: build the 2×2
flip matrix, run **McNemar**, list helped/hurt questions, and quantify parser
failures (malformed `\boxed{}`). Establishes the *real* paired effect and how
much of the "noise" is parsing. → answers: was +1 a wash or a reshuffle?

**E1 — SEAL sign + coef sweep (tests §6.1).** Cheapest GPU win.
```bash
# re-extract vector if missing, then:
for C in -2 -1 -0.5 0.5 1; do
  python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k \
    --prompt sequential --latent_steps 40 --latent_space_realign --think \
    --max_samples 200 --generate_bs 8 --max_new_tokens 2048 --device cuda:0 \
    --use_seal --seal_layer 28 --seal_coef $C --seal_mode latent
done
```
Discriminates: "SEAL sign was wrong" vs "activation steering on latents is inert."

**E2 — Cache steering, paper-faithful dose (tests §6.4, the documented decisive run).**
```bash
OUT=results/cache_steer_lasttoken SAMPLES=200 GEN_BS=8 \
  POSITIONS=last CV_SWEEP="0.5 1 2" bash scripts/run_cache_steer_pilot.sh
```
- Recovers to ≥ baseline → overdosing was the whole story (positive result).
- Still degrades → text→latent vectors don't transfer (clean negative, → E5).

**E3 — Wire + test Judger steering (surface C, fixes §6.3).** Requires a small
code change: call `seal.set_text_mask(...)` around the Judger
`generate_text_batch` in `methods/latent_mas.py`, then run `--seal_mode text` and
`both`. This is the first real test of steering the **answer path**.

**E4 — Layer sweep for SEAL** (the single layer-28 hook is arbitrary): sweep
`--seal_layer` over e.g. {10, 20, 28, 34} at the best coef from E1.

**E5 — Native latent-direction extraction (tests §6.5, H-MISMATCH).** Build the
steering direction from LatentMAS's *own* latent activations (correct vs
incorrect latent runs), then steer surfaces A/B/C with it. If a native vector
works where the text vector failed → mismatch confirmed and *fixable*.

**E6 — Task transfer.** GSM8K may be insensitive to reflection calibration. Re-run
the best config on **GPQA / AIME / ARC** (the cache-steering paper reports larger
gains on harder tasks). Loaders already exist in `data.py`.

---

## 10. Measurement & analysis (how to know it's real)

- **Always paired, same seed/samples.** Report **McNemar** (paired) not just
  two-proportion z; the report notes prior z-tests were conservative because arms
  were unpaired.
- **Adequate n.** n=100 is underpowered for ≤ few-pp effects; use **n≥200**,
  ideally 500+, for any claim. Pre-state the minimum detectable effect.
- **Separate parsing from reasoning.** The GSM8K `\boxed{}` parser
  (`utils.extract_gsm8k_answer`) is brittle and deflates *absolute* accuracy; it
  affects both arms equally so A/B deltas are valid, but quantify parser failures
  (E0) so you don't mistake parser noise for a steering effect.
- **Mechanism, not just the number.** For any effect, tie it to a measurable
  change (e.g. Judger execution/reflection ratio via `seal/thought_classifier.py`,
  token counts, or cache-norm shifts). A direction the mentor asked about:
  **per-role execution/reflection ratio** — but in pure LatentMAS only the Judger
  emits text, so a true per-role measurement needs a **TextMAS** run
  (`--method text_mas`) where every agent produces text.
- **Dose-response is your friend.** A clean monotonic curve (as in cache steering)
  is strong evidence the knob does real work; a flat curve suggests inertness/no-op
  (check for a wiring bug like §6.3).

---

## 11. Constraints & gotchas

- **HF backend only for cache steering** — vLLM doesn't expose `past_key_values`
  for editing (`run.py` errors on the combination). SEAL hooks work on the HF
  model (and on the 2nd HF model in the vLLM hybrid path).
- **Single GPU** in the pilots; the paper uses 2×GPU + vLLM. Absolute numbers
  aren't comparable to the paper — only internal A/B deltas are.
- **Batching uses left padding** (`models.py:_ensure_pad_token`). For cache
  steering prefer `positions=last_n` (clean trailing latent positions) over `all`
  (includes left-pad).
- **Transformers cache API drift** is already handled
  (`cache_steering/_cache_utils.py` covers old/new `DynamicCache` + legacy tuple);
  reuse `layer_kv_list` for any new cache edits.
- **Vectors are not in git** — every run must (re)extract them; the shell scripts
  do this automatically when the `.pt` is missing.
- **Smoke test first:** `python scripts/smoke_test_cache_steer.py` (no model
  download) verifies the KV edit hits exactly the targeted positions.

---

## 12. Hypotheses your strategy should target (rank-ordered)

Your §0 spec should be aimed squarely at one of these (and the fallback at
another). They are the live, discriminating questions about LatentMAS steering.

1. **H-SIGN:** Was the SEAL null result just an inverted coefficient? (E1)
2. **H-DOSE:** Does paper-faithful single-token cache steering recover/beat
   baseline? (E2)
3. **H-PATH:** Does steering the **Judger answer path** (surface C/D) — untested
   so far — move accuracy where latent-only steering didn't? (E3)
4. **H-MISMATCH:** Do text-derived steering vectors transfer to the realigned
   latent KV channel at all? Does a **native** latent-derived vector work where
   text vectors fail? (E5) — *this is the deepest scientific question.*
5. **H-TASK:** Is GSM8K simply insensitive, while harder tasks (GPQA/AIME) show an
   effect? (E6)
6. **H-ROLE:** Do agent roles specialize (Critic = reflection-heavy, etc.) such
   that *selective* per-role steering (surface E) beats global steering? (needs
   TextMAS run + thought classifier)

**What to hand back:** not the experiments themselves — the **§0 strategy spec**.
Pick the hypothesis with the best (impact × tractability), specify the one
strategy that most sharply tests it, and give the engineer everything in the §0
contract so they can implement and run it on the user's GPU without coming back
to you for design decisions.

---

## 13. Reproduce / environment

```bash
pip install -r requirements.txt
export HF_HOME=/workspace/hf_cache      # avoid re-downloads
python scripts/smoke_test_cache_steer.py            # sanity (no GPU download)

# SEAL pilot (auto-extracts vector if missing):
tmux new -s pilot ; bash scripts/run_pilot.sh

# Cache-steer A/B + sweep:
tmux new -s cachesteer
SAMPLES=200 GEN_BS=8 CV_SWEEP="1 3 6" bash scripts/run_cache_steer_pilot.sh
```

Key files: `models.py` (wrapper, latent loop, realignment, steering
registration) · `methods/latent_mas.py` (pipeline + intervention points) ·
`seal/` (activation steering) · `cache_steering/` (KV steering) · `run.py` (CLI) ·
`prompts.py` (agent prompts) · `docs/pilot_report.md` + `docs/cache_steering_report.md`
(prior write-ups).
