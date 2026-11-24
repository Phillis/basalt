# Last pass summary — W2 (drafter capture) and W1 (prefill pipeline)

Two workstreams from `NEXT-STEPS-SPEC.md`, worked in the order W2 → W1. One landed with a large gain, one closed honestly as
a null result. Nothing here changed model weights, quantisation or numerics: **both changes are pure scheduling**, so output
quality is untouched by construction.

| Workstream | Outcome | Effect |
|---|---|---|
| **W2** — capture the drafter block | **Implemented, correct, neutral** | 61.0 vs 61.5 tok/s — no gain; ships opt-in |
| **W1** — stage-ahead prefill pipeline | **Landed, default on** | **~1,450 → ~2,600 tok/s prefill (+73%)** |

Engine state after the pass: `loaded=True`, `ctx=196608`, no marker files present, ~10.7 GB VRAM, serving on `0.0.0.0:8080`.

---

# W2 — drafter capture: the blocker was one field, and the win was not there

## What the spec assumed

`NEXT-STEPS-SPEC.md` had W2 cutting the ~7.8 ms/window spent outside the main decode window down to ≤3 ms, by capturing the
drafter's block as a CUDA graph. The stated obstacle was that the drafter's attention "follows the position", so it had been
left running as direct launches between the captured stages.

## What reading the code showed

The window-attention kernel was **already** device-position-aware — `Range::pos_dev` plus a `row_pos0()` helper, added
earlier for `gqa_decode_attn` (qwen4exp's captured decode). The window kernel already used them for:

- the window's **low** bound (`row_lo` → `row_pos0`)
- the per-row chunk count, with `used = min(gx, chunks)` so surplus blocks exit early

The **only** quantity still baked at capture was `r.hi`, because `gqa_window_attn` sets `r.causal = 0` and `row_hi` returns
`r.hi` verbatim in that case. So the blocker was one field, not a mechanism.

## What was implemented

| File | Change |
|---|---|
| `src/kernels/cuda/gqa_attn.cu` | `Range::hi_off`; `row_hi` returns `row_pos0(r) + r.hi_off` **only when** `pos_dev != nullptr && hi_off != 0`; `gqa_window_attn` takes optional `pos_dev` + `hi_off`, skips its `hi < pos0 + T` validation when the position is device-side, and sizes the grid from `scratch_cells` (the shape bound) instead of the exact `hi` |
| `include/basalt/kernels/gqa_attn.hpp` | the two new parameters, defaulted (`nullptr`, `0`) |
| `src/models/inkling2/dspark.hpp` | `g_attn_[kDsMaxLayers][19]`, `attn_capture_`, `attn_captured()` |
| `src/models/inkling2/dspark.cpp` | captures layer k's block attention per **position bucket** (`2^s ≤ pos < 2^(s+1)`) and replays it; loud `warn` on capture-start, instantiate or launch failure |

Design notes that make it work:

- The position already reaches the device as `meta_d_ + 1` (the drafter's own `pos` word) — no new allocation.
- A bucket's floor `bl` supplies `pos0` for `r.base` (≤ the real one, so the range always covers the position) and, for a
  **full** layer, `window = bl + kDsBlock`, which makes `row_lo` clamp to 0 for every position in the bucket.
- A capture cannot resize its grid, so the captured call runs the scratch's *capacity* rather than the cells actually read —
  strictly more work at short context.

Inert by construction: every pre-existing caller passes neither new parameter, so the guarded branch is unreachable.

## Measurements (400-token free-form prose, `temperature 0`, engine relaunched per mode)

| Pair | Direct warm tok/s | Capture warm tok/s | Diag wall |
|---|---|---|---|
| 1 | 61.5 (4 runs) | 61.0 (4 runs) | — |
| 2 (+diag) | 61.1 — wall 30.35 ms | 61.9 — wall **29.29 ms** | 30.35 → 29.29 (−3.5%) |
| 3 | 60.50 (3 runs) | 55.37 (3 runs) | — |

Correctness throughout: `acc` 87–91% unchanged, outputs stable across repeats, verbatim-repeat 100% accepted, **no capture
ever fell back** (the new warn never fired), needle found on the shipped default.

## Why the gain is not there

1. **The 7.8 ms is GPU work, not launch overhead.** The stages and head were already graphs; only four attention calls ran
   directly (~5–10 µs each), and W-A had already removed the full-stream sync that used to dominate that span.
2. **Capturing a bucket on first sight stalls decode.** A 400-token reply crosses buckets `2^10` and `2^11` → up to 8
   capture+instantiate operations *inside* a request. That is pair 3's 63.1 → 50.8 → 52.2 collapse.
3. **Capacity-sized splits** make the captured attention slightly slower than the direct one at short context.

**Decision: ships opt-in** via `C:\LLM\dspark-capture.on`, default **off**. It is now a measurement switch, not a
performance one — leaving it on costs a few percent on long decodes.

---

# W1 — prefill pipeline: the wave-slicing was not needed

## What the spec assumed

W1 was specified as wave-pipelined staging **by slot range**: split a layer's spill into 2–4 waves, stage wave *i+1* while
computing wave *i*. That needs per-wave expert products, and `fp4::Product` addresses experts positionally
(`base + e * stride`, `bounds[e]`), so slicing is fiddly and a bug shows up as garbage.

## The insight that avoided it

Reading `experts()` and the spill layout showed the property that makes a much simpler pipeline possible:

> Every layer already owns its own expert region in VRAM — `w.experts`, with the staged spill at `w.experts + kept*stride`
> inside that layer's own range — and the staged bytes are identical on every visit.

So layer *L+1*'s staging can be copied into *L+1's own* slots while the card computes layer *L*. No product changes, no
extra VRAM, no wave bookkeeping. Two wins at once:

1. **Overlap** — issue *L+1*'s H2D on a second stream while *L*'s products run; *L+1* waits on an event.
2. **Reuse** — the staging backing survives across a prompt's chunks (`acquire_staging` is a no-op until something releases
   it), so one stage per layer serves every chunk. A generation counter bumped by `release_staging` invalidates it correctly.

## What was implemented

| File | Change |
|---|---|
| `src/models/inkling2/model.hpp` / `model.cpp` | `stage_layer(l, stream)` — a layer's whole staging (card copies where held, else one H2D) plus the guard memset, safe on any stream; `staging_gen_` + `staging_generation()`, bumped by `release_staging` |
| `src/models/inkling2/prompt.hpp` | `virtual void sync_staging() {}` on `PromptRunner` |
| `src/models/inkling2/prompt.cpp` | `copy_` stream + one event per layer; `stage_awaitable(l)` / `stage_issue(l+1)` / `sync_generation()`; `experts()` waits on the layer's event instead of staging inline |
| `src/models/inkling2/session.cpp` | `prompt_->sync_staging()` before `release_staging(0)` at the decode boundary |

Kill switch: `C:\LLM\prefill-no-prefetch.on` restores the serial staging exactly.

## Measurements (unique 10k-token prompts of random 6-char words, `max_tokens 4`, fresh engine per mode)

| Mode | Runs (tok/s) | Mean |
|---|---|---|
| Serial staging (`prefill-no-prefetch.on`) | 1,539 · 1,368 | ~1,454 |
| **Pipeline staging (default)** | **2,813 · 2,577 · 2,417** | **~2,600** |

Per 4,096-token chunk: **2.72 s → ~1.55 s**, i.e. essentially `max(H2D 1.33 s, compute 1.39 s)` — the copy engine and the SMs
are genuinely working at the same time. That clears `NEXT-STEPS-SPEC.md`'s 2,400 gate.

| Context | Prefill | Note |
|---|---|---|
| 10k unique, pipeline | ~2,600 tok/s | the win |
| 10k unique, serial | ~1,454 tok/s | baseline |
| 130,709 (needle doc) | 1,215 tok/s | **unchanged** — attention-bound, staging is not the constraint |

---

# Verification (both workstreams)

- **Needle found at 130,709 tokens** with the pipeline on: `ZEPHYR-4417`, 107.6 s prefill, `tg=92`. This is the gate that
  would catch staged-data corruption, and it passes across 32 chunks.
- Greedy outputs stable across repeats within a process; verbatim-repeat task 100% accepted.
- No marker files after the pass; defaults are the measured paths.
- Every A/B relaunched the engine (and killed any `C:\LLM` `python.exe`) — the project's standing rule, because a stale
  server silently serves a previous engine.

## Two traps caught during the pass

1. **Copy stream vs. VMM unmap.** The first pipeline version released the staging's backing while the copy stream still had
   copies in flight. That is exactly the class of silent failure this project has been hurt by (sticky CUDA error, no
   visible failure). `session.cpp` now quiesces the copy stream before `release_staging(0)`.
2. **A phantom decode regression.** W1 first appeared to cost ~6% decode. A control run of the *same* mode in two fresh
   processes spanned **62.5 vs 67.8 tok/s (±8.5%)** — larger than any difference attributed to the change. On a
   pinned-content workload the pipeline measured *faster* (52.5 vs 46.6 tok/s). Decode is within noise.

## One measurement caveat worth carrying forward

With the pipeline, `hit=` reads 48–51% instead of 64%. This is **not** lost residency: the promote log is byte-identical in
both modes (`1.01 GiB released … 0.72 GiB copied, 760 on the card`). It tracks `dl` (tokens per window, 7.1 vs 2.9 on the
verbatim task): the same resident experts serve more routed pairs per window, so the per-pair hit statistic moves down while
throughput rises. **Judge the front by `pp=` and `tg=`, not by `hit=`.**

---

# Where this leaves the engine

```
decode    39–67 tok/s by content (unchanged this pass); sustained 1,000-token ≈ 54
prefill   ~1,450 → ~2,600 tok/s on unique 10k prompts (+73%)
context   196,608 tokens; needle verified at 130,709
hit=      per-pair statistic, not comparable across changes that alter tokens/window
```

Prefill is now **compute-bound**: ~1.39 s of expert products per 4,096 tokens plus ~0.1 s of exposed first-layer staging.
The next lever is therefore the expert math, not more I/O:

- The CPU pool already runs 45.6 GB/s against a 51.9 GB/s harness ceiling, and `kElasticKeep`/promote tuning moves residency
  rather than rate.
- Further prefill gains need the products themselves — the quality-gated **D1 Q2_0 shadow** (real-weight SNR 3.56 dB cover
  bound / 7.90 dB clipping ladder, against a design prediction of 15–22 dB) remains blocked on the user's quality sign-off.
- W2's capture machinery is in place and correct, so if the drafter's stages ever stop being graphs, it is one marker away.

## Files touched in this pass

```
src/kernels/cuda/gqa_attn.cu              (+ backup gqa_attn.cu.orig)
include/basalt/kernels/gqa_attn.hpp
src/models/inkling2/dspark.hpp
src/models/inkling2/dspark.cpp
src/models/inkling2/model.hpp
src/models/inkling2/model.cpp
src/models/inkling2/prompt.hpp
src/models/inkling2/prompt.cpp
src/models/inkling2/session.cpp
```

## Documents written

| File | Contents |
|---|---|
| `W1-W2-ASSESSMENT.md` | the read-only assessment that identified both blockers before any code changed |
| `W1-PREFILL-PIPELINE-RESULTS.md` | W1's design, numbers and gates in isolation |
| `W2-W1-PASS-SUMMARY.md` | this document |
| `CHANGES.md` | Sessions 10, 11, 12 appended (now 87 KB) |
| `SUMMARY.md` | headline prefill updated to ~2.6k tok/s |
