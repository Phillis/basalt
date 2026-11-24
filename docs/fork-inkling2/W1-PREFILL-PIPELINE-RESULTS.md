> # ⚠ RETRACTED — Session 13
>
> **This document's result is withdrawn. The pipeline it describes corrupts the model.** It stages layer L+1 into what it
> assumed was L+1's own VRAM region, but all layers alias ONE shared 1034 MiB staging buffer (`model.cpp` maps the same VMM
> granules into every layer's range at the same offset), so prefetching L+1 overwrites the experts L is still reading.
> The engine then produced non-finite logits: the model answered `!!!` to everything, and the `+73%` below was measured on
> that broken build. Prefill is back to **~1,450–1,550 tok/s**; the fix and the full analysis are in `CHANGES.md`
> (Session 13). The document is kept because the *timing* observation (the H2D does hide behind compute when the buffers do
> not alias) and the machinery notes are still useful for the correct version — which needs either a second staging buffer
> or per-wave expert products.
# W1 results â€” prefill 1,454 â†’ ~2,600 tok/s (+73%)

The headline: **prompt prefill went from ~1,450 to ~2,600 tok/s** on unique 10k-token prompts, past the 2,400 target in
`NEXT-STEPS-SPEC.md`, with **no model changes and no quality impact**. Decode is unchanged within measurement noise.

## What the spec asked for, and what was actually needed

`NEXT-STEPS-SPEC.md` (and `W1-W2-ASSESSMENT.md` after it) framed W1 as *wave-pipelined staging by slot range*: split a
layer's spill into 2â€“4 waves and stage wave *i+1* while computing wave *i*, which requires slicing the expert products
per wave. That slicing is genuinely fiddly â€” `fp4::Product` addresses experts positionally (`base + e * stride`, `bounds[e]`).

**It was not necessary.** Reading `experts()` and the spill layout showed the property that makes a much simpler pipeline
possible:

> Every layer already owns its own expert region in VRAM â€” `w.experts`, with the staged spill at `w.experts + kept*stride`
> inside that layer's own range â€” and the staged bytes are identical on every visit.

So layer *L+1*'s staging can be copied into *L+1's own* slots while the card computes layer *L*. No product changes, no
extra VRAM, no wave bookkeeping, nothing to slice.

That gives two wins at once:

1. **Overlap** â€” issue *L+1*'s H2D on a second stream while *L*'s products run; *L+1* waits on an event.
2. **Reuse** â€” the staging backing survives across a prompt's chunks (`acquire_staging` is a no-op until something releases
   it), so one stage per layer serves every chunk. A generation counter, bumped by `release_staging`, invalidates it
   correctly.

## Numbers

Unique 10k-token prompts of random 6-char words, `max_tokens 4`, fresh engine per mode:

| Mode | Runs (tok/s) | Mean |
|---|---|---|
| Serial staging (`C:\LLM\prefill-no-prefetch.on`) | 1,539 Â· 1,368 | ~1,454 |
| **Pipeline staging (default)** | **2,813 Â· 2,577 Â· 2,417** | **~2,600** |

Per 4,096-token chunk that is 2.72 s â†’ ~1.55 s, i.e. essentially `max(H2D 1.33 s, compute 1.39 s)` â€” the copy engine and
the SMs are genuinely working at the same time.

| Context | Prefill | Note |
|---|---|---|
| 10k unique, pipeline | ~2,600 tok/s | the win |
| 10k unique, serial | ~1,454 tok/s | baseline |
| 130,709 (needle doc) | 1,215 tok/s | **unchanged** â€” attention-bound, staging is not the constraint |

## Correctness

- **Needle found at 130,709 tokens** with the pipeline on: `ZEPHYR-4417`, 107.6 s, `tg=92`. This is the gate that would
  catch staged-data corruption, and it passes across 32 chunks.
- Greedy output unchanged in character; verbatim-repeat task 100% accepted.
- Decode: **no reproducible change.** An early âˆ’6% reading was chased down properly â€” a control run of the *same* mode in
  two fresh processes spanned **62.5 vs 67.8 tok/s (Â±8.5%)**, which is larger than any difference attributed to the change.
  On a pinned-content workload the pipeline measured *faster* (52.5 vs 46.6 tok/s).

## The trap that was fixed

The first version released the staging's VMM backing while the copy stream still had copies in flight. That is exactly the
kind of silent failure this project has been hurt by before (a sticky CUDA error, no visible failure). `session.cpp` now
calls `prompt_->sync_staging()` before `release_staging(0)` at the decode boundary.

## Why `hit=` reads lower (and should be ignored here)

With the pipeline, `hit=` shows 48â€“51% instead of 64%. This is **not** lost residency â€” the promote log is byte-identical
in both modes (`1.01 GiB released ... 0.72 GiB copied, 760 on the card`). It tracks `dl` (tokens per window), which is 7.1
vs 2.9 on the verbatim task: the same resident experts serve more routed pairs per window, so the *per-pair* hit statistic
moves down while throughput goes up. Judge the front by `pp=` and `tg=`.

## Files

| File | Change |
|---|---|
| `model.hpp` / `model.cpp` | `stage_layer(l, stream)` (a layer's whole staging on any stream, plus guard memset); `staging_gen_` / `staging_generation()`, bumped by `release_staging` |
| `prompt.hpp` | `virtual void sync_staging() {}` on `PromptRunner` |
| `prompt.cpp` | `copy_` stream, one event per layer, `stage_awaitable` / `stage_issue` / `sync_generation`; `experts()` waits on the layer's event instead of staging inline; `C:\LLM\prefill-no-prefetch.on` restores the old path |
| `session.cpp` | `sync_staging()` before `release_staging(0)` |

## Where the remaining prefill time goes

At ~2,600 tok/s the chunk is compute-bound: ~1.39 s of expert products per 4,096 tokens, plus the first layer's exposed
staging (~0.1 s). The next lever is therefore **expert math, not I/O** â€” the CPU pool already runs 45.6 GB/s against a
51.9 GB/s harness ceiling, and `kElasticKeep`/promote tuning moves residency rather than rate. Further prefill gains need
the products themselves (e.g. a Q2_0 shadow, which is the quality-gated D1 item) rather than more pipelining.

