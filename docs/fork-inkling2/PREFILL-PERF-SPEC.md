# PREFILL-PERF-SPEC — Inkling-2-Mini prefill past 2,000 tok/s

**For an implementing agent.** Companion to `DECODE-PERF-SPEC-3.md` (read §0-§5 for the decode state; this file
is self-contained for prefill). Written from code inspection while the engine was in use — **no measurements
in here are new; every number is either from the existing logs or derived from the code and cross-checked
against them.** Working tree `C:\LLM\pr536\basalt-inkling-cpu-spill-review`, build `C:\LLM\rebuild.bat`,
canonical config `C:\LLM\Basalt\basalt_config.json` (copied into the tree by `Start-Basalt.cmd`).

**Baseline to beat:** unique-text 16-22k prompts run at **1,078-1,296 t/s** on a warm engine (first chunks
443 t/s while the decode-time staging reclaim is undone), ~1.5-1.65k on a fresh engine. Stock was 1,452.
**Target: sustained >2,000 t/s on unique 16k+ prompts.**

**Measurement discipline (unchanged from spec 3):** benchmark only with unique, never-seen text (random
digits work); the truth is the `s1 read pp=` lines in `server.out` — ignore `engine.log`'s "N reused in 0 ms"
lines, those are second-round restore stats. Kill all `basalt.exe` **and** the `C:\LLM` `python.exe` before
every relaunch (a stale python server silently serves A/Bs with a previous engine).

---

## 1. Where the time goes (code-derived, cross-checked against every observed rate)

Per chunk (`kPromptChunk` = 4096 rows, `prompt.hpp`), the walk (`walk.hpp`) runs **48 visits** — layers 0-5
once, **loop layers 6-17 three times**, 18-23 once. In `prompt.cpp` `experts()` (lines ~415-425), *every visit
of a layer with spilled experts stages its experts before its products run*:

```cpp
if (w.slot_map) {                                   // a layer with experts in RAM
    remap_ids(b_.ids, pairs, w.slot_map, stream_);
    const uint64_t kept = (n_expert - w.n_spill) * w.expert_stride;
    if (!m_.stage_from_card(l, stream_))            // from the elastic card copies (D2D) where they exist
        cudaMemcpyAsync(w.experts + kept, w.spill_host, n_spill * stride, H2D, stream_);   // else RAM -> card
    ...
}
moe_sort(...); mmq_/fp4_ products                    // the visit's expert GEMMs read the staged weights
```

Both copies run **on the compute stream, serially per visit** — the visit's expert products (and every later
visit) queue behind the transfer.

**Staging bytes per chunk** (~114 spilled experts/layer; blob 4.48 MiB NVFP4, 8.46 MiB Q8_0; the elastic
copies cover ~728 of 2,796 spilled, D2D):

| Visits | Layers | Bytes/visit | Total |
|---|---|---|---|
| 44 | NVFP4 (0-5; **6-17 x3**; 18-19) | ~511 MiB | ~22.7 GiB |
| 4 | Q8_0 (20-23) | ~965 MiB | ~3.9 GiB |

~26.6 GiB nominal — **70% of it the twelve loop layers copied three times each** — minus the D2D-covered part
(~26% of spilled experts today) leaves ~19.4 GiB of H2D per chunk at 15.75 GB/s ≈ **1.23 s of PCIe per
chunk**, issued as ~40-60 ms serial stalls per visit with only that visit's attention/dense (~5-15 ms)
overlapping. Budget: 1.23 s DMA + 0.3-0.5 s expert products (FP4 tensor cores, `experts_fp4` — not the wall)
+ 0.3 s dense/attention ≈ 2.3-2.5 s/chunk → 1,640-1,780 t/s fresh ✓, less with the reclaim tax ✓. One parameter
set reproduces every rate ever logged, including stock's 1,452.

**Corrections to Session 4 (both were invalid):**
- `--prefill-cpu-share` is **not parsed by inkling2** (qwen4exp-only; the server logs "this model has no such
  flag, skipped"). The 0.6/0.85/1.0 sweep measured a no-op. There is no CPU share in inkling2's prefill.
- The "prefill 8192 worse" test predates the doorbell build, and 8192 is infeasible anyway: the chunk arena
  (`carve()` — a dozen R x E float buffers, attention scratch, expert q8/FP4 rows) roughly doubles, ~+1 GiB of
  VRAM, and only ~768 MiB is free at 192k. Do not raise `kPromptChunk`.

## 2. The design: wave-pipelined staging (no extra VRAM; makes the copy engine the wall)

The physics: any scheme that keeps the PCIe copy engine 100% busy reaches the DMA-bound floor
`4096 / 1.23 s ≈ 3,300 t/s` (before P2's byte cuts). The cheapest scheme that does so **needs no additional
VRAM**, because the staging set is already sized for a whole layer:

**Split each visit's staging into W waves (2-4) and pipeline copy against compute *within the visit*:**

1. A dedicated copy stream (`cudaStreamNonBlocking`; the session already has side-stream precedent) plus one
   event per wave.
2. For wave i (a contiguous slot range of the layer's spilled experts):
   - issue `stage_from_card`-equivalent / `cudaMemcpyAsync` for wave i's slots **on the copy stream**
     (into the layer's own staging slots — wave i writes slots no product is reading);
   - record wave i's event; the compute stream waits on it and launches **wave i's expert products**;
   - the copy stream immediately proceeds to wave i+1.
3. The expert products must therefore run **per wave**: `mmq::Product`/`fp4::Product` take the per-expert row
   bounds (`b_.bounds`, built by `moe_sort`) — slice them per wave (a small prep kernel or host-side build of
   W compacted bound arrays over the wave's expert range; the sort's `bounds`/`tok`/`dst`/`exp` layout supports
   a contiguous expert-range slice without re-sorting). The FP4 path additionally reads each expert's
   fp32 tail at `base + L.bytes - 16` — unchanged, just per-wave base ranges.
4. `moe_combine` runs after the last wave, as today.
5. Start wave 0's copy as early in the visit as possible — right after the router (`router_select`), before
   the attention/dense chain, so it overlaps the visit's own ~5-15 ms of pre-expert work too.

The staging set is read only by the wave products scheduled against it, so no extra physical pages are needed;
a *later* visit's copies must wait for the *previous visit's last wave products* to finish reading the set
(~10 ms) — the scheme saturates the copy engine to within that tail.

**Projected per visit:** `max(W-copy total ~40 ms, dense 15 + products ~10 ms)` ≈ 40-45 ms → chunk ≈ 48 x 42 ms
+ chunk overhead ≈ **2.0-2.2 s → ~1,850-2,050 t/s** at today's byte volume. This alone is at the target's
edge; **P2 pushes it comfortably past it.**

Implementation notes:
- Keep the whole-copy path behind a flag (`--prefill-waves 0|N`, 0 = today's serial copy) for A/B and
  fallback; inkling2's flag parser is `dspark_parse_flag` (`dspark.cpp:722`) — add the flag there (and note it
  in `help.cpp`), it is *not* a config-json key (config only feeds known flags).
- `stage_from_card` (`model.cpp:562`) currently copies expert-by-expert on the caller's stream; give it a
  stream parameter and a slot range.
- The `cudaMemsetAsync(w.experts + n_expert*stride, 0, kSlotGuard)` guard write stays on the compute stream
  before the first wave's products.
- Correctness: this is bit-neutral data movement — the products, sort, router, and SwiGLU are untouched.
  Validate with greedy-identity A/B (same unique prompt, `--prefill-waves 0|4`) plus the repo tests; a wrong
  wave slice shows up immediately as garbage output, not as subtle drift.
- Chunk boundaries: also prefetch chunk n+1's first wave during chunk n's coda (the last visit's products
  finish reading the set ~10 ms before the chunk ends) — small, take it if free.

## 3. P2 — Cut the bytes: rank the elastic copies by staging frequency (cheap, compounds with §2)

Every elastic card copy stages D2D (in-VRAM, ~free) instead of H2D. Today the 728 copies are ranked by decode
routing heat only (`spill_order_`, `model.cpp promote`). But a **loop-layer expert's copy saves 3x its blob
per chunk** (its layer stages three times), a coda layer's only 1x. Weight the promote order by
`visits-per-chunk` (loop 3, others 1) so the copies concentrate on loop layers: coverage of loop-layer spilled
experts rises from ~26% toward ~50-60% (12 loop layers hold ~1,368 of the 2,738 spilled), cutting H2D to
roughly ~13-15 GiB/chunk → **0.85-0.95 s** of PCIe.

Combined with §2: chunk ≈ `max(0.85-0.95, compute ~0.7) + tail` ≈ 1.1-1.3 s → **3,150-3,700 t/s**; even
pessimistically 2,600-3,000. Watch the tradeoff: decode hit-rate also depends on the ranking — re-measure
decode t/s after changing it (the `diag` `pool busy` line is the canary; if decode regresses >5%, blend the
two rankings rather than replacing).

Verify with the existing `prompt_stage.cpp` log lines, which report staged bytes "on the card" vs "in
page-locked RAM" — they should show the D2D share rising.

## 4. P3 — The first-chunk reclaim tax (already spec-3 W-D; do it with this work)

A follow-up prompt after any decode runs its first chunks at 443 t/s while the W1 decode-time reclaim is
undone (`staging reclaim: ... taken back for the prompt chunk`). Keep a **standing staging floor**
(~512 MiB — one NVFP4 layer's worth, so short prompts never re-take) and move the re-take to idle time after a
decode burst instead of synchronously at the next prompt. This is worth ~200-300 t/s of *average* prefill on
conversational follow-ups and removes the TTFT cliff.

## 5. Order, gates, and validation

| Step | Gate |
|---|---|
| §2 wave pipelining | unique-16k `s1 read pp=` ≥ 1,900 sustained; greedy-identity parity vs `--prefill-waves 0`; TTFT unchanged |
| §3 promote ranking | H2D/chunk visibly down (prompt_stage lines); prefill ≥ 2,600; decode `diag pool busy` within 5% of baseline |
| §4 staging floor | follow-up-prompt first chunk ≥ 1,200 t/s; decode steady state unchanged |

Full regression after each step: repo tests, `test-long.ps1` (needle), a 16k unique-text prefill, a 5-minute
sustained decode run, VRAM/RAM lines clean (no `pinned staging alloc` failures — the plan's headroom is
~150 MiB, and none of this work should touch it: the wave scheme adds no VRAM; the standing floor *reduces*
what reclaim hands back, so re-check the promote line "N of 2738 copied" still fits).

Append results to `CHANGES.md` (Session 6).

## 6. What is deliberately NOT in scope

- **A CPU-prefill-expert path** (the qwen4exp-style pool assist): the pool's multi-token kernels cap at
  nt=8/expert, so prefill-scale CPU compute is a new kernel project (modeled ceiling ~4,500+ t/s by running
  the pool's ~45 GB/s alongside PCIe). Only if the PCIe wall must be broken after §2+§3 — the target does not
  need it.
- **Bigger chunks** (`kPromptChunk` 8192): ~+1 GiB chunk-buffer VRAM, unavailable at 192k; after §2 the DMA is
  mostly hidden anyway.
- **Attention growth at very long context**: the full-attention visits' per-chunk cost grows with the prefix
  (~0.1-0.2 s at 22k; several times that at 130k+). Re-benchmark §2 at 130k before promising 2,000+ there —
  the projections in this spec are for the measured range.
- GPU expert-kernel work (the FP4 path is ~0.3-0.5 s/chunk, not the wall); the drafter's prefill taps;
  `prefill-cpu-share` (does not exist for this model).

## 7. Risks

- The wave slicing of `moe_sort`'s bounds is the only fiddly part — keep waves large (2-4) so the per-wave
  launch overhead (~1-2 ms) stays well under the per-wave copy time (~10-20 ms).
- A second stream + events must be created per session and joined at chunk end (the session's side-stream
  precedent is `ra_st_`/`pf_st_` in `session.cpp`).
- The copy stream changes timing only; any output difference is a bug in the slicing, not an inherent
  trade-off — do not ship a "parity flag" as the default resolution.
- If `stage_from_card`'s per-expert D2D copies turn out to dominate its runtime (per-copy launch overhead for
  ~84 tiny copies), batch them: the wave's slot range is contiguous, so a single D2D/H2D covers it when no
  elastic copies intervene; only the copied experts need individual copies.
