# What to implement, from the Strata and DeepSeek-V4.1-Flash investigations

Two repos investigated: **niko1221/strata** (the app Basalt is a form of) and **yhfgyyf/vllm-deepseek-v4-sm89**
(V4.1-Flash: MoE + Engram, a vLLM fork). Everything below is judged against *our* engine, on this box.

---

## The headline: Strata's KV-to-RAM is already in our tree â€” for the *dense* models only

Strata's feature is **KV streaming**. `engine/include/strata/core/kv_host_dma.hpp` states the design precisely:

> With `--kv-resident` the authoritative K/V of a streamed QSA layer lives in **pinned, device-mapped RAM**; the prompt
> path's append also writes every new cell into the **one-layer staging pool in VRAM**.

So VRAM holds a *resident window* of cells plus one layer's staging buffer; everything older lives in host RAM and is read
through the mapping. `docs/KV_PREFETCH.md` adds the part that makes it fast: start the **next** layer's upload on a second
stream while the current layer computes â€” measured **âˆ’5.3% prefill at 32K and âˆ’4.1% at 64K** (939â€“1553 ms of serial
KV-stage wait moved off the critical path).

**Our Basalt tree already implements this** â€” in the shared dense engine:

- `Options::kv_resident`, `resident_cells(max_context, kv_resident)`,
- `shared_bytes()` / `session_host_bytes()` / `session_layer_bytes()` sizing a one-layer staging pool and a host copy,
- `src/dense/internal.hpp`: "per attention layer, with K/V streaming: the cells from `Options::kv_resident` on, in
  page-lock (device-mapped); `{}` without. `kc/vc/kq` then hold only the first `kv_resident` cells".

**inkling2 uses none of it.** Its caches are `VisitCache` + `core::ElasticArray`, which is a *VRAM*-elastic array (it
saves host RAM, not VRAM) â€” the opposite direction.

### Why that matters here

At 256k with `kv=fp8v4` our KV sits in VRAM: 1,664 B per token per full-attention layer (K 128 B + V 80 B, 8 heads), so
**~3.4 GiB** across the full-attention layers. Streaming with a 32k-cell resident window would leave ~0.5 GiB on the card
and move ~2.9 GiB to host RAM. The card then has **~3 GiB more for expert copies** â€” at ~5.5 MiB a blob that is ~550 more
resident experts, and residency is the lever we have measured repeatedly (192k â†’ 128k bought +5.9% decode; the 4-bit
format's +0.6 GiB of residency is what paid for 256k).

### The pinned question, answered by measurement

Can we pin more RAM from this machine? **No, not by any lever we have.** Re-probed with the engine stopped and 22.3 GiB of
RAM free: the CUDA host-allocation ceiling is **exactly 14.52 GiB** â€” byte-identical to the figure measured while the
engine held 12.8 GiB pinned. It is a fixed driver/WDDM quota (~47% of this machine's 31.1 GiB), *not* a function of free
memory, so nothing we free up can raise it.

The privilege is a dead end by construction: the engine's RAM tier is `cuMemCreate`/`cudaHostAlloc` (the load log's
`share ram ...` line is CUDA host VMM), and `MEMORY-LOCKING.md` established that the binary contains **zero** references to
`VirtualLock`, `SetProcessWorkingSetSizeEx`, `cuMemHostRegister` or `SeLockMemoryPrivilege`. The other mechanism *can* pin
more â€” `VirtualLock` after `SetProcessWorkingSetSizeEx(min 8 GiB)` was proven to lock 6 GiB â€” but the engine does not use
it, and an 18â€“24 GiB working set on a 31 GiB box was tried and reverted as a real regression risk.

**So item 1 is concretely constrained**: the weights pin **12.8 of the 14.52 GiB**, leaving 1.7 GiB, while a 256k KV tail
needs ~3.4 GiB. Streaming it into *pinned* memory would mean evicting ~2 GiB of weight spill, turning those experts into
CPU-computed ones â€” a straight trade, not a gain. Hence the three options in item 1 below.

---

## Ranked plan

### 1. KV streaming for inkling2 - **STARTED, AND RULED OUT BY MEASUREMENT**

I started the port and the first arithmetic stopped it. Inkling's full-attention layers scan **the whole prefix on every
decoded token**:

```
KV per decoded token, 8 full-attention layers, 256k context, v4 rows
  = 8 x 262,144 keys x 8 heads x (128 + 80) B   =  3.25 GiB
  from VRAM at ~800 GB/s                        =  4.4 ms      (our decode is 14.3 ms/token at 70 tok/s)
  over this machine's measured 15.75 GB/s link  =  222 ms/token  ->  ~4.5 tok/s
```

So streaming the tail - pinned *or* pageable - moves 31% of a decode token's work off an 800 GB/s bus onto a 15.75 GB/s
one: **a 15x decode regression** to free 3.4 GiB of VRAM. The pageable variant is worse still, because decode would have
to DMA a whole layer's tail into staging *per window*.

That is exactly why Strata's own docs promise prefill gains and state there is **"no decode speedup"** and that decode runs
"somewhat slower": their late layers are a gated recurrence plus QSA over a *resident window*, so their decode never scans
the streamed cells. Inkling's full-attention layers do, by construction.

**Conclusion: `--kv-resident` buys nothing for an architecture like Inkling's at a context that already fits.** It is the
right tool for *having* 1M context at slow decode - not our problem, since our 256k fits today.

**So item 1 is replaced by item 4** (paged, shared KV pool): it changes *allocation granularity*, not residency, so decode
keeps reading VRAM while sessions stop reserving their full ceilings - which is the thing that actually fails on this box
(2 x 128k was 1,383 MiB short).

The original design note follows, for the record.
### 1. KV streaming for inkling2 (`--kv-resident`) â€” the biggest single win, and mostly a port

**What**: give `VisitCache` a resident window plus a host tail for full-attention layers (the SWA layers keep their rings;
nothing to stream). Reuse the dense engine's sizing arithmetic and the existing VMM/ElasticArray machinery for the resident
part.

**Expected**: ~2.5â€“3 GiB of VRAM back at 256k â†’ ~450â€“550 more resident experts â†’ **plausibly +5â€“10% decode**, and 320k+
contexts become possible on this card.

**Cost**: the KV address is `[cell][head][dim]` inside every inkling2 attention kernel, in `save_cells/restore_cells` and
in the checkpoint transfer, so the resident/streamed split enters the address computation (a clamp plus a second base
pointer). Medium. **Risk**: low â€” streamed cells are read-only during a window.

**Three ways to hold the tail, given the pinned ceiling**:

1. **Pageable host copy + DMA into the one-layer staging pool** (recommended): zero pinned budget; item 2 hides the copies.
2. **Pinned at a smaller context**: 1.7 GiB is ~128k of tail.
3. **Add RAM**: the quota tracks physical memory at ~47%, so 64 GiB would pin ~30 GiB â€” weights *and* a 256k tail.

### 2. Cross-layer KV prefetch (Strata's `KV_PREFETCH`) â€” rides on 1, +4â€“5% prefill

Once the tail is in host RAM, prefill uploads each layer's prefix; starting the next layer's upload on a second stream
while the current computes took **5.3% of prefill at 32K** for them. We already use that exact recipe (second stream plus
`cudaEventRecord`/`cudaStreamWaitEvent`) for the expert staging waves. Small change, no quality risk. **Requires 1.**

### 3. Strata's `kv_host_dma` refinement â€” 50 Âµs a prompt-token on fast cards

Their measurement: carrying the prompt append to the host copy **per value over the link** costs ~50 Âµs per prompt token
(0.24 GB/s, a fifth to a third of prompt time at 8Kâ€“128K); doing it as **one DMA per array from the staging pool**
(`kv_unstage_to_host`) moves the same bytes at link rate. Item 1 must use the DMA form or it will lose more prefill than
the freed VRAM gains.

### 4. A paged, shared KV pool (from the DeepSeek fork) â€” the only way to get concurrency

They page at 128 cells over a shared pool where `max_model_len` is a **ceiling**, not a reservation. Ours sizes each
visit's KV contiguously to the full context, which is why **2 Ã— 128k was 1,383 MiB short** on this box. A paged pool would
let 2â€“4 sessions share one pool and turn v4's 44% KV saving into *concurrency* instead of only context.

**Cost**: larger than 1 â€” a block table plus indirection in every K/V address, and `save_cells/restore_cells` and the
checkpoint transfer both assume a contiguous run. **Best done together with 1**, since both change the same address path.

### 5. Strata's KVA (approximate late layers on long cold prompts) â€” real speedup, real trade, big job

`STRATA_KVA=1` reads a long prompt exactly through the early layers and then builds **only the recurrent state** for the
late ones, skipping their attention output, output projection and MoE half. Measured: **+11% prefill at 14K, +18% at 30K**
with split 36, and +12%/+20% on a P100. With split 24 it was 1.3â€“1.7Ã— but *lost records and agent loops*, which is why
split 36 is the shipped default and the feature stays opt-in.

**For us**: Inkling has linear-attention layers and SWA layers, so "state-only" maps â€” but it needs model-specific
state-only kernels and a quality campaign of its own. **Highest ceiling on prefill, highest risk, and only after 1â€“4.**

### 6. Already have, or already rejected

| their feature | our status |
|---|---|
| adaptive verification from the drafter's confidence head | **have it, stronger**: `spec::draft_policy.hpp` learns q(c) per confidence bin and maximises E(t)/ms, and arbitrates against the suffix drafter |
| cold-start warm-up (58.5 â†’ 83.3 tok/s) | **adopted**: launcher warms up, first request 67.0 vs 56â€“62 before |
| `POST /v1/vram` runtime reserve; shrinkable segmented expert cache | **have it**: `/v1/vram`, `reserve_/lend()`, VMM-granular map/unmap |
| graph the prefill (`FULL_AND_PIECEWISE`) | not worth it: prefill is weight-traffic bound (~200 TFLOPS effective), ~1 ms of launches against a 2.6 s chunk |
| persistent top-k, CPU activation LUTs, fused cache kernels, 4-bit CPU MoE | amortised by our decode graphs / weight-bound / already done / our rejected D1 |
| probabilistic draft sampling | a robustness option (fixed their cold-start garble); ours is greedy and clean |

---

## If I had to pick one

**1 + 2 together**, with the **pageable + DMA** tail: it is mostly a port of code we already ship for dense models, it
frees the VRAM that our own measurements say buys decode throughput, and it is the prerequisite for 3 and 4. The pinned
ceiling (14.52 GiB, 12.8 already used) rules out the pinned variant at 256k on this machine; pageable plus prefetch does
not need it.
