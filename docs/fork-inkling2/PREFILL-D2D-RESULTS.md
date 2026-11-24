# Prefill: per-slot D2D staging
# Session 62 - prompt processing, third pass: stage a slot from its card copy, not from RAM

## What the carve listing killed

The documented "next lever" from Session 61 was to split the expert products into two row halves to shrink the
prompt arena. `BASALT_DIAG_PROMPT` now lists every buffer's cost (one shot, in `Carve::take`), and it says the
idea was worthless:

| buffer | at 8,192 rows |
|---|---|
| `role[8]` (the residual states) | 8 x 72 = 576 MiB |
| `msg` (the four streams' messages) | 288 MiB |
| `mix` (three attention inputs) | 216 MiB |
| `nrm` (three normed rows) | 216 MiB |
| `yq` (the fused projection) | 205 MiB |
| `q` + `attn` + `o` | 264 MiB |
| **expert workspace** `xq` + `gu` + `h` | **1,044 MiB** |

`carve()` lays the expert workspace *from the arena's base* and ends with `c.at = max(end_a, x.at)` - and the
attention buffers (1,269 MiB) are bigger than the experts' (1,044 MiB), so the expert workspace is **already
hidden underneath them**. Halving it would have saved exactly nothing. The arena is attention-buffer bound.

## What actually pays: the link, not the buffer

`stage_from_card()` refuses while a loan is active (`loaned() > 0`), and a prompt runs *during* a loan - so the
~2,842 slots that already have a card copy (the elastic spill's, or the loan's) were being copied from RAM
anyway: **11.86 GiB of the 25.25 GiB per chunk crossed a 15.75 GB/s link for no reason.**

`stage_layer()` now walks a layer's RAM slots and takes each one from where it is: a slot with a copy is one
device-to-device copy (2 TB/s), and slots without one go as one H2D run per stretch of consecutive slots (host
and staging are both in slot order, so runs are the common case). It is used by `stage_ahead()` on the copy
stream and by the inline path, so a chunk's bytes no longer depend on which one runs.

```
per-slot staging 2842 copies D2D 11.86 GiB, 1191 runs H2D 13.40 GiB     (was 25.25 GiB, all H2D)
```

## Numbers (bench.ps1; 4,600 fresh words, 400 decode tokens, 2-3 processes per arm)

| chunk | prefill before | prefill now | decode before | decode now | hit |
|---|---|---|---|---|---|
| 4,096 | 1,976 | **2,300** | 99.5 | 91.5 | 94% |
| 6,144 | 2,331 | **2,570** | 92.6 | 87 | 76-88% |
| 8,192 | 3,018 | **3,153** (bench 3,033) | 84.8 | **92.9** | 93.3% |

The surprise worth keeping: decode came *back* at 8,192 rows (84.8 -> 92.9, hit 85% -> 93.3%). Halving the link
traffic during a prompt stops the prompt from demoting the copies decode wants. A 227-token prompt decoded at
**110 tok/s** in the same process. So 8,192 now dominates on both fronts instead of trading.

## Prefill, end to end

| arm | prefill | decode |
|---|---|---|
| start of session (chunk 4,096, serial staging) | 1,533 t/s | 101.8 tok/s |
| 8,192-row chunks | 2,226 | 83 |
| + staging on a copy stream | 3,018 | 84.8 |
| **+ per-slot D2D staging (now live)** | **3,153** | **92.9** |

**Prefill 2.06x, decode -9%.** Gate: prompt_parity 0 failures, kernels_parity 0 failures, Gate 0 six questions
correct, needles 872 / 1,634 / 3,971 / **70,585** all EXACT.

## What is left on this front

1. **The remaining 13.40 GiB of H2D per chunk** is the pack's RAM-resident experts, and it can only shrink with
   more card room for copies. The arena is attention-buffer bound (above), so the lever is the attention
   buffers themselves: `msg`/`mix`/`nrm` are 3-4 copies of `R x E` floats and `role[8]` is eight more. Sharing
   two of them (e.g. `nrm` is only ever `R x E` in use although carved `3 x R x E`) would free ~200 MiB.
2. **2,842 D2D copies per chunk** at ~5-10 us of launch each is ~20-30 ms of the chunk; coalescing copies that
   sit in the same elastic granule into one would take most of that.
3. The compute side: with staging now mostly D2D, the chunk is closer to compute-bound (attention + MMQ), so
   the #579 leftovers (`attn_prompt` double-buffered tiles, -11.6% on that kernel) are worth a look again.
