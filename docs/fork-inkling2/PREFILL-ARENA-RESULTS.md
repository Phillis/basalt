# Prefill: the arena's contents, and the compute wall
# Session 63 - prompt processing, fourth pass: what the arena is made of, and where the time went next

## The carve list

`BASALT_DIAG_PROMPT` now prints every buffer of a chunk (one shot, from `Carve::take`):

| buffer | at 8,192 rows |
|---|---|
| `role[8]` the residual states | 8 x 72 = 576 MiB |
| `msg` the four streams' messages | 288 MiB |
| `mix` three attention inputs | 216 MiB |
| `nrm` the normed rows | 216 -> **72 MiB** |
| `yq` the fused projection | 205 MiB |
| `q` + `attn` + `o` | 264 MiB |
| expert workspace `xq` + `gu` + `h` | 1,044 MiB, *hidden under the attention buffers* |
| **total** | 1,982 -> **1,838 MiB** |

Two conclusions, both acted on:

- The expert workspace is **already covered**: `carve()` lays it from the arena base and ends with
  `c.at = max(end_a, x.at)`, and the attention buffers (1,269 MiB) are bigger than it (1,044 MiB). The
  "split the expert products in half" idea from Session 61 would have saved nothing. Recorded so it is not
  tried again.
- `nrm` was carved `3 x R x E` and only ever used as `R x E` (`rms_norm_rows` writes it; `f32_rows_dot`,
  the mmq/fp4/mxfp8 quantizers and the peer gather read it). **Shipped**: 1,982 -> 1,838 MiB, `cuda0`
  1.6 -> 1.7 GiB. Gate 0 correct, needles 872 / 1,634 / 3,971 / 70,585 EXACT.

## The attn_prompt port: measured, then reverted - and the lesson is the useful part

Upstream `cfd41894` (`#579`) double-buffers every K/V tile and replaces the ring's 64-bit `key % cap` with a
power-of-two mask. The double-buffering does not port cheaply (its loop restructure meets our V4 fp8-keys /
4-bit-values loader in three conflict regions), so only the mask was taken.

It **broke prompt_parity 10 of 10** on the first build: the substitution rewrote the helper's own fallback
into `wrap_key(key, cap)` - infinite recursion. Fixed, parity returned to **0 failures**, and prefill measured
**3,064 / 3,033 t/s** against 3,033 / 3,153 without it. Neutral, so reverted (one exception to "no neutral
changes" was already spent on #636; not spending another).

The conclusion is worth more than the change: **the attention loop's modulo is not where a chunk's time goes.**
With staging now mostly D2D, the 8,192-row chunk is:

| term | per chunk |
|---|---|
| H2D staging (13.40 GiB at 15.75 GB/s) | ~851 ms |
| D2D staging (11.86 GiB, 2,842 copies, ~5-10 us of launch each) | ~30 ms |
| compute | ~1,540 ms |

**Compute is 63% of prefill now.** The FP4 expert products (gate/up 2F x E and down E x F per pair, 65,536
pairs per layer, 48 layers = ~50 TFLOP per 15.4k-token prompt) run at an effective **~20 TFLOPS**, far under
the card's peak. That is the next real lever, and it is kernel work in `prefill/fp4` (tile shape, occupancy,
the two-term activation trick's cost), not a flag.

## Where prompt processing stands

| arm | prefill | decode |
|---|---|---|
| start of session | 1,533 t/s | 101.8 tok/s |
| 8,192-row chunks | 2,226 | 83 |
| + staging on a copy stream | 3,018 | 84.8 |
| + per-slot D2D staging | 3,153 | 92.9 |
| + `nrm` R x E (live) | 3,100 (unchanged, +0.1 GiB of copies) | ~85-93 (noisy) |

**Prefill 2.02x** over the start of the session, decode within noise of where it began.
