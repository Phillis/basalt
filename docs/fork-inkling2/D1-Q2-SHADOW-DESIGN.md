# D1 design — the Q2_0 shadow quantizer and its integration

**For the deepseek worker.** This unblocks Session 8's scoped-down item: the tree has every Q2_0 *reader*
(CPU pool `q2_avx2_rows.inc`, GPU `mmq_q2_0.cu`, reference `dequant.hpp`) and **no writer**. This document
pins the format contract from the tree's own code, gives a reference quantizer, and corrects three numbers in
`NEXT-WORK-SPEC.md` §3 that were written before the geometry was checked (they do not change the plan, only
its arithmetic).

---

## 1. The format contract — from `include/basalt/artifact/dequant.hpp:68` (the referee)

**This fork's Q2_0 is not ggml's classic 32-value Q2_0.** `ggml_common.h` defines `QK2_0 = 64`,
`block_q2_0 { ggml_half d; uint8_t qs[16] }` — **18 bytes per 64 values, 2.25 bits/value**, and the reference
decode is:

```
d     = fp16_to_fp32(block[0..1])                       // little-endian half
code  = (qs[j / 4] >> (2 * (j % 4))) & 3                // value j's 2-bit code
value = (code - 1) * d                                  // symbols {-d, 0, +d, +2d}
```

Two properties that a naive quantizer gets wrong:

- **The symbol set is asymmetric** (−1, 0, +1, +2). The negative side only reaches −d while the positive
  side reaches +2d. The scale must therefore satisfy `d ≥ max(|min_j x|, max_j x / 2)`, *not* the symmetric
  `amax/2` habit.
- The code sits in the **low bits of byte j/4 first** (`2*(j%4)`), same packing order as the reference — the
  CPU and GPU kernels in-tree both consume this exact layout (`q2_avx2_rows.inc` `unpack64`, mirrored by
  `mmq_q2_0.cu`). There is exactly one layout to match; no variants.

## 2. The quantizer (reference implementation)

```cpp
// NVFP4 blob -> Q2_0 shadow blob. `src` is the dequantised f32 of one 64-value block (NO per-blob
// s_gate/s_up/s_down applied - they stay in the tail), `dst` is 18 bytes out.
void quantize_q2_0_block(const float* src, uint8_t* dst) {
    float max_pos = 0.f, max_neg = 0.f;
    for (int j = 0; j < 64; ++j) {
        if (src[j] > max_pos) max_pos = src[j];
        if (src[j] < max_neg) max_neg = src[j];
    }
    float d = std::max(max_neg * -1.f, max_pos * 0.5f);   // cover -d .. +2d exactly
    if (d == 0.f) d = 1.f;                                // all-zero block
    const float id = 1.f / d;
    uint16_t half = fp32_to_fp16(d);                      // F16C round-to-nearest; the readers decode this back
    dst[0] = (uint8_t)(half & 0xFF); dst[1] = (uint8_t)(half >> 8);
    uint8_t qs[16] = {};
    for (int j = 0; j < 64; ++j) {
        int c = (int) std::lrintf(src[j] * id) + 1;       // -1 -> 0, 0 -> 1, +1 -> 2, +2 -> 3
        c = std::clamp(c, 0, 3);                          // saturate: only |x| > 2d or < -d can reach here
        qs[j / 4] |= (uint8_t)(c << (2 * (j % 4)));
    }
    std::memcpy(dst + 2, qs, 16);
}
```

Optional refinement (worth one experiment, not required): scan `d` over a few multipliers of the cover bound
(1.0, 0.9, 0.8) and keep the minimum per-block MSE. The cover bound alone is already safe — saturation is
impossible by construction.

**Where the per-blob `{s_gate, s_up, s_down, 0}` tail fits:** the NVFP4 kernels apply those scales *outside*
the block decode (`nvfp4_512_gu_rows(..., sg, su, ...)`), and the blob tail carries them. Convert the shadow
from NVFP4 values *without* sg/su/sd applied (the reference dequant's raw output), then **copy the original
16-byte tail verbatim** to the shadow blob's end. Both the CPU pool (`NativeFmt` gate/up scales) and the GPU
SwiGLU form (`mmq::swiglu` reads `base + blob_bytes - 16`) then behave identically per type.

## 3. Corrected geometry and arithmetic (supersedes NEXT-WORK-SPEC §3's numbers)

| | NVFP4 | **Q2_0 shadow** |
|---|---|---|
| bits/value | 4.5 (36 B / 64) | **2.25 (18 B / 64)** |
| gate/up row (2304 values) | 1,296 B | **648 B** |
| down row (1152 values) | 648 B | **324 B** |
| blob | 4,478,992 B | **2,239,504 B (exactly half)** |
| blob layout | — | **different row bytes**: NOT a drop-in — needs its own descriptors |

Consequences:

- **Shadow RAM ≈ 6.1 GiB** (2,738 × 2.14 MiB), not the 4.4 GiB previously estimated. It fits the ~10 GiB free
  (pageable, never pinned — first-touch at build), but re-check free RAM under the user's normal desktop
  load before committing to the flag default.
- **Prefill staging halves, not ÷2.7**: H2D per chunk ~21 → **~10.5 GiB → ~0.67 s** (with the Q8_0 layers
  20-23 unchanged). On its own that is ~2.2 s/chunk ≈ 1,850 t/s; with wave pipelining it is fully hidden
  behind the 1.39 s compute anyway — the shadow's prefill value is margin and long-context insurance.
- **The shadow needs parallel per-layer descriptors**, because `LayerWeights` carries one
  `experts/stride/blob_bytes` per layer: add `{shadow_blobs, shadow_gu_row = 648, shadow_down_off,
  shadow_stride, shadow_blob_bytes}` (a small struct beside `LayerWeights`, or fields on the spill state —
  not on LayerWeights, which the GPU paths share). `CpuSpill` builds a second `NativeFmt` via the existing
  `native_fmt(GGML_TYPE_Q2_0, ...)` (the pool path already dispatches `q2_native_kernels`), and `host_blob`
  answers with the shadow pointer when the flag is on. The prefill side (`prompt.cpp` `experts()`) takes the
  shadow base/stride/type for the staging copy and `mmq::Product` (`p.type = GGML_TYPE_Q2_0`) — the Q8_0
  layers and any layer without a shadow keep today's path.

## 4. Build and plumbing order

1. **Round-trip harness first** (kills the "garbage not error" risk before the engine is touched):
   a small standalone (`tools/` or a `tests/kernels` case) that, per layer, samples N spilled blobs,
   dequantises NVFP4 (`dequant.hpp` reference) → quantises Q2_0 → decodes back with **the tree's own
   `dequantize_q2_0`** → reports per-layer SNR (expect ~15-22 dB; report the distribution, gate at a floor
   you choose with the user). Also assert the CPU pool's `q2_rows_any` output matches the reference decode on
   a synthetic block set. This harness is also the quality evidence for sign-off.
2. The converter at load: after the spill copies land (know them from `CpuSpill::open`, which already walks
   every spilled (layer, expert)), dequant→quant the **NVFP4 layers 0-19 only** (Q8_0 layers 20-23 keep their
   blobs — already cheap) into the pageable shadow, **first-touch every page**, parallel over ~12 threads
   (~9.1 G values; expect ~10-20 s added load time). Copy each source blob's 16-byte tail verbatim.
3. `--cpu-shadow q2|off` via `dspark_parse_flag` (`dspark.cpp:722`), default off; a log line with the shadow
   bytes and the harness's SNR summary so every load self-reports quality.
4. Decode switch (`CpuSpill`), then prefill switch (`prompt.cpp` staging + `mmq::Product` type), A/B'd
   separately behind the one flag.
5. Validation per NEXT-WORK-SPEC §5: decode `diag` pool busy ≤ 8 ms (today 15.4), TCP-prompt decode ≥ 65,
   unique-16k prefill ≥ 1,850 (with shadow alone), needle, sustained run, and **side-by-side generation
   samples for the user** — the quality trade is theirs to accept, and the q4 fallback (Q4_0, same bytes as
   NVFP4, decode-only ~1.4× cheaper ops, no prefill win) remains the escape hatch if Q2_0 text quality
   offends.

## 5. Projected end state (corrected numbers)

| Step | Decode (TCP prompt) | Prefill (unique 16k+) |
|---|---|---|
| Today | 53 | 1,561 |
| + shadow (D1) | ~65-70 (pool 15.4 → ~7-8 ms) | ~1,850-2,000 |
| + wave pipelining (P1) | unchanged | ~2,700-2,900 (compute-bound) |
| + drafter capture (D2) | ~70-75 | unchanged |

The pool's shadow reads are ~half the bytes *and* the 2-bit decode is cheaper than NVFP4's (no fp8
sub-scale LUT, no nibble-LUT shuffle — the `q2_dot4` path is plain masked extracts), so the ~2× on the pool
term is conservative.
