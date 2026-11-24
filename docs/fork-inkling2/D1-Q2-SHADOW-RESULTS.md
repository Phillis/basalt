# D1 — Q2_0 shadow: format proven, quality measured, shadow not built

Outcome report for `D1-Q2-SHADOW-DESIGN.md`. Self-contained: what was implemented, what was measured, why the shadow was
**not** built, and what a future session would need if the quality trade is accepted anyway.

**Summary: the writer is correct and verified (0 packing mismatches, 0 saturations); the quality on the model's own weights
is 3.56 dB with the design's scale and 7.90 dB at best — against the design's predicted 15-22 dB. That is the reason the
6 GiB shadow was not built.**

---

## 1. The format contract (as verified, not as assumed)

This fork's Q2_0 is **not** ggml's classic 32-value Q2_0:

| | |
|---|---|
| Block | **18 bytes per 64 values** (2.25 bits/value) |
| Layout | `uint16 d` (fp16, little-endian) then `uint8 qs[16]` |
| Code position | value `j` is `(qs[j / 4] >> (2 * (j % 4))) & 3` — low bits first, 4 codes per byte |
| Symbol set | code `{0,1,2,3}` → **`{-d, 0, +d, +2d}`** — **asymmetric** |
| Consequence | the scale must cover `-d .. +2d`, i.e. `d = max(|min|, max/2)`, **not** the symmetric `amax/2` habit |
| Type id / geometry | `GGML_TYPE_Q2_0 = 42`, `QK2_0 = 64` (`include/basalt/ggml/ggml_common.h:140`) |
| Referee | `include/basalt/artifact/dequant.hpp:69` (`dequantize_q2_0`), mirrored by the CPU (`q2_avx2_rows.inc`) and GPU (`mmq_q2_0.cu`) readers |

The writer, as implemented in `src/models/inkling2/cpu_spill.cpp`:

```cpp
static void quantize_q2_0_block_scaled(const float* src, uint8_t* dst, float mult) {
    float max_pos = 0.f, max_neg = 0.f;
    for (int j = 0; j < 64; ++j) {
        if (src[j] > max_pos) max_pos = src[j];
        if (src[j] < max_neg) max_neg = src[j];
    }
    float d = std::max(max_neg * -1.f, max_pos * 0.5f) * mult;   // mult < 1 clips outliers for resolution
    if (d <= 0.f) d = 1.f;
    const float id = 1.f / d;
    const uint16_t half = f32_to_f16(d);                          // the tree ships only fp16 -> f32
    dst[0] = (uint8_t)(half & 0xFF); dst[1] = (uint8_t)(half >> 8);
    uint8_t qs[16] = {};
    for (int j = 0; j < 64; ++j) {
        int c = (int) std::lrintf(src[j] * id) + 1;               // -1 -> 0, 0 -> 1, +1 -> 2, +2 -> 3
        qs[j / 4] |= (uint8_t) std::clamp(c, 0, 3) << (2 * (j % 4));
    }
    std::memcpy(dst + 2, qs, 16);
}
```

The per-blob `{s_gate, s_up, s_down, 0}` tail is **not** part of the block: NVFP4 kernels apply those scales outside the
decode, so the shadow copies the source blob's last 16 bytes verbatim and both readers behave identically.

Blob geometry (this is why the shadow needs its own descriptors — it is not a drop-in):

| | NVFP4 | Q2_0 |
|---|---|---|
| gate/up row (2,304 values) | 1,296 B | **648 B** |
| down row (1,152 values) | 648 B | **324 B** |
| blob (incl. 16 B tail) | 4,478,992 B | **2,239,504 B** |

---

## 2. What was proven (step 1a) — the format is right

`C:\LLM\bwtest\q2_roundtrip.cpp` (+ `build_q2.bat`, `q2_roundtrip.exe`): generates synthetic NVFP4 blocks, then per block runs
`dequantize_nvfp4` → writer → **two independent decoders** — the tree's `dequantize_q2_0` and one written straight from the
contract above — compares them, and counts values the clamp had to move.

```
blocks                200000
packing mismatches    0   (independent decoder vs the tree's dequantize_q2_0)
saturations           0   (must be 0: the cover bound makes them impossible)
```

Both failure modes that would have produced garbage instead of an error are excluded.

---

## 3. What was measured (step 1b) — and it changes the decision

An opt-in pass in the engine samples the **model's own spilled expert blobs** rather than synthetic data: on the first forward
after load it walks all 20 NVFP4 layers, takes 4 spilled experts per layer, strides through their gate/up/down blocks, and
round-trips each block through the tree's dequantizers.

| Scale strategy | Real-weight SNR |
|---|---|
| Cover bound (design's reference: no saturation) | **3.56 dB** (per-layer 2.77 – 4.37) |
| Best of a clipping ladder, `d × {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}`, min per-block error | **7.90 dB** |
| **The design's prediction** | **15 – 22 dB** |

The estimate does not hold. A 2.25-bit grid with four *asymmetric* levels cannot reach 15-22 dB on weight-like data; the
design's figure looks like it was carried over from a symmetric-scale 2-bit scheme. Even the error-minimising scale reaches
only 7.9 dB, where the error is roughly 40% of signal amplitude.

This distortion would apply to the **~27% of routed lookups that miss the card** — on decode, and to every staged expert on
prefill. It is a real quality regression, not a rounding detail.

Sample lines from the run:

```
cpu-shadow measure: layer 0   Q2_0 round-trip SNR 3.45 dB (4 experts sampled)
cpu-shadow measure: layer 9   Q2_0 round-trip SNR 4.20 dB (4 experts sampled)
cpu-shadow measure: layer 19  Q2_0 round-trip SNR 3.74 dB (4 experts sampled)
cpu-shadow measure: 20 NVFP4 layers, aggregate Q2_0 round-trip SNR 3.56 dB (the shadow is OFF: nothing changed)
cpu-shadow measure: with a clipping-ladder scale the aggregate SNR is 7.90 dB
```

---

## 4. Why the shadow was not built

The design gates on exactly this measurement — "report the distribution, gate at a floor you choose with the user" — and
the quality trade is explicitly the user's to accept. On 3.56 / 7.90 dB against an expected 15-22 dB, building it would mean:

- **~6 GiB of pageable RAM** held for the life of the process (the spill's own sizing is the constraint this machine already
  sits near), plus ~10-20 s of added load time,
- for a shade under **half the prefill H2D** (which P1's wave pipelining largely hides anyway — the chunk becomes
  compute-bound either way),
- at a distortion level roughly 2-3× worse than the plan assumed.

That trade is not obviously worth it, and it is reversible: nothing was built, so nothing has to be undone.

---

## 5. What is in the tree now

| Path | What |
|---|---|
| `C:\LLM\bwtest\q2_roundtrip.cpp` / `.exe` / `build_q2.bat` | the standalone harness from §2 |
| `src/models/inkling2/cpu_spill.cpp` | `f32_to_f16()`, `quantize_q2_0_block()`, `quantize_q2_0_block_scaled()`, and the measurement pass |

**All of it is inert.** The measurement only runs when the marker file exists, and the shadow is never allocated, so the
engine's behaviour and numbers are unchanged (verified after removing the marker: 35.9 / 45.2 tok/s on 250-token runs,
`hit=` 69.4 / 69.6%).

### Re-running the measurement

```powershell
"on" | Out-File C:\LLM\cpu-shadow-measure.on -Encoding ascii
C:\LLM\Basalt\Start-Basalt.cmd          # then send one request
Get-Content "$env:USERPROFILE\.cache\basalt\55cfc8c42932d132\engine.log" | Select-String 'cpu-shadow'
Remove-Item C:\LLM\cpu-shadow-measure.on        # back to normal (or the next load re-runs it once)
```

The re-run costs nothing but the sampling pass: it allocates no shadow and touches no output.

---

## 6. If you decide to go ahead anyway

The design's remaining steps, in the order it gives them:

1. **Build the shadow at load** for the spilled experts of layers 0-19 (~2,738 × 2.14 MiB ≈ 6.1 GiB, **pageable RAM, never
   pinned**; first-touch every page at build or the first decode after load pages in GiBs and stalls the pool). Convert from
   each spilled expert's RAM copy (`model.host_blob(l, e)`), copy the 16-byte tail verbatim, parallelise over ~12 threads.
2. **Use the clipping scale** — 7.90 dB instead of 3.56 is free, and there is no reason to ship the cover bound.
3. **Parallel descriptors per layer** — `LayerWeights` carries one `experts/stride/blob_bytes`; the shadow needs its own
   `{blobs, gu_row = 648, down_off, stride = 2,239,504}`, held beside `LayerWeights` rather than on it (the GPU paths share
   that struct).
4. **Decode first** (`CpuSpill::run()` picks the shadow blob + a Q2_0 `NativeFmt` from the existing `native_fmt(Q2_0, …)`),
   A/B'd on its own; **then prefill** (`prompt.cpp` staging from the shadow + `mmq::Product` type `Q2_0`, keeping Q8_0 layers
   20-23 as they are).
5. **Gate**: pool busy ≤ 8 ms (today 15.4), decode ≥ 65 on the TCP prompt, unique-16k prefill ≥ 1,850, needle test, and
   **side-by-side generation samples** — quality is the whole question here.

Preferable alternative if the goal is prefill: **skip this and do P1 wave pipelining** (no quality cost, ceiling ~2,400-2,900
tok/s). D1's prefill benefit is mostly hidden by P1 anyway; its distinct value is the decode pool term.

---

## 7. Where the numbers sit

| Build | Decode | Prefill (unique 16k+) |
|---|---|---|
| Today (no shadow) | 39–67 tok/s by content | 1,536–1,561 |
| + Q2_0 shadow (if accepted, 7.9 dB) | ~65–70 | ~1,850–2,000 |
| + P1 pipelining | unchanged | ~2,400–2,900 |
| Hardware (16 GB+ card) | ~100+ | — |
