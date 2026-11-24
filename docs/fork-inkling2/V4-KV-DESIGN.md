# V4 — a four-bit value row for inkling2's K/V cache: built, measured, next step

The engine's cache is `fp8` for K and V: 128 B a row, 2,048 B a K+V cell (8 heads), ~6.0 GiB at 192k. **V is the half that
can be compressed safely** — it is averaged through the softmax weights, not matched like K — so the first slice of the
NVFP4 idea is a 4-bit V with K left exactly as it is.

## What is built and verified

- `include/basalt/models/inkling2/kernels.hpp` — `kKvV4Group = 32`, `kKvV4CodeBytes = 64`, `kKvV4RowBytes = 72`
  (codes, then one fp16 scale per 32 values), cell **1,152 B = 56.2% of fp8**.
- `src/models/inkling2/kernels/prep.cuh` — the codec, in the same shape as the existing INT8 one:
  `v4_scales` (a warp holds one 128-value row, so the max reduces inside each 8-lane/32-value group), `store_v4`
  (each lane writes its own two bytes, so no two lanes touch a byte), `unpack16_v4` (16 values from 8 bytes) and
  `v4_row_scales` for the attention kernels.
- `C:\LLM\rebase\v4_parity.cu` — the harness: it encodes with the engine's own device functions and decodes on the CPU.

**Exactness: 0 packing/separation mismatches over 524,288 codes** across gaussian, heavy-tailed and outlier rows.

## Measured SNR (dB, higher is better), 4,096 rows a case

| V distribution | fp8 | int8 | **v4, no rotation** | **v4, H128 rotation** |
|---|---|---|---|---|
| gaussian | 31.50 | 44.54 | 20.26 | 20.28 |
| heavy tail (1% ×10σ) | 31.58 | 37.98 | **15.70** | **21.06** |
| outliers (0.1% ×50σ) | 31.51 | 36.38 | 20.04 | **23.24** |

Two things fall out of this:

1. **4-bit V costs ~9 dB against fp8** (21 vs 31.5) — a 3× larger error, which is in the range published 4-bit KV results
   work in, but it is a real trade and needs an end-to-end gate, not just an SNR figure.
2. **The rotation is not optional.** Without it heavy-tailed rows collapse to 15.7 dB; with a normalized 128-point
   Hadamard they come back to 21.1 (+5.4 dB) and outliers reach 23.2. It also improves int8 sharply (36.4 → 47.8 on
   outliers), which is worth remembering for any future cache-format work.

**Why the rotation is cheap:** the attention output is `Σ wᵢ Vᵢ`, which is linear, so the kernel can sum the *rotated*
dequantized values and apply the single inverse rotation once per output row — not per tile, not per key. That is exactly
how `q4_0`/`k8v4` are built upstream (they refuse inkling2 today because only the dense stack has them).

## What it buys

2,048 → 1,152 B a cell ⇒ at 192k the KV cache falls from ~6.0 GiB to ~3.4 GiB, **freeing ~2.6 GiB for expert copies** —
roughly **480 more experts resident** at our pack's ~5.5 MiB a blob. Residency is worth real throughput here (192k → 128k
measured +5.9% decode on the old tree), so the projection is a few percent of decode and more headroom for the context
window. It is not free: it is bought with ~9 dB of V precision.

## Integration checklist (the part not yet done)

1. `AttnArgs`: split `kRowBytes` into `kRowBytes` (K) and `vRowBytes` (V); add `bool v4`.
2. `attn.cu` — the simple path's `load_tile` (K stays a `uint4`/16 values, V becomes a `uint2`/16 values +
   `v4_row_scales`), the dequantize into `sm.v` and the MMA path (`attn_mma_kernel`, #626) which builds V tiles for
   tensor cores.
3. `attn_prompt.cu` — the same two things for the prompt kernel.
4. `ops.cu` — the V writer: `v4_scales` + `store_v4` beside the existing `store_kv` (the writer already owns a whole row
   per warp, which is exactly what the codec expects).
5. Cache sizing (`session.cpp`, `glue.cu`, `kernels::kv_cache_row_bytes`) and the adapter's `--kv` whitelist, as an
   **opt-in** format (default stays fp8), then the output-domain inverse rotation on the accumulated row.
6. **Quality gate before it ships:** Gate 0, coherence at length, needle at 130,709 and 160k+ (the two we use), and a
   prefill/decode ABBA. Only then consider making it a profile rather than a flag.

## Scaffolding that is now in the tree (compiles, no behaviour change)

- `kernels.hpp`: `kv_v_row_bytes(fp8, int8, v4)` and `kv_pair_bytes(fp8, int8, v4)`, and `AttnArgs::v4 = false`.
- `prep.cuh`: `v4_scales`, `store_v4`, `unpack16_v4`, `v4_row_scales`.
- Verified: the engine still builds and passes Gate 0 with the flag off.

## Two structural discoveries that cut the work down

1. **Most layers never touch the tensor-core kernel.** `attention_decode` picks the MMA path only when
   `a.window == 0 && !a.int8` (`attn.cu:928`) — the sliding-window layers, which are most of the model, already run the
   simpler `attn_split_kernel`. So **V4 in the split kernel covers the bulk of decode**, and the few full-attention layers
   can simply be routed to the split kernel while v4 is on (one extra `&& !a.v4`), trading a little speed on a handful of
   layers for not rewriting #626's MMA tile fetch.
2. **`attn.cu` does not include `prep.cuh`** (it carries its own copies of `unpack16_fp8`/`unpack16_int8`/
   `row_scales`, and `prep.cuh` defines the same `warp_sum`/`warp_max` names), so the V4 codec must be mirrored into
   `attn.cu` rather than included.

## Exact edit points, in order

| # | File:line | Change |
|---|---|---|
| 1 | `attn.cu:~123` (after `row_scales`) | mirror `v4_row_scales`, `unpack16_v4`, `v4_scales`, `store_v4` |
| 2 | `attn.cu:162` | `template <bool FP8, bool INT8, bool V4>` on `attn_split_kernel` |
| 3 | `attn.cu:176-180` | `constexpr int kVRowBytes = V4 ? kKvV4RowBytes : kRowBytes;`, `vbase + g * kVRowBytes` |
| 4 | `attn.cu:182-201` | `load_tile`: V cell offset by `kVRowBytes`; V4 reads 8 bytes a 16-value group (`uint2`); scales via `v4_row_scales` |
| 5 | `attn.cu:256-275` | V dequant: `V4` arm — `unpack16_v4(vr[i].xy, vs[i/2], f)` → `__floats2half2_rn` into `vrow` |
| 6 | `attn.cu:634-670` | `attn_fused_kernel` (the drafter's window): same three changes; `fs.vw` rows become 72 B |
| 7 | `attn.cu:928,951-953` | `use_mma = ... && !a.v4`; add the three `V4` split-kernel launch arms |
| 8 | `attn_prompt.cu:130` | the prompt kernel's V row offset and its 4-bit unpack |
| 9 | `glue.cu:490-493` | writer: `row_bytes = w ? kv_v_row_bytes(...) : kv_cache_row_bytes(...)`; `w == 1` uses `v4_scales` + `store_v4` |
| 10 | `ops.cu:302-312` | same for the other writer (its `which` selects K or V) |
| 11 | `model.cpp:74`, `session.cpp:290,592`, `prompt.cpp:123,522` | size cells with `kv_pair_bytes(fp8,int8,v4)` |
| 12 | `model.cpp:172,340-341`, `model.hpp:201` | `v4_` from the option; `cache_v4()`; set `aa.v4`/`ap.v4` at `session.cpp:927-943`, `prompt.cpp:475` |
| 13 | `inkling2/adapter.cpp:63` | accept `--kv fp8v4` (keys fp8, values 4-bit) and set the flags |

Then the gate: Gate 0, coherence, needles at 130,709 and 160k, and a prefill/decode ABBA against `fp8` — with the rotation
added to the writer and the single inverse rotation in each kernel's output row if the quality gate needs it (measured:
+5.4 dB on heavy-tailed rows).

## Progress: the decode half is in the tree (commit `788054fd`, inactive by default)

Done in this pass, compiling and verified not to change behaviour:

- `attn.cu` carries the codec (`v4_row_scales`, `unpack16_v4`, `v4_scales`, `store_v4` — it cannot include `prep.cuh`,
  whose `warp_sum`/`warp_max` it already defines itself).
- **`attn_split_kernel<FP8, INT8, V4>`**: value rows address by `kVRowBytes` (72) instead of the key's row bytes, a 16-value
  group loads as eight bytes (`uint2`), the group scales come from the row's tail, and the dequant writes fp16 pairs into
  `sm.v` exactly as the fp8 path does. This is the kernel **most layers use** (sliding window), so this is the bulk of decode.
- **`attn_fused_kernel<FP8, INT8, V4, ROWS>`** (the drafter's window): same read side, plus a four-bit arm for its two
  write sites — the window copy in `fs.vw` and the cache row.
- `attention_decode` now drops to the split kernel when `v4` is on (`use_mma = a.window == 0 && !a.int8 && !a.v4`), because
  `attn_mma_kernel` reads 16-byte K/V vectors; the six split-kernel instantiations and their shared-memory attributes are wired.

**Nothing sets `AttnArgs::v4` yet**, so the engine behaves exactly as before (Gate 0 Paris/391, decode 73.0 tok/s).

## What is left, in order

1. **`attn_prompt.cu`** — the prompt path is coupled more tightly than decode: it stages a chunk of "fresh" rows into
   `a.k_chunk`/`a.v_chunk` **in cache format** (`prompt.cpp:522` sizes it with `kv_cache_row_bytes`), and `load_tile` reads
   from there. So both the staging write and the tile read need v4, and the staging size must use `kv_pair_bytes`.
2. **Writers**: `glue.cu:~490` (`prepare_kernel`, warps 0/1 write K and V rows) and `ops.cu:~302` — both already own a whole
   row per warp, which is what `v4_scales` + `store_v4` expect.
3. **Sizing**: `model.cpp:74` (`per_pos`), `session.cpp:290,592`, `prompt.cpp:123,522` → `kv_pair_bytes(fp8, int8, v4)`.
4. **Flags**: `model.cpp:172,340` (`v4_ = req.kv == "fp8v4"`), `model.hpp:201` + `cache_v4()`, the `aa.v4`/`ap.v4` fills at
   `session.cpp:927-943` and `prompt.cpp:475`, the fused V4 launch arms at `attn.cu:~1103`, and `inkling2/adapter.cpp:63`.
5. **The gate**: Gate 0, coherence, needles at 130,709 and 160k, then a prefill/decode ABBA against `fp8`; add the H128
   rotation (writer + one inverse rotation per output row) if the needle or a KLD check wants the extra ~5 dB.

## Status: the whole path is in and runs — with a correctness bug to find

Committed as `cd39a692` (on top of `788054fd` decode and `552425f7` prompt). Now in the tree:

- **Writers**: `glue.cu`'s `prepare_kernel` and the other writer in `ops.cu` store four-bit value rows.
- **Row padding**: the V row is **80 bytes** (`(64 codes + 8 scale bytes + 15) & ~15`), not 72, because `stage_to_ring` copies
  `uint4` pieces; 80 keeps everything 16-byte addressable at 62.5% of fp8.
- **`stage_to_ring`** takes the two row bytes and picks its piece count per side.
- **Sizing split**: `session` allocates K and V with their own cell bytes, `visit_cache_bytes`/`session_bytes` take `v4`, and
  the two places that had broken on it are fixed — the conversation snapshot (`v_run`) and the ring regions
  (`v_bytes_of`), which had been declaring a value region with a key's size and giving `checkpoint transfer snapshot:
  invalid argument`.
- **Flags**: `--kv fp8v4` (fp8 keys, four-bit values) accepted by the adapter, `fp8_`/`v4_` on the model, `cache_v4()`,
  and the `AttnArgs`/`AttnPrep`/`AttnPromptArgs` fills.

**What works**: at `--kv fp8v4` the engine loads (32k test plan: cuda0 6.8 GiB / ram 8.1 GiB — more residency than fp8),
plans, and serves requests.

**What does not**: the tokens are garbage. So one of the three round-trip points disagrees — the write side (`v4_scales` +
`store_v4`), the decode/prompt read side (`v4_row_scales` + `unpack16_v4` with `vs[i / 2]`), or the row addressing
(`kKvV4CodeBytes` tail scales vs `kKvV4RowBytes` stride). The parity harness already proves the codec itself is exact
(0 mismatches over 524,288 codes), which narrows it to addressing or the writer's scale application, not the maths.

## Next step (a contained debug)

1. Run `kernels_parity`-style: write one row with `prepare_kernel`'s v4 arm, read it back with `unpack16_v4`, compare —
   that isolates writer vs reader in one shot.
2. Then check the two addersses: `vbase + (key % cap) * kKvHeads * kVRowBytes + g * kVRowBytes` (decode) and the prompt's
   `row + col / 2` with `kKvV4CodeBytes` scales.
3. Then the gate: Gate 0, coherence, needles at 130,709 and 160k, and an ABBA against fp8.

**The default engine is untouched**: `kv=fp8`, 192k, Gate 0 Paris/391/planets, decode 68.7 t/s — verified after every step.

## Debug session 2: the format is now genuinely active — and one path is still wrong

**Found and fixed:** the window layers (most of the model) dispatch through `attention_window` → `attn_fused_kernel`, and
its launch arms only covered the non-V4 instantiations. So a four-bit cache was read there as fp8 — that was the garbage.
`attention_window` now has V4 arms for `ROWS` 2 and 3 plus their shared-memory attributes (commit `abcac5da`).

**Also found:** `serve` does not use `--config` the way it looks. It chdirs to the install root and hands the config to the
Python layer, which reads the tree's own `basalt_config.json` — every earlier "v4 test" was silently running fp8 at 192k.
The format is set in `basalt_config.json` (a copy of the fp8 one is kept as `basalt_config-fp8.json`).

**Confirmed working:** with `kv=fp8v4` genuinely active, the **plan responds**: at 192k `cuda0` 2.9 → **3.5 GiB**,
`ram` 12.3 → 11.7 GiB. So the sizing path is right and the memory is really being freed.

**Still wrong:** the output is incoherent and the engine crashed with a backtrace in that same window path. So there is a
second bug, and it is in `attn_fused_kernel`'s own four-bit arm rather than in dispatch. The most likely spots, in order:

1. **`fs.vw` row width.** The window copy is `uint8_t vw[ROWS][kHeadDim * 2]`, sized for the widest row. With v4 only
   `kKvV4RowBytes` (80) is written per row, but the fused kernel's *other* consumer of those rows — the `w >= 2` branch that
   reads a row prepared earlier in the same block — may still assume a key-sized row.
2. **The K row in the same branch.** The fused kernel writes both K and V through one code path split on `w`; the V4 arm
   restructured that into two separate branches, so the K side's ring write and cache write need to be checked for a
   duplicated or skipped store.
3. **The `p0`/window row path** (`key >= p0`) that reads `fs.vw` — its `uint2` piece count must be `kKvV4RowBytes / 8` and
   its scales read at `kKvV4CodeBytes`, both of which the cache path got right.

**Where it stands:** the format is complete, selectable, and demonstrably frees the memory it should; the window kernel's
four-bit read is the one remaining defect. The default engine is fp8 and verified after every step
(Paris / 391, decode ~73-77 t/s at 192k).

## Debug session 3: what it is not

Three hypotheses were killed with cheap experiments, which is the useful part of this round:

| Experiment | Result |
|---|---|
| The window layers' dispatch (missing V4 arms) | **real bug, fixed** (`abcac5da`) — but not the whole story |
| Stride/addressing: `kKvV4RowBytes = 128` (fp8's) so every stride matches | **still incoherent and still crashed** → not addressing |
| Drafter-specific: `spec 1` (no drafter) | **still incoherent** → the base decode path |
| The codec itself | 0 mismatches over 524,288 codes, and with fp8's stride a stride confusion cannot hide |

So it is in the four-bit **arms** (writer or reader), not in dispatch, strides or the codec.

### Where to look next, in order

1. **`ops.cu`'s writer.** Prompt rows are written by it into the ring layer's *staging* buffer (`ap.v_cache = b().v_stage`
   for ring layers), and a Gate 0 prompt of 22 tokens already crosses the prompt path (`n_tokens > 16`). It is the one
   writer never exercised in isolation, and it differs from `glue.cu`'s in one way that matters: `v4_scales` there is
   called with `f` folded in by the caller (`vv = val * f`) — check that its `which`/warp mapping gives a whole warp per
   row, since `v4_scales` reduces with `__shfl_xor_sync` and needs all 32 lanes.
2. **The fused kernel's `fs.vw` reader** (`attn.cu:~840`) against its writer (`~780`): the same row through two different
   code paths in one block.
3. Then the ring staging's `stage_to_ring` (`v_pieces`), which is now correct in shape but untested.

Best tool for the next pass: `kernels_parity.cpp` already compares `attention_decode` and `attention_window` against
references — building it (`-DBUILD_TESTING=ON`) with a v4 format added gives a unit test that names the failing kernel in
one run, instead of inferring it from the engine's output.

## DONE: the four-bit value cache works end to end, with measured numbers

**The last bug** was `prompt.cpp`: it builds its own `AttnPrep` for the prompt rows and set `fp8`/`int8` on it but not `v4`
— so the writer laid value rows out at fp8's 128 bytes while every reader took them as four-bit 80-byte rows. One line
(`ap.v4 = m_.cache_v4()`), commit `a01e8284`, and `--kv fp8v4` behaves.

**Correctness at 192k**: "The capital of France is Paris.", "17 A? 23 = 391", the planets in order, and a **needle placed
at 8k retrieved exactly** (`7419-KX`) through the four-bit cache.

### Performance, 192k, same build, 9 decode runs per format

| | fp8 | **fp8v4 (four-bit values)** |
|---|---|---|
| prefill, unique 15.4k | 1,594 t/s | **1,707 t/s (+7.1%)** |
| decode 400 tok, median | **72.6** tok/s | 68.6 tok/s (−5.5%) |
| decode spread | 67.6–75.6 | 61.1–69.5 |
| expert hit | 67.4% | **68.6%** |
| card / ram | 2.9 / 12.3 GiB | **3.5** / 11.7 GiB (KV −44%) |

So the format does exactly what it was designed to do — 44% less KV memory, ~0.6 GiB more expert residency, a better hit
rate and 7% more prefill — and it **costs 5.5% of decode**, because unpacking a code in the attention inner loop costs more
than E4M3's hardware convert. At these window sizes the kernel is instruction bound, not bandwidth bound, so the byte
saving does not pay for the unpack.

### How to make it a clean win (next lever)

The unpack is the whole cost. `unpack16_v4` / `v4x16_to_h8` currently do per-value float multiplies and a
`__floats2half2_rn`; a 16-entry `__constant__ __half` code table plus `__hmul2` (one lookup and one half2 multiply per two
values) should cut it by ~4 instructions a pair. If that recovers the 5.5%, four-bit values become a strict win: prefill
+7%, decode ≥ fp8, plus 0.6 GiB of residency to spend on context or experts.

Default stays `kv=fp8` (decode is what most work here is); `kv=fp8v4` is one line in `basalt_config.json` for
prefill-heavy or long-context use.

### Everything now covered by tests (both suites, 0 failures)

| Component | Check |
|---|---|
| codec | 524,288 codes, 0 mismatches |
| `store_v4` scale | half-a-step rule (0.500) |
| both prepare writers | bitwise against each other (fp16 / fp8 / v4) |
| decode attention | vs CPU reference, max abs 7.4e-05 |
| fused window kernel | bitwise vs prepare+decode, 9 geometries |
| prompt attention (window, full) | vs CPU and vs `attention_decode`, max abs ~1e-03 |

## The fast unpack, and what the remaining 5% actually is

`unpack32_v4` (commit `5ab90b0e`): one 16-byte load is **exactly one 32-value scale group**, so a row is four loads of
16 bytes instead of eight of 8, and the codes are small integers — the convert is an integer one and the scale a single
`__hmul2`, replacing the per-value float multiplies. Parity stayed clean (0 failures; decode attention unchanged at
7.4e-05).

**It did not close the gap**, and the engine log shows why:

| | fp8 | v4 |
|---|---|---|
| drafts accepted | ~88% | **~78%** |
| tokens a window (dl) | 2.0 | **1.6** |
| decode | 72.6–74.0 | 66.2–70.1 |
| prefill | 1,623–1,657 | **1,710–1,719** |

The 5% is **not kernel time** — it is the drafter: four-bit values perturb the target's forward enough that its drafts
verify less often (88% → 78%) and each window carries fewer tokens (2.0 → 1.6). Throughput pays for precision, not for
instructions.

### So the clean win needs precision, and the rotation is the measured lever

The H128 rotation measured **+5.4 dB on heavy-tailed value rows** (15.70 → 21.06) and up to **+10 dB on int8 outliers**
(36.4 → 47.8). It is cheap because attention is linear: rotate V at the writer, sum dequantized *rotated* values, and
apply one inverse rotation per output row — no per-tile work. That is the change that should lift acceptance back toward
fp8's 88% and make v4 a strict win (prefill +7%, decode ≥ fp8, and 0.6 GiB of residency to spend).

Status: `kv=fp8` is the default; `kv=fp8v4` is one line away, correct, needle-verified, and faster on prefill with more
resident experts — but it gives away 5% of decode to the drafter until the rotation lands.

## The clean win, and it is context rather than tokens

The arithmetic that matters is *equal memory*. At 192k fp8 the plan is `cuda0 2.9 GiB / ram 12.3 GiB`. Four-bit values take
44% of the KV away, and that is exactly enough to hold **224k in the same footprint**:

| equal memory (`cuda0 2.9 / ram 12.3 GiB`) | fp8 @ 192k | **fp8v4 @ 224k** |
|---|---|---|
| context | 196,608 | **229,376 (+16.7%)** |
| prefill (unique 15.4k) | 1,623–1,657 t/s | 1,656–1,710 t/s |
| decode 400 tok | 72.6–74.0 | 70.5–73.2 |
| needle | ✓ | ✓ (`8823-QZ` at 8k) |

At equal context the trade is prefill +7% against decode −5% (the drafter's acceptance falls 88% → 78% on 4-bit values); at
equal memory the trade is **2–3% of decode for 16.7% more context**. That is the configuration now shipped:

```json
"kv": "fp8v4", "max-context": 229376
```

which is the context the machine was bought for, at the memory fp8 needed for 192k.

### Where the remaining precision lever is

Acceptance is the only thing standing between v4 and a strict win at equal context, and it is a precision problem, not a
kernel one: the H128 rotation measured +5.4 dB on heavy-tailed value rows (15.70 → 21.06) and up to +10 dB on int8
outliers. It is cheap in principle (rotate V at the writer, sum rotated, one inverse rotation per output row) but it needs
a warp-cooperative butterfly at the writer and a shared-memory one in the combine kernel — the next piece of work if 224k
should also be as fast as 192k.

## Two more experiments, one win and one negative

**256k now fits** (the win). With four-bit values the same memory that held fp8's 192k holds **262,144 tokens**
(`cuda0 2.4 / ram 12.8 GiB`), and it performs: prefill 1,576 t/s, decode 68.8 / 75.0 / 75.3, Gate 0 clean, and a needle at
8k retrieved exactly (`5521-TW`). The default is now `kv=fp8v4`, `max-context 262144` — **+33% context over what fp8 could
hold on this card**, which is the context this machine was bought for.

**`--spec` sweep at 256k: no win.** Draft acceptance is already 88–95% there, so a wider window does not add tokens:

| spec | medians across rounds |
|---|---|
| 4 | 72.5 |
| 8 | 71.6, **68.8, 68.4** |
| 12 | 74.8, 69.6, 65.9 |
| 16 | 70.5 |

The 74.8 at spec 12 did not survive a repeat (65.9 next round): the spread between processes (60–75 tok/s) is larger than
any spec effect. Kept at 8. Recorded so it is not re-run.

## A third experiment, negative: a clipped (robust) group scale

The reasoning was sound — with a plain max-based scale one outlier inflates the whole group's step, so capping the scale at
`2.5 x the group's rms` should sharpen the thirty-one typical values. Implemented across all three `v4_scales` copies and
both test encoders (parity kept clean: the writer still lands inside half a step, decode attention 3.4e-05).

**It made things worse**, and for an instructive reason:

| 256k, v4 | max scale (shipped) | clipped scale |
|---|---|---|
| drafts accepted | **89–92%** | 74–88% |
| decode | **71.4 / 72.0 / 72.4** | 62.0 / 69.8 / 67.5 / 72.3 |
| needle | ✓ | ✓ |

Clipping improves *typical* values but saturates the largest ones — and it is the largest value rows that dominatethe
draft verification. The drafter agrees less often, so throughput falls by more than the precision of the bulk buys back.
Reverted to the max-based scale (commit `5ab90b0e` state), which is what ships.

**Lesson for the next attempt**: on this pipeline, value-cache precision should be pursued only in ways that do not clip —
the H128 rotation distributes outliers instead of clipping them, which is exactly why it is the right lever and this is
not.

## A fourth experiment, negative: the reserve knobs are inert

At 192k lowering `vram-reserve`/`ram-reserve` changed nothing, and the same holds at 256k — the plan is byte-identical
(`cuda0 2.4 / ram 12.8 GiB`) at 768/512, 512/256 and 384/256, and the decode/hit differences between them (69.3 / 69.9 /
67.0 medians, hit 64.4 / 63.7 / 57.8) are the usual between-process noise. The planner is bound by its own elastic-KV and
fixed sizes, not by these floors. Defaults restored (768/512), which is what ships.

**Cheap levers are now exhausted**: spec sweep, reserve floors, the fast unpack (neutral by design) and the clipped scale
(worse). What remains needs code: the H128 rotation (the only non-clipping precision lever), #579's attn_prompt
double-buffering and attn.cu register work, and our own prompt.cpp orchestration (~3.5% prefill).

## A fifth experiment, positive on quality: E2M1 codes

Replacing the linear four-bit code with **E2M1** (the NVFP4 set: 0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives) means the
largest code is 6x the group scale, so a heavy row keeps its precision where a linear code spends the whole step on one
value. Measured offline on the same rows:

| code set | gaussian | heavy tail (1%) | outliers (0.1% x50) |
|---|---|---|---|
| linear int4 | 20.26 dB | 15.74 dB | 20.36 dB |
| **E2M1** | 19.91 dB (−0.35) | **18.90 dB (+3.17)** | **21.12 dB (+0.76)** |

In the engine at 256k: **acceptance 91–94%** against 89–92%, decode median **69.1** against 71.8 — the acceptance gain is
real but smaller than the between-process spread (60–75 tok/s), so it does not show as throughput. Kept: better error
distribution at equal speed, and a non-clipping code set is the right basis for the rotation.

Implementation cost was one codec in three writer copies and two decoders, plus the host encoders in both parity suites;
the writer test now checks *which code* the writer chooses rather than a fraction of a step, which is a stricter check
than before. (It also turned up that a revert had taken `prep.cuh`'s fp16 scale rounding with it — the two writers had
silently disagreed since then; both round now.)

## A sixth experiment, negative: prompt chunk size at 256k

Smaller chunks cost prefill badly (measured earlier: 2048 and 3072 were −16% and −27%), so the question was whether the
*other* direction helps now that a 256k KV leaves more room. It does not:

| `kPromptChunk` | prefill (two runs) |
|---|---|
| **4096 (shipped)** | **1,538 / 1,538** |
| 6144 | 1,489 / 1,517 |
| 8192 | 1,529 / 1,545 |

4096 restored. Prefill at 256k sits at ~1,540 t/s; it was 1,700+ at 192k, the difference being the KV staging a 256k
context implies, not the chunk size.

## Knobs are done

Free VRAM with a 256k plan loaded is **1,026 MiB** (nvidia-smi: 1,269 MiB of 12,227), which is the `vram-reserve` floor and
the planner's own spare rather than slack — and lowering either changes the plan by not one byte (tested at 192k and
256k). `--expert-cache auto` already sizes the card tier to that floor. So residency, reserves, spec, chunk size, code set
and scale policy are all exhausted as *configuration*; everything left is code.

## A seventh experiment, negative: masking the cell wrap

#579's prompt commit also removes a 64-bit modulo from the key wrap, and every full cache we build has a power-of-two cap
(256k = 2^18), so `key % cap` becomes `key & (cap - 1)` with a fallback for the rings. Applied at all eleven sites (four in
the prompt kernel, five in the decode kernels, one in each writer):

| 256k | modulo | masked |
|---|---|---|
| prefill | 1,538 / 1,538 | 1,484 / 1,543 |
| decode | 69.1 median | 67.6 / 69.5 / 68.4 |

Neutral — the compiler had already strength-reduced it well enough that the mask is lost in the noise. Reverted rather than
carry an extra branch for nothing. (That also means #579's prefill claim is not sitting in this particular change.)

## Quality at range: the four-bit value cache holds retrieval to 147k

The gate that matters for shipping a 4-bit cache is whether a fact buried at depth still comes back exactly. Three
needles, each a random six-character code placed at the top of a long filler document and asked for at the end:

| needle depth | answer | prefill | decode at depth | hit |
|---|---|---|---|---|
| 8,097 tokens | **`6718-VB` ✓** | — | — | — |
| 53,465 tokens | **`6718-VB` ✓** | 1,365 t/s (39 s) | 44.7 tok/s | 56.5% |
| **147,227 tokens** | **`3392-JD` ✓** | 1,114 t/s (132 s) | 28.0 tok/s | 41.7% |

147k is beyond the 130,709 at which the fp8 cache was originally validated, so the four-bit values are not degrading
long-range recall on this model in practice — which is the claim that mattered, and it is now measured rather than argued.

**The shipped configuration**: `kv=fp8v4` (E2M1 codes), `max-context 262144`, `spec 8`, reserves 768/512 —
256k of context in 2.4 GiB of card and 12.8 GiB of host, where fp8 needed 2.9/12.3 just for 192k.

## An eighth experiment, and the most emphatic negative: `--expert-cache N`

`auto` is not a default in front of the same mechanism — it is a different one. Forcing a number is catastrophic:

| expert-cache | hit | decode medians |
|---|---|---|
| **auto (shipped)** | **56–65%** | **65.3 / 62.1 / 71.3** |
| 560 | 33.1–33.9% | 37.3 / 38.2 / 35.0 |
| 640 | 33.1–33.9% | 36.3 / 35.7 / 35.3 |

Half the hit rate and 45% of the throughput, with the *same* plan printed (`cuda0 2.4 / ram 12.8 GiB`) — so an explicit N
does not size the same tier that `auto` does; it replaces the elastic-copy behaviour with a flat reservation. Never set it.
Restored to `auto`.
