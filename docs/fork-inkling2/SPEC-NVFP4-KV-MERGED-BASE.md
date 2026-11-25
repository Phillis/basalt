# SPEC: NVFP4 K/V for Inkling 2 Mini on the merged base (`--kv nvfp4`)

Written 2026-11-25 (about 11:00 local) by the review session, for the deepseek worker. This replaces
`SPEC-INKLING2-NVFP4-KV.md`, which was written against the old fork tree. Base: branch `merge-upstream`,
commit `bf330c03` (upstream `31174039` as the base, our `kCardSpare 768` and the KV plumbing staged);
`032ab699` (the `fp8v4` port, work in progress) is the map, not the base. Line numbers are that branch's.
Scope rule unchanged: a quality trade only behind a runtime switch, gated so it can be refused.

--------------------------------------------------------------------------------
## 0. Read this first

Phil's decision: stop debugging the `fp8v4` port onto upstream's kernels; build NVFP4 for both planes
instead. Two reasons make that the right order, not just a preference:

1. **It removes the class of fault the port is fighting.** With `fp8v4` the key row is 128 bytes and
   the value row 80, so every K/V address in the engine has two strides: the elastic arrays open and
   grow with unequal cells, the prompt loader keeps a piece count per plane, the read-ahead a cell per
   plane, `MixNorm` a `kv_cell_v`. The open fault is exactly that shape (sticky illegal access on long
   prompts; at the grow with two slots, in the first chunk with one; `fp8` clean on the same build).
   With NVFP4 on both planes every row is 80 bytes: K and V cells are equal, `cell_bytes_at` is one
   number, the piece counts agree, and a reader that mixes up the planes is harmless instead of fatal.
2. **On the merged base this format is worth more than on our old tree.** Upstream's kernels have no
   four-bit cache, so the merged build runs `fp8`: 3.9 GiB of KV at 2 x 128k, hit 63-79%, decode
   40.6 / 51.6. NVFP4 is 2.43 GiB: 1.47 GiB back, about 350 more expert copies, which is what puts
   the hit rate back where our tree's is (83-96%). The merged base only beats our live engine at two
   slots if its KV is small.

What it is not: a speed switch. The card does not gain tensor-core attention from it; the rows are
expanded to fp16 at load. Decode gains come through the copies.

Quality stance: keys at four bits are the risk (values at four bits we have run since the fork's V4).
Upstream's own measurement on Qwen3.8-27B (`--kv nvfp4`, 100 chunks): KLD 0.0283 against int8's 0.0259
and fp16's 0.0262; top-1 92.69 against 92.89 and 92.82. Small, visible, opt-in. The gate in section 7
decides for Inkling with Inkling's numbers. Until the merged build passes it, the live engine stays on
`pr650-int` with `fp8v4`.

--------------------------------------------------------------------------------
## 1. The bar, and where the engines are

| engine | prefill per stream | decode per stream | hit |
|---|---|---|---|
| merged base, `kv fp8` (`bf330c03` + kernels) | 1,753 / 1,606 t/s | 40.6 / 51.6 tok/s | 63-79% |
| our tree, `kv fp8v4` (`pr650-int` @ `04baae23`, live now) | 1,705 / 1,500 by depth; 1,231 across two 30k prompts | 46.2 / 44.9; prose 41.5 / 47.0 | 83-96% |

The merged `nvfp4` build has to beat our tree on both axes at equal or better hit rate, and pass
every gate in section 7, before it replaces the live engine.

--------------------------------------------------------------------------------
## 2. Item 0: the numerics baselines (first; no kernel code)

The reference is public: `inkling2-bf16-base.kld` in the Hugging Face dataset `jesdga/basalt-kld-bases`
(not gated; verify against the dataset's `SHA256SUMS`). The tool: `tools/ppl_gate.py --model <pack>
--architecture inkling2 --base inkling2-bf16-base.kld --out <dir> --chunks 100 --t 3 --kv <format>`,
then `--report <dir>/rows.tsv` and `--diff A/rows.tsv B/rows.tsv`. It scores the decode path on
wikitext-2. Run it where the format exists:

- on `pr650-int` (live tree): `fp16`, `fp8`, `fp8v4`. This is the first measurement of what the
  fork's V4 costs; Phil has never seen it.
- on the `nvfp4` build, when it passes parity: `fp16`, `fp8`, `nvfp4`.

Write every line (PPL, KLD mean / median / p99, top-1) into CHANGES.md. Upstream's Inkling numbers are
the reference points: fp16 0.00305 / 97.76%, int8 0.00343 / 97.47%. If `ppl_gate.py` needs the pack's
tokenizer path or a flag it does not set for inkling2, say what it needed in the report.

--------------------------------------------------------------------------------
## 3. The format

`--kv nvfp4` (the name upstream's dense stack uses). K and V both:

    80-byte row per head: [0..63] E2M1 codes, value 2i in byte i's low nibble
                          [64..71] one E4M3 scale byte per 16 values (group g covers values 16g..16g+15)
                          [72..79] zero

- **K**: the convolved, L2-clipped key (prep.cuh `kv_row`, clip at about L106, unconditional for every
  format) turned by the normalized 128-point Hadamard transform, then per-16 amax, scale = E4M3(amax /
  6), codes = E2M1(v / scale). Use upstream's helpers in `include/basalt/kernels/kv_fp4.cuh`
  (`kv4_group_scale`, `kv4_group_code`, `kv4_e2m1_value14`, `kv4_e4m3_value`, `kv4_decode8`) so K uses
  the dense stack's exact rounding.
- **V**: the normed value (prep.cuh `kv_row`, `rsqrtf(ss / kHeadDim + eps)`), same encoding, no
  rotation (values are averaged, not matched).
- **Q**: turned by the same transform, once, where the query is produced: prep.cuh `q_heads`
  (about L238-268): one warp holds one head-row (32 lanes x 4 values); after the clip factor `f`
  (L266) and before the store (L267). Both paths read `AttnPrep.q`, so this one site covers decode and
  prompt. The transform is orthonormal, so dot products are unchanged before rounding; the norm is
  preserved, so `qk_clip` still bounds them.
- The 128-point transform is warp-level and in registers: two stages inside the lane (h = 1, 2 over
  the four values), five across lanes (`__shfl_xor_sync` masks 1, 2, 4, 8, 16 = h 4, 8, 16, 32, 64),
  then 1/sqrt(128). Add `kv4_hadamard128` beside upstream's `kv4_hadamard256`.
- Inkling has no rotary positions and nothing else reads a key's coordinates: the relative bias and
  the sink are added to scores by position, the gate multiplies the output. The drafter keeps its own
  fp16 caches from its own projections (`dspark.cpp`): do not rotate those.
- **Equal cells**: K cell = V cell = 8 x 80 = 640 bytes; a position across the 8 full-attention
  visits is 10 KiB (fp16 32, int8 28, fp8 16, our fp8v4 13).

--------------------------------------------------------------------------------
## 4. The port, by site, on a new branch from `bf330c03`

Branch `nvfp4-kv` from `bf330c03`. Use `git show 032ab699` as the list of sites (it found them all);
do not cherry-pick it: its codec is V4's (fp16 scale per 32, value plane only). `fp8v4` is retired
on this branch; the adapter takes `fp16`, `fp8`, `int8`, `nvfp4`.

Strategy for every reader: **stored four-bit, expanded to fp16 at load into the layout the fp16 path
already uses.** Only writers and loaders change; the math after a loader is the fp16 path's, which
upstream's kernels already run.

### 4.1 Constants, flags, sizes
- `include/basalt/models/inkling2/kernels.hpp`: `kKvF4RowBytes = 80`;
  `kv_cache_row_bytes(fp8, int8, f4)`, `kv_pair_bytes(fp8, int8, f4)` = 160 under f4 (the V4
  constants and `kv_v_row_bytes` go away on this branch); `AttnPrep::f4`, `AttnArgs::f4`,
  `AttnPromptArgs::f4` beside `.v4`; `MixNorm.kv_cell` only (`kv_cell_v` deleted: the cells are equal).
- `src/models/inkling2/model.cpp` (about L197, L366, L670): parse `nvfp4`; `f4_` true, `fp8_` and
  `int8_` false; `model.hpp` `cache_f4()`; `adapter.cpp` L63: the whitelist.
- `src/models/inkling2/session.cpp`: `row_bytes()` and the value sizing (about L319, L768-779) both
  80; `cell_bytes() == v_cell_bytes()`, so the open (L337), the grow (L835, L862), `count_backed`
  (L798), the trim (L1014) and `cell_bytes_at` (L779) are one number. `mn.kv_cell` (L1237) serves both.
- `prompt.cpp`: the staging carve (L125-129) is sized for the widest format and stays; the per-plane
  row bytes passed to `stage_to_ring` (L621-622) are 80 and 80; `ap.f4` (L598) and `aa.f4` (L617).
- `model.cpp` `visit_cache_bytes` (L86): `kv_pair_bytes(..., f4)`.

### 4.2 Writers (both planes, both the cache row and the chunk row)
- `prep.cuh`: the decode path's store (the `store_kv` site at about L159-226; a four-bit store for
  K and V); q rotation in `q_heads` (section 3).
- `glue.cu` `attn_prepare` (L498-511, the prompt path's writer): `row_bytes` 80 for w = 0 and 1;
  the four-bit store for both planes.
- `ops.cu` (its own copy of the store, about L282 and L338).
- One device header for the codec helpers, included by each of those files (they do not all include
  `prep.cuh`): `k4_scales` (per-16 amax over the four lanes of a group: `__shfl_xor_sync` masks 1 and
  2), `store_k4` (rotate, scales, codes, the scale byte by the group's first lane), `store_v4nv`
  (the same without the rotation), `k4_unpack32(piece, s0, s1, h[16])` (32 fp16 from one 16-byte code
  piece and its two scale bytes: four `kv4_decode8`), `k4_row_scales(row, s[8])`.

### 4.3 Readers
- `attn.cu` `attn_split_kernel<FP8, INT8, SkipDeep, F4>` (L246): `kRowBytes = kVRowBytes = 80`,
  `kLoads` 5 for both planes (four code pieces, one scales piece); the loader (L274-319) decodes both
  planes to the fp16 smem tiles the `!FP8 && !INT8` branch fills.
- `attn.cu` `attn_fused_kernel<FP8, INT8, ROWS, F4>` (L1215): the same constants and loader, and its
  inline store (the four-bit store for both planes, q rotated).
- `attn.cu` L1849: `use_mma = !a.int8 && !a.f4`. The MMA kernels (`attn_mma_kernel` L470,
  `attn_mma_split_kernel` L1069 and their helpers) read raw 16-byte rows; the first version leaves
  them to the other formats. Section 4.5 is the extension.
- `attn_prompt.cu`: `RawKvTile<FP8, F4>` (L44-49) with `kRowB = kRowV = 80`; the raw loader
  (L198-250) with `kVecs = kVecsV = 5` and an F4 decode that writes the fp16 tiles `load_tile_f16`
  (L150) writes; the kernel template (L303) and `attention_prompt`'s dispatch (the site of the port's
  second bug: the dispatch must launch the F4 instantiation for `nvfp4`); the slow path's address form
  (`(slot * kKvHeads + kvh) * rb + col`, the port's first bug).
- `glue.cu` read-ahead (L226-240): one cell for both planes.
- `prompt_ops.cu` `stage_to_ring` (L99-108, L158-166): equal piece counts.

### 4.4 The inventory, and the proof that nothing was missed
Every place a K or V address is formed on this branch is one of the sites above or in this list; grep
each pattern after the port and account for every hit:
`kKvInt8RowBytes|kHeadDim \* 2|kv_cache_row_bytes|kv_v_row_bytes|kv_pair_bytes|kv_cell|kRowBytes|kRowB|kRowV|kLoads|kVecs|cache_row|k_cache|v_cache|k_chunk|v_chunk|k_stage|v_stage`.
Then, behind `BASALT_DIAG_KV=1`, a bounds assertion in every F4 loader (the port's own next step):
the byte offset against the array's `backed()`. With equal cells a wrong plane cannot overrun, but a
stale 128-byte assumption can; the assertion names it in one run.

### 4.5 Later, measured: the MMA arm
Upstream's MMA kernels take int8 rows by copying raw pieces plus the scale word and dequantising in
shared memory before `ldsm` (attn.cu about L755-770: `mma_copy16`, `copy_scale`). NVFP4 rows fit the
same route: four code pieces and the scales piece, `kv4_decode8` into fp16, then the existing MMA. Do
it only if section 7's numbers show the split path costing decode at depth; our live engine already
runs attention on the split path (`fp8v4` excludes MMA), so the first version is no regression against
today.

--------------------------------------------------------------------------------
## 5. Parity

A `kv_f4_parity` beside `kernels/kv_int8_parity.cu`: the decode attention (`attention_window`, sliding
and full, one row and a 32-row batch, a batch of two segments) and the prompt attention with `f4`
against the fp32 reference at the int8 suite's tolerance; the writers against a host reference of the
codec; and the identity that matters here: a query rotated and encoded against a rotated cache gives
the same scores as the unrotated fp32 pair up to the rounding. `kernels_parity` and `prompt_parity`
gain f4 cases beside their fp8/int8 ones (the harnesses iterate the formats); `gqa_window_parity`
unchanged (the drafter).

--------------------------------------------------------------------------------
## 6. The two things not to do
- Do not run `nvfp4` with sized arrays and an unported loader to "see if it works": it answered Gate 0
  correctly in the port while writing rows at the wrong stride. Verify with needles, never Gate 0.
- Do not copy upstream's dense trick of an int8 copy of every position for prompt reads (2 KiB a
  position of max-context: 512 MiB at 256k). Read the four-bit rows.

--------------------------------------------------------------------------------
## 7. Gates, in order, each able to stop the item

1. Parity (section 5) and the three suites under the flag: 0 failures. The inventory (4.4): every hit
   accounted for. The bounds assertion: silent through one 64k read in each slot.
2. The numerics gate (Item 0) for `nvfp4` on this build, beside this build's `fp8` and our tree's
   `fp8v4`: mean KLD at most `fp8` + 0.0010 and at most `fp8v4` + 0.0007; top-1 within 0.15 points
   of `fp8`; p99 KLD within 10%.
3. Needles: three at 10/50/90% read into one slot while the other decodes, at 64k and at 120k, and
   the reverse: exact. Ten long-context questions needing two facts from far apart: all correct, the
   same answers `fp8v4` gives.
4. Gate 0; agent turns plain and tool at 40k and 100k, both slots, header on and off; cancel
   mid-generation x5; stress 20/20 in one process: 0 crashes and restarts. The 83k prompt and the
   262,144-token load.
5. Numbers against the bar (section 1): both decode harnesses at 2k and 64k, the curve 30k to 120k,
   prefill per stream, hit per slot, the plan line. Report them; they do not stop the item unless
   they are below our tree's.

--------------------------------------------------------------------------------
## 8. The stop rule, and what ships

If gate 2 or 3 fails: `nvfp4` does not ship as a profile. Phil decides with the numbers whether the
merged base then ships with `fp8` (hit 63-79%) or whether the `fp8v4` port is finished instead (your
own ordered list in `UPSTREAM-REVIEW-31174039.md`, stage 2: name the kernel with the bounds
assertion, bisect the read-ahead, one slot with a 5k prompt, the park/restore path).
If it passes: `basalt_config-2x128k-nvfp4.json`, `basalt_config-2x160k-nvfp4.json`,
`basalt_config-2x192k-nvfp4.json`; BEST-CONFIG.md gains a row per format with KLD, top-1, VRAM,
decode; and the recommendation to Phil which profile to load. The live engine moves to the merged
build only when he says so.

--------------------------------------------------------------------------------
## 9. Expectations (estimates)

| | |
|---|---|
| VRAM | merged fp8 3.9 -> 2.43 GiB at 2 x 128k: +1.47 GiB, about +350 copies; our fp8v4 3.16 -> 2.43: +0.73 GiB |
| decode per stream | on the merged base, from hit 63-79% toward 90%+: the item that lets the merged kernels match our tree's 46 / 45; beyond that, +3-10% at depth through the extra copies |
| prefill | small: 80-byte rows in the prompt attention; the chunk writer does more work per row |
| attention | on the split path (no MMA) in the first version: measure at 120k before 4.5 |
| quality | upstream's Qwen cost for both halves: +0.0024 KLD over int8. Keys are the risk here; Item 0 and gate 2 give Inkling's number. |
| effort | about a day and a half: the codec and stores (half a day), the loaders and dispatch (half a day), parity (a few hours); the gates half a day more (five numerics runs at 100 chunks) |

--------------------------------------------------------------------------------
## 10. Cheat sheet

- codec and transform: `include/basalt/kernels/kv_fp4.cuh`; upstream's users: `src/kernels/cuda/dense_ops.cu`
  about L435-439 (queries turned), `src/dense/session.cpp` about L575-582, `include/basalt/kernels/gqa_attn.hpp` L25-39
- clip and queries: `kernels/prep.cuh` `kv_row` L82-108 (clip L106), `q_heads` L238-268 (clip L266,
  store L267)
- writers: `prep.cuh` store L159-226; `kernels/glue.cu` `attn_prepare` L498-511; `kernels/ops.cu` L282, L338
- readers: `kernels/attn.cu` split L246-319, fused L1215, `use_mma` L1849, MMA int8 route L755-770;
  `kernels/attn_prompt.cu` `RawKvTile` L44-49, `load_tile_f16` L150, raw loader L198-250, template L303;
  read-ahead `glue.cu` L226-240; `stage_to_ring` `prompt_ops.cu` L99-166
- sizes: `kernels.hpp` L21-36, L103-146, L283-284; `session.cpp` L319-337, L768-779, L798, L835,
  L862, L908, L1014, L1237; `model.cpp` L86, L197, L366, L670; `adapter.cpp` L63; `prompt.cpp` L125-129, L598, L617, L621-622
- quality: `tools/ppl_gate.py`; `docs/precision/quality.md`; the base file in `jesdga/basalt-kld-bases`
- the port's record: `C:\LLM\Basalt\UPSTREAM-REVIEW-31174039.md` stage 2 (L178-298)
- harnesses: `docs/fork-inkling2/tools/agent_turns.py`, `stress_two_slot.py`;
  `C:\LLM\Basalt\needles3.py`, `curve.py`, `scenA.py`
