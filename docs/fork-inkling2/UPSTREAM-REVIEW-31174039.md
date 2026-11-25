# Review: upstream `31174039` against our fork (`f0e377ff`) - analysis only, nothing merged

## 1. What "the new upstream build" actually is

There is **no new GitHub release**: the latest is still **v0.5.7** (`400b411c`, 25 Nov 02:18 UTC). The
`v0.5.8/9/10-basalt` tags on that releases page are *ours*, not upstream's.

What did move is **`origin/main`**, now `31174039` (25 Nov 04:06 UTC) - **79 commits past v0.5.7**, unreleased. That
is the new code, and it is a large body of work: **76 commits since our merge base `3ce7d52e`**, by area:

| area | commits |
|---|---|
| docs / tooling / ci / arm64 | 21 |
| other (community, factory, spec) | 19 |
| **inkling2** | **16** |
| **kernels / core / prefill** | **10** |
| **serve** | **10** |

## 2. What upstream brings that lands on exactly our bottlenecks

| commit | what | why it matters here |
|---|---|---|
| `6450a411` #652 | **16-row batch windows, multi-session batched DSpark**, bitwise multi-tile Q8/GQA MMA | the two-slot drafter problem (our acceptance/dl loss) |
| `56473ab5` #674 | **16-row single-pass verify window**, hybrid DSpark/SuffixDrafter continuation | fewer verify rows = less wasted pool traffic (our Item 2's target) |
| `f9e689df` #673 | shared-KV MMA decode attention extended to **sliding-window layers**, fused batch segments | decode attention |
| `071018a6` #669 | staging wait deferred to the first expert product, **cross-chunk staging overlap** | our Item 6 / P2 - done properly |
| `5cb82de4` #657 | **128x128 TMA Q8 GEMM**, SMEM-coalesced controller epilogue, cp.async `f32_rows_dot` | prefill compute |
| `a035d98c` #651 | **4-stage cp.async pipeline + LPT scheduling** in prompt attention | prompt attention |
| `01762a7a` #579 | the Blackwell sm_120 prefill/decode kernels (+7.3-8.3% prefill, +1.2-4.1% decode, upstream's claim) | we took only 5 of its 8 commits |
| `95024237` #606 | INT8-cache decode attention | we skipped it |
| `#605`, `#697` | room-bound RAM copy; `--vram-reserve-mib` honoured at load | our VRAM findings |
| `fae1575d` #654 | a read of up to 64 tokens runs as decode windows | our `kPromptPathMin` |
| `b843142d` #642, `130a6d76` | park chats on a shared prefix; disk tier room check | our R1/R2 area |

## 3. What of ours upstream already has (redundant, no need to carry)

- **#650 / #649** - the same hunks appear on both sides with identical sizes (e.g. `core/elastic_array.cpp` +42/-10
  on each, `progress.hpp` +13/0 on each): upstream merged the PR we had ported by hand.
- **#609 routed staging, the staged-ahead copy stream, per-slot D2D staging** - upstream's `b78aadaa`, `72961b05`,
  `071018a6`.
- **#636** - upstream's `9acd845f`.
- **#579** - upstream has all 8 commits (`01762a7a`); we took 5.
- **The reply-role token and the `n-1` turn checkpoint** - present on both sides.
- Constants that match: `kGqaMaxWindow 32`, `kElasticKeep 512`, `kTasksPerThread 12`, `kBatch 32`.

## 4. What is uniquely ours and must be re-ported

1. **The four-bit value cache (`kv=fp8v4`)** - seven markers in our tree, **zero upstream**. It is the reason 262,144
   tokens fit; without it our pack's 256k config is not possible. Touches `kernels.hpp`, `kernels/attn.cu` (ours +230
   lines), `kernels/attn_prompt.cu` (+83), `session.cpp`, `prompt.cpp`, the config schema and the two parity suites.
2. **`cell_bytes_at()` - the K/V cell-size fix.** Upstream charges `cell_bytes()` to *every* elastic array
   (`a->ensure(cell_bytes() * target)`, `a->growth(cell_bytes() * positions)`). Harmless for their fp8-only KV where a
   V row equals a K row; **wrong under our v4**, where a V row is 80 B against 128 - a slot would stop at 62.5% of its
   context exactly as ours did before the fix. So the v4 port *must* carry this.
3. **`kCardSpare` 768 MiB** (upstream 1024): our measured +6% decode / +2% prefill, taken before #697/#605 rewrote
   the plan - it must be re-measured on their plan, not assumed.
4. **`--prompt-chunk`** (upstream: no such flag, a fixed 4096): our runtime knob and its 6144/5120 default. Their
   `071018a6` cross-chunk staging overlap changes that trade, so the value needs re-measuring.
5. **Item 2 - the round's cost charged per slot by rows** (`ms_share` in `batch_round.cpp`). Upstream barely touched
   that file (12/6), so the idea still applies, but #652/#674 replace the window-shape logic it sits next to: re-derive
   and re-measure rather than patch blind.
6. **R1's yardstick** (`restore_image` comparing against what `running_state()` produces): upstream still computes
   `running_bytes()` in that path, so their mount may carry the same bug - check before porting.
7. Our harnesses (`scenA.py`, `curve.py`, `stress_two_slot.py`, `agent_turns.py`, `needle*`), the configs and these docs.

## 5. The conflict surface, and why a naive merge is dangerous

- **64 files changed on both sides.** The heavy ones, upstream vs ours (added/removed lines):
  `session.cpp` +574/122 vs +534/72 · `model.cpp` +677/117 vs +636/100 · `attn.cu` +957/141 vs +230/54 ·
  `attn_prompt.cu` +480/222 vs +83/11 · `prompt.cpp` +189/28 vs +198/20 · `kernels.hpp`, `batch_round.cpp`,
  `session_engine.cpp`, `request_*.cpp`, `spill.*`, `dspark*`.
- Net of the two: **201 files, +4,582 / -12,588** - we are *missing* ~12.6k lines of their work and carry ~4.6k of
  ours, a third of which duplicates theirs.
- The overlap is exactly the machinery where a wrong resolution silently feeds wrong weights - the failure mode this
  session hit twice (the router buffer, and the stale graph key). A merge must be gated, not eyeballed.

## 6. Recommended plan (for when you want it done)

**Take upstream as the base, then re-port our four unique items** (their new work is the superset, and it targets the
two bottlenecks we measured: pool-bound rounds and 38%-exposed staging):

1. Branch from `origin/main`; build and gate it *as upstream ships it* first, and measure (that number is the bar).
2. Port, one commit each, each with its own gate and measurement:
   a. **v4 KV + `cell_bytes_at`** - the biggest and most invasive (6 source files + parity tests). Gate:
      `kernels_parity`, `prompt_parity`, needles 3/3 at 64k and 120k, 262,144 still loads, and the 83k-token prompt.
   b. **`kCardSpare` 768** vs their 1024 on the merged plan - keep only if it wins outside 3-5%.
   c. **`--prompt-chunk`** and its value, at their new staging overlap.
   d. **Item 2 re-derived** on their batch path.
   e. **R1's yardstick** if their mount still rejects with "another size".
3. Full gate after each: parity suites (theirs and ours), needles, stress 20/20, both decode harnesses, the 30k-120k
   curve. Only then choose the config defaults and cut a release.

The alternative - `git merge origin/main` into our branch, preferring theirs wherever they overlap - reaches the same
place with more conflict work and less control over which of their rewrites lands.

## 7. What to expect

Their #652/#674/#673/#657/#651/#579 work attacks precisely our two measured bottlenecks. The numbers to beat are ours:
**43-46 tok/s per stream** at two slots (prose: 35-47 by depth), **1,190-1,705 t/s prefill per stream**, 38% of a
prefill chunk exposed to staging, and the round budget's 42.5% CPU-pool merge wait. If upstream's batch windows and
batched DSpark are what they claim, that wait is the first thing that should move.

Nothing in this review has been merged. The tree is clean at `f0e377ff`; `origin/main` is fetched and available as
`31174039`.
## Stage 1 EXECUTED: upstream as the base, our card spare, the v4 plumbing staged (branch `merge-upstream`, bf330c03)

Branch `merge-upstream` from `origin/main` (`31174039`), pushed to the fork as `merge-upstream-31174039`. The live
engine stays on `pr650-int` (`7e2f3c54`) until the v4 codec lands - the merged build is not yet strictly better.

### What upstream gets by itself, measured at the best configuration it can run

Upstream only knows `kv == fp8|int8|fp16` (its adapter rejects anything else), and at two slots its fp8 plan does
not fit 2 x 128k: with `vram-reserve-mib 0` it is **83 MiB short** ("even with every expert in RAM the card is
83 MiB short"). Its best 2-slot configuration here is **2 x 96k fp8** (`cuda0=3.3GiB ram=11.9GiB`):

    gate        Gate 0 correct
    prose       40.8 / 39.1 tok/s per stream, hit 63-71%
    prefill     1,190 / 1,186 t/s per stream (two 30k prompts at once)
    curve       30k: prefill 1,814, decode 38.2      64k: prefill 1,660, decode 50.9

### What the merge adds

1. **`kCardSpare` 1024 -> 768 MiB.** The value we measured on our tree (+6% decode at two slots). It is also what
   makes 2 x 128k reachable on upstream at all: at 1024 the plan is 83 MiB short, at 768 it loads with
   `cuda0=2.1GiB ram=13.1GiB` and passes Gate 0 (391, Tokyo, planets, 144/12).
2. **The four-bit value cache's plumbing, staged and dormant** (~105 lines, no codec): `kernels.hpp`'s V4 constants,
   `kv_v_row_bytes`/`kv_pair_bytes` and the two `v4` flags; `model.cpp`'s `kv == "fp8v4"` parse, `fp8_`/`v4_` and the
   per-position pair bytes in `visit_cache_bytes`/`session_bytes`/the split spec; `session.cpp`'s `v_row_bytes()`,
   `v_cell_bytes()`, `cell_bytes_at(i)` and the per-array elastic open/ensure/count/trim (upstream charges **every**
   array the key's row size, which is what capped a v4 slot at 62.5% of its context); `stage_to_ring`'s separate
   value row size (its kernel copied both planes with one piece count, so an 80-byte value cell needed its own);
   the adapter accepting `kv=fp8v4`. **With `kv=fp8` every one of these is inert** (`v_row_bytes()` returns the key
   row, `cell_bytes_at()` returns `cell_bytes()`), which is why stage 1 can ship without the codec.

### Gates on the merged build (2 x 128k, kv=fp8, chunk 5120, reserve 256)

    parities    prompt_parity 0 failures, kernels_parity 0 failures, gqa_window_parity PASS
    needles     3/3 at 64,016 tokens, 3/3 at 119,960 tokens, each with a concurrent decoder
    stress      10 rounds, 0 crashes
    gate 0      correct

### Numbers, and the reason the merge is not live yet

    prefill     1,753 t/s per stream at 30k, 1,606 at 64k; two 30k prompts at once 1,146 per stream
    decode      40.6 per stream at 30k, 51.6 at 64k (curve); prose 38.0 / 38.4; hit 63-79%

Against our tree (fp8v4, 2 x 128k): prefill 1,705 / 1,500 - **merged +2.8% / +7.1%**; decode 46.2 / 44.9 -
**merged -12% at 30k, +15% at 64k**.

The 30k loss is entirely the copies. fp8 value rows are 128 B against fp8v4's 80, so the KV of two 128k slots costs
~1.1 GiB more and the expert copy room shrinks; the hit rate falls from our 83-96% to 63-79% and decode follows.
That is exactly the gap the four-bit codec closes - which is why **stage 2 is the codec, not more tuning**.

### A false pass worth recording

Running `kv=fp8v4` with the staged sizes but upstream's unported kernels answers Gate 0 correctly (391, Tokyo).
It is **wrong**: those kernels store and load value rows at the key's stride, so they write 128 bytes into an
80-byte cell and read it back self-consistently, overrunning into neighbouring allocations. Two right answers is
not evidence. This is the reason the v4 port is gated on parity tests and needles rather than on Gate 0.

### Stage 2, the remaining work (the codec in upstream's rewritten attention)

Upstream rewrote `attn.cu` (+957) and `attn_prompt.cu` (+480): a `template <bool FP8, bool INT8, bool SkipDeep>`
tile loader with **one** `kRowBytes` for both planes, `template <bool FP8, int BM>` kernels, an MMA
`mma_store_value_piece` path, and dispatch instantiations `attn_split_kernel<...>` x4 / `attn_fused_kernel<...,2/3>`
x6. The port is:

  a. `prep.cuh`'s value store: a `store_v4` that packs E2M1 codes and the per-row FP16 scale at 80 bytes a row.
  b. Both kernels' value load: a `V4` template parameter threaded beside `FP8`/`INT8` (doubling the instantiations),
      with the v4 unpack in place of the E4M3 cast and `vbase` on the value row size.
  c. `attn_prompt.cu`'s value load, same treatment.
  d. `kernels_parity` / `prompt_parity` v4 cases, and then the full gate: Gate 0, parities, needles 3/3 at 64k and
     120k, 262,144 loads, an 83k prompt, stress 20.

The codec itself is self-contained device code already written for our tree (`v4_e2m1_code`/`mag`/`value`,
`v4_row_scales`, `unpack32_v4`, `unpack16_v4`, `v4_scales`, `store_v4`) and can be re-used verbatim.

## Stage 2 porting map: the four-bit codec into upstream's rewritten attention

The reference implementation already exists in **our tree** (`pr650-int`, `7e2f3c54`) - the codec is a `V4` template
parameter there, and upstream's rewrite is a different shape, so this is a re-application rather than a cherry-pick.

### The reference, by symbol and line (ours @ 7e2f3c54)

    src/models/inkling2/kernels/prep.cuh      (the store side; attn.cu keeps its own copy because it does not include prep.cuh)
      L184  v4_e2m1_mag(i)                     the E2M1 magnitude table 0, .5, 1, 1.5, 2, 3, 4, 6
      L187  v4_e2m1_code(x)                    nearest code, bit 3 is the sign
      L192  v4_e2m1_value(code)                code -> float
      L196  v4_scales(v, f, lane, scale)       one FP16 scale per 32 values, from a lane's 4 values
      L208  store_v4(row, lane, v, scale)      packs 8 lanes' 4 values into 16 bytes + the group's scale at kKvV4CodeBytes
      L224  unpack16_v4(u, scale, f[16])       the decode, 16 values from one 16-byte load
      L237  v4_row_scales(row, s[8])           the 8 group scales of a row

    src/models/inkling2/kernels/attn.cu       (the load side)
      L47-58                                   the codec again (v4_e2m1_code/mag/value)
      L139-195                                 v4_row_scales, unpack32_v4 (32 values of one 16-byte load as fp16 pairs),
                                               unpack16_v4, v4_scales, store_v4
      L233  template <bool FP8, bool INT8, bool V4>   the split kernel
      L249  constexpr int kVRowBytes = V4 ? kKvV4RowBytes : kRowBytes;
      L250  constexpr int kNScales   = V4 ? kKvV4Groups   : 2;
      L258, 266-273                            the tile loader: `if constexpr (V4)` 64 code bytes = four 16-byte loads,
                                               each one 32-value group, plus v4_row_scales into vs[8]
      L343-347                                 the dot: `unpack32_v4(vr[i], vs[i], h)` instead of the E4M3 cast
      L728  template <bool FP8, bool INT8, bool V4, int ROWS>   the rows kernel
      L738-739, 754-769                        the same two constexprs and the same loader branch
      L811-816                                 the store: v4_scales(vv, 1.f, lane, s4) then store_v4(row, lane, vv, s4)

    src/models/inkling2/kernels/attn_prompt.cu  the prompt path: the same three pieces (kVRowBytes, the loader branch,
                                                the decode) inside the prompt tile loop.

### The edits, in order, on branch `merge-upstream`

1. **`prep.cuh`** - copy our L184-240 block verbatim after upstream's INT8 store helpers, and thread a `V4` template
   parameter into upstream's value store the way we did (the store writes `kKvV4RowBytes` = 80 B per value row).
2. **`attn.cu`** - our L47-58 + L139-195 blocks verbatim (they are self-contained device code). Then, at upstream's
   L247 (`kRowBytes`/`kLoads`, split kernel) and L492 (`kRowBytes`, its second kernel), add `kVRowBytes`/`kNScales`
   alongside, add `V4` to the template parameter list, branch the tile loader and the value math on `if constexpr (V4)`,
   and put `vbase`/`vcell` on `kVRowBytes`. Upstream's new `mma_store_value_piece` path must be left to the non-v4
   formats: our own rule already is `use_mma = window == 0 && !int8 && !v4`.
3. **Dispatch** - upstream instantiates `attn_split_kernel<false,false>/<true,false>/<false,true>/<false,true,true>` and
   `attn_fused_kernel<...,2>/<...,3>`; each needs a `V4=false` argument and a `V4=true` sibling (4 -> 8, 6 -> 12), and
   the launcher picks it from `a.v4`.
4. **`attn_prompt.cu`** - same three pieces in its prompt loop.
5. **Tests** - `kernels_parity` and `prompt_parity` gain v4 cases beside their fp8/int8 ones (the existing harnesses
   already iterate over the formats; ours adds `"fp8v4"` to that list).

### The gate, unchanged from the v4 work

`kernels_parity` 0 failures, `prompt_parity` 0 failures, `gqa_window_parity` PASS, Gate 0, needles 3/3 at 64k and
3/3 at 120k with a concurrent decoder, a 262,144-token load, an 83k prompt, then stress 20 rounds - and only then the
numbers against the bar in the stage-1 section (merged fp8: prefill 1,753 / 1,606, decode 40.6 / 51.6, hit 63-79%;
our tree: 1,705 / 1,500, 46.2 / 44.9, hit 83-96%).

### The one thing NOT to do

Do not run `kv=fp8v4` with the staged sizes and an unported attention to "see if it works". It answers Gate 0
correctly while writing 128-byte value rows into 80-byte cells (see the false pass above). Verify with needles.

## Stage 2 EXECUTED (branch `merge-upstream`, 032ab699): the codec is in place, one fault keeps it off the live engine

The four-bit value cache is now ported into upstream's rewritten kernels - re-applied, not cherry-picked, because
upstream rewrote `attn.cu` (+957) and `attn_prompt.cu` (+480). Five writers and three readers now agree on an
80-byte value row:

    attn.cu         the codec (v4_e2m1_code/mag/value, unpack32_v4, unpack16_v4, v4_row_scales, v4_scales, store_v4);
                    a V4 parameter on attn_split_kernel<FP8,INT8,SkipDeep,V4> and attn_fused_kernel<FP8,INT8,ROWS,V4>;
                    kVRowBytes/kNScales; the tile loader's four-16-byte-load branch; the value unpack; the store;
                    use_mma = !int8 && !v4 (the MMA kernels read 16-byte K/V rows); 3 + 6 v4 instantiations
    attn_prompt.cu  the codec; RawKvTile.kRowV; a loader with a piece count per plane; the four-bit unpack; the
                    kernel template; and the dispatch (which is where the second bug was)
    prep.cuh        the codec the store paths share
    glue.cu         the prepare kernel's value store; and the KV read-ahead's per-plane cell (the third bug)
    ops.cu          its own copy of the codec and its value store
    kernels.hpp     MixNorm.kv_cell_v

### The three defects it caught (all silent-corruption class, all found by reading or by a fault, not by Gate 0)

1. **`attn_prompt.cu`'s slow path**: I wrote `base + slot * kKvHeads + rb` where the address is
   `base + (slot * kKvHeads + kvh) * rb + col`. Symptom: "misaligned address" on the first long prompt.
2. **`attention_prompt`'s dispatch was never templated**: the kernel took a V4 parameter but the dispatch still launched
   `attn_prompt_kernel<true,false>` for `fp8v4`, so the prompt read value rows at the key's stride. Symptom: illegal
   memory access during the first chunk.
3. **The KV read-ahead** (`mix_norm_kernel`'s prefetcher, upstream's new L2 warm-up) strode *both* planes by `kv_cell`,
   so with 80-byte value rows it prefetched past the end of the value array. It now takes `kv_cell_v`.
   **Caveat that matters for the next session:** our own tree has the same over-read and passes needles, so a bulk
   prefetch past a mapped region does not appear to fault on this driver. This fix is correctness hygiene; it is NOT
   the proven cause of the remaining fault below, and should not be assumed to be.

### What still fails, precisely

With `kv=fp8v4` on this build, long prompts die with "staging a layer's RAM experts failed: an illegal memory access"
- a sticky error surfacing at the next copy, not the faulting kernel:

    two slots, 2 x 128k:  dies at the 32768 -> 65536 elastic grow ("cache grow to 65536 positions at 32768")
    one slot:             dies earlier, during the first chunk's prefill (positions 27 and 229 in the log)
    short prompts (28-79 tokens): fine.  Gate 0: answers correctly throughout - which is why it is not evidence.
    control on the same build, kv=fp8: needles 3/3 at 64,016.  So the fault is in the v4 path only.

Repro: `bin\basalt.exe serve --config basalt_config.json ...` with `kv: fp8v4`, then
`python C:\LLM\Basalt\needles3.py 40000 200` (one slot fails even earlier).

### Next steps, in order

1. **Name the kernel.** `compute-sanitizer --tool memcheck` could not attach here ("No attachable process found",
   the engine re-execs and WDDM hides the child). Run the engine in a *console* with `--target-processes all` and
   `--launch-timeout 1800`, or add a bounds assertion behind `BASALT_DIAG_KV` in the v4 readers (compare
   `cell * kKvHeads * kVRowBytes` against the array's `backed()`).
2. **Bisect the remaining suspects cheaply**, each one a rebuild: (a) set `mn.kv_pos = nullptr` to remove the
   read-ahead entirely; (b) `kv=fp8v4` with `parallel 1` and a 5k prompt (does the *first* chunk alone fail?);
   (c) check the batch path (`save_cells`/`restore_cells`) by parking a long conversation between two requests.
3. Then the full gate: Gate 0, parities (with v4 cases added), needles 3/3 at 64k and 120k, 262,144-token load,
   an 83k prompt, stress 20 - and only then the numbers (bar: merged fp8 1,753 / 1,606 prefill, 40.6 / 51.6 decode;
   our tree 1,705 / 1,500, 46.2 / 44.9).

### Where the live engine is

`pr650-int` (`3dd59a13`), rebuilt and verified after this work: plan cuda0 2.0 / ram 13.2 GiB, Gate 0 correct,
needles 3/3 at 64,002, prose 41.5 / 47.0 tok/s per stream at hit 83-96%, prefill 1,231 t/s per stream.
