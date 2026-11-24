# Port spec: #579's all-tile double-buffering in `attn_prompt` (prefill)

Claimed +7.3–8.3% prefill, +1.2–4.1% decode. Commit `cfd41894` of PR #579 ("double-buffer all tiles and remove 64-bit
modulo in attn_prompt"), 103 lines changed in `src/models/inkling2/kernels/attn_prompt.cu`. It was skipped during the rebase
because its `load_tile` is FP8-only against main's `template <bool FP8, bool INT8>`, and ours now has a third arm (`V4`).

## It fits this card — checked, not assumed

From our own constants (`kKeys 32`, `kLd 136`, `kBand 96`, `kBandLd 24`, `kWarps 12`, `kGLd 55`):

| | bytes | |
|---|---|---|
| `KvTile` (K + V) | 2 x 32 x 136 x 2 = 17,408 | 17.0 KiB |
| `band` | 96 x 24 x 2 = 4,608 | 4.5 KiB |
| `gx` (bias exchange) | 12 x 16 x 55 x 4 = 42,240 | 41.25 KiB |
| **today**: `t0 + band + max(t1, gx)` (union) | | **62.75 KiB** |
| **after**: `t0 + t1 + band + gx` (no union) | | **79.75 KiB** of 99 KiB |

79.75 ≤ 99 on sm_120, and it matches their measured "~79.8 KiB ≤ 99 KiB sm_120 SMEM" exactly. Our `kThreads` = kWarps x 32 =
384 = `kRel * 24` = 16 x 24 and `kBand` = 96 = 24 x 4, so their band thread-map `static_assert`s hold on our constants.

## What the change is

1. **`Smem`**: drop the union; `KvTile t1` becomes its own field beside `gx`. That is the whole point — the union only
   allowed a second tile when the exchange was unused, i.e. only for *far* tiles (beyond the relative bias's reach).
   With both, **every** tile (window and near tiles included) can overlap its `cp.async` with tensor-core compute.
2. **Loop restructure**, in this order per iteration:
   - request the *next* tile into the alternate buffer, `cp.async.commit_group`;
   - load the bias band for the current tile **while that copy is in flight** (hoisted thread map:
     `bt = tid / 24`, `bb = (tid % 24) * 4`, four consecutive distances a thread);
   - `cp.async.wait_group 1` when a next tile exists, else `wait_group 0`;
   - `__syncthreads()`, then compute;
   - `__syncthreads()`, then `cur ^= 1`.
   The second sync is new and required by the ping-pong (the current tile is still being read while the next is written).
3. **`continue` → `if (active)`**: a warp that skips a tile must still reach the trailing sync, or the block deadlocks.
   This is the subtle part of the port.
4. **Modulo removal** (`wrap_key` with `cap_pow2`/`chunk_pow2`, `rem >> 4` / `rem & 15` instead of `/` and `%`): ported and
   measured **neutral** on this box, so **skip it** — it adds a branch per piece for nothing.
5. Prologue: the first tile is requested before the loop (`k_first = lo / kKeys * kKeys`, loaded once).

## Our port must reconcile the third arm

`load_tile` here is `template <bool FP8, bool INT8, bool V4>` with three bodies. The double-buffering is orthogonal to the
format, so the change is:
- keep all three arms, add the `cap_pow2`/`chunk_pow2` parameters only if item 4 is taken (it is not, so leave the signatures
  alone — fewer diffs, fewer chances to break the V4 arm);
- in the `V4` arm the value side already reads 16-byte pieces of 32 values with one scale each, and the key side is E4M3;
  both are pure loads, so they double-buffer exactly as the `FP8` arm does;
- the `!FP8` guard around the prologue and the `cur ^= 1` flip must become `!FP8 && !INT8 && !V4`-aware: the fp16 arm is the
  only one that uses `cp.async` (the fp8/int8/v4 arms convert and store synchronously), so **the ping-pong only benefits the
  fp16 cache**, which we do not use. **That is the catch**: with `kv=fp8v4` the tiles are converted, not cp.async'd, so
  there is nothing to overlap.

## Consequence

**For our shipped configuration (`kv=fp8v4`) this commit would buy little.** Its win comes from overlapping `cp.async`
copies with compute, and our four-bit/FP8 arms do the conversion inline. To get the same effect for v4, the V-arm would
have to *stage raw code bytes* into shared memory with `cp.async` and convert on read — a different change, and the honest
one to write if prefill is the target.

**Recommendation**: before porting, measure whether our prompt attention is even copy-bound. `prompt_parity` already prints
it: `attention_prompt full at 32K e4m3+v4, 2048 rows: 20208 us (42.1 TFLOP/s of QK and PV)` — 42 TFLOP/s is close to what
this card sustains for the tensor-core QK/PV, so the kernel looks **compute-bound, not copy-bound**, exactly as our earlier
prefill finding (weight-traffic bound, ~200 TFLOPS effective across the layer) suggested.

So: `20208 us` for 2048 rows at 32K. If a port moved that by 8%, prefill would move by the attention kernel's share of the
prompt step — which the bench lets us compute without writing the port.
