# W1 / W2 assessment — both need kernel-interface changes, and here is the exact unlock for each

Read this before starting `NEXT-STEPS-SPEC.md`'s two workstreams. It records what I found by reading the call sites, **the
specific mechanism that makes each one hard**, and — for W2 — the precedent in the same header that makes it tractable. No
code was changed in this pass, so the tree is exactly as Session 9 left it.

Both items are the spec's own 1–2 days each, and both live in the hottest paths (prefill expert products; the decode
drafter). Neither is flag work, and a mistake in either produces **silent garbage output**, not an error.

---

## W1 — wave-pipelined staging: the blocker is the expert indexing, and it is positional

### What the staging and products actually do today

`prompt.cpp` `experts()` (~L406-423), per visit of every chunk:

```cpp
if (w.slot_map) {                                  // spilled layer
    remap_ids(b_.ids, pairs, w.slot_map, stream_);
    const uint64_t kept = (n_expert - w.n_spill) * w.expert_stride;
    if (!m_.stage_from_card(l, stream_))           // per-expert D2D where a card copy exists, else a whole-spill H2D
        cudaMemcpyAsync(w.experts + kept, w.spill_host, n_spill * stride, H2D, stream_);
    cudaMemsetAsync(w.experts + n_expert * stride, 0, kSlotGuard, stream_);
}
moe_sort(b_.ids, rows, k, n_expert, b_.sort_scr, b_.bounds, b_.tok, b_.dst, b_.exp, stream_);
```

and the NVFP4 product (`experts_fp4`, ~L473):

```cpp
fp4::plan_blocks(b_.bounds, n, b_.tile_off, b_.blk_off, stream_);
fp4::Product gu;
gu.base = w.experts;            // expert e's blob = base + e * stride
gu.stride = w.expert_stride;
gu.n = n;                       // walks ALL n experts
gu.bounds = b_.bounds;          // indexed positionally from expert 0
gu.ids = b_.tok; ...
```

**The blocker:** experts are addressed *positionally* — `base + e * stride` with `bounds[e]` — so a wave over slots
`[a, b)` cannot be expressed by passing a smaller `n`. It needs either

- **(a)** `base += a * stride` **and** a re-indexed bounds/id/dst slice, or
- **(b)** the spec's fallback: filter the pairs to the wave's expert range and run `moe_sort` + products per wave.

Either way you are editing the function that does 1.39 s of every 2.6 s chunk, and (a) also touches the FP4 kernel's notion
of expert index (its per-blob `tail_off = L.bytes - 16` read is fine, but `dst_ids`/`tok` remapping is not obviously
offset-safe). Get it wrong and the chunk emits plausible-looking garbage.

### What makes it winnable

- The staging set is already whole-layer sized, so waves need **no VRAM** — that part of the design holds.
- `moe_sort`'s output (`bounds`, `tok`, `dst`, `exp`) is a compact per-expert layout, so (b) is mechanically simple: it only
  needs a pair filter and a second sort per wave. More launches, same math.
- The failure is loud in a **greedy-identity A/B** (`--prefill-waves 0` vs `N` on one unique prompt) — that test is the
  guard rail, and it must be run before any timing is trusted.

### Suggested order if you take it

1. Implement **(b)** first (per-wave `moe_sort` + product) behind `--prefill-waves N`, default 0. It is less elegant and
   costs extra launches, but it is correct-by-construction and proves whether pipelining pays at all.
2. Measure. If the launch overhead eats the win, move to (a) with the `base += a * stride` variant and re-verify with
   greedy identity, because (a) is where the subtle bugs live.
3. Only then start wave 0 right after `router_select` (overlapping the visit's own attention/dense), and consider the
   chunk-boundary prefetch.

Gate unchanged from the spec: unique-16k `s1 read pp=` ≥ 2,400, TTFT unchanged, greedy identity exact, needle, decode canary.

---

## W2 — drafter capture: the blocker is a by-value position, and the fix already exists one function over

### Why `draft_begin`'s attention is uncaptured today

`dspark.cpp` `draft_begin` (~L571-601) runs, per drafter layer, a **direct** launch:

```cpp
const int64_t window = c.ring ? (int64_t) d.window + 1 : pos + kDsBlock;
const int64_t cells  = pos + kDsBlock - std::max<int64_t>(0, pos - window + 1) + 64;
bk::gqa_window_attn(qr_, c.k, c.v, kHeads, kKvHeads, kHeadDim,
                    pos, kDsBlock, pos + kDsBlock, window, c.cap,
                    att_scr_, std::min(att_cells_, cells), attn_, stream_, true);
```

`pos`, `hi` (`pos + kDsBlock`) and `window` are **by-value kernel arguments**, so a captured graph would bake the position
and replay the wrong window — not "a bit slow", but wrong drafts. That is the whole reason for the direct launch, and why
the sliding-window layers (`window = d.window + 1`, constant `cells`) are capturable while the four full-attention layers
are not.

**The unlock is in the same header.** `gqa_decode_attn` (`include/basalt/kernels/gqa_attn.hpp` ~L46-52) already solves
exactly this:

> `pos_dev` (optional): pos0 read from the device instead, the grid sized for `max_cells` (**a captured window**).

So W2 is: **mirror `pos_dev` (and a `max_cells`-sized grid) into `gqa_window_attn`**, key the drafter's graphs on a
position-bound rather than the exact position, write the actual `pos` into a device word per launch, and capture the block
as one graph. That is a kernel-signature change of the same shape the codebase already ships, not a new mechanism.

### Notes that will save you a day

- Keep the direct path behind `--dspark-direct` as the spec says — the first doorbell attempt hung the card on a bad capture,
  and the bail-out must be one flag away.
- The capture must not disturb the W-A event: `draft_end` waits on the drafter's own event (`cudaEventSynchronize(ev_)`),
  which stays correct across a graph replay as long as the event is recorded after the graph launch.
- Acceptance is the correctness signal: a capture bug shows up as **wrong drafts** (`acc` and `dl` move, and greedy identity
  breaks), not as slowness. Gate on `acc ≈ 73%` and `dl` unchanged, plus a verbatim-repeat prompt still reaching `dl ≈ 3.9`.

---

## Why neither was attempted in this pass

Both are one-to-two-day kernel-interface changes in paths where a mistake is silent, and this session had neither the budget
to finish one nor the room to validate it afterwards. Starting W1 and stopping half-way would leave prefill in a state where
`--prefill-waves 0` (the default) must be trusted to be the old path — which is only true if the change is complete and the
A/B has been run.

**Recommended next session order: W2 first, not W1.** Despite the spec ranking W1 first, W2 is the smaller blast radius (one
kernel parameter plus a capture key, no change to how experts are indexed), it is fully guarded by the `--dspark-direct`
flag, and its correctness signal (`acc`/`dl`) is immediate. W1's payoff is larger but it edits the chunk's expert products,
where a slicing error is only visible through a greedy-identity diff.

## Current state (unchanged)

```
decode   53.2 tok/s TCP prompt (39–67 by content); pool 15.4 ms/window, GPU chain 9.3, head 0.9, hand-off 0, drafter ~7.8 outside
prefill  1,536–1,561 t/s unique prompts; 2.58–2.62 s/chunk = H2D ~1.33 s + compute ~1.39 s
hit=     ~69%, stable (Session 8's fix)
tree     no changes from this pass; marker files absent; engine healthy at ctx=196608
```
