# SPEC: wire `stage_plan` into the prompt path (the flat 1.8 s of time to first token)

## What is measured

A prompt read costs a **flat ~1.8 s** whatever its length, from 106 to 448 tokens (measured: 106 → 1.8 s, 204 → 1.8 s,
448 → 1.8 s, 8,947 → 6.9 s, 43 → 0.5 s on the decode path). `pp=32/0c 1.8s` says the read of a 32-token prompt runs at
18 tok/s, and turning the prompt cache off changes nothing, so it is not the checkpoint.

It is the staging: before a prompt chunk computes a layer, it copies **every** RAM-resident expert of that layer into the
shared staging buffer.

```
prompt.cpp L558-564 today (pre-#609):
    if (w.slot_map) {
        remap_ids(b().ids, pairs, ...);
        const uint64_t kept = (n_expert - w.n_spill) * w.expert_stride;
        if (!m_.stage_from_card(l, strm(), !peer))       // all of the layer's card copies, if it has them all
            cudaMemcpyAsync(w.experts + kept, w.spill_host,   // else EVERY RAM expert of the layer, in one copy
                            (size_t) m_.ram_n(l) * w.expert_stride, H2D, strm());
        cudaMemsetAsync(... kSlotGuard ...);
    }
```

With ~58 RAM experts a layer at ~5.5 MiB, that is ~320 MB a layer over 48 layers = **~15 GB on the host link, ~1 s**, before
any attention or expert product runs. A 33-token chunk routes to a handful of experts per layer, so almost all of it is
wasted.

## What is already in the tree

`7d90e66d` ("a prompt chunk of at most 1,024 rows stages only the RAM experts it routes to") landed its **plumbing**:

- `src/models/inkling2/spill.hpp`: `StageOp {Kind::Ram|Card|Guard, slot, count}` and
  `std::vector<StageOp> stage_plan(int n_spill, const uint8_t* card, const uint8_t* want, bool kept);`
  — `want[j]` says whether the chunk reads slot `j` (null: every one); a wanted slot is staged from its card copy, or with
  its adjacent copy-less wanted slots in one RAM copy; a slot left out right after one the products may read gets the guard
  memset (the products read past a matrix's end into the next slot, and multiply by zero activations, so those bytes must be
  finite).
- `src/models/inkling2/spill.cpp` implements it and `spill_test.cpp` covers it (card, ram, guard, all-slots).

**Nothing in the engine calls `stage_plan` — only its test does.** That is the whole gap.

## The change

In `prompt.cpp`, for a chunk of `rows` rows and a spilled layer:

1. the router's picks for the chunk are already in `b().ids` / `w.slot_map` (remapped by `remap_ids`) at that point, so a
   `uint8_t want[n_spill]` can be built by marking every slot the chunk's pairs name — `rows * n_used` entries, at most
   `1,024 * 8`;
2. `card[j]` is the layer's per-slot card copy mask (`copy_` / the elastic spill's), already available where
   `stage_from_card` reads it;
3. walk `stage_plan(n_spill, card, want, kept != 0)` and issue each op: `Card` → per-slot D2D from the copy,
   `Ram` → one host-to-device copy of `count` contiguous slots, `Guard` → `cudaMemsetAsync(kSlotGuard)`;
4. keep the "every slot" form when `rows > 1,024` (a long chunk reads most slots anyway, and the plan's guardbook-keeping
   then costs more than it saves) — that is what the commit's title means.

Note the two places that must stay consistent: the guard slot at the *end* of the layer (`cudaMemsetAsync` at
`n_expert * expert_stride`, L563) still applies, and the per-slot guard from `stage_plan` replaces the old assumption that
`cudaMemsetAsync` covered everything.

## Expected value, and the gate

- **up to ~1.5 s off TTFT for every prompt from ~100 tokens to a few thousand** (the staging falls from ~15 GB to
  `rows * n_used` experts a layer, ~1-3 MB);
- prefill *rate* is unchanged for long prompts (it is weight-traffic bound), so this is a latency change, not a throughput
  one — measure TTFT by prompt size with `bench.ps1`-style discipline (fresh prompts, engine timings, median of 3, two
  processes) at 43 / 106 / 448 / 4,600-word prompts;
- the gate before timing: `spill_test` passes, `kernels_parity` passes, Gate 0 coherent, and a needle at 130,709 — because
  a staging mistake silently feeds the expert products the wrong weights.

## Implementation detail worked out from the code (so this can be typed in, not re-derived)

**The slot layout.** `spill_.slot[l][e]` is expert `e`'s slot in layer `l`'s `experts` range; the RAM experts are the last
`n_spill` slots, so `kept = n_expert - n_spill` and RAM slot `j` lives at
`w.experts + (uint64_t) (kept + j) * w.expert_stride` (which is exactly what `stage_from_card` writes and what the bulk
copy at `prompt.cpp:562` writes to `w.experts + kept`).

**The routed mask.** At `prompt.cpp:559` the pairs have just been remapped:
`remap_ids(b().ids, pairs, peer ? w.slot_map : w.slot_full, strm())`, so **`b().ids[0..pairs)` holds slot indices**, not
experts. That makes the mask trivial, but they are on the device: `cudaStreamSynchronize(strm())` then one D2H of
`pairs * 4` bytes (`rows * n_used`; 33 x 8 = 264 entries for a short prompt, and the routing work just issued is small, so
the sync is cheap - about 48 of them a chunk).

**The card mask and addresses.** `stage_plan` wants a byte a slot (`card[j]`), but issuing a `Card` op needs the slot's copy
address, which is `copy_[(size_t) l * n_expert + e]` (a device pointer, non-null when the elastic spill holds a copy). The
model exposes neither, so two accessors are needed on `InklingModel`:

```cpp
const int32_t* slots_of(int l) const { return spill_.slot[(size_t) l].data(); }
const void*   copy_of(int l, int e) const { return copy_.empty() ? nullptr : copy_[(size_t) l * geo().n_expert + e]; }
```

and one host-side pass over the layer's experts builds both `card[j]` and a `copy_addr[j]` array.

**Issuing the ops** (all on `strm()`, in plan order, which is slot order, so the ops are naturally ordered):

| op | what to issue |
|---|---|
| `Card`, slot `j` | `cudaMemcpyAsync(dst(kept+j), copy_addr[j], stride, DeviceToDevice)` |
| `Ram`, slot `j`, `n` | one `cudaMemcpyAsync(dst(kept+j), w.spill_host + j*stride, n*stride, HostToDevice)` |
| `Guard`, slot `j` | `cudaMemsetAsync(dst(kept+j), 0, kSlotGuard)` |

The trailing `cudaMemsetAsync(w.experts + n_expert * stride, 0, kSlotGuard)` at `prompt.cpp:563` stays: that is the guard
after the layer's **last** slot, which `stage_plan` deliberately does not emit.

**Where the old path stays.** `rows > 1,024` (a long chunk routes to most slots, and the per-slot plan then costs more than
it saves), and the peer path (`peer_send` / `slot_full` / `with_peer`) - the mask there spans two cards' slots and
`stage_from_card`'s `with_peer` form is the tested one.

**Order of work**: `spill_test` (unchanged, it already covers the plan) -> the two accessors -> the caller behind a
`rows <= 1024 && !peer` guard -> build -> Gate 0 -> `kernels_parity` -> needle at 130,709 -> then TTFT by prompt size
(43 / 106 / 448 / 4,600 words, fresh prompts, median of 3, two processes). Expect ~1.5 s off TTFT between ~100 and a few
thousand tokens, no change in prefill rate for long prompts.

---

# Next latency item: the long-chunk staging on a second stream (commit `91bdae93`, skipped)

## What it does

The routed-only staging above covers chunks of at most 1,024 rows. A **long** chunk still copies every spilled layer's RAM
experts, and it does so **serially**: layer l's copy, then its kernels, then layer l+1's copy, and so on. That serial
staging is the remaining ~1.8 s of time to first token for prompts above ~1,000 tokens, and it is also ~15% of a 15k prompt's
whole read (6.9 s of prompt time, ~1 s of it staging).

`91bdae93` overlaps it: the next spilled layer's copies run on a dedicated stream **under the kernels of the layer before**.
Its shape, from the commit:

- a `copy` stream per lane plus two events, `freed` (follows the last products that read the staging) and `staged` (follows
  the next layer's copies), created at setup when the model has any spilled layer and destroyed in the teardown;
- `next_spill_[step]`, a precomputed map from each step of the visit walk to the next step whose layer is spilled, so a step
  knows what to prefetch without searching;
- at each spilled step: wait `freed`, issue the copies on `copy`, record `staged`; the step's products wait `staged` (and
  anything that later reuses the staging waits `freed`);
- a `drain()` on a failed read, so every queue it touched runs out before the chunk buffers, the staging or a card copy is
  reused, and `broken_` refuses later reads if the drain itself cannot complete.

## Why it is safe to overlap

Each spilled layer has **its own reserved slot range** inside `experts`, so a copy into layer l+1's range cannot disturb
layer l's products; the only ordering that matters is (i) the previous reader of l+1's range is done (`freed`, or the
same-layer loop visit before it) and (ii) l+1's products wait for its copies (`staged`).

## Expected, and the gate

For a 15k prompt: ~1 s of staging overlapped under ~5 s of compute, so prefill goes from ~1,547 t/s to ~1,800 t/s (+15%),
and TTFT for prompts above 1,024 rows falls by up to ~1 s. Measure with `bench.ps1` (three runs, two processes) at 4,600
words plus TTFT by prompt size at 1,200 / 2,000 / 4,600 / 15,000 tokens.

Gate, in this order: build -> Gate 0 -> `kernels_parity` -> three mid-prompt needles at 621 / 1,198 / 2,968 tokens (the same
ones that gated the routed path) -> then timing. A stream-ordering mistake here reads a half-copied expert, which is a wrong
answer, not a crash.
