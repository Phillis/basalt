# Day hand-off - two concurrent 128k sessions, what is in place and what it measures

Known good: **c10b3862** (branch `pr650-int`, pushed to `Phillis/basalt`). Engine up at 2 x 128k. Tree clean.

## Launch

    C:\LLM\Basalt\Start-Basalt.cmd

Copies `basalt_config.json` into the engine tree, launches `run-pr.cmd --prompt-chunk 5120`, and warms up three
single requests, then two concurrent ones and one overlap. `BASALT_PROMPT_CHUNK` overrides the chunk.
`BASALT_DIAG_SLOT=1` adds the restore/boundary diagnostics; `BASALT_DIAG_PROMPT=1` the staging/arena one.

Config: 2 x 128k, `parallel 2`, `kv fp8v4`, `vram-reserve-mib 256`, `spec 8`, `suffix-draft` default, **no
`reasoning-effort`** (the pack template's own default applies; a client-sent value wins). Profiles:
`-2x128k-fast.json` (`reasoning-effort: medium`), `-2x160k.json` (2 x 160k, the largest that fits).

## The numbers in one place

Two slots, both decoding, per stream:

| context | prose harness | curve harness | hit |
|---|---|---|---|
| ~2k | **35.4 / 36.3** tok/s | - | 79-94% |
| ~30k | - | 46.2 tok/s | 83-93% |
| ~64k | **44.8 / 46.5** tok/s | 44.9 tok/s | 77-96% |
| ~96k | - | 43.3 tok/s | |
| ~120k | - | 43.3 tok/s | |

Prefill per stream 1,190-1,705 t/s by depth (1,217-1,283 at chunk 5,120 across 30k prompts). Overlap (one slot reads
while the other decodes): decoder **38.2-39.6** tok/s, reader **1,440-1,629** t/s. Agent turns (real roles, replies
sent back): **0.9-3.7 s** at 40k, **3.4-5.4 s** at 100k, first read 63-65 s. Needles 3/3 at 64k and 120k with a
concurrent decoder. Stress 20/20, parities 0 failures.

## The round budget (Item 1 of the last spec), which drives everything

`nsys profile --trace=cuda` + `nsys stats` of a two-slot decode: **42.5% of GPU kernel time is
`moe_cpu_merge_kernel`** - the merge waiting on the CPU pool's experts - against 24% for the card's own expert
kernels (`native_gu_multi` 14.6%, `native_down_sub8` 9.0%). On the host side `cudaStreamSynchronize` is 49.7% of the
wall. **The round is pool-bound, not kernel-bound**, which is why the two items that reduce misses paid and the
kernel-side ones did not.

## What was kept, and what was measured and rejected

Kept: **Item 2** - the batch round's time is charged to each slot by its rows, so the draft policy stops over-pricing
its window (2k decode 33.2/32.3 -> 35.4/36.3, 64k 44.6/41.9 -> 44.8/46.5, acceptance up, `dl` down).

Rejected with numbers: `kElasticKeep` 256 MiB (**64k decode 44.6 -> 30.2**); the lookahead prefetch with an epoch
guard (**crashes**, 5 crash lines, so the epoch alone is not enough); double-buffered staging (**the plan cannot fit
it at 2 x 128k**: "needs 2366 MiB a session"); the disk conversation cache (**writes 1.24 GB per parked conversation
but cannot mount**: the image's running state is 98,050,048 bytes against the restoring session's 113,901,568,
because `running_bytes()` sums a `vc_` sized from each session's own conversation). Tier 1's flags (`spec`,
`suffix-draft`, `prompt-chunk`, `pool-workers`) were all measured neutral outside the 3-5% noise band.

## pi client notes

Context **131,072** per slot; send **no** `max_tokens` or at least 16,384 (a small one caps the thinking budget);
send `x-session-affinity` so a conversation keeps its slot - without it the server alternates and prefix reuse cannot
happen; `reasoning_effort` per request overrides the config.

## Known limits

2 x 192k does not fit (2 x 160k does). Filling both contexts costs ~2.5 GiB of expert copies and decode eases from
46 to 43 tok/s. Pinned RAM is a fixed 14.52 GiB and the two-slot plan sits on it. Prefill is ~63% compute with an
exposed ~33% of staging that **cannot be double-buffered on this card**. A card sitting at 92 of 93 W and 78 C is
power-throttled: a cooling pad or the vendor's performance mode is worth an estimated 5-15% of prefill, no code.

## Not done, in the order I would take it

1. **Item 8 interleaving** (a flag, default off): the decoder is still at zero while the other slot reads. Needs the
   per-session decode tap split first.
2. **The disk cache's mount** (a real fix, upstream): record the running state's layout in the image, or size the
   session's `vc_` from the request's caches before the mount.
3. **Item 7.2**, a second slot reusing the first slot's system-prompt cells (`slots_share_cells()` is false and
   `copy_from_slot`/`copy_to_slot` are stubs in inkling2).
4. **The FP4 GEMM** (~20 effective TFLOPS, prefill's compute half) and an `ncu` pass on the card expert kernels.
5. The tool-call and reasoning-echo harness variants (code exists in `tools/agent_turns.py`).