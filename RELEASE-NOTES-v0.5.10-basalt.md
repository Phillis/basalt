# v0.5.10-basalt - the round budget, and the one change that paid: batch rounds charged per slot by rows

Follows `v0.5.9-basalt`. Nothing here changes what the model computes: the one shipped change is a draft-policy fix,
and everything rejected is recorded with its numbers.

## The round budget (measured, nsys)

A two-slot decode profiled with `nsys profile --trace=cuda` and read with `nsys stats`:

| kernel | share of GPU kernel time | calls | avg |
|---|---|---|---|
| **`moe_cpu_merge_kernel`** - the merge waiting on the CPU pool's experts | **42.5%** | 1,704 | 442 us |
| `native_gu_multi_kernel` - card gate/up | 14.6% | 1,705 | 152 us |
| `native_down_sub8_kernel` - card down | 9.0% | 1,704 | 94 us |
| dense projections, the head, attention, router, controllers, read-ahead | ~24% | | |

Host side: `cudaStreamSynchronize` is **49.7% of the wall**. **The round is CPU-pool bound, not kernel bound** - which
is why items that reduce misses pay and kernel work does not.

## Shipped: the batch round's cost charged to each slot by its rows

`batch_round()` gave every slot in a group the same `round0`, so each slot learned its window cost the whole shared
round - twice its real cost - and the draft policy shortened it (dl 2.0-2.3 solo against 1.6-1.9 at two slots,
acceptance down to 30-56%). The group's time is now split over its windows by rows.

| | before | after |
|---|---|---|
| ~2k context, per stream | 33.2 / 32.3 tok/s | **35.4 / 36.3** (+7% / +12%) |
| ~64k context, per stream | 44.6 / 41.9 tok/s | **44.8 / 46.5** (+0.4% / +11%) |
| acceptance | 30-70% | **66-90%** |

## Measured and rejected, with the numbers

- **Lookahead prefetch with an epoch guard** (the spec's Item 3): implemented as specified - the guess runs with
  copies present, stamped with `placement_epoch()`, skipped when the epoch moved. **It crashes the engine** (5 crash
  lines, requests 503 in two processes), so the epoch alone does not make a guess safe. Reverted.
- **`kElasticKeep` 512 -> 256 MiB**: 64k decode **44.6 -> 30.2 tok/s per stream**. Reverted.
- **Double-buffered staging** (to hide the ~33% of prefill that is exposed copying): the plan refuses to load -
  "`--max-context 131072 needs 2366 MiB a session`". The exposed staging cannot be hidden on a 12 GB card at 2 x 128k.
- **The disk conversation cache**: it works (`disk-cache=4096MiB`, 34 files of 46-68 MB, 60,017 tokens read back in
  278 ms) but **cannot mount**: the image's running state is 98,050,048 bytes against the restoring session's
  113,901,568, because `running_bytes()` sums a `vc_` sized from each session's own conversation. Left off (it writes
  1.24 GB per parked conversation for no reuse).
- **Tier 1 flags** (`spec` 8/6/5/4, `suffix-draft` 3/8/16, `prompt-chunk` 5120/6144/7168, `pool-workers`): no arm
  beat the base outside the 3-5% run-to-run noise band; the defaults stand.

## Gates

Stress 20/20 in one process, 0 crashes; `inkling2_prompt_parity` and `inkling2_kernels_parity` 0 failures; three
needles at 10/50/90% exact at 64,049 and 119,985 tokens read into one slot while the other decodes; cancel
mid-generation five times clean. `DAY-HANDOFF.md` carries the launch recipe, every number and the ranked list of what
is left (interleaving as a flag, the disk cache's mount, cross-slot system-prompt reuse, the FP4 GEMM).
