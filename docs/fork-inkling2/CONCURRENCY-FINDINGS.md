# Concurrency findings
# Concurrency: measured, and the two bugs it exposes

Two slots is the case that matters here, so it was measured directly rather than extrapolated. Batching does
not currently pay, and the reasons are concrete.

## Numbers (2 x 128k profile, 2 x 300-token generations, 131k context each)

| run | aggregate | per stream | acceptance | dl | hit |
|---|---|---|---|---|---|
| 2 concurrent, chunk 4,096 | **60.3 tok/s** | 30.1 | 34-38% | 1.8-2.0 | 65% |
| 2 sequential, chunk 4,096 | 68.0 tok/s | 34 | 62-67% | 1.8-2.0 | 76% |
| 2 concurrent, chunk 2,048 | 61.6 tok/s | 30.8 | 44-56% | 1.5-1.7 | 80-81% |
| 2 sequential, chunk 2,048 | **63.9 tok/s** | 32 | - | - | - |
| 4 concurrent, chunk 4,096 (4 x 64k) | **57.2 tok/s** | 14.6 | - | - | - |

For comparison, one slot at 262k with chunk 8,192 measures **92.9 tok/s** and hit 93%.

**Batching is -3.7% to -11.4% in aggregate at 2 slots, and 4 slots collapse to 57 tok/s aggregate against the
230 tok/s the 4 x 64k profile gave before the copy machinery landed.** That is a regression, not a tuning gap.

## Why: two separate defects

1. **Draft acceptance collapses under batching**: 62-67% alone, 34-38% with two slots. `dl` stays ~2, so
   exactly as many drafts are produced and then thrown away - the verifier does the work and commits one
   token. That is the dominant loss. A per-session drafter exists (`sl.sfx` is per slot), so the suspicion is
   the batched verify window packing two slots' drafts into one window with one slot's positions or
   confidence bins, and `DraftPolicy`'s per-bin learning being fed two different streams. This needs #232-style
   instrumentation to attribute per slot, then a look at `session_engine`'s batch round.
2. **Copy residency is halved**: hit rate 93% alone, 76-81% with two slots even after shrinking the prompt
   arena to 2,048 rows (which recovered `cuda0` 2.5 -> 3.0 GiB). Two KV caches and two arenas come out of the
   same 12 GB that holds the expert copies, and `loan_ready()` requires `sessions() == 1`, so the prompt loan
   - the machinery that pushes residency to 1,593 copies during a prompt - is **off exactly when it is needed**.
   That gate is the cheapest thing to relax.

## Also found

- `--prompt-chunk 1024` crashed with `inkling2 prompt: the FP4 planes outgrow xq` (the NVFP4 activation planes
  need more room than the q8 row buffer they share at that size). The flag's floor is now **2,048**, so the
  knob cannot reach a size that aborts.
- `--prompt-chunk 12288` does not fit beside a 262k cache: "the card is 774 MiB short". 8,192 is the ceiling.
- A 30,160-token prompt reads at **2,951 t/s** at chunk 8,192 (4 staging passes).

## What to do about it, in order

1. **Relax the `sessions() == 1` gate on `loan_ready()`** (and on rebalance), so two slots still get the
   prompt loan's extra copies. Expected: hit rate back toward 90%, which is where the per-stream loss is.
2. **Find the acceptance collapse** (per-slot accounting first). If drafts are being verified against the
   wrong slot's context, that is a correctness-adjacent bug worth fixing for its own sake; if the policy is
   simply shared, making it per slot is a few lines.
3. Only then re-measure the profiles: `2x128k` is the one that matters, `4x64k`/`5x52k` are for fleets.
