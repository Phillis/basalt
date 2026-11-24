# SPEC 5 >= Next steps: prefill past 2,000, decode to ~60, then the optional trade

**For the deepseek worker.** The consolidated, up-to-date plan after Session 8 (tally fix landed; Q2_0 shadow
measured and closed on quality) and Session 9 (the shadow family mapped; the clean path chosen >=
`D1-SHADOW-VERDICT.md`). **Two workstreams, in order. Nothing here touches model quality.** The Q3_K
middle-ground shadow was offered and **declined by the user** >= see "Closed by user decision" at the end:
do not build, propose, or re-measure it.

**Ground rules (from `IMPLEMENTATION-NOTES.md` >= every A/B in this tree has been ruined at least once by
skipping one of these):**
- Kill all `basalt.exe` **and** the `C:\LLM` `python.exe` between A/B relaunches; a stale python server serves
  requests with a previous engine and silently invalidates the result.
- One change, one measurement. Unique text only for prefill (random words; repetition is exploited by the
  suffix drafter and the prompt cache). The truth is `server.out`'s `s1 read pp=` / `tg=` lines >= never
  `engine.log`'s "N reused in 0 ms" restore stats.
- `BASALT_DIAG_VISIT=1` for the decode budget (`diag: N windows ...` lines in `engine.log`); `hit=` is now
  trustworthy (Session 8's fix) and must stay >= 100%.
- Standing constants, not knobs: `kCardSpare 768`, `vram-reserve-mib 768`, the ~14.5 GiB pinned ceiling
  (~150 MiB headroom, probe `C:\LLM\bwtest\hostalloc_seq.exe`), the direct launcher (never the scheduled task).
- Validation after every step: repo tests, greedy-identity within one process, `test-long.ps1
  -TargetTokens 131072`, a 5-minute sustained decode run, and re-check *both* fronts.

**Current state (verified):** decode 53.2 tok/s on the TCP prompt (39-67 by content; per window: pool 15.4 ms,
GPU chain 9.3, head 0.9, hand-off 0, drafter+engine ~7.8 outside); prefill 1,536-1,561 t/s on unique prompts
(2.58-2.62 s per 4,096-token chunk: H2D staging ~1.33 s + compute ~1.39 s); `hit=` ~69%.

---

## W1 >= Prefill: wave-pipelined staging (target >= 2,400; ceiling ~2,900)

**The wall, measured:** every one of a chunk's 48 visits stages its layer's spilled experts over PCIe **on
the compute stream, serially**, before its expert products (`prompt.cpp` `experts()`, the
`cudaMemcpyAsync(w.experts + kept, ...)` / `stage_from_card` at ~L419-421). ~21 GiB of H2D per chunk issued
as ~40-60 ms stalls per visit; compute (~1.39 s: attention + dense + expert products) waits behind it. Both
terms are ~1.3 s, so pipelining one behind the other yields `max(1.33, 1.39) + tails >= 1.4-1.5 s` per chunk.

**The design (VRAM-free):** split each visit's staging into **2-4 waves** (contiguous slot ranges) and
pipeline copy against compute *within* the visit:

1. A dedicated copy stream (`cudaStreamNonBlocking`; precedent `ra_st_`/`pf_st_` in `session.cpp`) plus one
   event per wave. Create/join it per session; join at chunk end.
2. Wave *i*: issue that slot range's `stage_from_card`-equivalent / H2D on the copy stream >= record the
   event >= the compute stream waits on it and launches **wave *i*'s expert products** >= the copy stream
   proceeds immediately to wave *i+1*. The staging set is whole-layer sized, so no extra physical pages: wave
   *i* writes slots no product reads.
3. **The fiddly part >= slicing `moe_sort`'s bounds per wave.** The products take per-expert row bounds
   (`b_.bounds` from `moe_sort`); a wave runs the FP4/MMQ product over a contiguous expert-slot range only.
   Two acceptable implementations:
   a. *Preferred:* run `moe_sort` once as today, then build W compacted bound slices (host-side or a tiny
      prep kernel) >= the bounds/tok/dst/exp layout supports a contiguous expert-range slice without
      re-sorting.
   b. *Simpler fallback if (a) fights you:* run `moe_sort` + products per wave over the wave's rows
      (`pairs` filtered by expert range) >= more launches, still correct.
   The FP4 path's per-blob tail read (`base + L.bytes - 16`) works per wave unchanged.
4. Start wave 0's copy **right after the router** (`router_select`), before the attention/dense chain, so it
   overlaps the visit's own ~5-15 ms too.
5. `stage_from_card` (`model.cpp` ~L562) needs a stream parameter and a slot range (today: expert-by-expert
   on the caller's stream). If its per-expert copy overhead dominates, batch: the wave's slot range is
   contiguous, so a single H2D covers it when no elastic copies intervene.
6. The `cudaMemsetAsync(..., kSlotGuard)` guard write stays on the compute stream before wave 0's products.
7. Flag `--prefill-waves 0|N` (0 = today's serial copy) added to `dspark_parse_flag` (`dspark.cpp` ~L722) >=
   **it is not a config-JSON key**; note it in `help.cpp`.
8. Chunk-boundary bonus if free: prefetch chunk *n+1*'s first wave during chunk *n*'s coda (the staging set
   is free for the last ~10 ms of a chunk).

**Gates:** unique-16k prefill >= 2,400 t/s sustained on the `s1 read pp=` lines; TTFT on short prompts
unchanged; **greedy-identity A/B** (`--prefill-waves 0` vs N on the same unique prompt >= this is bit-neutral
data movement, so *any* output difference is a slicing bug, not a trade-off); needle; decode canary unchanged
(`diag` pool busy and tok/s); sustained 5-min run.

**Do not attempt alongside this:** the "loop-major" reordering (withdrawn >= `NEXT-WORK-SPEC.md` ??1: the walk
is already pass-major; the loop is a dependency cycle; staging reuse across passes needs ~6.2 GiB),
`kPromptChunk` 8192 (+~1 GiB arena, unavailable), `--prefill-cpu-share` (not parsed by inkling2), CPU-prefill
kernels (only to break the ~4,800 t/s PCIe ceiling later), GPU expert-kernel work (products are ~0.3-0.5 s of
the 1.39 s >= not the wall), `kElasticKeep` trims (the demote sizing was verified legitimate).

**After it lands:** the chunk is compute-bound (~1.39 s). Profile that split (attention vs dense vs products)
at 53k *and* 130k context before optimising anything further >= attention grows with the prefix and will bound
long-context prefill; that profile is the input for any future round.

## W2 >= Decode: capture the drafter block (target >= 58)

**The wall, measured:** ~7.8 ms per window is spent *outside* the decode window >= the DSpark drafter block
that runs between windows. The W-A event fix already removed the full-stream sync (`dspark.cpp` ~L607-619,
worth 37 >= 31.2 ms/window); what remains is the block's own serial time: `draft_begin` runs its attention
**uncaptured** ("its window follows the position" >= ~3 direct `gqa_window_attn` launches per block), then
per-layer stage graphs, plus the screened head and the meta H2D.

**The change:** make the drafter block a single captured graph per position-shape, like the main window
already does:

1. Key drafter graphs on the same launch-bound shape the main session uses (`full_keys`-style: the position
   rounded to a power of two). `draft_begin`'s attention window depends only on `pos` + the fixed block
   geometry, so a shape key of `(kDsBlock rows, pos-bound)` suffices; capture once per shape, replay after.
2. Batch the meta H2D into the captured block (a `copy_i32_from_mapped`-style kernel node >= the doorbell
   machinery already avoids memcpy nodes mid-chain).
3. Keep the direct-launch path as a fallback flag (`--dspark-direct`) >= a bad capture hangs the first window
   (the first doorbell attempt's failure mode; see `IMPLEMENTATION-NOTES.md` ??2), so A/B and bail-out must be
   one flag away.
4. The W-A event bookkeeping stays: `draft_end` waits on the drafter's own event, not the stream.

**Gates:** the outside-window span >= 3 ms per window (measure by the serve-line closure: total per token >=
window wall ?? tokens/window, or add host timers around `draft_begin`/`draft_end` in the solo request path);
decode >= 58 on the TCP prompt with no acceptance change (acc must stay ~73% ?? noise >= a capture bug shows up
as wrong drafts, not slow ones); needle; the `dl` content spread spot-checked (a verbatim-repeat prompt should
still show dl ~3.9).

## Closed by user decision ? the Q3_K shadow (do not build)

The Q3_K middle-ground shadow (13.78 dB SNR at 3.44 bits, blob ?1.31, decode ~57-59 alone / ~62-64 with W2 ?
`D1-SHADOW-VERDICT.md` ?3-4) was offered with side-by-side sign-off as the only remaining in-tree decode
lever beyond W2, and the user declined it. **The entire shadow family is now closed:** Q2_0 on measured
quality (7.85 dB vs a 9.71 dB family ceiling), Q4_K on no byte savings, Q3_K by user decision. No future
session should build, propose, or re-measure a weight-shadow format for this model on this machine without a
new instruction from the user.
## Order, effort, and the end state

| Step | Effort | Gate | After it |
|---|---|---|---|
| W1 wave pipelining | 1-2 days | prefill >= 2,400; greedy-identity | prefill ~2,700-2,900 ceiling |
| W2 drafter capture | 1 day | outside-window >= 3 ms; decode >= 58 | decode ~58-60 |

**The end state on this hardware (zero quality cost): ~58-60 tok/s decode (39-67 by content) and
~2,400-2,900 t/s prefill on unique prompts.** The ranked list beyond it: a 16 GB card (removes the spill,
>100 tok/s), the CPU-prefill kernel project (only to break the ~4,800 t/s PCIe ceiling), and
the post-W1 compute-split profile as the input for whatever comes next. Append all results to `CHANGES.md`
(Session 10).
