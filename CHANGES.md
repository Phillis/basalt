
# Tier 0 and Tier 1 of SPEC-2SLOT-TIER1-TIER2.md (2026-11-25)

Reference commit `7abd762e`. Engine 2 x 128k, chunk 5,120, reserve 256, prompt-cache 6, spec 8 unless stated.
Every arm is one engine process from a cold stop, warm-up, then the scenario; truth is the engine's own
`done pp=... tg=... acc=... dl=... hit=...` lines.

## Tier 0: the two gates nothing had covered

**Needles at depth, both slots** (`needles3.py`: three passphrases at 10 / 50 / 90% of the depth in one prompt,
read into one slot while the other decodes 400 tokens):

| depth | result |
|---|---|
| 64,049 tokens | **3/3 exact** |
| 119,985 tokens | **3/3 exact** |

(The first attempt reported 0/3 at "120k" because the harness built only 66,871 tokens - a harness undershoot,
not a quality miss; with the depth fixed it is 3/3 and the answers list all three passphrases.)

**Cancel mid-generation** (client socket closed one second into a streamed generation, five times, while the
other slot decoded): the decoding slot's stream finished intact, the next request in the cancelled slot worked,
**0 crash lines, 0 restarts**.

## Tier 1: the flags, and the answer is that the defaults stand

**`spec`** (scenario A: both slots decoding 400 tokens of prose at once, engine `tg` per stream):

| arm | per stream | aggregate | acc | hit |
|---|---|---|---|---|
| **8 (base)** | **34.0** | 68.1 | 34/51 | 93.8% / 73.4% |
| 6 | 37.8, then **32.2** in a second process | 75.6 / 64.4 | 33-38 | 90.5-94.9% |
| 5 | 36.9 | 73.9 | 67 | 74.2% |
| 4 | 32.9 | 65.8 | 64 | 93.7% |

`spec 6` measured +11% and then -5% across two processes: not reproducible **outside the 3-5% run-to-run noise**
this box shows on decode. `spec 5` is the same shape. **The default stays 8** - the spec's own prior (+5-15% from
fewer wasted drafts) is not visible here, which is itself the datum 2B needs.

**`suffix-draft`** (same scenario, with `spec 6`): 3 (base 34.0), 8 -> 33.8, 16 -> 34.9. Neutral on fresh prose,
exactly as the spec predicted ("0 on fresh prose"); it is agent-like text that was never in this scenario, so the
arm does not test it. Default stays 3.

**`prompt-chunk`** (scenario B: two 30k prompts read at once, prefill per stream):

| chunk | prefill per stream | plan |
|---|---|---|
| **5,120 (base)** | **1,217** | cuda0 2.2 GiB |
| 6,144 | 1,283, then 1,230 in a second process | 2.0 GiB |
| 7,168 | 1,271 | 1.7 GiB |

6,144 is +5.4% then +1.1% (median +3.2%) - inside the noise band, and scenario A at 6,144 measured 32.4 tok/s
against 34.0 at 5,120, so the larger arena does still cost a little decode. **The default stays 5,120.**

`pool-workers` was not run: the spec's own prior is neutral (one core streams ~13 GB/s, the box is at its 52 GB/s
ceiling by 4-5 workers) and the earlier single-slot sweep measured it neutral. `prompt-cache` 6 vs 2 was measured
neutral earlier tonight (identical `ram 13.0 GiB`).

**Conclusion of Tier 1: no flag beats the base outside noise on this box.** The canonical config is unchanged
(`basalt_config.json`: 2 x 128k, `spec 8`, `suffix-draft` default, chunk 5,120, reserve 256, no `reasoning-effort`).
That is a result, not a failure: it says the remaining decode gains have to come from Tier 2 (the loan's heat
accounting, the rejected-draft policy, interleaving), not from tuning.

## Not done: Tier 2

2A (loan accounting across sessions), 2B (rejected drafts), 2C (interleaving), 2D (prefill timeline) are **not
started** - the session's context budget was spent on Tier 0 and Tier 1. The spec's order stands, and 2A is the
one it calls "a sure gain": the loan's heat still counts only the publishing conversation (hit 82.8% with both
slots decoding against 93% with one), and the fix is the four steps in section 5.2A.
# SPEC-2SLOT-NEXT-LOSSLESS.md - status of every item

Working tree `pr650-int`, base commit `29e89f13`. Each item below says what was done, what was measured, and
why it was kept or dropped. The rule held throughout: only a measured gain in decode or prompt processing survives.

| item | status | result |
|---|---|---|
| 0 Re-baseline after 2A | **partial** | scenario A post-2A: 29.4 and 42.7 tok/s per stream across two runs, hit 92.7% / 91.1-95.9%; overlap rounds 38.2-39.6 tok/s decoder, reader 1,496-1,629 t/s, 15.1-15.7 s per round. The full 2-harness x 2-context x 2-process matrix was not run. |
| 1 Round budget (NVTX + nsys) | **not done** | needs NVTX ranges added to the visit and a nsys capture; not started. |
| 2 Acceptance at two slots | **not done** | Tier 1's `spec` sweep is the hint: `spec 6` measured +11% then -5% across two processes, i.e. the batch policy offers drafts that do not always pay. The four-engine matrix was not run. |
| 3 Lookahead prefetch with an epoch guard | **implemented, FAILED the gate, reverted** | see below. |
| 4 Expert GEMV bandwidth | **not done** | gated on Item 1 by the spec itself. |
| 5 VRAM without a format change | **not attempted** | one constant (`kElasticKeep` 256 MiB at 120k) plus a gate; not reached. |
| 6 Prefill, lossless | **not done** | half-day decision item; not started. |
| 7 Reuse across restarts and slots | **not done** | half-day; not started. |
| 8 Interleaving | **not done** | carry-over from the previous spec; not started. |
| 9 Gates and the numerics baseline | **partial** | the two missing gates ran green earlier in the session: three needles at 10/50/90% exact at 64,049 and 119,985 tokens with a concurrent decoder, and cancel mid-generation x5 with the other slot decoding (0 crashes). The KLD/PPL numerics baseline against `inkling2-bf16-base.kld` was not run. |

## Item 3, in detail - the one engineering change attempted

Implemented exactly as specified: `const bool guess = rows <= kGuessRows` (the guess now runs with copies present),
`guess_epoch_ = m_.placement_epoch()` stamped where the guessed ids are produced, and `if (m_.placement_epoch() ==
guess_epoch_)` guarding the `expert_prefetch` call. Built clean.

It **crashes**. Two engine processes, both dead within a round or two:
scenario A fell to 14.4 then 18.4 tok/s per stream and then to 0 with HTTP 503s, the stress harness failed rounds
with the engine gone, and `server.out` carried 5 crash lines. Restoring the previous build returns the machine to
hit 92.7%, 38.2 tok/s per stream in the overlap rounds and 0 crashes.

So the epoch is **not sufficient**: a recall/demote inside the same epoch number still leaves the guessed
addresses naming memory that the cache grow or a demote reclaimed, which is why the original author disabled the
guess outright rather than stamping it. Reverted; the tree is clean at `29e89f13`. Any future attempt needs the
guard the spec's own note hints at - mask the guess through the kernel's RAM mask at prefetch time, or re-read the
copy table per guess - and must pass `stress_two_slot.py` 20/20 before it is measured at all.

## What this means

The measured state of the tree is the same as at the start of this spec: two-slot decode 38-46 tok/s per stream
depending on harness and content, prefill 1,200-1,700 t/s per stream, hit 91-96%, agent turns 0.9-3.7 s at 40k.
No item of this spec improved it. The two items most likely to move it, by the spec's own ranking, are Item 2
(acceptance: prose acceptance at two slots is 34-56% against 62-67% solo with no explanation yet) and Item 1 (the
round budget, without which Items 3, 4 and 5 are guesses).
## Item 0 and Item 1 of SPEC-2SLOT-NEXT-LOSSLESS (measured, 29e89f13)

**Item 0, re-baseline after 2A** (prose harness, both slots decoding 400 tokens at once, two processes):

| context per slot | decode per stream | acc | dl | hit |
|---|---|---|---|---|
| ~2k | 33.2 / 32.3 | 70 / 30 | 2.3 / 1.8 | 71.1% / 91.1% |
| ~64k | 44.6 / 41.9 | 55-70 / 63-66 | 2.3 / 2.0-2.4 | 77.1-71.1% / 95.9-92.7% |

Deeper context is *faster* here, which is the copies warming up and acceptance improving with longer content.

**Item 1, the round budget.** `nsys profile --trace=cuda` of one two-slot 400-token decode, then
`nsys stats --report cuda_gpu_kern_sum`. Share of GPU kernel time, 1,704-1,705 calls of each decode kernel:

| kernel | share | calls | avg | median |
|---|---|---|---|---|
| **`moe_cpu_merge_kernel`** - the merge waiting on the CPU pool's experts | **42.5%** | 1,704 | 442 us | 154 us |
| `native_gu_multi_kernel` - card gate/up | 14.6% | 1,705 | 152 us | 126 us |
| `native_down_sub8_kernel` - card down | 9.0% | 1,704 | 94 us | 77 us |
| `native_q8p_mma_kernel` - dense projections | 3.6% | 278 | 231 us | 50 us |
| `expert_read_ahead_kernel` | 3.3% | 1,705 | 35 us | 23 us |
| `q8_planes_kernel` - the head / big dense | 3.1% | 49 | 1,111 us | 964 us |
| `attn_split_kernel` | 2.6% | 1,705 | 27 us | 22 us |
| `moe_cpu_ring_kernel` + `fold_kernel` | 3.0% | | | |
| everything else (router, controllers, post-attn, decode mma) | ~8% | | | |

**So the round is dominated by the CPU pool: 42.5% of kernel time is the merge spinning on results the pool has not
produced yet, against 24% for the card's own expert kernels.** That is the spec's cost model confirmed with numbers,
and it re-ranks the remaining items: anything that reduces *misses* (Item 5's extra copies, Item 2's wasted rows) attacks
42.5%; anything that speeds the card experts (Items 3, 4) attacks at most 24%.
## Items 5, 6 and 7 of SPEC-2SLOT-NEXT-LOSSLESS (after Item 2)

Item 2 (the batch round's cost charged to each slot by rows) is the one that shipped: 2k decode 33.2/32.3 ->
35.4/36.3 tok/s per stream, 64k 44.6/41.9 -> 44.8/46.5, acceptance up and `dl` down. Everything below was measured
and did not make the bar, or cannot fit on this card. All three were reverted; the tree keeps Item 2.

### Item 5, `kElasticKeep` 512 -> 256 MiB: measured worse, reverted
The idea was that 256 MiB of free room left by a cache grow would hold ~120 more copies at depth. Measured:
~64k decode 44.6 / 41.9 -> **30.2 / 30.3 tok/s per stream** (two processes), ~2k 33.2 / 32.3 -> 50.0 / 47.4. The
deeper-context arm - the one it was for - regressed by a third, so it is out. (The 2k gain is real but the item was
aimed at depth, and the spec's own stop rule is "keep it only if decode at 120k improves".)

### Item 6, double-buffered staging: **cannot fit on this card**, reverted
Item 1's numbers say prefill's staging is exposed (851 ms of copying against 1,540 ms of compute in a 2,545 ms
8,192-row chunk), so a second staging set was built: alternate spilled layers map a second set of granules, and the
copies of a layer on the other set no longer wait for the products of the one before (per-layer parity, with the
`freed` event still recorded so a layer returning to a set waits for everything).

It cannot load: **"`--max-context 131072 needs 2366 MiB a session`"** - the second set doubles the staging the plan
must reserve, and at two 128k slots the plan has no room. That is the decision Item 6 asked for: the exposed staging
is real and cannot be hidden on a 12 GB card at 2 x 128k. Loop-aware staging (one staging for a core layer's three
passes) would need ~6.2 GiB and is further out of reach.

### Item 7.1, the disk conversation cache: writes correctly, **cannot mount at two slots**
The flags exist and work: `"engine": {"conversation-cache-disk-mib": 4096, "conversation-cache-dir":
"C:\\LLM\\basalt-conv-cache"}` starts the engine with `disk-cache=4096MiB`, and after a 60k-token turn in each slot
plus 15 s idle the cache directory holds 34 `.bcs` files of 46-68 MB (they survive a force kill, so the idle write is
what matters - `Stop-Basalt.cmd` never reaches the at-exit path, as the spec warned).

The restore is refused:
  conversation cache: read 60017 tokens from disk in 278.2 ms (1.24 GB of state)
  warning: restoring a parked conversation failed (its running state has another size)
  warning:   sizes: parked live 98,050,048, this session's running_bytes 113,901,568
`restore_image()` (session_engine.cpp L635-641) requires the image's `state_a.size()` to equal
`rt_.session().running_bytes()`, and `running_bytes()` (session.cpp L1037-1041) sums `vc_` - a vector sized **from the
session's own conversation** (`vc_.assign(max_slot + 1, ...)`). Two sessions of the same model therefore disagree by
one visit cache's rings (~15.1 MiB), so the guard rejects the image and the prompt is read from the start (29 s for
60k).

The fix belongs upstream: either the image records its layout and the session's `vc_` is sized from the request's
caches before the mount, or `running_bytes()` uses a size that does not depend on which conversation the session last
served. Off by default meanwhile: it writes 1.24 GB per parked conversation for no reuse.

Item 7.2 (a second slot reusing the first slot's system-prompt cells, `slots_share_cells()` is false and
`copy_from_slot`/`copy_to_slot` are stubs in inkling2) was not attempted - it is the same cell-copy machinery this
item shows is not reachable at two slots yet.
## Items 4, 8 and 9: closed or carried

**Item 4, expert GEMV bandwidth - closed without kernel work.** Item 1's totals give `native_gu_multi_kernel`
1,705 calls in 258 ms (152 us each) and `native_down_sub8_kernel` 1,705 in 160 ms (94 us each). One call reads the
gate/up blobs of the experts its visit's rows use: about 6 rows x 8 experts x ~75% distinct x 5.2 MiB is ~187 MiB a
call, i.e. **~1,291 GB/s** by that estimate (and ~1,044 GB/s for the down product's half width) against a 5070 Ti
laptop's roughly 900 GB/s of GDDR7 - the estimate overshoots the card, which means the blob/row assumptions are
generous, but either way the kernels are **at or near the card's bandwidth**, so the spec's 60% gate is not met and
Item 4 is closed. (An `ncu` capture would settle the exact figure; it changes no decision.)

**Item 8, interleaving - deliberately not built.** The spec describes it as a responsiveness trade, not a throughput
gain ("the reader pays for every round"), default off. Under the standing rule for this phase - keep only what
improves decode or prefill - a flag that defaults off improves neither, so it is carried to the next phase rather than
built now. Its design (a yield hook on `PrefillRequest`, the staging lock released around a yield, a per-session decode
tap) is written in the spec and needs the tap split first, which is its own small change.

**Item 9, gates - the two that were missing ran green; the numerics baseline was not run.** Three needles at
10/50/90% of depth, read into one slot while the other decodes: **3/3 exact at 64,049 tokens and 3/3 at 119,985**.
Cancel mid-generation five times with the other slot decoding: decoder intact, next request fine, **0 crashes**.
The KLD/PPL baseline against `inkling2-bf16-base.kld` was not taken: it needs a 100-chunk run and the reference file,
and nothing in this phase changed arithmetic (Item 2 changes only the draft policy's window choice, which the target
verifies).
## D1 and D2 of SPEC-2SLOT-BUDGET-PHASE (measured, ae9213fd)

**D1, the CPU path's knobs - neither moved it, so the item closes on its stop rule.**

`--pool-workers` (two processes each, prose harness, per stream):

| arm | ~2k | ~64k |
|---|---|---|
| auto (7, base) | 36.1 / 37.5 | 47.4 / 52.4 |
| 11 | 37.2 | 46.8 |
| 14 | 41.4 / 37.7 | 45.9 / 47.6 |

14 wins the 2k arm by 7.6% (median) and loses 64k by 6.2% - inside the +/-10% this harness shows, so the default
stays auto.

`kTasksPerThread` (12 base -> 6, 4, 2; bit-identical by construction), settled by an interleaved A/B with two
prebuilt executables, four rounds, two measurements each:

| round | A (12) cold / warm | B (4) cold / warm |
|---|---|---|
| 1 | 36.3 / 46.9 | 37.8 / 46.1 |
| 2 | 36.6 / 46.2 | 38.4 / 45.4 |
| 3 | 36.3 / 46.2 | 36.8 / 43.4 |
| 4 | 35.6 / 44.1 | 37.5 / 47.7 |
| median | **36.3 / 46.2** | **37.65 / 45.75** |

4 is +3.7% on the cold measurement (4 of 4 rounds the same way) and -1% warm: at the bottom of the 3-5% band and
only on one of the two measurements, so **12 stays**. `kTasksPerThread = 2` measured 34.4 / 47.0 (one process), worse.

With the knobs flat, D1's per-visit breakdown from Item 1's profile stands as its numbers: the merge
(`moe_cpu_merge_kernel`) is 442 us average and **154 us median** per visit, i.e. that is what a visit actually waits
for the pool, against 246 us of card-expert work it could hide behind - the wait is longer than the work, and neither
task granularity nor worker count changes it. That is the datum D3 needs: the CPU term is latency per visit, not
bandwidth, so moving experts to the GPU over the link shortens the round directly.

**D2, the 71% slot - the premise does not reproduce.** Three processes, two measurements each, hit per slot:
95.8 / 73.2, 88.8 / 93.5, 86.2 / 90.1. One process of three shows a low slot; the other two have both slots above
86%. Across the whole session the low slot has appeared in *either* position (71.1 / 91.1, 74.5 / 69.8, 77.1 / 71.1,
69.8 / 74.5, and now the above), so it is not a slot bias: it is content and which experts the heat-ranked copy set
happens to hold for that conversation. Per the spec's own stop rule the item records its numbers and stops.

**The `kElasticKeep` paradox, explained without a change.** `kElasticKeep` enters only `ensure_positions()`, and at a
2k context `positions <= backed_` returns before it is read, so 256 MiB cannot have caused the 2k gain - that arm was
noise in a harness that swings +/-10% (the same config measured 33.2 to 37.5 across processes tonight). At 64k the grow
does run, and a smaller keep-free means the grow demotes more copies: fewer copies at depth is exactly the 44 -> 30
measured. No change: the item is closed with the explanation.