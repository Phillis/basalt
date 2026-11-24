# SPEC 4 — The consolidated path forward: decode and prefill

**For the deepseek worker.** One document, both fronts, superseding the open items in `DECODE-PERF-SPEC-3.md`
and `PREFILL-PERF-SPEC.md`. Everything below was re-measured or re-verified today on the current build
(post W-A-sync-fix, lookup drafting, P2 promote ranking, the `r.n_tokens > 16` re-take gate) — the budgets in
the older specs are stale where this disagrees.

Working tree `C:\LLM\pr536\basalt-inkling-cpu-spill-review`, build `C:\LLM\rebuild.bat`, canonical config
`C:\LLM\Basalt\basalt_config.json` (192k ctx, `spec 8`, `kv fp8`, `vram-reserve 768`, `ram-reserve 512`),
launch `Start-Basalt.cmd`, stop `Stop-Basalt.cmd`. **Measurement discipline from `IMPLEMENTATION-NOTES.md`
§1 applies throughout** (kill basalt.exe *and* the C:\LLM python.exe between A/Bs; unique text only; the truth
is `server.out`'s `s1 read pp=` / `tg=` lines; `BASALT_DIAG_VISIT=1` for the decode budget; the needle test as
correctness). Standing constraints: the pinned-host ceiling (~14.5 GiB, ~150 MiB headroom;
`C:\LLM\bwtest\hostalloc_seq.exe`), `kCardSpare 768` / `vram-reserve 768` are measured constants, and the
scheduled task cannot launch this build.

## 0. Verified current state (measured today, diag on)

**Decode** (greedy, TCP-congestion prompt, 600 tokens): **53.2 tok/s**, acc 73%, dl ~1.7-2.2 (content-dependent
39-67 overall). Per window (~2 rows, body ~25.5 ms + ~7.8 ms outside):

| Term | ms/window | Notes |
|---|---|---|
| CPU pool expert compute | **15.4** | ~0.85 GiB blobs/window at ~45-50 GB/s; the dominant term, unchanged |
| GPU pre-expert chain | 9.3 | flat-ish in rows |
| Head | 0.9 | |
| Drafter + engine (outside window) | **~7.8** | improved from ~9 by the W-A event fix (their 37→31.2 ms note); the drafter block itself remains |
| Hand-off | 0 | doorbell, host-funcs/window = 0 |

**Prefill** (unique 31,717-token prompt, warm): **1,536 t/s** average, 2.58-2.62 s per 4,096-token chunk
(early chunks 647 t/s while staging re-take + KV growth settle). Corrected chunk budget (handover §4, verified):
**H2D staging ~1.33 s (~21 GiB at 15.75 GB/s) + compute ~1.39 s** (dense + attention + expert products; the
older spec's 0.8 s compute estimate was wrong — this matters, see §2). KV growth evicts elastic copies during
long prompts (verified: `the caches grow to 57344 positions; 0.64 GiB of copies go back to RAM`; the sizing is
legitimate — ~0.46 GiB real KV growth + the 512 MiB `kElasticKeep` margin — so the P2 promote ranking cannot
help long prompts; +2.4% was its real value and that is all there is).

**Known-bug reminder:** `hit=` reads 193% in today's log — the cumulative `ex_tally_` ÷ per-request-count bug
(W-E.1 below).

## 1. Correction to HANDOVER-PREFILL §5 — "loop-major" is not implementable as written

The handover's recommended next step ("process the chunk pass-major; each loop layer's staging set is built
once and reused for all three passes") contains a contradiction:

- **The walk is already pass-major.** `walk.hpp` runs prelude → [loop 6-17] × 3 passes → coda, all 4,096 rows
  per visit. The loop is a *cycle*: pass *p+1*'s layer 6 consumes pass *p*'s layer-17 output (`BankOut`) and the
  bank messages, so no other order (e.g. layer-major, a layer's three passes adjacent) is legal.
- **"Reuse the staging across passes" needs all 12 loop layers resident simultaneously** (~12 × 511 MiB ≈
  **6.2 GiB**), which does not exist. The 38 MB figure in the handover is the hidden-state size, not the
  staging.

Do **not** attempt it. The ~8-9 GiB/chunk of re-staging it targets is real, but the only legal way to avoid it
is to make the staged bytes *smaller* (§3) — which the same work delivers anyway.

## 2. Prefill: wave pipelining is still the top lever, with the corrected ceiling

With DMA ~1.33 s and compute ~1.39 s per chunk, pipelining the staging behind the compute stream yields
`max(1.33, 1.39) + tails ≈ 1.4-1.5 s` per chunk → **~2,700-2,900 t/s** (the handover's corrected 2,400-2,700
is the pessimistic tail-inclusive end; either way the 2,000 target is cleared with margin). The design is
unchanged from `PREFILL-PERF-SPEC.md` §2 (waves of 2-4 slot-ranges on a dedicated copy stream, `moe_sort`
bounds sliced per wave, `--prefill-waves 0|N` flag via `dspark_parse_flag`, bit-neutral data movement, greedy-
identity A/B) with the handover's refinements (start wave 0 right after the router; give `stage_from_card` a
stream + slot range). **After** it lands, the wall is the 1.39 s compute — profile its split (attention vs
expert products vs dense) at 53k *and* 130k before optimising anything further; attention grows with the
prefix and will bound long-context prefill.

## 3. The one feature that moves BOTH fronts: the Q2_0 shadow (D1/P-shadow)

**New finding: `src/prefill/mmq/instances/mmq_q2_0.cu` exists — the GPU can compute Q2_0 experts natively.**
So the Q2_0 shadow of spilled experts (spec-3 W-C) is worth building once and using twice:

- **Decode:** the CPU pool reads the shadow (existing Q2_0 CPU kernels, `q2_avx2_rows.inc`) —
  ~2.7× fewer bytes and cheaper dequant → pool term 15.4 → ~6-8 ms/window → **~65-75 tok/s** (the ≥75 gate
  from spec 3 stands; treat the earlier 90/80 projections as upper bounds — the dl-3.9 run measured 65, the
  row-coefficient is worse than linear beyond ~2 rows).
- **Prefill:** stage the *shadow* blobs (1.65 MiB vs 4.48 MiB per NVFP4 expert; Q8_0 layers 20-23 keep their
  own blobs) and compute with the GPU Q2_0 MMQ instance → H2D per chunk **~21 GiB → ~7.7 GiB → 0.49 s**.
  Without pipelining that alone takes a chunk from 2.6 → ~2.0 s (~2,050 t/s); with §2 pipelining the chunk
  becomes compute-bound either way — so the shadow's prefill value is insurance and long-context margin
  (the 0.49 s DMA no longer grows relative to compute as attention does).

Design (from spec-3 W-C, corrected and sharpened):
1. Build at load: requantize the **spilled experts of the NVFP4 layers** (0-19) NVFP4→Q2_0 into **pageable
   RAM** (~2,738 × 1.65 MiB ≈ 4.4 GiB of the ~10 GiB free; **no pinning** — nothing on the card touches the
   shadow while the CPU-spill path is active; re-verify with `hostalloc_seq.exe` that the pinned total is
   unchanged). First-touch the whole shadow at build time (a mid-run page-in of GiBs would stall the pool).
   Layers 20-23 (Q8_0) are untouched.
2. Decode path: `CpuSpill::run()` uses shadow blob + Q2 `NativeFmt` when present (flag-gated).
3. Prefill path: `experts()` stages from the shadow and dispatches the Q2_0 MMQ instance when the shadow is
   on; the per-blob `{s_gate, s_up, s_down}` tail the SwiGLU form reads is NVFP4-specific — carry equivalent
   scales for Q2_0 blobs (Q2_0 has per-block f16 scales; the gate/up/down softcaps need per-blob
   `sg/su/sd` — store the same 16-byte tail at each Q2 blob's end so the existing `mmq::swiglu` form works).
4. Flag `--cpu-shadow q2|off` (default off — **it is a real quality trade**: ~27% of routed lookups go from
   NVFP4 (4-bit, fine scales) to Q2_0 on decode, and prefill *staged* experts likewise). The user opts in.
   If quality hurts: a `q4` middle exists for decode-only (no byte win — Q4_0 ≈ NVFP4 size), so prefill
   simply keeps staging the original blobs when `q4` is selected.
5. Gates: decode `diag` pool busy ≤ 8 ms and tok/s ≥ 65 on the TCP prompt; unique-16k prefill ≥ 1,900 with the
   shadow alone; needle test; the user's judgement on generation quality (show side-by-side samples).

## 4. Decode remainder, in order after D1

- **D2 — finish W-A: capture the drafter block.** The remaining ~7.8 ms/window outside the window is the
  drafter block (3 direct attention launches per draft_begin + stage graphs + screened head + host readback).
  Capture the block per position-shape (the main session already keys graphs on the launch bound) or at least
  batch the direct launches; target ≤ 3 ms → **+6-9% decode on top of D1**. Read the drafter events (already
  added by the W-A sync fix) before touching anything.
- **D3 — W-E.1 tally fix (deliberate, small).** `hit=` (193% and climbing) divides cumulative `ex_tally_`/
   `lookups_` by per-request counts. Fix the lifetime (reset per request, or accumulate the denominator the
   same way) — but the tally also feeds the expert map, so change the accounting at one site and re-run the
   needle test plus a decode canary. Until fixed, read `experts: N of M in RAM` for cache truth.
- **D4 — optional, risky micro:** trim `kElasticKeep` (512 MiB) during large prefills to keep ~256 MiB more
  copies resident — the demote sizing is otherwise legitimate; only worth it if D1+§2 leave prefill short of
  target, and the driver/graph headroom must be watched.

## 5. Implementation order, gates, and what not to do

| Order | Work | Gate |
|---|---|---|
| 1 | **D1/P-shadow Q2_0 shadow** (decode first, then prefill staging) | pool busy ≤ 8 ms; decode ≥ 65; prefill ≥ 1,900; needle; user quality sign-off |
| 2 | **P1 wave pipelining** | unique-16k ≥ 2,400; TTFT unchanged; greedy-identity vs `--prefill-waves 0` |
| 3 | **D2 drafter capture** | outside-window ≤ 3 ms (diag/serve closure); decode ≥ 70 (with D1) |
| 4 | D3 tally fix; profile the 1.39 s compute split at 53k/130k | `hit=` ≤ 100%; a ranked list for the next round |

**Do not:** loop-major chunk reordering (§1); `prefill-cpu-share` (not parsed); `kPromptChunk` 8192 (~+1 GiB
arena); CPU-prefill-expert kernels (the pool caps at nt=8 — only if the PCIe wall must be broken *and* D1+P1
have landed); any mapped-RAM GPU compute (6×, re-refuted); SIMD kernel work (closed, ≤12% headroom); sweep
`kCardSpare`/`vram-reserve` (measured constants). Keep `--relaxed-drafting` off (it cannot pay until D2 lands
and only for sampled requests — spec 3 §2).

**Validation protocol** (every step): repo tests, greedy-identity within-process, `test-long.ps1
-TargetTokens 131072`, a 16k+ unique prefill, a 5-minute sustained decode run, VRAM/RAM lines clean (no
`pinned staging alloc` failures), and re-check *both* fronts — every lever here touches decode and prefill
through the same experts. Append results to `CHANGES.md` as Session 8.

## 6. Where this ends up

| Build | Decode (TCP prompt) | Prefill (unique 16k+) |
|---|---|---|
| Today | 53 (39-67 by content) | 1,536 |
| + D1 shadow | ~65-70 | ~2,000 |
| + P1 pipelining | unchanged | **~2,700-2,900** |
| + D2 drafter | **~70-75** | unchanged |

Past that: the pool is instruction-bound at the harness ceiling, the drafter is DSpark's horizon, prefill is
compute-bound at long context, and the next structural steps (16 GB card, or a CPU-prefill-expert kernel to
break the ~4,800 t/s PCIe ceiling) are hardware or major-project territory. Hardware remains the only route
to ~100+ tok/s decode.
