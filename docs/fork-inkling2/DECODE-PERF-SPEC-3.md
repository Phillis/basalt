# Specification 3 — Inkling-2-Mini: the drafter tax, the real remaining levers, and prefill

**For an implementing agent.** Follows `DECODE-PERF-SPEC-2.md` (W1/W2 landed; W3 closed; W4 implemented but
inert). Working tree `C:\LLM\pr536\basalt-inkling-cpu-spill-review`, build `C:\LLM\rebuild.bat`, canonical
launcher `C:\LLM\Basalt\Start-Basalt.cmd` (copies `C:\LLM\Basalt\basalt_config.json`; **that** file is the one to
edit), stop `Stop-Basalt.cmd`. The per-visit diag harness is permanent: launch with
`cmd /c "set BASALT_DIAG_VISIT=1&& ...\run-pr.cmd"` and read `engine.log` for `diag: N windows ...` lines —
**kill all basalt.exe and the C:\LLM python.exe before every A/B relaunch** (a leftover python server serves
requests with a stale engine and silently ruins the measurement).

## 0. What this session re-measured (all clean, no interference; corrects the record)

The 62.7-68.4 tok/s in `SUMMARY.md` is **content-dependent**: it was measured on tune.ps1's highly predictable
TCP-congestion prompt (96% draft acceptance). On ordinary content the same build does:

| Workload (greedy, warm, 600-token runs) | tok/s | acc | dl |
|---|---|---|---|
| Predictable technical prose (tune.ps1 prompt) | 56.4 | 76% | ~2.05 |
| Story / free-form prose | 47-51 | 69-81% | 1.7-1.8 |
| **`spec 0` (no drafting), any content** | **54.5** | — | 1.0 |

**DSpark drafting is now worth +3.5% at best.** The doorbell hand-off made single-row windows so cheap
(19.3 ms wall) that the drafter's per-window tax eats almost the whole benefit. Deepseek's W4 numbers
(sampled + relax: 55.3/44.8 vs 60.8 no-relax) are consistent with this: **any drafter whose fixed cost
exceeds ~13 ms/token of body time needs dl > ~2.5 just to break even — this one delivers 1.6-2.0.**

## 1. The current budget, measured two ways (clean diag, steady state)

**`spec 8` window (~2.05 tokens, ~37 ms total):**

| Term | ms/window | ms/token |
|---|---|---|
| CPU pool expert compute (~0.85 GiB blobs/window at ~45-50 GB/s) | 15.2 | 7.4 |
| GPU pre-expert chain (attention, dense, router, on-card experts) | 10.2 | 5.0 |
| Head | 0.95 | 0.5 |
| Host enqueue + host side | ~1.6 | 0.8 |
| **DSpark drafter block + engine, outside the window** | **~9** | **4.4** |

**`spec 0` window (1 token, 17-19 ms):** pool 7.5-8.2, GPU chain 9.75, head 0.97, host ~1.6.

Two structural facts fall out of the A/B:
1. **The GPU pre-expert chain is nearly flat in rows** (9.75 ms at 1 row vs 10.2 at 2) — the window kernels
   batch rows, so every extra accepted token amortises ~10 ms of GPU work for free.
2. **The pool term is linear in rows** (~7.5 ms/row at 45-50 GB/s) and the drafter block is a **fixed ~9 ms
   serial tax between windows** (`dspark.cpp` `draft_end` does a full `cudaStreamSynchronize`, and
   `draft_begin`'s attention runs uncaptured, per position).

**Token-time model (greedy):** `t = (13.5 + 7.5·rows + D)/accepted` where D is the drafter tax (9 now).
Today: `(13.5 + 15.4 + 9)/2.05 = 18.5 ms` → 54-56 t/s ✓. The model says exactly where the next gains are:
cut `D`, raise `accepted` beyond ~2.5, and cut the per-row pool term.

## 2. Why W4 (relaxed drafting) did not work — the deep dive

W4 was implemented correctly (host-side, opt-in, `session_engine.cpp` `accept_drafts`, no new readback —
`conf_` was already read back by `draft_end`). It failed for four reasons, none of them the implementation:

1. **The workload is greedy.** At temperature 0 the acceptance test *is* the argmax match; there is no
   "marginal draft" to relax — greedy acceptance was already 96% on the benchmark content. Relaxation only
   exists for sampled requests.
2. **Acceptance was never the binding constraint — dl was.** DSpark proposes a 3-layer drafter's next ~2
   tokens; after that its confidence collapses. Raising acceptance of the *first 1-2 drafts* cannot move dl
   past ~2.0. The ceiling is the drafter's horizon, not the acceptance policy.
3. **The drafter tax exceeds the marginal value of a kept draft.** At temp 1.0 without relax the engine
   drafts nothing (acceptance ≈ 0) and runs pure single-row decode at 60.8 t/s. Turning relax on drafts
   windows whose kept drafts are worth ~7 ms of body amortisation while the drafter block itself costs ~9 ms
   — a net loss, exactly the 55.3/44.8 they measured. Relaxation *widened* the funnel into a loss-making
   drafter.
4. **The kept drafts distort the sample for nothing**: `outv[a] = window[a+1]` replaces the target's own draw
   with the drafter's proposal whenever the drafter was ≥P confident — the output distribution shifts toward
   the drafter, and the relaxed keeps are counted as "accepted" in the stats, masking that real acceptance
   stayed near zero.

**Verdict: keep W4's code, keep it off, and stop pursuing acceptance policy as a speed lever.** The two
things that *would* raise dl are a cheaper drafter (W-A) and a drafter with a longer horizon on repetitive
content (W-B). If both land, relax becomes worth re-testing — at dl 3+, keeping marginal drafts finally pays.

## 3. Workstreams, ranked by measured value

### W-A — Cut the drafter block's ~9 ms/window (restores spec-8's value; ~1-2 days)
Profile one window's inter-window span first (add host timers around `draft_begin`/`draft_end` in the solo
request path, or reuse the serve-line closure). Suspects, in order:
1. **The full-stream sync in `draft_end`** (`dspark.cpp:611`) — replace with the doorbell pattern the main
   path already uses: the drafter's ids/conf are written to mapped pinned memory by its (captured) head, a
   host thread publishes them, no sync. At minimum, sync only the drafter's own event rather than the whole
   stream.
2. **`draft_begin`'s uncaptured attention** — it runs directly "because its window follows the position";
   capture per `full_keys`-style shape instead (the main session already keys graphs on the position's
   launch bound), so the drafter block is one graph launch instead of ~8 direct kernels + 3 graphs.
3. The screened head over `kDsBlock` rows (201k vocab, screened) and the meta H2D — batch them into the
   captured block.
Gate: drafter tax ≤ 3 ms/window → **spec 8 ≈ (13.5 + 15.4 + 3)/2.05 = 15.6 ms/token ≈ 64 t/s** on
predictable content; on story content (dl 1.75) ≈ 58. Serve-level 60+ on all content.

### W-B — Enable `--suffix-draft` for inkling2 (one-line flag parse; the only cheap source of dl > 2.5)
**Deepseek's "suffix-draft 8 → no gain" test was invalid: inkling2 does not parse `--suffix-draft`**
(`adapter.cpp` accepts only `--kv`, `--spec`, `--draft-log`; passing it trips the unknown-flag discipline or
is silently dropped), so nothing actually ran. The machinery is generic and already wired:
`session_engine.cpp:56/391/618` builds the `SuffixDrafter` whenever `o.suffix_draft > 0 && copy_drafts()`
(inkling2's `copy_drafts()` is true with `spec ≥ 1`).

1. Add `--suffix-draft` (and `--mtp-window` if trivial) to `dspark_parse_flag` in `src/models/inkling2/dspark.cpp`.
2. Validate on repetitive content: send a request whose answer must repeat a long passage from the prompt
   ("repeat the passage verbatim"); the lookup windows should give dl 3-8. With W-A landed, at dl 4:
   `t = (13.5 + 7.5·4 + 3)/4 ≈ 11.9 ms/token ≈ 84 t/s` — and the GPU chain amortises better the longer the
   match.
3. Free-form content keeps falling back to DSpark (the policy picks per window which drafter's rows pay).
4. If the PLE/n-gram table machinery (`--ple-*`, `src/ngram/`) is what qwen uses for cross-request lookup,
   evaluate it separately; the SuffixDrafter alone covers in-context repetition.

### W-C — The Q2_0 CPU shadow, *without* pinned RAM (corrects W5's blocker; ~1-2 days, quality-flagged)
`SUMMARY.md` closes W5 with "needs ~5.5 GB more **pinned** RAM, past the ~14.5 GiB ceiling". **That is
wrong**: the shadow is read only by the CPU pool's kernels (`nvfp4_avx512.cpp` etc. read plain host memory);
nothing on the card ever touches it while the CPU-spill path is active, and the doorbell hand-off publishes
*inputs*, not blobs. Allocate it as **ordinary pageable RAM** (commit, first-touch at build time); no
`cudaHostAlloc`, no quota, ~5-6 GB of the ~10 GB headroom. Only the spill copies themselves must stay
CUDA-pinned (the card maps them as a fallback path and prefill DMAs from them).
Mechanics otherwise as `DECODE-PERF-SPEC.md` §W1.3: requantize the spilled experts of the NVFP4 layers
(20-23 stay Q8_0, already cheap) at load, `--cpu-shadow q2|q4|off` flag-gated, q2 ≈ 2.7× fewer bytes and
cheaper decode. Per-row pool term 7.5 → ~3 ms:
- spec 8 + W-A + W-C: `(13.5 + 6.2 + 3)/2.05 ≈ 11.1 ms/token ≈ 90 t/s` predictable; ~80 story.
- plus W-B on repetitive content (dl 4): `(13.5 + 12 + 3)/4 ≈ 7.1 ms/token` — the first three-digit regime.
Quality is the price (2-bit on the ~28% of routed lookups that miss; q4 as the conservative middle if q2
hurts). User opt-in.

### W-D — Prefill (currently 1.1-1.3k t/s on unique 22k prompts; was 1.5-1.8k)
1. **The W1 reclaim/retake cycle taxes the first chunks**: the 22k run's interim reads show 443 t/s for the
   first ~3.7k tokens (staging re-taken, coldest copies demoted) before settling at ~1.0-1.3k. Keep a
   **standing staging floor** (e.g. 512 MiB never reclaimed) so short prompts never re-stage, and start the
   re-take asynchronously at the *end* of a decode burst (idle time) instead of synchronously at the next
   prompt.
2. Re-tune the prefill CPU share with the now-fast pool (the `prefill-cpu-share 0.85` override was removed;
   the engine default was tuned before the doorbell — sweep 0.6/0.85/1.0 again on a 16k unique prompt).
3. Retry `kPromptChunk` 4096 → 8192 on the current build (fewer staging cycles; the old "worse" measurement
   predates the doorbell and reclaim).
4. Measure with unique text only — the `s1 read pp=` lines in `server.out` are the truth; `engine.log`'s
   "N reused in 0 ms" lines are second-round restore stats and fool the eye.

### W-E — Diagnostics/accounting fixes (half a day)
1. **The hit-rate line can read >100%** (`194.5%` observed) — the per-(layer,expert) tally is double-added
   somewhere in the doorbell merge / reclaim promote path. Diagnostic-only, but it corrupts every "hit=" stat
   the server reports; fix before any further cache tuning is trusted.
2. Add drafter-span host timers (W-A's gate) and a `pool busy` line to the `diag` summary at `spec 0`
   (already correct) and prefill (absent — `prompt_.read` has no stamps; add chunk-level host timers if W-D
   is pursued).

### Explicitly not worth pursuing (re-confirmed)
- W3 kernel micro-opts (closed: engine pool within 12% of the harness ceiling on this dual-pumped Zen 5).
- Acceptance/drafting knobs (`spec` sweep, `draft-experts`, `mtp-window`) — all flat; the drafter's *cost*
  and *horizon* are the problems, not its tuning.
- Multi-card, mapped-RAM compute (6×), smaller pack (needs llama.cpp).

## 4. Order and gates

| Step | Gate |
|---|---|
| W-A drafter tax | timers show ≤3 ms/window; serve t/s ≥ 60 on story content; parity via greedy identity |
| W-B suffix-draft | flag accepted at load; repetitive-content run shows dl ≥ 3; free-form unchanged |
| W-C shadow (opt-in) | load unchanged in pinned bytes (verify with the `share ram` line); user accepts quality; t/s ≥ 75 predictable |
| W-D prefill | 16k unique ≥ 1.5k t/s sustained, TTFT not regressed on short prompts |
| W-E tally | `hit=` ≤ 100% across a mixed workload |

Full validation protocol as before (tests, greedy identity, `test-long.ps1`, sustained 5-min run, VRAM/RAM
lines, no `pinned staging alloc` failures) after every step; append results to `CHANGES.md` as Session 5.

## 5. Risks
- W-A touches the drafter's capture scheme (position-dependent attention) — keep the direct-launch path as a
  fallback flag; a bad capture hangs the first window (seen in the first doorbell attempt).
- W-C's pageable shadow must be first-touched at build time or the first decode after load pages in ~5 GB
  (a one-off stall); also verify the working set doesn't push it out mid-run (watch for sudden pool-time
  spikes in `diag`).
- All projections use the greedy model; the user's real workload mix (coding? chat? quoting?) decides whether
  W-B pays at all — measure on *their* content before investing beyond the flag parse.

---

## 6. Prefill: how to get past 2,000 tok/s (inspected, not benchmarked � written while another agent works the engine)

Current: **1,078-1,296 t/s** on unique 22k-token prompts (first chunks 443 t/s while the decode-time staging
reclaim is undone), 1.5-1.65k on a fresh engine. Stock was 1,452. Target: >2,000.

### 6.1 Where the time goes � from `prompt.cpp`, closed against the measured rates

Per chunk (`kPromptChunk` = 4096 rows), the walk runs **48 visits**, and *every visit of a layer with spilled
experts stages them before its expert products* (`prompt.cpp:419-421`):
`stage_from_card(l, stream_)` (from the elastic card copies, D2D) or `cudaMemcpyAsync(w.experts + kept,
w.spill_host, n_spill x stride, H2D)` � **on the compute stream, serially per visit**. The chunk's expert
products then wait on that copy.

Staging bytes per chunk (48 visits, ~115 spilled experts/layer, blob 4.48 MiB NVFP4 / 8.46 MiB Q8_0):

| Visits | Layers | Bytes/visit | Total |
|---|---|---|---|
| 44 | NVFP4 (0-5 once, 6-17 three passes, 18-19) | ~515 MiB | ~22.7 GiB |
| 4 | Q8_0 (20-23) | ~973 MiB | ~3.9 GiB |

**~26.6 GiB per chunk nominally � 70% of it the same 12 loop layers copied three times over.** The elastic
copies (728 after the reclaim) shave the D2D-covered part off PCIe; the rest is H2D at 15.75 GB/s � 1.25-1.7 s
per chunk, issued as ~40-60 ms serial stalls per visit with only that visit's attention/dense (~5-15 ms)
overlapping. Budget: DMA 1.3-1.7 s + expert products ~0.3-0.5 s + dense/attention ~0.3 s � 2.3-2.5 s per chunk
? 1,640-1,780 t/s on a fresh engine, less with the reclaim tax � **matches every measured number.**

Two corrections to Session 4's prefill conclusions:
- The `prefill-cpu-share` sweep (0.6/0.85/1.0 "flat") was **invalid: inkling2 does not parse
  `--prefill-cpu-share`** (qwen4exp-only; the server even logs "this model has no such flag, skipped"). There
  is no CPU share in inkling2's prefill � all experts stage to the card and compute there (FP4 tensor cores,
  `experts_fp4`, or MMQ). The flat sweep measured a no-op.
- GPU expert compute is **not** the wall (the FP4 block-scaled path reads staged weights once, ~0.3-0.5
  s/chunk). PCIe staging is.

### 6.2 The plan, ranked

**P1 � Take the staging off the compute stream and pipeline it (the whole win; ~1-2 days).**
1. A dedicated copy stream (or the existing `pf_st_`/copy-engine path) for `stage_from_card`/the H2D
   `cudaMemcpyAsync`, joined by an event before the visit's expert products.
2. **Half-layer ping-pong staging**: carve the existing ~1.03 GiB staging into two halves; while visit v's
   products read half A, prefetch visit v+1's staging into half B. Total staging bytes are unchanged (this is
   what makes it fit in VRAM), and the blob-table base for the products is already a parameter
   (`p.w`/`gu.base`), so alternating it per visit is mechanical. Prefetch across the chunk boundary too
   (chunk n+1's visit 0 during chunk n's coda).
3. Expected chunk time: `max(H2D 1.25-1.7, compute ~0.7) + tail` � 1.4-1.9 s ? **2,150-2,900 t/s** � clears
   2,000 even at the pessimistic end. Gate: `server.out`'s `s1 read pp=` on a unique 16k prompt = 2,000
   sustained, TTFT not regressed.

**P2 � Make the elastic copies cover what is staged most (cheap, compounds with P1).**
Each promoted copy stages D2D instead of H2D; a **loop-layer (6-17) expert's copy saves 3x its blob per
chunk**, a coda layer's only 1x. The promote ranking is decode-heat only today (`spill_order_`); weight it by
staging frequency (loop > coda) so the 728 copies concentrate where they save the most PCIe. Also: the
`prompt_stage.cpp` log lines already report staged bytes ("on the card" vs "in page-locked RAM") � read them
before/after to verify coverage.

**P3 � First-chunk fix (already spec-3 W-D):** a standing staging floor (~512 MiB) + re-take at idle instead of
at the next prompt. This alone is worth ~200-300 t/s of *average* prefill on follow-up prompts (the 443 t/s
first-chunk reads).

**P4 � Chunk size: do not raise it.** 8192 halves the staged bytes per token but the chunk arena
(`carve()`: ~a dozen R x E float buffers + attention scratch + expert q8/FP4 rows) roughly doubles � about
+1 GiB of VRAM, which does not exist at 192k (768 MiB free). After P1 the DMA share is hidden anyway; spend
the effort there. (Session 4's "prefill 8192 worse" test also predates the doorbell build.)

**P5 � Only if the PCIe wall must be broken: a CPU-prefill-expert path** (the qwen4exp-style pool assist does
not exist for inkling2 and cannot be copied as-is � the pool's multi-token kernels cap at nt=8, and a
4096-token chunk re-reads a blob once per 8-token group). It is a real kernel project (tile tokens per expert
with the decoded weights resident); modeled ceiling with the pool at ~45 GB/s in parallel with PCIe is
~4,500 t/s. Not needed for the 2,000 target.

**P6 � Re-measure at 130k+ context after P1.** The full-attention visits' cost grows with the prefix
(rows x prefix reads per chunk); at 130k it may add several hundred ms per chunk and become the second wall.
The 2,000+ figures above are measured-range (22k) projections.

### 6.3 Feasibility of the 2,000 target

The budget model reproduces all four observed rates (443 first-chunk, 1,078-1,296 with reclaim, 1,5-1.65k
fresh, stock 1,452) from one parameter set, and P1's projected 2,150-2,900 t/s is the `max(DMA, compute)`
collapse of the same budget � **2,000+ is reachable with P1 alone**; P2-P3 add margin; the PCIe wall caps the
pipelined design near ~2,400-2,900 until P5.
