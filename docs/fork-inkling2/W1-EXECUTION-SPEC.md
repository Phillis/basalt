# W1-EXECUTION-SPEC — wave-pipelined prefill staging, end to end

**For the implementing agent, written to be followed without further context.** Companion docs:
`NEXT-STEPS-SPEC.md` (the two-workstream plan), `HANDOVER-PREFILL.md` (the corrected budget),
`IMPLEMENTATION-NOTES.md` (the trap list — read §1 before touching anything).

**Mission:** take unique-prompt prefill from ~1,450 t/s to **>= 2,400 (expected ~2,700-2,900)** with
wave-pipelined expert staging, with **zero quality cost** — and with a verification protocol designed so that
the last attempt's failure mode (**NaN logits -> constant-token "!!!!!" output**) cannot pass unnoticed.

**Communication protocol for this session:** a second agent (pi) is monitoring your session transcript and
the engine state while you work. It will verify your results independently and write corrections/answers into
`C:\LLM\Basalt\W1-EXECUTION-NOTES.md` as needed — check that file if you hit a wall; it may already contain
the answer. Do not restart the engine while it is mid-verification (it will say so in that file).

---

## 0. Ground rules (every A/B in this tree has been ruined by skipping one of these)

1. **Kill everything between A/B relaunches** — all `basalt.exe` AND the `C:\LLM` `python.exe`
   (`ExecutablePath -like 'C:\LLM\*'`). A stale python server serves requests with a previous engine.
2. **One change, one measurement.** No bundling.
3. **Unique text only** for prefill benchmarks (random 6-char words; ~87 kB ≈ 53k tokens). The truth is
   `server.out`'s `s1 read pp=<new>/<total> <t/s>` lines — NEVER `engine.log`'s "N reused in 0 ms" restore stats.
4. Cross-process greedy divergence is NORMAL on this machine (expert placement differs per start). Do not
   chase bitwise A/B across engine restarts; use rate lines + coherence + the needle test.
5. Standing constraints: pinned ceiling ~14.5 GiB (~150 MiB headroom, probe `C:\LLM\bwtest\hostalloc_seq.exe`);
   the scheduled task cannot launch this build (launch directly); `ram-reserve-mib 512`; the model is **v2**
   (`Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt`, needle answer `ZEPHYR-4417`).
6. Current build state to know about: PR #555 (upstream prefill optimization) is merged; `kCardSpare` is 736
   and `vram-reserve-mib` is 672 — **experimental leftovers from the #555 debugging, see Phase 0**.

---

## 1. Why the retracted "W1 pipeline" does not kill this design (read before implementing)

The withdrawn attempt (Session 13) prefetched **layer L+1's** experts while layer L's products still read the
staging. Invalid: the staging is ONE physical 1034 MiB buffer aliased into every layer's virtual range
(`model.cpp`: the same granule handles mapped into every `Range`), so the prefetch overwrote the bytes being
read -> NaN logits -> the sampler emitted a constant token ("!!!!!!!"), and draft acceptance read 100%
because a constant token is trivially predictable.

**This design cannot hit that failure mode, by construction.** It stays **within one visit**: the current
layer's spilled experts are split into waves; wave *i*'s copy writes slot ranges that this visit's already-
enqueued products are not reading, and **no other layer touches the buffer until this visit completes**. The
cross-layer aliasing that killed the prefetch is never engaged. The retracted build's +73% timing is,
moreover, direct evidence that hiding the DMA delivers at least as much as this spec projects.

**Hard rules that fall out of this:**
- Never issue a staging copy for a layer whose products are not all enqueued (no cross-layer prefetch, ever).
- Any suspicion of corruption: stop and run the anti-NaN checks (§5) IMMEDIATELY — do not trust acc/dl.

---

## 2. Phase 0 — resolve the baseline first (half a day; gates everything)

The tree carries unresolved experimental state: `kCardSpare 736`, `vram-reserve-mib 672`, and an unexplained
~8% decode regression attributed to the #555 build (52.5-53.4 vs 57.9-58.2 warm on the pre-#555 v2 build —
though v2 itself costs ~3 t/s vs v1's drafter drift).

**Do this A/B, on the current #555 build:**
1. Variant A (as-is): `kCardSpare 736` (rebuild if changed — it is in `model.cpp:35`), `vram-reserve-mib 672`.
2. Variant B: `kCardSpare 768`, `vram-reserve-mib 768` (the previously measured constants; needs a rebuild).
3. Each variant: confirm the load plan line (`experts: N of 3072 in RAM` — must stay <= 2746; if 768 fails to
   load reliably, that is the answer, stay at 736), then warm decode x3 (400-token prose, unique), unique-16k
   prefill x3, and watch for `pinned staging alloc ... failed` / `cannot select device 0` (the misleaders).
4. Pick the variant with better warm decode, load reliability FIRST (a marginal prefill/decode win is worth
   nothing if loads are flaky), and record the choice in `CHANGES.md` as the baseline constants. Update the
   `model.cpp:35` comment to state the final value and why.

**Gate for Phase 0:** a chosen baseline with load reliability confirmed (3 consecutive clean loads), warm
decode and unique-16k prefill numbers recorded in `CHANGES.md`. Do not start Phase 1 without this.

---

## 3. Phase 1 — wave-pipelined staging (1-2 days)

### 3.1 What exists already (reuse, do not rebuild)

The retraction left the machinery in place (currently marker-gated behind
`C:\LLM\prefill-prefetch-experiment.on`, documented as known-invalid):
- `InklingModel::stage_layer(int l, void* stream)` (`model.hpp:108`) — the per-visit staging call with a
  stream parameter, called at `prompt.cpp:459`.
- `Prompt`'s `copy_` stream, `stage_ev_` events, and `sync_staging()` (`prompt.cpp:137-214, 568-571`).

**Remove the cross-layer prefetch machinery entirely** (the `stage_issued_`/`sync_generation`/prefetch
plumbing is dead and dangerous to leave lying around), then rebuild the same stream/events as the wave
machinery. Keep `sync_staging()` — anything that frees or re-takes the staging (the decode-time reclaim,
chunk boundaries before a re-take) must call it; it already exists and is already called.

### 3.2 The design

Per visit with spilled experts (`prompt.cpp` `experts()`, the `if (w.slot_map)` block at ~L459):

1. **W waves of the layer's spilled slots** (start W=3; sweep 2/3/4 in §3.5). The layer's `n_spill` spilled
   experts divide into W contiguous slot ranges. Wave boundaries: multiples of `n_spill / W` — prefer
   granule-aligned (the staging granularity is 2 MiB VMM granules; aligning wave edges avoids one straddling
   copy, cosmetic only).
2. **Enqueue order per visit (the heart of it):**
   ```
   router_select(...)                          // as today
   for i in 0..W-1:
       stage_layer(l, i-th slot range, copy_)  // the wave's H2D/D2D on the COPY stream
       cudaEventRecord(stage_ev_[i], copy_)
       enqueue wave-i products on stream_       // stream_ waits stage_ev_[i], then the product launch
   cudaMemsetAsync(guard) before wave-0 products (compute stream, as today)
   moe_combine after the last wave, as today
   ```
   The copy stream walks ahead of the compute stream: wave-1's DMA overlaps wave-0's products, wave-2's
   overlaps wave-1's, etc. The first wave's DMA overlaps only this visit's attention/dense (~5-15 ms) — the
   per-chunk cost is one wave's DMA (~13-20 ms) as the pipeline fill, amortized over the chunk.
3. **Wave-i products: the fiddly part — two acceptable implementations; try (a) first.**
   (a) **Zero-range filtered bounds (preferred, no re-sort):** run `moe_sort` once as today; then build W
   small per-wave bounds arrays (each `n_expert` ints, host or a tiny prep kernel) where experts outside wave
   i have zero-length ranges. The MMQ product (`mmq::Product`, `p.bounds`) iterates experts via bounds and
   skips empty ones; the FP4 path additionally needs per-wave `tile_off/blk_off` — re-run
   `fp4::plan_blocks` per wave on the filtered bounds (it is a cheap planning kernel). The
   `tok/dst/exp/expert-byte` arrays are untouched. Assert total pairs over waves == pairs (a checksum in
   debug).
   (b) **Fallback if (a) fights you:** run `moe_sort` + products per wave over the wave's expert range
   (filter `ids` before the sort by slot). More launches, still correct, slower — only acceptable if it
   still clears the gate.
4. **`stage_layer` extension:** add `(int e0, int e1)` slot-range parameters; it copies only the experts in
   range — from the elastic card copies (D2D) where they exist, else RAM (H2D), on the given stream. The
   per-expert copy loop stays; if its launch overhead dominates at W=3, batch the contiguous RAM-only runs.
5. **The SwiGLU tail, the Q8_0 layers, the FP4 path:** all unchanged — per-blob tails are read at
   `base + L.bytes - 16` regardless of waves; Q8_0 layers (20-23) use the same wave treatment (their staging
   is ~965 MiB/visit, so they benefit most); `experts_fp4` and the MMQ path both get the wave loop.
6. **Chunk boundaries:** the last wave's products of the chunk's last visit complete before the next chunk's
   first staging is issued — nothing to do. The decode-time staging reclaim / prompt re-take already calls
   `sync_staging()`; verify with one grep that every `release`/`acquire` path still does.
7. **Flag:** `--prefill-waves N` in `dspark_parse_flag` (`dspark.cpp:~722`), default **3**, `0` = today's
   serial staging; note it in `help.cpp`. It is an engine CLI flag, **not a config-JSON key** — pass it via
   `run-pr.cmd %*` for tests. After the gates pass, also honor it in whatever launcher the engine uses daily
   and record that in CHANGES.

### 3.3 What NOT to do

- No cross-layer prefetch (the retracted design) — ever.
- No second staging buffer (2x1034 MiB does not fit VRAM).
- No `kPromptChunk` change; no CPU-share anything; no shadow quantization (user-declined); no decode-kernel
  changes. If you find yourself touching decode kernels, stop and re-read §1.

### 3.4 Build order

1. Rip out the dead prefetch plumbing; keep `copy_`/`stage_ev_`/`sync_staging` (rebuilt as always-on, not
   marker-gated — the marker file's meaning changes: see §3.2.7).
2. `stage_layer` slot-range + stream parameters.
3. The wave loop in `experts()` with **serial-fallback semantics** (`--prefill-waves 0` must reproduce
   today's behaviour byte-for-byte — this is your A/B control and your escape hatch).
4. Bounds slicing (a), fallback (b) only if (a) fails.
5. Rebuild (`C:\LLM\rebuild.bat`), then straight to §5 before any benchmarking.

---

## 4. The anti-!!!!! protocol (Gate 0 — run before ANY benchmark)

The failure signature of corrupt staging: NaN logits -> `logprob=-9999.0` on every token (`logprobs.hpp:25`'s
non-finite marker) -> the sampler emits a constant token -> output is "!!!!!!..." -> and the stats LIE
(draft acceptance reads 100% because a constant token is perfectly predictable; `dl` looks healthy).

**After every build and every flag change, in this order, before touching throughput:**
1. **Coherence eyeball (non-negotiable):** three factual prompts, read the answers —
   "What is the capital of France?" (Paris), "What is 17*23?" (391), "Name three rivers of Europe."
   If every answer is the same repeated character — STOP, the staging is corrupt; check wave boundaries first.
2. **logprob sanity:** one request with `logprobs` on — no `-9999.0` markers, alternatives present.
3. **acc sanity:** on 400-token *prose*, acceptance must be < 100% (typically 73-83% on v2). acc=100% with
   stable dl is a corruption signature, not a success.
4. **Needle:** `test-long.ps1 -TargetTokens 131072` — must answer `ZEPHYR-4417`.
Only after all four: proceed to throughput.

---

## 5. Measurement plan and gates

**Primary (prefill):** unique-text 16k prompt (and one 53k for the long run), 3 runs each, `--prefill-waves
0` vs `3` (and sweep `2`/`4` if time permits), reporting the `s1 read pp=` t/s lines.
- **Gate: >= 2,400 t/s sustained at W=3** (expected ~2,700-2,900). If you land 2,000-2,400, stop and record —
  do not chase the ceiling by breaking rules from §3.3; report the gap and the diag evidence instead.

**Secondary (must not regress):**
- Warm decode, 3x 400-token prose, unique: within ±5% of the Phase-0 baseline.
- TTFT on a short (50-token) unique prompt: unchanged.
- 5-minute sustained decode + one long conversation (prefill -> decode -> new prefill -> decode): no stalls,
  no `staging reclaim` errors, `hit=` sane (it is trustworthy now — Session 8 fixed the tally; must stay
  <= 100%).
- Load reliability: 3 consecutive clean loads on the final constants.

**Instrumentation if you need to attribute a gap:** per-chunk host timers around the wave loop, or one-off
CUDA events at first/last wave — but remember the budget: DMA ~1.33 s, compute ~1.3-1.39 s per chunk; the
pipelined floor is `max(...)` + one wave-fill. If measured chunks sit at ~1.5 s, you are done; if they sit at
~2.2-2.4 s, the copy stream is not walking ahead (event ordering bug) — check that wave-i+1's copy is issued
BEFORE wave-i's products are enqueued, not after.

---

## 6. On completion

Append Session N to `CHANGES.md` with: the Phase-0 constants decision, the wave implementation summary, the
anti-!!!!! results, the prefill/decode/TTFT/needle numbers at each W, and the final flag default. Update
`SUMMARY.md`'s prefill row and `NEXT-STEPS-SPEC.md`'s W1 section to "done". Leave `--prefill-waves 0` working
as the documented fallback. The monitoring agent will independently re-run: coherence, needle, unique-16k
prefill, warm decode, and a sustained run before signing off.
