# Specification — Decode performance for Inkling-2-Mini on the custom Basalt fork

**Goal:** raise decode throughput from the current ~35 tok/s (256k context) to **50+ tok/s**, software only
(no RAM/GPU hardware changes). This document is written for an implementing agent: it contains the full
measured context, the derived token-time budget, ranked workstreams with exact code pointers, a validation
protocol, and the risks. All measurements below were taken on this machine (ASUS G14 GA403WR, Ryzen AI 9 HX
370 12C/24T, 4×8 GB LPDDR5X-8000 = 128-bit, RTX 5070 Ti Laptop 12 GB, Windows 11).

**Working tree:** `C:\LLM\pr536\basalt-inkling-cpu-spill-review` — Basalt fork @ PR #536 branch
(`inkling-cpu-spill-review`) with two local patches already applied (keep them, see §1.4).
Build with `C:\LLM\rebuild.bat`. Run with `C:\LLM\Basalt\Start-Basalt-Fast.cmd` (scheduled task `BasaltPR`).
This document supersedes the "50 tok/s is impossible without new hardware" conclusion in
`C:\LLM\pr536\PR-FINDINGS.md` §"What would actually reach 50": that conclusion was based on the belief that
the CPU expert path was DRAM-bandwidth-bound at ~29 GB/s. Direct micro-benchmarking (§2.3, §2.4) shows this
machine streams 47 GB/s on **one** core and that the NVFP4 CPU expert kernel is **instruction-throughput
bound**, not bandwidth bound. There is roughly 2–3× of recoverable headroom in software.

---

## 1. Baseline and inventory

### 1.1 What runs today
| Item | Value |
|---|---|
| Engine | Custom build of the fork, `-DBASALT_MODELS=inkling2`, MSVC 19.44 / CUDA 13.5 / Ninja |
| Model | `C:/LLM/Basalt/models/Inkling-2-Mini-Unbound-NVFP4-q8.basalt` (18.38 GiB, 3072 experts) |
| Config | `basalt_config.json` in the tree: ctx 262,144, kv fp8, spec 0, expert-cache auto, vram-reserve 1024 MiB, ram-reserve 512 MiB, prompt-cache 6 |
| Decode | **34–38.8 tok/s** (server log; short cold chats dip to 28) |
| Prefill | 1,770–1,804 tok/s (16k prompt) |
| Expert hit rate (decode) | 74.8–78.8% long-run, 66.4% cold |

### 1.2 Model geometry (read from the pack's `meta` section, verified)
- 24 layers; **loop layers 6..17 are visited 3× per token** (`loop.count = 3`) → **48 expert visits per token**.
- 128 experts/layer, top-8 routed (`expert_used_count = 8`) → **384 expert lookups per token**.
- Expert blobs: layers 0–19 **NVFP4** (gu_type = d_type = 40), **4,478,992 B** each
  (gate 1152×2304 + up 1152×2304 + down 2304×1152 values @ 36 B per 64-value block, +16 B fp32 tail);
  layers 20–23 **Q8_0**, **8,460,288 B** each. Total experts section 14.71 GiB of 17.98 GiB pack.
- n_embd 2304, n_ff_exp 1152, vocab 201,024, attention: 24 Q / 8 KV heads × 128, SWA window 1024,
  sliding-window pattern on 20 of 24 layers (4 full-attention layers per pass), rel states 16 over 1024.
- DSpark drafter present in the pack (3 blocks, dense), currently `spec 0`.

### 1.3 Memory placement at 256k context (from engine log)
- 2,698 of 3,072 experts spill to pinned RAM at load; the elastic spill then **promotes 703 hot ones back as
  card copies (3.11 GiB)** → ~1,077 of 3,072 experts (35%) readable on-card.
- VRAM: experts ~2.6 GiB + kv 4.0 GiB (fp8) + "other" ~6.7 GiB + free 1.0 GiB (the reserve).
- Decode lookups: ~21% miss → **~81 CPU expert computations per token**.

### 1.4 Patches that must not regress
- **Patch A** (the 3× win): the link gate removal in `src/models/inkling2/cpu_spill.cpp`
  `CpuSpill::make()` — keep the `[local] link gate removed` line; `cpu_spill.cpp.orig` is the pristine PR file.
- **Patch B** (PR #525, prompt-chunk staging from card copies): `model.cpp/.hpp`, `prompt.cpp` hunks.

---

## 2. Measured facts (all on this machine, engine stopped unless noted)

### 2.1 Raw DDR5 streaming read (new benchmark, `C:\LLM\bwtest\bw_bench.cpp`, AVX-512 sum over 4 GiB)
| threads (pinned 0..n-1) | GB/s |
|---|---|
| 1 | 47–49 |
| 8 | 60 |
| 12 | 76–78 |
| 16 | 86 |

→ The chip's practical aggregate read bandwidth is **60–86 GB/s**, not ~29 GB/s as previously assumed.

### 2.2 Engine CPU-spill cost (prior instrumentation, `C:\LLM\pr536\PR-FINDINGS.md`)
- ~0.155 ms per expert visit, linear in experts; 8-on-CPU visit = 1.26 ms; 0-expert visit = ~1 µs.
- The card never waits on the CPU (`wait = 0.000 ms`); GPU util 21–43% during decode.

### 2.3 Isolated NVFP4 CPU expert kernel (new benchmark, `C:\LLM\bwtest\nvfp4_bench2.cpp`)
Compiles the engine's own `src/kernels/cpu/nvfp4_avx512.cpp` and drives one real-shape expert
(gu rows 0..1152, down rows 0..2304, nt = 1) with persistent pinned worker threads:
| workers | ms/expert | aggregate GB/s |
|---|---|---|
| 1 | 0.332 | 13.5 |
| 4 | 0.149 | 30.0 |
| 6 | 0.119 | 37.6 |
| 8 | 0.129 | 34.7 |
| 12 | **0.093** | **48.0** |

Readings:
1. The engine's ~0.155 ms/expert **is** the kernel cost (7 workers + host ≈ 8-thread row above + pool overhead).
2. **Single-core kernel throughput is 13.5 GB/s vs 47 GB/s pure-stream on the same core** → the kernel is
   instruction-throughput bound (4-bit decode + fp8-scale reconstruction + VNNI dot ≈ 0.5 op per blob byte),
   NOT memory-bandwidth bound. The PR-FINDINGS "0.155 ms = the machine's practical DDR5 bandwidth" inference
   was wrong; 0.155 ms is ~29 GB/s only by coincidence.
3. **More workers help** (12 workers = 1.67× over 7). The engine's earlier `pool-workers 11` regression
   (33.6 vs 35.1) is a scheduling/affinity artifact (see W1.1), not a hardware wall.

### 2.4 Link and placement facts
- Host↔card link 15.75 GB/s; GPU-compute-over-mapped-RAM is ~6× worse than CPU-compute-from-RAM
  (measured, PR-FINDINGS "GPU_PCT" experiment, reverted). **Do not revisit GPU-side compute of spilled experts.**
- Pool workers spin 20 ms before sleeping (`kSpinBeforeSleep`), so during continuous decode there is no
  wake-up cost; between long idle gaps workers sleep (fine).
- Auto worker count picks 7 (topology: 4 Zen5 + 8 Zen5c "hybrid" heuristic → `p_cores-1 + e_cores/2`).
- The session already overlaps: next-visit router on a side stream (`ra_st_` + `kGuessExperts`), GPU-L2
  prefetch of guessed experts (`expert_prefetch`, `pf_st_`), and previous-window expert read-ahead
  (`expert_read_ahead`, `rec_blobs_`). Nothing more to win there on the GPU side.
- Speculative decoding windows do **not** amortize CPU expert reads for this model: with top-8-of-128
  routing, a k-row window touches ~128·(1−(1−8/128)^k) distinct experts per visit (k=2 → 15.6 vs 16);
  the blob read and decode work scale with distinct experts. DSpark measured neutral (34.9 vs 34.6) —
  it *steals VRAM from expert copies*, which is a net loss at this VRAM pressure.

---

## 3. Token-time budget at 256k (per token, ~28 ms, derived; confirm in Phase 0)
| Component | Est. | Basis |
|---|---|---|
| CPU expert compute | **~12.6 ms (45%)** | ~81 misses × 0.155 ms |
| GPU compute (attention, glue, router, GPU experts, head) | ~9 ms (32%) | GPU util ~31% avg |
| Per-visit hand-off, launches, D2H/H2D, sampling, host loop | ~6–7 ms (23%) | remainder |

Attack order follows this table: the CPU term (W1), then the hit-rate term that shrinks it (W2), then GPU/
overhead trims (W3). Projected compounding:
- W1.1+W1.2 (CPU path 2×+): token ≈ 21 ms → **~48 tok/s**
- +W2.1 (hit 78.8→~85%): token ≈ 19.5 ms → **~51 tok/s**
- +W1.3 optional (shadow quant, 2× CPU again) or W3 trims: **55–60 tok/s**

---

## 4. Dead ends — do not spend time here (all measured)
- GPU computing spilled experts over PCIe/mapped RAM (6× worse).
- Splitting spilled experts between CPU and card paths (21.1 vs 35.1 tok/s).
- `route-prior`, `route-tail-skip` (not implemented for inkling2), `adapt-swaps`, `prefill 8192`,
  pool-affinity p-cores as-is, DSpark at current VRAM pressure.
- Multi-row decode windows as a CPU-bytes lever (§2.4 last bullet). (W4 re-evaluates windows only as a
  GPU-side amortization *after* W1/W2 land, and only if profiling shows GPU time still matters.)
- Any hardware/VRAM-quantity change (out of scope by requirement).

---

## 5. Workstream W1 — make the CPU expert path 2–3× faster

### W1.1 Pool sizing and core placement (quick win, hours)
**Why:** isolated bench shows 12 pinned workers = 0.093 ms/expert vs engine's 0.155 at 7 workers; the engine's
`pool-workers 11` regression is a placement artifact, not a wall.

Changes (in `src/models/inkling2/cpu_spill.cpp::make` + `src/kernels/cpu/pool.cpp` + topology detection in
`src/kernels/cpu/…` where `detect_cpu_topology` lives):
1. Default inkling2 CPU-spill workers to **all physical cores (12 here)**, not the hybrid heuristic's 7.
   Keep honoring an explicit `--pool-workers` override.
2. **Pin the worker threads to logical CPUs 0..11 (physical cores)** with `SetThreadAffinityMask` at thread
   start (pool threads are created in `ExpertPool::ExpertPool`); currently no worker pinning exists — the
   OS floats them across SMT siblings and the scheduler collides them with the CUDA host-func thread,
   the Python server, and GPU driver threads.
3. Leave SMT siblings (12..23) to the host: the CUDA callback thread that runs `Impl::execute` →
   `run_split_multi_native` and the server. Consider pinning that thread (it is a CUDA driver thread;
   if it cannot be pinned directly, prefer keeping workers off SMT and letting the OS balance).
4. Re-tune task granularity for 12 workers: `kTasksPerThread` (pool.hpp, = 12) with
   `fused_tail_ = threads` quarter-parts — re-measure 8/16/24 parts-per-thread; smaller parts reduce tail
   imbalance on the heterogeneous Zen5/Zen5c mix, but each part currently costs a claim + a
   `wait_ready`/`prefetch_until_ready` gate.
5. Guard against thermal regression: the pool now saturates the die. Verify **sustained** (≥5 min) decode
   tok/s, not 30-second bursts; the laptop may clock down. If sustained regresses, expose
   `--pool-workers` default = 8 and document.

**Acceptance:** engine-level decode ≥ 42 tok/s at 256k sustained, quality gates pass. If in-engine gain is
< 1.3×, stop and instrument (Phase 0 I3) before continuing — the gap vs the isolated bench is then pool
overhead, not worker count.

### W1.2 NVFP4 kernel micro-optimization (1–2 days)
**Why:** single core does 13.5 GB/s of expert work vs 47 GB/s streaming; ~0.5 instructions per blob byte, so
instruction throughput is the cap. The engine repo holds the kernel in
`src/kernels/cpu/nvfp4_avx512.cpp` (decode + rows_dot + gu_rows/down_rows), reached through
`native_expert.cpp`'s `native_gu_rows`/`native_down_rows` (check the dispatch in
`include/basalt/kernels/cpu/native_expert.hpp` / `src/kernels/cpu/expert_layout.cpp`).

Optimization targets, in expected-yield order (measure each against the harness in §Validation):
1. **Split the per-block scale reconstruction out of the row loop's critical path.** `decode()` burns ~6 of
   its ~12 ops on the 4 scalar `kLut.v[x->d[i]]` loads + `_mm_setr_ps` + `permutexvar_ps`. Replace with a
   vector reconstruction of the UE4M3→fp32 bits (`((e+119)<<23)|(m<<20)` is already what the LUT encodes;
   for e==0 the denormal path needs the LUT or a fixup) or a two-level `vpermi2b` over 128-entry tables.
   Both scale lanes and codes are constant per block; restructure so the *code* expansion (shuffle-port
   bound) and the *scale* expansion (ALU-port bound) interleave without serializing.
2. **Two-block unroll** in `rows_dot` (process ib and ib+1 per iteration) to break the `acc` FMA latency
   chain: today each (row, token) accumulator has a ~5-instruction dependent chain per 36 B block; with
   R=2, NT=1 there are only 2 independent chains.
3. **R=4 in gu_rows** (two gate rows + two up rows per iteration) — halves the per-block `y`/`dv` loads and
   doubles ILP; `down_rows` similarly (it already pairs 2 rows). Register pressure check on Zen5c.
4. Software-prefetch the next block pair (`_mm_prefetch T0`, 1–2 blocks ahead) — rows are 1296 B (not
   cache-line aligned per row; gu_row = 20.25 lines) so hardware prefetch is imperfect.
5. Only if the above stalls on memory: non-temporal (`_mm512_stream_ps`) stores for `ff`/`out` are NOT
   wanted (they are re-read immediately); leave loads alone.
6. Mirror the same wins into the AVX2 path only if trivial (Zen5c executes AVX-512 dual-pumped; the AVX-512
   build is the one that runs here).

**Parity requirement:** the repo's stated contract is *bitwise equality with ggml's dot* (see the header
comment of `nvfp4_avx512.cpp`). Restructuring must preserve per-token accumulation order across blocks
(sum over blocks in the same order, same dpbusd lane grouping). Validate with the existing CPU kernel
tests (`tests/kernels/`, `tools/greedy_identity.py`) plus a new unit test that compares old vs new kernel
outputs on randomized blobs (see §Validation). Bit-identical is achievable because the changes above only
reorder *independent* work; if a variant cannot be made bit-identical, ship it behind a flag
(e.g. `--cpu-expert-kernel fast|exact`) and default to the parity-preserving one.

**Acceptance:** isolated harness: single-core ≥ 20 GB/s (from 13.5), 12-worker ≥ 65 GB/s (from 48).
Engine: with W1.1, decode ≥ 48 tok/s.

### W1.3 (optional, quality-flagged) Q2_0 CPU shadow copies of spilled experts
**Why:** half the bytes *and* cheaper per-byte decode (Q2_0: ~10 ops/32 B ≈ 0.31 op/B vs NVFP4's 0.53) →
~2× the CPU expert path on top of W1.1/W1.2. The repo already ships fast Q2_0 CPU expert kernels
(`src/kernels/cpu/q2_avx2.cpp`, `q2_avx2_rows.inc`, dispatched by `q2_native_kernels` in `pool.cpp`).

Design:
- New flag `--cpu-shadow q2|off` (default off). At load, after the spill copies land in pinned RAM, build a
  Q2_0 shadow of **only the spilled experts of the NVFP4 layers** (2698 experts × ~2.05 MB ≈ **5.5 GB
  additional pinned RAM**; Q8_0 layers 20–23 keep their blobs — already cheap to decode). Include the shadow
  in the RAM floor accounting (`ram-reserve`, the `cannot pin` guard in the spill plan) and refuse to enable
  when RAM headroom is insufficient.
- `CpuSpill::Impl::open` builds a second `NativeFmt` (Q2_0 layout, same softcaps) for those layers;
  `run()` uses the shadow blob pointer + shadow format when present (add an accessor on `InklingModel`,
  e.g. `shadow_blob(l,e)`, mirroring `host_blob`).
- Quantize from the NVFP4 data (dequant → min/max per block). Do it load-time in parallel (a few seconds).
- Quality: the shadow replaces the weights only for experts that miss the card cache — the coldest ~20%
  of routed lookups. Expose `--cpu-shadow q4` as a middle option (same bytes as NVFP4, ~1.5× cheaper ops,
  negligible extra loss) if q2 proves too lossy.

**Acceptance:** greedy-identity spot checks remain acceptable to the user (this is a *tunable quality*
feature, off by default; document measured perplexity deltas if a cheap eval can be run, otherwise stick to
greedy-identity + needle test). Decode ≥ 55 tok/s with the shadow on.

---

## 6. Workstream W2 — convert idle VRAM into expert hit rate

Every ~4.4 MB of VRAM moved into elastic-spill copies ≈ one more expert on-card ≈ −0.155 ms (post-W1:
−0.06..0.09 ms) on every token that routes to it. Long-run hit is 78.8% at ~35% residency.

### W2.1 Reclaim the prompt staging during decode (~1.05 GiB, 1 day)
The spill plan reserves, on the card, a per-layer staging sized to a *whole layer's expert set*
(`src/models/inkling2/spill.cpp`, `tail(l) = round_up(n_expert·stride + kSlotGuard) − own(l)`,
`staging = max over spilled layers`) — that is ~1.05 GiB (largest layer is a Q8_0 one: 128 × 8.46 MB).
It exists so a **prompt chunk** can copy that layer's RAM experts in for GPU prefill; during pure decode it
is idle (`model.cpp` maps it via `core::vmm` handles, `staging_`; the load log even prints it as part of the
device budget, and `model.cpp:346` enforces it for `--max-context`).

Implement demand-growth:
1. At load, allocate staging as today (prefill needs it immediately).
2. After prefill completes for the running request (or on first decode-only window), **release the staging
   mappings** and hand the freed bytes to `InklingModel::promote()` (the elastic-spill copy machinery,
   `model.cpp:517`) via the existing `keep_free` mechanism.
3. When a new prompt chunk needs staging again, call `demote()` to free copies (the ranking already knows
   the least-routed first), remap the staging, prefill, then re-promote. The elastic spill already
   grows/shrinks around the KV cache this way (`session.cpp:349–378`) — reuse the same lifecycle hooks.
4. Keep a small standing staging (one chunk's typical need, e.g. 256 MiB) if remap latency proves bad;
   measure remap (`core::vmm`) cost once and choose the threshold.

**Acceptance:** `elastic spill: N of 2698 RAM experts copied back` line grows by ~200–240 experts at steady
decode; long-run hit ≥ 85%; decode +2–4 tok/s on top of W1.

### W2.2 Online re-ranking of promoted copies (cheap, high leverage for cold starts)
The promote ranking is the shipped static one; a fresh conversation that routes differently starts at
66.4% hit. The session already counts per-visit lookups on the card (`ex_tally_`, used for the hit-rate log,
`session.cpp:62–68`). Add: every N decode windows (e.g. 512) while otherwise idle, feed the live tally into
the promote/demote ranking (same eviction order) so the copy set tracks the *current* conversation.
Do not do this concurrently with active prefill; reuse the elastic-spill sync points.

### W2.3 VRAM audit (half a day, informational)
Instrument and print the components of the "other" ~6.7 GiB (session fixed 768 MiB, dense weights, head
492 MB, staging 1.05 GiB, CUDA graphs, cuBLAS/workspace, driver). Anything idle ≥ 256 MiB gets the same
treatment as W2.1. Do not touch the KV cache format (fp8 is the model's floor) or `vram-reserve-mib` below
what `expert-cache auto` needs (512 MiB over-commits today).

---

## 7. Workstream W3 — GPU and per-visit overhead trims (after Phase 0 profiling)
Do these only with profile evidence (Phase 0 I4). Candidates, in order of suspicion:
1. **Per-visit hand-off cost**: each of 48 visits does `cudaEventRecord` + 2 async D2H copies (input 9 KB +
  ids 32 B) + `cudaLaunchHostFunc` + a finish-merge kernel launch. 0-expert visits cost ~1 µs (measured), so
  the win here is bounded — confirm before touching.
2. **Head**: 201,024 × 2304 Q8_0 = 492 MB read per sampled token (~0.8 ms at VRAM speed). Unavoidable per
  sample, but check it is not re-run for draft windows / logprobs more than once.
3. **Full-attention visits at 262k**: verify the KV read cost of the 4 non-SWA layers dominates the GPU term;
  if the glue/attention kernels under-utilize (tail-latency at 8-row windows), batch shape is a likely
  cheap fix (kMaxRows = 8, `glue.cu` RMAX comment).
4. Reduce `kSessionFixed`-shaped scratch if the audit shows slack (768 MiB per session is generous for
   ctx≤262k, single session).

---

## 8. Workstream W4 — speculative decoding re-evaluation (stretch, after W1+WW2)
Only worth revisiting once the CPU term is ≤ ~5 ms/token:
- DSpark remains VRAM-negative at 256k. At 128k it fits; re-measure after W2.1 (staging reclaim may make
  room for the drafter that did not exist before).
- Prompt-lookup (`--suffix-draft N`, `src/ngram/ple_reader.cpp`, `src/spec/`) needs **no drafter VRAM**, but
  `inkling2`'s `max_window()` returns 1 without DSpark (`session.cpp:327`) — enabling suffix-draft windows
  for inkling2 without a drafter is a small, self-contained change (the plumbing exists:
  `copy_drafts()` already comments "prompt lookup: any window's rows verify alike"). Windows amortize the
  GPU term only; CPU cost is unchanged to first order (§2.4). Model it: gain ≈ (accepted − 1) × GPU-term;
  break-even at ~1.5 accepted tokens/window post-W1.
- If enabled, add the inkling2 window shapes to the captured-graph keys and re-verify the
  `kDecodeRows`/`kGuessRows` machinery at rows>1 (the guess router already handles `rows <= kGuessRows`).

---

## 9. Phase 0 — diagnostics to build first (half a day; the implementing agent should not skip this)
1. **Commit a git baseline** in the working tree (`git init && git add -A && git commit`) — the tree is
   currently an unzipped snapshot; every workstream should land as a commit.
2. **Add the two micro-benchmarks as repo tools** (port `C:\LLM\bwtest\bw_bench.cpp` and
   `nvfp4_bench2.cpp` into `tools/` or a `bench/` directory with a CMake target) so kernel/pool changes are
   measured against the same harness; report ms/expert and GB/s at 1/8/12 workers.
3. **In-engine per-visit split timer** (temporary, `#ifdef BASALT_DIAG`): per visit, record
   (a) CPU expert wall time, (b) GPU busy time between cpu start and finish, (c) hand-off gap. Emit one
   summary line per 1k tokens. This replaces the deleted per-visit instrumentation with a permanent,
   compilable-off tool. Use it to fill §3's budget table with real numbers and to attribute the
   engine-vs-isolated-bench gap (0.155 vs 0.129 ms at 8 workers).
4. **GPU profile** one decode second with Nsight Systems (`nsys profile` on the engine under
   `Start-Basalt-Fast`): attention vs glue vs router vs head vs expert-kernel time, launch gaps, and the
   idle fraction per visit. Confirms/refutes §7 candidates.

## 10. Implementation order
| Step | Item | Expected tok/s after | Effort |
|---|---|---|---|
| 0 | Phase 0 diagnostics + git baseline | 35 | 0.5 d |
| 1 | W1.1 pool size/pinning | 40–42 | 0.5 d |
| 2 | W1.2 kernel micro-opts | 46–49 | 1–2 d |
| 3 | W2.1 staging reclaim | 49–52 | 1 d |
| 4 | W2.2 live re-rank (+W2.3 audit) | 50–54 | 0.5–1 d |
| 5 | W3 trims (profile-guided) | 52–57 | 1 d |
| 6 | W1.3 Q2 shadow (optional, flagged) | 55–62 | 1–2 d |
| 7 | W4 spec re-eval (optional) | +5–15% | 1 d |

Each step is independently shippable and measurable; stop when the user's target is met with margin.

---

## 11. Validation protocol (run after every step; keep logs in `C:\LLM\perf-logs\<step>\`)
1. **Build:** `C:\LLM\rebuild.bat` (MSVC 19.44 + CUDA 13.5, `-DBASALT_MODELS=inkling2` already configured in
   `build\`). Zero warnings introduced.
2. **Unit/parity:** repo test suite (`tests/`, incl. `tests/kernels` CPU parity tests) +
   `tools/greedy_identity.py` if its fixtures cover inkling2; plus a new randomized old-vs-new kernel
   equality test for W1.2 (same blobs/acts, bitwise-compare `ff`/`out`).
3. **Decode benchmark (primary metric):** with the engine on the optimised build and 256k config:
   - short chat (the server log's `tg=` lines; ≥ 3 runs, report median tok/s and the logged hit rate);
   - 16k-prompt request then ≥ 250 generated tokens (matches the historical 38.8 t/s measurement);
   - ≥ 5-minute sustained run to expose thermal/SMT regressions.
4. **Long-context correctness:** `C:\LLM\Basalt\test-long.ps1` (needle at >130k positions) must pass.
5. **No VRAM/RAM regressions:** startup log shows `elastic spill: N of 2698 ... copied back` (N must not
   shrink), `1024 MiB of VRAM free`, and no `cannot pin` failures; with W1.3 on, RAM headroom ≥ 2 GiB after
   load under normal desktop load.
6. **Update `C:\LLM\Basalt\CHANGES.md`** with each landed step (measured before/after), keeping the existing
   document's style. Do not regress Patch A/B (§1.4).

## 12. Risks
- **Thermal/power**: 12 streaming workers + GPU on a laptop may sustain-clock lower; the sustained benchmark
  (§11.3) is the gate, and `--pool-workers` remains the fallback knob.
- **Scheduling interactions**: pinning workers to 0..11 can collide with the CUDA host-func thread; if the
  host-func thread lands on a worker core the pool stalls (its `wait_done` runs on that thread). Verify the
  callback thread's placement empirically; worst case reserve one physical core for it (11 workers).
- **Kernel parity**: W1.2 must stay bitwise or be flag-gated (§ W1.2).
- **Staging reclaim (W2.1)**: prompt-arrival remap cost could add TTFT; measure `core::vmm` map/unmap
  latency before committing to full release, and keep a standing floor if it is slow.
- **Q2 shadow (W1.3)**: real quality loss on the coldest experts; default off, user opt-in, and the
  acceptance criteria are the user's judgement, not a number.

## 13. Environment notes for the implementing agent
- Engine start/stop: `C:\LLM\Basalt\Start-Basalt-Fast.cmd` (optimised build, task `BasaltPR`),
  `C:\LLM\Basalt\Stop-Basalt.cmd`. Kill `strata.exe` if it holds ~10.8 GB VRAM.
- Monitor: `C:\LLM\basalt-monitor.cmd` (reads `/metrics`, needs the API key in `C:\LLM\Basalt\api-key.txt`).
- Engine log: `C:\Users\Phil\.cache\basalt\55cfc8c42932d132\engine.log` (elastic-spill lines, hit rates).
- Benchmarks/tuning scripts that already exist: `C:\LLM\Basalt\tune.ps1` (throughput), `sweep.ps1`,
  `test-long.ps1`. New micro-bench sources: `C:\LLM\bwtest\{bw_bench,nvfp4_bench2}.cpp` (+ compiled exes).
- Config files: live `basalt_config.json` in the source tree (256k). Do not edit the stock install's configs
  for this work.
- Known-bad flags for inkling2 (no-ops or unimplemented): `route-prior`, `route-tail-skip`, `prefill-cpu-share`
  (silently skipped for this model), and `--spec > 0` at 256k (drafter does not fit). See `C:\LLM\Basalt\CHANGES.md` §8.
