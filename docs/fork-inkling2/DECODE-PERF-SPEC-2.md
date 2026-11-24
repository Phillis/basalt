# Specification 2 — Inkling-2-Mini decode: 38 → 50+ tok/s

**For an implementing agent.** Builds on the results in `PERFORMANCE-RESULTS.md`, `CHANGES.md` (Session 3) and
`DECODE-PERF-SPEC.md` (superseded where noted). Working tree:
`C:\LLM\pr536\basalt-inkling-cpu-spill-review` (MSVC 19.44 / CUDA 13.5 / `-DBASALT_MODELS=inkling2`), build with
`C:\LLM\rebuild.bat`, launch with `C:\LLM\Basalt\Start-Basalt-Fast.cmd` (direct launch — **not** the `BasaltPR`
task, see Appendix A), stop with `Stop-Basalt.cmd`.

**Current state:** 192k context + DSpark (`spec 8`), decode **37-40.7 tok/s** (128k: 40.7), prefill 1,643 t/s,
hit 65-70%, 2767 of 3072 experts in RAM, `kCardSpare = 896` MiB and `vram-reserve-mib = 768` (do not change
these without re-reading CHANGES.md Session 3 — the plan sits ~150 MiB under the host-allocation quota).

## 0. Review verdict on the last round (what stands, what was wrong)

| Claim in PERFORMANCE-RESULTS.md | Verdict |
|---|---|
| "CPU expert work is ~3.2 ms/token (11%)" | **Wrong, and it mattered.** The 3→11 worker scaling derived it from `t(3w) − t(11w)`, but per-window CPU work scales with *distinct spilled experts per window* (~0.95 GiB of blobs), not just the kernel rate. Direct pool-timer measurement: **~18 ms/window ≈ 10 ms/token ≈ 40%**. |
| "~24.7 ms = 48 serialised visits, mostly launch/serialisation gaps" | **Half right.** Measured with GPU-timeline stamps: per-visit hand-off overhead is real (~0.2 ms × 48 = **10 ms/window**), but the larger share of the post-expert segment is actual CPU expert compute (~18 ms). |
| "W1.2/W1.3 capped at +5%, not started" | **Wrong** — the cap arithmetic was based on the 3.2 ms figure. Kernel optimisation and hit-rate work are worth **~2× on the dominant term** again. |
| "dl saturates at ~1.6-1.7 with this drafter; drafting knobs exhausted" | Correct for DSpark's *exact* acceptance, but `--relaxed-drafting` was never available to inkling2 (qwen4exp-only) — W4 below ports it. |
| "Nothing left but a smaller pack or a bigger card" | **Wrong** — see the measured budget below. |

The engine's own `diag` harness (Session 3, permanent in the tree) produces the budget below; **run it before and
after every change** (`cmd /c "set BASALT_DIAG_VISIT=1&& ...run-pr.cmd"`, then read `engine.log` for
`diag: N windows...` lines).

## 1. Measured decode budget (steady state, 192k + DSpark, ~39.5 ms per ~1.8-token window ≈ 22 ms/token)

| Term | ms/window | Share | How measured |
|---|---|---|---|
| **CPU pool expert compute** | **~18** | 46% | `pool busy` counter; ≈0.95 GiB of spilled blobs/window at the isolated-kernel rate (~0.084 ms/expert, 12 pinned workers) |
| **Per-visit hand-off** | **~10** | 25% | post-expert span (28.0) − pool busy (18); 48 host-func round trips × ~0.2 ms |
| GPU pre-expert chain (attention, dense, router, on-card experts) | ~9.7 | 25% | GPU-timeline stamps |
| Head (201k vocab) | ~1.0 | 2.5% | stamps |
| Drafter + engine, outside the window | ~1.5 | 4% | serve-line vs window-math closure |

Independent floors: the pool kernel runs at 53 GB/s at 12 workers (`C:\LLM\bwtest\nvfp4_bench2.exe`); pure DDR5
streaming is 47 GB/s per core / 86 GB/s at 16 threads (`bw_bench.exe`); the host link is 15.75 GB/s and
GPU-compute-over-mapped-RAM stays a dead end (6×, measured twice now).

**Token-time model:** `t = (18·m + 10 + 9.7 + 1 + 1.5)/dl` where `m` is the miss-factor (1.0 today) and `dl`
≈ 1.7 accepted tokens/window. Every workstream below attacks one factor; the projections use this model.

---

## 2. Workstreams, ranked by measured value

### W1 — Convert the idle 1,034 MiB prompt staging into expert copies (halves the CPU term; ~1 day; no quality risk)
The spill plan reserves ~1 GiB of VRAM so a **prompt chunk** can stage a spilled layer's experts; during pure
decode it is idle (`spill.cpp`, `tail(l)`, mapped by `model.cpp` `staging_`). The elastic spill already owns the
lifecycle hooks (`promote`/`demote`, `session.cpp:349-378` grows/shrinks around the KV cache — reuse them):

1. After a prefill completes (or on the first decode-only window), release the staging mappings and hand the
   freed bytes to `promote()` via its existing `keep_free` mechanism → **~220 more hot experts on the card**.
2. On the next prompt chunk, `demote()` the coldest copies, remap staging, prefill, re-promote. Measure
   `core::vmm` map/unmap latency once; keep a standing floor (e.g. 256 MiB) if remapping is slow.
3. Careful: `kCardSpare` 896 + the staging release must not break the `so --max-context fits` plan check — the
   staging is counted in the load-time plan; only the *runtime* lifecycle changes.

Expected: hit 65-70% → ~80%; pool term 18 → ~10 ms/window. **Verify with `diag`**: `pool busy` should halve.

### W2 — Port the qwen4exp doorbell hand-off to inkling2 (kills the 10 ms hand-off term; 2-3 days; medium risk)
The repo already contains a proven zero-driver-call hand-off design ("Plan v0.3", `src/models/qwen4exp/decode/`):
`doorbell_publish` (a captured kernel that copies x + ids into **mapped pinned host memory** and bumps a sequence
number), a **host thread that polls** and runs the pool (`decode/session.cpp:689-860`), and
`wait_flag_ge`/`doorbell_wait` (captured kernels that spin on host-written flags) plus
`scatter_rows_f32`/`copy_from_mapped` for the return path (`include/basalt/kernels/elementwise.hpp`). The
qwen4exp code even splits its graph per layer at the router — inkling2 does **not** need that: its whole window
body is one captured graph, and every polling kernel is capturable, so the 48-visit chain stays one graph.

Replace, per visit (`cpu_spill.cpp` `start`/`finish`):
- `cudaMemcpyAsync` D2H + `cudaLaunchHostFunc` → a `doorbell_publish`-style kernel writing the normed input
  (2304 × rows floats), the ids, and a per-visit ring bump into mapped pinned memory (inkling2 needs a
  **48-slot ring** so all visits of a window can be outstanding; ~0.5 MiB payload + output ring
  48 × ~0.6 MiB — allocate once in `CpuSpill::open`).
- The CUDA-callback host → a dedicated engine thread that polls the ring in order (the pool's
  `start_split_multi_native` two-phase API already lets the host publish work and join later), runs the pool
  directly, writes the merged rows into the output slot and sets the done flag. No `cudaLaunchHostFunc` anywhere.
- `cudaStreamWaitEvent(done)` + `moe_cpu_merge` → a `wait_flag_ge` node (captured) + the merge kernel reading the
  mapped output (it already reads mapped memory today — only its event wait changes).
- Keep the old path behind a flag (`--cpu-spill-handoff hostfunc|doorbell`, default doorbell after parity) so
  A/B is trivial. **Bitwise output parity** is achievable (same pool work, same merge kernel); assert it with a
  greedy-identity A/B in addition to the repo tests.
- The host thread must live outside CUDA callback context: create it in `CpuSpill::open`, park it with the pool's
  idle machinery (`idle_start`), and make it stop cleanly in `~Impl` (see `qwen4exp` `session_loop` for the
  shutdown protocol).

Expected: hand-off 10 → ~2 ms/window. **Verify with `diag`**: `post-expert` should drop to ≈ pool busy + ~2 ms.

### W3 — NVFP4 CPU kernel micro-optimisation (spec 1 §W1.2, unchanged; ~1-2 days)
Still worth ~6-8 ms/window after W1 (pool 10 → ~6-7). The guidance in `DECODE-PERF-SPEC.md` §W1.2 stands:
vectorise the UE4M3 scale reconstruction out of the shuffle-port path, 2-block unroll to break the `acc` FMA
chain, R=4 rows in `gu_rows`, prefetch next block pair; **bitwise parity or flag-gated**. Benchmarks:
`C:\LLM\bwtest\nvfp4_bench2.exe` (1 thread 13.5 GB/s → target ≥20; 12 threads 53 → target ≥70).

### W4 — Port `--relaxed-drafting` to inkling2 (raises dl; ~1 day)
`default_relax()` is overridden only by qwen4exp (`src/program/serve/request_parse.cpp:36`,
`request.hpp:132`); the per-request `relax` plumbing exists generically. Inkling2 needs: its request type to
override `default_relax()`, and its verify path to honour it (the acceptance test keeps a draft the target
scores ≥ P instead of discarding). At temp 1.0 this is a legitimate sampling trade-off, off by default.
Measure dl before/after (`drafts accepted` + `diag` tokens/window): dl 1.7 → 2.2 turns every per-window term
into ÷1.25 per token (≈ +25% tok/s after W1+W2 ≈ **70+ tok/s**).

### W5 — (optional, quality-flagged) Q2_0 CPU shadow of spilled experts (spec 1 §W1.3, unchanged; ~1-2 days)
Halves pool bytes and decode cost for the RAM tier; the repo's Q2_0 CPU kernels already exist
(`q2_avx2_rows.inc`, `q2_native_kernels`). Worth revisiting **after** W1 (fewer misses left to convert).

### Explicitly deferred
- Multi-row window *scheduling* changes (dl is acceptance-bound, not window-size-bound; all knobs measured flat).
- GPU-side anything for spilled experts (mapped-RAM compute, 6× worse, re-confirmed).
- Prefill: 1,643 t/s is adequate; if desired later, investigate skipping DSpark's per-chunk drafter taps
  during pure-prefill requests (~10% of prefill), but the drafter cache is needed for later decode.

## 3. Implementation order and gates

| Step | Gate before continuing |
|---|---|
| W1 staging reclaim | loads clean at 192k (watch the quota — Appendix B), `diag` pool busy halved, tok/s ≥ 44 |
| W2 doorbell | bitwise/greedy parity vs hostfunc path, `diag` post-expert ≈ pool + 2 ms, tok/s ≥ 50 |
| W3 kernels | parity tests pass, bench targets met, tok/s ≥ 55 |
| W4 relaxed drafting | dl ≥ 2.0 measured, user accepts the sampling trade-off, tok/s ≥ 60 |

Re-run the full validation protocol from `DECODE-PERF-SPEC.md` §11 after each step (tests, greedy identity,
300/1000-token runs, 16k-prefill + long decode, `test-long.ps1` needle, VRAM/RAM lines, sustained 5-min run),
and append results to `CHANGES.md` as Session 4.

## 4. Risks
- **Load-margin regressions:** any change that grows VRAM demand at plan time (more experts on card) shrinks
  the ~150 MiB pinned headroom; keep `kCardSpare` 896 + `vram-reserve-mib` 768 in sync and watch for
  `pinned staging alloc ... failed` (now logged with the real CUDA error).
- **Doorbell port:** WDDM flag-visibility across captured graphs is subtle; the repo's own
  `tools/zero_doorbell_check.py` and `src/core/graph.cpp:40`'s comment ("captured graphs make a doorbell visible
  and direct launches do not") document the trap — read both first, and A/B against the hostfunc path behind
  the flag rather than replacing it outright.
- **Thermal:** W1 raises GPU-side expert work; the sustained-run gate catches it.

## Appendix A — Launcher anomaly (open question)
With the current build, engines launched by the `BasaltPR` scheduled task fail during the loader's device select
(`cannot select device 0 for the load`) while direct launches of the same `run-pr.cmd` from the interactive
session load every time; a plain CUDA probe (allocs) succeeds under the task. Something in the engine's plan
phase (VMM/elastic maps, drafter staging, ring memsets?) interacts badly with the task context and leaves a
sticky CUDA error that the first `cudaDeviceSynchronize` surfaces. `Start-Basalt-Fast.cmd` now launches directly.
If this is root-caused later, the task can return as the launcher (it survives shell death); until then treat
"scheduled task = broken" as a known constraint.

## Appendix B — Pinned-memory quota (measured)
- The engine's RAM tier is CUDA host VMM (`cuMemCreate`), capped by a **~14.6 GiB WDDM quota** — not by
  `VirtualLock`/`SeLockMemoryPrivilege` (inert; see `MEMORY-LOCKING.md`).
- 192k+DSpark pinned demand: spill (~13.43 GiB at 2767 experts) + embedding (0.86) + read staging (0.25) ≈
  14.5 GiB — ~150 MiB under the quota. The spill size moves with free VRAM at plan time (desktop usage swings
  it by ~128 MiB), which is why 1024 sometimes loaded and sometimes didn't; 896 is the reliable point.
- Fresh-process probes: `C:\LLM\bwtest\hostalloc_seq.exe` (full engine sequence succeeds), `hostalloc_fill.exe`
  (finds the ceiling). Use them if load failures reappear.
