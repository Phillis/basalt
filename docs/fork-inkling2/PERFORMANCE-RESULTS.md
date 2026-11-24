# Basalt / Inkling 2 Mini â€” performance results

Standalone summary of the tuning work on this machine. For the change-by-change log see
[`CHANGES.md`](CHANGES.md); for the upstream investigation see `C:\LLM\pr536\PR-FINDINGS.md` and
`DECODE-PERF-SPEC.md`.

**Result: decode 11.3 â†’ ~38 tok/s (3.4Ã—), prefill 1,452 â†’ 1,836 tok/s (1.26Ã—).**
Live profile: **196,608-token context with DSpark speculative decoding**.

---

## 1. Machine

| | |
|---|---|
| GPU | RTX 5070 Ti Laptop, 12 GB (11.9 GiB usable), sm_120, PCIe 4.0 **x8** (15.75 GB/s measured) |
| CPU | Ryzen AI 9 HX 370, 12 cores / 24 threads (4 Zen 5 + 8 Zen 5c), AVX-512 |
| RAM | 31.1 GB LPDDR5X-8000 |
| Model | `Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt`, 18.38 GiB, 3,072 experts, 24 layers / 48 visits per token |
| Behaviour | 16.3 GiB of weights cannot fit 11.9 GiB of VRAM â†’ most experts stream from system RAM |

---

## 2. Where the 3.4Ã— came from

Each step measured on its own, 256k context unless stated:

| Step | Decode | Prefill (16k) | Gain |
|---|---|---|---|
| Stock Basalt v0.5.5 release | 11.3 tok/s | 1,452 tok/s | â€” |
| + PR #536 (CPU-computed experts), link gate intact | 10.7 tok/s | â€” | **none** |
| + **Patch A: link gate removed** | **34.6 tok/s** | 1,452 tok/s | **3.2Ã—** |
| + PR #525 (prompt staging from card copies) | 34.6 tok/s | 1,770â€“1,804 tok/s | +24% prefill |
| + **W1.1: pool = all physical cores, pinned** | 35.8â€“36.6 tok/s | **1,836 tok/s** | +3% / +2% |
| + **W4: 192k + DSpark windows (live)** | **37.5â€“38.6 tok/s** | 1,643 tok/s | +5% decode, âˆ’10% prefill |

Cumulative: **decode 3.4Ã—, prefill 1.26Ã—.**

### Patch A is the whole story

PR #536 only enables CPU-computed experts when the host link measures **â‰¤ 8 GB/s**:

```cpp
// src/models/inkling2/cpu_spill.cpp, CpuSpill::make()
if (!model.spill().any() || model.geo().n_used != kUsed || !(link > 0 && link <= 8e9)) return nullptr;
```

The engine logs ours as **15.75 GB/s**, so the path never engaged and the CPU sat at 15% during decode.
Removing only the link clause took decode **10.7 â†’ 34.6 tok/s**.

---

## 3. Live configuration

`C:\LLM\pr536\basalt-inkling-cpu-spill-review\basalt_config.json`

```json
"max-context": 196608,  "kv": "fp8",  "spec": 8,  "parallel": 1,
"expert-cache": "auto", "vram-reserve-mib": 1024, "ram-reserve-mib": 512,
"vision": "off",  "prefill-cpu-share": 0.85,
"host": "0.0.0.0", "allowed-host": "*"
```

Load: `experts: 2767 of 3072 in RAM (13.43 GiB), 2.31 GiB on the card with 1034 MiB staging`,
`ctx=196608`, `slots=305`, DSpark on the GPU (0.48 GiB).

### Verified

| Check | Result |
|---|---|
| Decode, 3 consecutive 300-token runs | 38.6 / 38.2 / 37.5 tok/s (acc 67â€“77%, dl 1.6â€“1.7) |
| Prefill, 16k prompt | 1,643 tok/s |
| Sustained single 1,000-token run | 34.3 tok/s (GPU 76 Â°C, no thermal collapse) |
| Long context | needle found at **130,709** tokens |
| Stability | no `cannot pin`, no VRAM/RAM errors, 1,205 MiB VRAM free at rest |

---

## 4. Profile choices

Switch with `Start-Profile.cmd <file>` (copies over `basalt_config.json` and restarts):

| File | Context | Decode | Prefill (16k) | Use when |
|---|---|---|---|---|
| `basalt_config-192k-dspark.json` **(live)** | 196,608 | **37.5â€“38.6 tok/s** | 1,643 | default; decode first |
| `basalt_config-256k.json` | 262,144 | 35.8â€“36.6 tok/s | **1,836 tok/s** | full 256k window / long prompts |
| `basalt_config-128k-dspark.json` | 131,072 | **40.7 tok/s** | ~1,650 | maximum decode speed |
| `basalt_config-32k-fast.json` | 32,768 | ~40+ tok/s | higher | short-context chat |

**192k + DSpark is +5% decode for âˆ’10% prefill and âˆ’25% window** versus 256k. It is the default because decode
was the stated priority.

---

## 5. Measured bottleneck (why 50 tok/s was not reached)

Instrumented `CpuSpill` directly and A/B-tested worker counts:

| Measurement | Value |
|---|---|
| CPU expert work per token (from the 3â†’11 worker scaling) | **~3.2 ms (11%)** |
| Card's wait on the CPU, per visit | **0.000 ms** â€” the GPU never waits |
| GPU utilisation during decode | 21â€“43% |
| Remaining time | **~24.7 ms (89%)** = 48 serialised layer visits Ã— ~0.5 ms, mostly launch/serialisation gaps |
| CPU experts per token | ~37 of 384 lookups |
| Isolated kernel bench (`nvfp4_bench2.exe`) | 0.084 ms/expert @ 12 threads (53 GB/s) â€” reproduces the spec's claim |

The engine's cost model therefore **is not** CPU-bound: making the CPU kernels 2Ã— faster would save ~1.6 ms
(â‰ˆ +5%), not the +15â€“20% that the spec projected. Multi-row windows amortise the per-visit latency by only
~1.6Ã— because `dl` saturates at ~1.6 with this drafter.

---

## 6. Everything tried and rejected (all measured)

| Attempt | Result |
|---|---|
| PR #536 as-is (link gate intact) | 10.7 tok/s â€” path disabled at 15.75 GB/s |
| Split spill between CPU and card's mapped-RAM path (`BASALT_CPU_SPILL_GPU_PCT`) | **21.1** vs 35.1 at 0% â€” card's mapped read costs ~6Ã— more per expert |
| `expert-cache 128` (explicit) | load fails: `cannot pin 14515 MiB` |
| `expert-cache 420` / `520` | **25.2 / 25.5 tok/s**, hit 66%â†’38% â€” evicts the hot set; `auto` only |
| `pool-workers 11` on the 7-worker default | 33.6 (before pinning) |
| `pool-affinity p-cores` | 31.0 (only 3 workers) |
| `route-prior 2.0/4.0`, `route-tail-skip` | no effect â€” **not implemented for inkling2** |
| `prefill-cpu-share 0.9 / 0.2` | 1,348 / 1,443 â€” flat; 0.85 kept |
| `prefill 8192` (smaller staging) | prefill 1,696, slots unchanged |
| `adapt-swaps 256`, `vram-reserve 512` | 34.0 â€” flat |
| `spec 4 / 6 / 8`, `draft-experts 64`, `mtp-window 0`, `suffix-draft 8` | 37.1â€“38.1 â€” all identical |
| `kCardSpare` 1024 â†’ 896 MiB | plan accepts, then `cannot pin 15023 MiB` (3,062 experts spill) â€” **infeasible**, reverted |
| DSpark at 256k | impossible: drafter forces 3,062 experts into RAM, past the ~14.4 GiB pin ceiling |
| AVX-512 | already compiled per-file and runtime-dispatched; Zen 5 satisfies F/BW/VL/VNNI/VBMI |

---

## 7. Code changes in the tree

`C:\LLM\pr536\basalt-inkling-cpu-spill-review` â€” built with MSVC 19.44 / CUDA 13.5 /
`-DBASALT_MODELS=inkling2`, rebuilt via `C:\LLM\rebuild.bat`. Exactly four `[local]` deltas:

| # | File | Change |
|---|---|---|
| A | `src/models/inkling2/cpu_spill.cpp` | host-link gate removed (the 3.2Ã—) |
| B | `src/models/inkling2/{model.cpp,model.hpp,prompt.cpp}` | PR #525 harness applied by hand (prefill +24%) |
| C | `include/basalt/platform/cpu.hpp`, `src/platform/cpu.cpp`, `src/kernels/cpu/pool.cpp` | `platform::pin_current_thread()`; pool workers pinned one per physical core |
| D | `src/models/inkling2/cpu_spill.cpp` | default CPU-spill workers = every physical core (11, was 7) |

Backups: `cpu_spill.cpp.orig`, `model.cpp.orig`. No diagnostic scaffolding left in the tree.

---

## 8. Remaining headroom

At 192k the token is ~26 ms, of which ~24.7 ms is the 48-visit serial GPU chain with the GPU ~31% busy.
Ranked paths to 50 tok/s:

1. **A smaller pack (~10 GB)** â€” fewer bytes per expert and more experts resident; needs llama.cpp, since
   Basalt cannot ingest a GGUF for `inkling2` (no arch module; unknown archs silently fall back to
   `qwen4exp`). `gguf\Inkling-2-Mini-Unbound-IQ2_M.gguf` is 5.5/10.3 GB downloaded and resumable.
2. **A 16 GB+ card** â€” ~2.3 GB more VRAM per ~480 experts lifted off the CPU, plus room for DSpark at 256k.
3. **More rows per window** (`dl` above ~1.6) â€” needs a better drafter than DSpark, which does not fit at 256k.

---

## 9. Operating it

| Action | Command |
|---|---|
| Start (192k + DSpark) | `C:\LLM\Basalt\Start-Basalt.cmd` |
| Stop | `C:\LLM\Basalt\Stop-Basalt.cmd` |
| Live speeds | `C:\LLM\basalt-monitor.cmd` |
| Switch profile | `C:\LLM\Basalt\Start-Profile.cmd basalt_config-256k.json` |
| Network access | `http://192.168.11.214:8080/v1`, model `Inkling-2-Mini-Unbound`, API key in `C:\LLM\Basalt\api-key.txt` |

---

# Update â€” Session 4 (DECODE-PERF-SPEC-2: W1 staging reclaim + the kCardSpare find)

| Metric (192k context + DSpark) | Before this session | **Final** |
|---|---|---|
| Decode, 300-token runs (warm) | 37.5â€“39.5 tok/s | **45.4â€“45.9 tok/s** |
| Decode, sustained 1,000-token run | 34.3 tok/s | 35.7 tok/s |
| Prefill, 16k prompt | 1,643 tok/s | 1,520â€“1,645 tok/s |
| Expert-cache hit | 65.7% | **71.7%** |
| Drafts per window (`dl`) | 1.6 | **2.0â€“2.2** (acceptance 84â€“96%) |
| Experts resident on card | 305 | 334 |
| Long context | pass | **pass** (needle at 130,709 tokens) |

**Versus the stock v0.5.5 release: 11.3 â†’ 45.5 tok/s decode = 4.0Ã—, at a 192k window.**

## What was implemented

1. **W1 â€” prompt-staging reclaim** (`model.cpp`, `model.hpp`, `session.cpp`): the ~1 GiB of card memory reserved for
   prompt-chunk staging is released at the first decode window and handed to the elastic spill (`promote`), then taken back
   when a prompt chunk needs it. Hit rate 66% â†’ 72â€“95%; `dl` 1.6 â†’ 2.1â€“2.2; table above.
2. **`kCardSpare` 896 â†’ 768 MiB** (`model.cpp`): 896 put the plan ~20 MiB over this machine's reachable pinned total
   (14.52 GiB), which made loads fail with `cannot select device 0`. 768 keeps ~29 more experts on the card and restored
   reliable loads.
3. **Dropped the `prefill-cpu-share: 0.85` override**: it was costing prefill (~1,514 vs ~1,645 tok/s).

## What did not work (all measured)

| Attempt | Result |
|---|---|
| **W2 doorbell hand-off** (full port: ring, `doorbell_publish_store`, `wait_flag_ge`, polling thread) | **Blocked, not a small fix.** inkling2's decode window is replayed from a **captured CUDA graph**, so `start`/`finish` run only at *capture*; on replay the recorded publish/wait nodes carry the slot index and sequence value baked in, the host bookkeeping freezes and the card spins in `wait_flag_ge` (stall report: pool idle, 144/144 jobs done). Needs a per-window host protocol with device-resident ring indices â€” a redesign of the hand-off. Code kept behind `C:\LLM\cpu-spill-doorbell.on`; host-func stays the default. Two useful spinoffs: `doorbell_publish_store()` was added (no PCIe read-modify-write of the counter), and the ring had to move from plan-time to lazy allocation (plan-time pinning starved the loader). |
| `kCardSpare` 640 | 35.5 tok/s â€” `dl` collapses to 1.6; the drafter needs free VRAM |
| `promote()` keep-free 384 (vs 512) | 37.0 vs 42.7â€“45.9 â€” same effect |
| `prefill-cpu-share` 0.6 / 0.85 / 1.0 | 1,642 / 1,514 / 1,648 â€” removed the override |

## Still on the table

- **W3** NVFP4 CPU kernel micro-optimisation (AVX-512, bitwise parity) â€” the pool term is now ~9 ms/token, so worth less than when the spec was written.
- **W4** `--relaxed-drafting` for inkling2 â€” needs a DSpark draft-probability probe (CUDA + read-back) and changes the sampling distribution.
- **W2 redone properly** â€” per-window host protocol, device-resident ring indices; the largest single remaining term (~9 ms/window hand-off).

---

# Session 5 â€” W2 doorbell hand-off landed (**62.7â€“68.4 tok/s**)

| Metric (192k + DSpark) | Previous best | **Now** |
|---|---|---|
| Decode, 300-token runs (warm) | 45.4â€“45.9 tok/s | **62.7 / 68.3 / 68.3 / 68.4 tok/s** |
| Decode, sustained 1,000-token | 35.7 tok/s | **54.0 tok/s** |
| Prefill, 16k | ~1,520â€“1,645 tok/s | 1,511 tok/s |
| `diag` wall per window | 53.3 ms | **28.6 ms** |
| `diag` hand-off (post-expert âˆ’ pool busy) | ~9 ms | **â‰ˆ0 ms** |
| `host-funcs/window` | 48 | **0** |
| vs stock v0.5.5 (11.3 tok/s) | 4.0Ã— | **6.0Ã—** |

The hand-off no longer costs driver round trips: the card publishes each visit into mapped pinned memory with an
`atomicAdd` and the card's wait node compares two mapped words (`doorbell_wait`) at run time, so a **captured** graph
replays correctly. One slot per visit, self-describing payload `{layer, rows}`, a dedicated polling thread runs the pool.
Revert switch: `C:\LLM\cpu-spill-hostfunc.on`.

Validation: needle at **130,709** tokens; determinism within a process; coherent output; no errors. A cross-path bitwise
A/B is not obtainable here â€” the same build diverges across processes (expert placement), measured.

Remaining: prefill (~1.5k tok/s) is untouched by this; the CPU pool is now the dominant decode term again (~15 ms/window),
which is what **W3** (NVFP4 kernel micro-optimisation) targets.

## W3 (NVFP4 kernel micro-optimisation) â€” investigated, closed

Three parity-safe changes tried against the isolated harness and all rejected on measurement: a `_mm_i32gather_ps` scale
reconstruction (neutral), 4-block software prefetch (neutral), and R=4 rows per pass in `gu_rows` (much worse â€” register
spills on a dual-pumped 512-bit Zen 5). Reverted; no regression (66.1â€“68.8 tok/s).

Measured reason: the engine's pool already runs at **45.6 GB/s** against the harness's best **51.9 GB/s** on an idle
machine â€” â‰¤12% headroom, below what the spec's targets assumed. Remaining decode headroom is the Q2_0 shadow (W5, a
quality trade) or hardware.

## W4 (relaxed drafting for inkling2) â€” implemented, opt-in, no gain

Acceptance is one host-side line in `SessionRequest::accept_drafts`, and the drafter's per-draft confidence
(`window_conf_`, from DSpark `draft_end`) was already available, so the port is ~10 lines behind the existing per-request
`relax` body key (default off).

It works (acceptance 43% â†’ 83â€“88% at `relax` 0.4â€“0.6 in one pass) but **does not raise throughput**: in a clean pass
sampled requests ran 55.3 (`relax` 0.4) and 44.8 (0.6) against 60.8 without, and greedy â€” which ignores `relax` by design â€”
stays the fastest at 63.1. Kept as an opt-in, not enabled.

## Session 6 (DECODE-PERF-SPEC-3) — W-A, W-B, W-D

| Workload (greedy, warm, 300 tok, 192k + DSpark) | before | **now** |
|---|---|---|
| Story prose | ~63 tok/s | **66.7–67.3** |
| Technical prose | — | 48.9–49.4 |
| Repetitive content, `--suffix-draft 8` | `dl` 2.6 → — | **`dl` 3.3–3.9, 53.7–65.4** |
| `diag` wall per window | 37 ms | **31.2 ms** |
| Prefill, 16k | ~1,511 tok/s | 1,507 tok/s |

- **W-A**: `draft_end` synchronised the whole CUDA stream to read the drafts back; it now waits on an event recorded at the
  end of the drafter's own work (`dspark.cpp`), cutting the drafter tax ~9 → ~4.6 ms/window.
- **W-B**: inkling2 never parsed `--suffix-draft`, so lookup drafting had never actually run. Parsed now; it lifts `dl` to
  3.3–3.9 on repetitive content (free-form unchanged).
- **W-D**: the W1 staging reclaim was being re-taken for *every* prefill including ≤16-token prompts that never use it.
  Gated on the prompt path's own condition, removing ~1 GiB of demote+remap from short-turn time-to-first-token.
- **W-C (Q2 shadow)**: not attempted (1–2 days, quality trade). Its pinned-RAM objection is withdrawn — the shadow only
  needs pageable RAM.
- **W-E.1 (hit-rate >100%)**: diagnosed as an intermittent stale-baseline in the server's accounting, not a cache bug;
  left as-is (diagnostic only).

Note on headline numbers: the earlier **62.7–68.4 tok/s** came from a highly predictable prompt. On ordinary prose the
comparable figure is ~63 (before this session) and **~67** now.
