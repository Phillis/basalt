# Inkling 2 Mini on Basalt ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â what was done, in one page

A summary of the whole effort on this machine. Detail lives in the linked documents.

**Outcome: decode went from 11.3 tok/s (stock Basalt v0.5.5) to ~63ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ68 tok/s ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â a 6.0ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â gain ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â while holding a
196,608-token context window, with prefill at ~1.5k tok/s.** The model is an 18.38 GiB NVFP4 MoE pack running on a
12 GB card, so most experts stream from system RAM; everything below is about making that path cheap.

---

## 1. Final numbers

| Metric | Stock release | **Now** |
|---|---|---|
| Decode, 300-token calls (greedy, warm) | 11.3 tok/s | **63.1 / 65.3 / 66.1 / 68.3 / 68.4 / 68.8** |
| Decode, sustained 1,000-token call | ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â | **54.0 tok/s** |
| Prefill, 10k unique prompt | 1,452 tok/s | ~1,450-1,550 tok/s (the Session 12 stage-ahead claim was **retracted**: it corrupted the model) |
| Context window | 8,192 (default) | **196,608** |
| Drafts per window (`dl`) | ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â | 1.9ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ2.0 (acceptance 84ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ96%) |
| Expert-cache hit | ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â | 71.7% |
| Experts resident on card | ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â | 334 of 3,072 |
| Long-context retrieval | ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â | needle found at **130,709** tokens |

---

## 2. The journey, step by step

Each row was measured on its own; the gain column is against the row above.

| Step | Decode | Prefill | Gain |
|---|---|---|---|
| Stock Basalt v0.5.5 + release notes | 11.3 | 1,452 | ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â |
| **PR #536 + removing its host-link gate** (computes spilled experts on the CPU) | **34.6** | 1,452 | **3.2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â** |
| PR #525 (prompt chunks stage from card copies) applied by hand | 34.6 | 1,770 | +24% pp |
| Pool = all 11 physical cores, pinned one per core | 35.8 | 1,836 | +3% |
| 192,608 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 192k context + DSpark drafting (drafter needs VRAM headroom) | 37.6ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ41 | ~1,650 | +5% decode |
| **W1: prompt-staging reclaimed for expert copies** | 39.3ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ42.7 | ~1,520 | +5ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ8% |
| **`kCardSpare` 896 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 768 MiB** (fixes load reliability *and* keeps more experts on card) | 45.4ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ45.9 | ~1,520 | +6% |
| **W2: doorbell hand-off** (no driver calls per visit) | **62.7ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ68.4** | ~1,511 | **+45%** |

The two structural wins were the link-gate removal (3.2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â) and the doorbell hand-off (+45%). Everything else is tuning.

---

## 3. What worked

**W1 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â prompt-staging reclaim.** The spill plan reserved ~1 GiB of VRAM so a *prompt chunk* could stage a spilled layer's
experts; during decode that memory sat idle. It is now released at the first decode window and handed to the elastic spill
(hit rate 66% ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 72ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ95%, `dl` 1.6 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 2.1), and taken back when a prompt chunk needs it.

**`kCardSpare` 896 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 768 MiB.** 896 left the plan ~20 MiB over this machine's *reachable* pinned total (14.52 GiB), which
was silently causing `cannot select device 0` load failures. 768 both restored reliable loads and keeps ~29 more experts
resident.

**W2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â doorbell hand-off (the big one).** Each of a window's 48 layer visits used to cost a `cudaMemcpyAsync` pair, a
`cudaLaunchHostFunc` and an event wait. Now the card publishes the visit into mapped pinned memory (payload + ids +
`{layer, rows}`) with an `atomicAdd`, and its wait node compares two mapped words at run time (`doorbell_wait`); a host
thread polls and runs the pool. `host-funcs/window` went **48 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 0**, and `diag` wall per window **53.3 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 28.6 ms**.
The critical detail: the wait must compare at *runtime*, because inkling2's window is a **captured CUDA graph** ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â a value
fixed at capture replays stale and hangs the card (that was the first attempt's failure).

---

## 4. What did not work (all measured, all reverted)

| Attempt | Result |
|---|---|
| PR #536 as shipped | 10.7 tok/s ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â its `link <= 8 GB/s` gate disabled the CPU path on our 15.75 GB/s link |
| Splitting spilled experts between CPU and the card's mapped-RAM path | 21.1 vs 35.1 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â the card's mapped read costs ~6ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â more per expert |
| `expert-cache` set explicitly (128 / 420 / 520) | load failure, or 25.2 tok/s (hit 66% ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 38%); `auto` only |
| `kCardSpare` 640 | 35.5 tok/s ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â `dl` collapses to 1.6, the drafter needs headroom |
| `promote()` keep-free 384 (vs 512) | 37.0 vs 42.7ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ45.9 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â same effect |
| **W3: NVFP4 kernel micro-optimisation** | **closed.** `_mm_i32gather_ps` scale path and prefetch: neutral. R=4 rows: much worse (0.364 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 0.557 ms/expert) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â register spills on a dual-pumped 512-bit Zen 5. The engine already runs the pool at 45.6 GB/s against the harness's best 51.9, so there is ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€šÃ‚Â¤12% headroom |
| **W4: relaxed drafting for inkling2** | **implemented but no gain.** Acceptance does rise (43% ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ 83ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ88%), throughput does not (sampled 55.3/44.8 vs 60.8); greedy ignores it by design. Kept opt-in, not enabled |
| DSpark at 256k | impossible ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â the drafter forces 3,062 experts into RAM, past the pinned ceiling |
| `route-prior`, `route-tail-skip` | not implemented for inkling2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â no-ops |
| `pool-workers` 11 on the old 7-worker default, `pool-affinity p-cores` | 33.6 / 31.0 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â the auto pick was fine |
| `prefill-cpu-share` 0.6/0.85/1.0, `prefill 8192`, `adapt-swaps 256` | flat; the 0.85 override was actively costing prefill and was removed |

---

## 5. Machine changes made

| Change | Why |
|---|---|
| **NVIDIA CUDA Toolkit 13.5** (winget) | the Basalt release ships no CUDA libraries ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â `basalt.exe` would not start |
| **VS 2022 Build Tools + CMake + Ninja** | to build the engine from source for W1/W2/W3/W4 |
| Python 3.11 `.venv` + `requirements.txt` | the server is Python |
| Scheduled task ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ **direct launch** (Appendix A of the spec) | with the current build the task context fails device selection |
| `SeLockMemoryPrivilege` granted to `Phil` | later found **inert** ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â the engine uses CUDA host VMM, not `VirtualLock`; see `MEMORY-LOCKING.md` |
| `strata.exe` stopped | the Strata app auto-starts an engine holding ~10.8 GB of VRAM |
| Server binds `0.0.0.0` + API key | LAN access (firewall already allowed 8080) |
| llama.cpp b12631 installed | side experiment for GGUF quants; Basalt cannot ingest them |

---

## 6. Where things are

**Engine (optimised):** `C:\LLM\pr536\basalt-inkling-cpu-spill-review` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Basalt fork at PR #536, rebuilt with
`C:\LLM\rebuild.bat`, plus four local deltas: the link-gate removal, PR #525's hunks, worker pinning + full-core pool,
and W1 + W2 + W4. Rebuild switch for the hand-off: create `C:\LLM\cpu-spill-hostfunc.on` to go back to the old path.

**Model:** `C:\LLM\Basalt\models\Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt` (18.38 GiB, SHA-256 verified).

**Live config:** `basalt_config.json` in the source tree ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â 196,608 context, `kv fp8`, `spec 8`, `expert-cache auto`,
`vram-reserve 768`, `ram-reserve 512`, `host 0.0.0.0`.

### Daily use

| Action | Command |
|---|---|
| Start | `C:\LLM\Basalt\Start-Basalt.cmd` (same as `Start-Basalt-Fast.cmd`; both run the optimised build) |
| Stop | `C:\LLM\Basalt\Stop-Basalt.cmd` |
| Live speeds | `C:\LLM\basalt-monitor.cmd` - live **PP (prompt processing) and TG (decode)** rates, read position + ETA during a prefill, GPU load, and a per-request table |
| Long-context test | `powershell -File C:\LLM\Basalt\test-long.ps1 -TargetTokens 131072` |
| Switch profile | `C:\LLM\Basalt\Start-Profile.cmd basalt_config-256k.json` |
| Legacy stock build | `powershell -File C:\LLM\Basalt\Start-Basalt-Stock.ps1` |
| API | `http://192.168.11.214:8080/v1`, model `Inkling-2-Mini`, key in `C:\LLM\Basalt\api-key.txt` |

### Documents

| File | What it holds |
|---|---|
| **this file** | the one-page summary |
| `PERFORMANCE-RESULTS.md` | performance tables and the profile trade-offs |
| `CHANGES.md` | 43 KB chronological log of every change, with measurements |
| `MEMORY-LOCKING.md` | the page-locking episode, including why it was unnecessary |
| `DECODE-PERF-SPEC.md`, `DECODE-PERF-SPEC-2.md` | the two specifications worked through |
| `C:\LLM\pr536\PR-FINDINGS.md` | the first investigation (budget, CPU-vs-GPU measurements) |

---

## 7. What is left

- **W5 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Q2_0 shadow copies of spilled experts.** Halves the pool's bytes (~15 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ ~8 ms/window), the largest remaining
  software term. Needs ~5.5 GB more pinned RAM, and this machine already sits at its ~14.5 GiB ceiling, plus it is a real
  quality trade (2-bit where the pack is 4-bit). Not attempted.
- **W2 done as the spec imagined** (a per-window host protocol with device-resident ring indices) is no longer needed: the
  runtime-compared doorbell already delivers its win.
- **Hardware.** A 16 GB+ card removes the spill entirely and would put this model well past 100 tok/s.
- **Prefill (~1.5k tok/s)** is untouched by any of this. It is bound by staging experts over the 15.75 GB/s host link, not
  by the CPU or the kernel.
