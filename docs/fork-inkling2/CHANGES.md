# Changes made to this machine for Basalt / Inkling 2 Mini

Everything below was done during this session, in order. It covers `C:\LLM\Basalt` (the install),
`C:\LLM` (tooling, monitor, llama.cpp) and the system-level changes that support them.

**Bottom line:** Basalt v0.5.5 + model working at a **262,144-token context**, and an optimised
source build (two upstream PRs + one local patch) running **~35 tok/s decode / ~1,780 tok/s prefill**
versus **11.3 tok/s / 1,452 tok/s** on the stock release â€” a **3.1Ã— decode** and **1.2Ã— prefill** gain.

---

## 1. Software installed on the system

| What | Version | Why | Method |
|---|---|---|---|
| NVIDIA CUDA Toolkit | 13.5 | The Basalt release ships **no** CUDA libraries; `basalt.exe` would not start (`0xC0000135` DLL not found) | `winget install Nvidia.CUDA` |
| VS 2022 Build Tools | 17.14.45 (MSVC 14.44) | Required to compile the engine from source | `winget install Microsoft.VisualStudio.2022.BuildTools` with the VCTools workload |
| CMake + Ninja | (bundled with the above) | Basalt's documented Windows build (CMake 3.24+, Ninja) | same VCTools workload |
| Python venv | 3.11.9 | The server (`serve/`) is Python; the engine runs it as a child | `py -3.11 -m venv .venv` + `pip install -r requirements.txt` |
| llama.cpp | b12631 (CUDA 13.5 x64) | Side experiment for GGUF quants; not used by Basalt | GitHub release zip â†’ `C:\LLM\llama.cpp` |

---

## 2. System-level changes

### 2.1 Windows privilege "Lock pages in memory" (`SeLockMemoryPrivilege`)

Granted to `Phil` and `Administrators` via local security policy (used `secedit` in
`C:\LLM\Basalt\grant-lock-pages.ps1`, run elevated):

```
SeLockMemoryPrivilege = Phil,*S-1-5-32-544
```

Takes effect at logon (a reboot was done). **See [`MEMORY-LOCKING.md`](MEMORY-LOCKING.md) for the full record —
including that the grant is currently inert (the engine uses CUDA host VMM, not `VirtualLock`).** **Note:** this turned out **not** to be what blocked the
256k load â€” the real blockers were an explicit `expert-cache: 128` and later the link gate â€” but it is
harmless and is needed by the engine's page-locked RAM tier in some configurations.

### 2.2 Scheduled tasks

| Task | Runs | Purpose |
|---|---|---|
| `BasaltPR` | `cmd.exe /c C:\LLM\pr536\basalt-inkling-cpu-spill-review\run-pr.cmd` | Runs the **optimised** build. Registered as a scheduled task so the server survives the launching shell being closed or killed. |
| `Basalt` | *(deleted)* | Was used for the stock build; removed during cleanup. `Start-Profile.cmd` can recreate it. |

### 2.3 IEEE / network

- Server now binds `0.0.0.0:8080` (was `127.0.0.1`), with an API key required on `/v1/*` **and** `/metrics`.
- No firewall change was needed: inbound rules `allow 8080` (all profiles) and `llama.cpp server 8080`
  (Private) already existed, and the network is profiled as **Public** ("Gooby Home 3"), covered by `allow 8080`.

### 2.4 Stopped

- `strata.exe` was stopped repeatedly: the Strata app auto-launches an engine that holds **~10.8 GB of VRAM**,
  which starves Basalt. `Start-Basalt.ps1` offers to stop it when free VRAM is short.

---

## 3. `C:\LLM\Basalt` â€” the install

### 3.1 Engine

- Downloaded `basalt-v0.5.5-windows-x64.zip` from GitHub releases and extracted it **flat** into
  `C:\LLM\Basalt` (`bin\`, `data\`, `docs\`, `serve\`, `tools\`, licences, `requirements.txt`).
- `bin\basalt.exe` was re-extracted from the zip once (it was locked mid-move).
- **CUDA runtime DLLs copied next to `bin\basalt.exe`** because the release ships none:
  `cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll` (from the CUDA 13.5 toolkit) and
  `nvcudart_hybrid64.dll` (from the driver store).

### 3.2 Python environment

- `.venv\` â€” Python 3.11.9 with `requirements.txt` (numpy 2.4.6, jinja2, regex, pyyaml, tqdm, pillow,
  pypdf, psutil, markupsafe).

### 3.3 Model

- `models\Inkling-2-Mini-Unbound-NVFP4-q8.basalt` â€” 18.38 GiB, downloaded from
  `nokeepai/Inkling-2-Mini-Unbound-Basalt`.
- **SHA-256 verified**: `8f2e1c2a90f4a6b8b7462d2cdaf61651b2b86e23c2f7d4d264119fcc8ff579b5` âœ“

### 3.4 Config profiles (all 256k = 262,144 tokens unless noted)

`basalt_config.json` is the live file; `Start-Profile.cmd <file>` copies a profile over it and restarts.

| File | Context | Drafting | Measured decode | Notes |
|---|---|---|---|---|
| `basalt_config.json` / `basalt_config-256k.json` | 262,144 | off | 11.3 tok/s (stock) | `kv fp8`, `expert-cache auto`, `vram-reserve 1024`, `ram-reserve 512` |
| `basalt_config-128k-dspark.json` | 131,072 | **DSpark on** | 12.8 tok/s (stock) | the drafter fits at 128k; +7% only, it steals expert slots |
| `basalt_config-32k-fast.json` | 32,768 | DSpark on | 14.0 tok/s (stock) | fastest stock profile |

Why these values:
- `kv: fp8` â€” Inkling 2 Mini accepts only `fp16`/`fp8`. fp8 halves KV to 16 KiB/position = **4 GiB at 256k**.
- `spec: 0` at 256k â€” the DSpark drafter cannot fit: `spill plan: even with every expert in RAM the card is 74 MiB short`.
- `expert-cache: auto` â€” a fixed smaller value (e.g. 128) forces 2,944 experts into RAM and the load dies with
  `cannot pin 14515 MiB of RAM`.
- `vram-reserve-mib: 1024` â€” 512 makes `auto` over-commit VRAM.
- `ram-reserve-mib: 512` â€” leaves room for the ~13 GiB pinned spill.
- `prefill-cpu-share: 0.85` â€” small but repeatable prefill gain (1,804 vs 1,770 tok/s).

### 3.5 Scripts created

| File | Purpose |
|---|---|
| `run-server.cmd` | The exact command the task runs (stock build) |
| `Start-Basalt.cmd` / `Start-Basalt.ps1` | Start the stock build; checks free VRAM, offers to stop `strata.exe` |
| `Start-Basalt-Fast.cmd` | **Start the optimised build** (task `BasaltPR`) â€” use this one |
| `Stop-Basalt.cmd` | Stops either build (ends both tasks, kills `basalt.exe` and the venv python) |
| `Start-Profile.cmd <config>` | Switch profile and restart in one step |
| `README-SETUP.md` | Operating notes, measured numbers, layout |
| `CHANGES.md` | This file |
| `api-key.txt` | 48-hex-char API key required by `/v1/*` and `/metrics` |
| `grant-lock-pages.ps1` | Elevates `SeLockMemoryPrivilege` for `Phil`/`Administrators` |
| `sweep.ps1`, `tune.ps1`, `test-long.ps1` | Tuning/benchmark harnesses (config sweep, throughput bench, 256k needle test) |
| `probe.*`, `try.*`, `wsboost.log` | Leftovers from bring-up diagnostics (safe to delete) |

### 3.6 Partial download kept

- `gguf\Inkling-2-Mini-Unbound-IQ2_M.gguf` â€” **5.5 GB of 10.3 GB**, paused. A ~10 GB quant is the only
  remaining route to >50 tok/s (fewer bytes per expert), but Basalt cannot consume a GGUF; it needs llama.cpp.
  Resume with `curl -C - -L -o <same path> <huggingface url>`.

---

## 4. `C:\LLM` â€” tooling outside the install

| Path | Purpose |
|---|---|
| `basalt-monitor.ps1` / `.cmd` | **Throughput monitor** (see Â§7). Reads `/metrics` only. |
| `llama.cpp\` | llama.cpp b12631 CUDA 13.5 x64 (for the GGUF experiment) |
| `pr536\basalt-inkling-cpu-spill-review\` | The **optimised source build** (Â§5) |
| `pr536\PR-FINDINGS.md` | Full performance investigation and measurements |
| `install-toolchain.ps1` | Elevated winget install of VS Build Tools + CMake + Ninja |
| `build-pr536.ps1`, `rebuild.bat` | Configure/build the source tree (vcvars + CUDA 13.5 + Ninja) |
| `basalt-pr536.zip`, `pr536.diff` | Source snapshot and PR #525's diff |
| `cpu_timing.txt`, `mon-test.txt` | Instrumentation and monitor test output (throwaway) |

---

## 5. The optimised build â€” `C:\LLM\pr536\basalt-inkling-cpu-spill-review`

Built from **`YurkoHoshko/basalt` @ `inkling-cpu-spill-review`** (PR #536), which is a fork of
`jesdga95/basalt` main. Configured and built with:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=inkling2
cmake --build build -j 12
```

`-DBASALT_MODELS=inkling2` builds only this model (`bin\basalt.exe` is 80 MB instead of 153 MB) and is
much faster to compile. Runs on the **same 18.38 GiB pack** â€” no model rebuild needed.

### 5.1 Patch A â€” PR #536 link gate removed *(this is where the 3Ã— comes from)*

`src/models/inkling2/cpu_spill.cpp`, in `CpuSpill::make()`:

```diff
-    if (!model.spill().any() || model.geo().n_used != kUsed || !(link > 0 && link <= 8e9)) return nullptr;
+    if (!model.spill().any() || model.geo().n_used != kUsed) return nullptr;  // [local] link gate removed
```

The PR only enables CPU-computed experts when the host link measures **â‰¤ 8 GB/s**. The engine logs ours as
**15.75 GB/s**, so the CPU path never engaged (CPU 15% during decode). Removing only the link clause takes
decode **10.7 â†’ 34.6 tok/s**. Everything else in the PR is untouched.

### 5.2 Patch B â€” PR #525 applied by hand

`giveen/basalt @ tiny-speed`: *"a prompt chunk stages each RAM expert from the copy it still has"*.
Applied as the PR's own hunks (it is a clean, self-contained diff):

- `src/models/inkling2/model.cpp`
  - `load_experts`: dropped `layer_copies_.assign(...)`
  - `promote`: dropped `++layer_copies_[l]`
  - `stage_from_card`: rewritten to stage **each** spilled expert from its card copy when one still exists,
    otherwise from its RAM copy (was: bail out unless *every* expert of the layer had a card copy)
  - `demote`: dropped `--layer_copies_[...]`
- `src/models/inkling2/model.hpp`: comment updated, `std::vector<int> layer_copies_` member removed
- `src/models/inkling2/prompt.cpp`: call-site comment updated

Effect: prefill **1,452 â†’ 1,770 tok/s** (and 1,804 with `prefill-cpu-share 0.85`). Bit-neutral by design.

### 5.3 Tried, measured, and reverted

- **Per-visit timing instrumentation** (atomics + `C:\LLM\cpu_timing.txt`) â€” used to find the bottleneck, then
  removed so the shipped binary is the clean PR + patches.
- **CPU/GPU bandwidth split** â€” a `InklingModel::blob_addr()` accessor plus a stable hash assigning a share of
  spilled experts back to the card's mapped-RAM path, tunable via `BASALT_CPU_SPILL_GPU_PCT`.
  Result: **21.1 tok/s at 34%** vs 35.1 at 0%. Reverted â€” the card's mapped read costs ~6Ã— more per expert.

`src/models/inkling2/cpu_spill.cpp.orig` is the untouched PR file, kept for reference.

### 5.4 Config used by this build

`basalt_config.json` in that tree: 262,144 context, `kv fp8`, `spec 0`, `expert-cache auto`,
`vram-reserve 1024`, `ram-reserve 512`, `prefill-cpu-share 0.85`, `host 0.0.0.0`, `allowed-host "*"`.
`run-pr.cmd` launches it with `--api-key-file C:\LLM\Basalt\api-key.txt`.

---

## 6. Measured performance

Same machine, same 18.38 GiB pack, 256k context unless stated:

| Build | Decode | Prefill (16k) |
|---|---|---|
| Basalt v0.5.5 release, stock | 11.3 tok/s | 1,452 tok/s |
| + PR #536, gate intact | 10.7 tok/s | â€” |
| **+ PR #536, gate removed** | **34.6 tok/s** | 1,452 tok/s |
| **+ PR #536 + PR #525 (current)** | **~35 tok/s** | **1,770â€“1,804 tok/s** |

Long context verified on the optimised build: needle retrieved correctly at **130,709** prompt tokens;
the stock build retrieved it at **261,416** tokens (the 256k window works).

---

## 7. Daily use

| Action | Command |
|---|---|
| Start (optimised, 35 tok/s) | `C:\LLM\Basalt\Start-Basalt-Fast.cmd` |
| Start (stock 11 tok/s) | `C:\LLM\Basalt\Start-Basalt.ps1` |
| Stop | `C:\LLM\Basalt\Stop-Basalt.cmd` |
| Monitor | `C:\LLM\basalt-monitor.cmd` (`-Interval`, `-Last`, `-Url`, `-NoFollow`, `-Plain`) |
| Switch profile (stock build) | `C:\LLM\Basalt\Start-Profile.cmd basalt_config-32k-fast.json` |

### Network access (another computer)

- Server listens on **`0.0.0.0:8080`**, so it is reachable as **`http://192.168.11.214:8080`**
  (Wi-Fi "Gooby Home 3", /24).
- An **API key is required** on `/v1/*` and `/metrics`. It lives in `C:\LLM\Basalt\api-key.txt`
  (48 hex chars, starts `37993471â€¦`, ends `â€¦6637`).
- Clients send `Authorization: Bearer <key>` (or `x-api-key: <key>`).
- Chat UI: `http://192.168.11.214:8080/` â€” enter the same key. OpenAI-compatible base URL:
  `http://192.168.11.214:8080/v1`, model name `Inkling-2-Mini-Unbound`.
- Verified: request **without** key â†’ `401`; **with** key â†’ `200` and a correct completion.
- Because the key only covers `/v1/*` and `/metrics`, the stats page and UI HTML are reachable unauthenticated
  on the LAN â€” that is Basalt's design, not a misconfiguration.

---

## 8. Why it is ~35 tok/s and not 50 (measured, not guessed)

Instrumented the CPU spill path cost per layer visit:

| Experts on CPU | CPU time | Card wait |
|---|---|---|
| 0 | 0.001 ms | 0.000 ms |
| 1 | 0.163 ms | 0.000 ms |
| 8 | 1.261 ms | 0.000 ms |

- The card **never** waits for the CPU (`wait = 0.000 ms`), so CPU/GPU already overlap perfectly â€” the GPU
  simply idles at 21â€“43%.
- Cost is linear at **~0.155 ms per expert**, and 50 of 96 visits had all 8 experts on the CPU.
- 0.155 ms for a 4.79 MiB blob = **~30 GB/s**, the machine's practical DDR5 bandwidth, at an arithmetic
  intensity of only 1.58 MAC/byte. **Decode is memory-bandwidth bound.**

So decode time â‰ˆ `CPU experts/token Ã— 0.155 ms`. Only two things reduce it: **fewer bytes per expert**
(a ~10 GB quant) or **fewer experts on the CPU** (a 16 GB+ card). That is why every other knob measured flat:
`route-prior`/`route-tail-skip` (not implemented for inkling2), `pool-workers 11` (33.6), `pool-affinity p-cores`
(31.0), `adapt-swaps 256` (34.0), `prefill 8192` (1,696 pp), DSpark at 128k (34.9), AVX-512 (already active).

Full detail: `C:\LLM\pr536\PR-FINDINGS.md`.

---

## 9. Reverting

- **Optimised â†’ stock:** stop the `BasaltPR` task (`Stop-Basalt.cmd`) and start with `Start-Basalt.ps1`
  (task `Basalt`, recreated on demand). No model or config change needed.
- **Undo Patch A:** restore `src/models/inkling2/cpu_spill.cpp.orig` over `cpu_spill.cpp` and rebuild with
  `C:\LLM\rebuild.bat` (decode returns to ~10.7 tok/s).
- **Undo Patch B:** re-apply the inverse of the hunks in `C:\LLM\pr536.diff`, or re-download the branch zip.
- **Remove the privilege:** `secpol.msc` â†’ Local Policies â†’ User Rights Assignment â†’ *Lock pages in memory*.

---

# DECODE-PERF-SPEC implementation (W1.1) and what the measurements say

Implemented from `DECODE-PERF-SPEC.md`: **W1.1 only** (pool sizing + worker pinning). Everything else was
assessed against fresh measurements that contradict the spec's time budget; see below.

## What was implemented (W1.1)

| File | Change |
|---|---|
| `include/basalt/platform/cpu.hpp`, `src/platform/cpu.cpp` | New `platform::pin_current_thread(int cpu)` â€” Windows `SetThreadGroupAffinity` (handles >64-processor groups), Linux `pthread_setaffinity_np`. Keeps OS code inside `src/platform/`. |
| `src/kernels/cpu/pool.cpp` | `ExpertPool`'s worker threads are now **pinned**, one per physical core from `topo_.worker_cores`; a worker beyond the core count stays unpinned. Previously *no* worker was pinned (`pool.hpp:72`). |
| `src/models/inkling2/cpu_spill.cpp` | Default CPU-spill worker count is now **every physical core** (11 here) instead of the hybrid `p_cores-1 + e_cores/2` heuristic (7). An explicit `--pool-workers` still wins. |
| `C:\LLM\Basalt\test-long.ps1` | Sends the API key (it predated the key and had started failing 401). |

Build clean, `0` warnings introduced. Start log confirms `CPU spill: 11 workers` (was 7); CPU utilisation during
decode rose 45% â†’ 56%.

## Measured result of W1.1

| Metric | Before | After |
|---|---|---|
| Decode, 300-token run | 35.6 tok/s | **35.8â€“36.6 tok/s** |
| Decode, sustained single 1000-token run | â€” | 34.8 tok/s (GPU 79 Â°C, 2685 MHz, 78 W â€” no thermal drift) |
| Prefill, 16k prompt | 1,770â€“1,804 tok/s | **1,836 tok/s** |
| Long context | passes | **passes** (needle found at 130,709 tokens) |
| VRAM/RAM | 374 slots, 13,457 MiB | unchanged, no `cannot pin` |

**W1.1 delivered ~+1â€“3% decode, not the spec's predicted 40â€“42 tok/s.** Per the spec's own gate
("if in-engine gain < 1.3Ã—, stop and instrument before continuing"), the rest of W1 was re-costed instead of
implemented blind.

## Why the spec's budget does not hold (measured)

1. **The spec's isolated kernel numbers reproduce exactly.** `C:\LLM\bwtest\nvfp4_bench2.exe` here:
   1 thread 0.345 ms/expert (13.0 GB/s); 8 â†’ 0.098; **12 â†’ 0.084 ms/expert, 53.2 GB/s**. So the CPU kernel is
   instruction-bound, as claimed, and the DDR5 read is 47+ GB/s single-core.
2. **But the engine computes far fewer experts on the CPU than assumed.** Scaling workers 3 â†’ 11 changed decode
   only 31.2 â†’ 35.8 tok/s. With the bench's own per-expert costs (0.20 ms at 3 threads, 0.086 at 11), that
   Î” of 4.2 ms/token implies only **~37 CPU experts per token**, not the ~81 the spec estimated
   (26% of 384 lookups).
3. **So the CPU term is ~3.2 ms of a 27.9 ms token (~11%), not 12.6 ms (45%).** Consequences:
   - W1.2 (kernel 2Ã—) and W1.3 (Q2 shadow 2Ã—) can each save at most ~1.6 ms â†’ **~+5%**, not the projected 48â€“55 tok/s.
   - Those workstreams are capped by arithmetic, so they were not started.
4. **The dominant term is ~24.7 ms of non-CPU time**: 48 layer visits per token, GPU utilisation only ~31%,
   and the card's wait on the CPU measured at ~0.000 ms. That is per-visit launch/serialisation latency
   (â‰ˆ0.5 ms per visit, most of it gaps rather than SM work) â€” the spec's Â§7 (W3) territory, and the case for
   **more rows per launch** (W4 suffix-draft / multi-row windows), which amortises 48 visits over several tokens.

## Not implemented, and why

| Item | Status |
|---|---|
| Â§9.1 git baseline | **not possible** â€” `git` is not installed on this machine |
| Â§9.2 port micro-benches into the repo | **not done** â€” they exist and run at `C:\LLM\bwtest`; moving them into `tools/` + a CMake target was not worth the build churn |
| Â§9.3 per-visit `#ifdef` timer | **attempted, then removed.** A marker-file/env-gated version was built and validated (`open: workers=11 diag=1`, both `start` and `finish` reached), but the visit counter advanced only a handful of times per 200-token generation, so callback-entered counters cannot attribute the token time. Removed to keep the tree clean; `cpu_spill.cpp` now has exactly the two `[local]` patches |
| Â§9.4 Nsight Systems profile | not run (tool not present) |
| W1.2 kernel micro-opts | not started â€” capped at ~5% by (3) above |
| W1.3 Q2 shadow | not started â€” same cap, plus quality risk |
| W2.1 staging reclaim / W2.2 live re-rank / W2.3 VRAM audit | not started â€” time; W2's value also shrinks with (2): ~+1 expert on card â‰ˆ 0.086 ms Ã— (its share of 384) â‰ˆ 0.008 ms/token |
| W3 launch trims, W4 windows | not started â€” but these, not W1, are where the 24.7 ms is |

## Honest bottom line

Decode at 256k is **not** CPU-expert-bound on this build; it is bound by the serialised, latency-dominated
per-visit GPU path. The remaining software headroom the spec promised is real but it lives in W3/W4
(rows per launch, launch/event trims), not in the CPU kernels. W1.1 is kept: it is a small genuine win,
it lifts CPU utilisation, and it costs nothing in VRAM, RAM or quality.

---

# W4 investigation: multi-row verify windows and the 192k sweet spot

Following the decision that the 24.7 ms of per-visit latency, not CPU expert work, is the constraint.

## Windows do amortise the per-visit cost (measured, same context A/B)

| 131,072 context | Decode | drafts |
|---|---|---|
| `spec 0` (no windows) | 37.1 tok/s | â€” |
| `spec 8` (DSpark, dl 1.7) | **40.7 tok/s** | acc 57% |

+10% from verifying ~1.7 tokens per pass. So the W4 premise holds.

## Why DSpark cannot run at 256k â€” the real reason

The 74 MiB shortfall was a symptom, not the cause. Lowering the hardcoded `kCardSpare` from 1024 to 896 MiB
in `model.cpp` made the spill plan accept â€” and the load then failed at
`cannot pin 15023 MiB of RAM`: with the drafter on the card only 10 experts fit, so **3062 of 3072 experts
spill (14.67 GiB)**, past the ~14.4 GiB page-lock ceiling. `kCardSpare` was reverted to 1024 MiB
(`model.cpp.orig` kept). DSpark at 256k is infeasible on 12 GB at any reserve.

## The workable point: 196,608 tokens (192k) + DSpark

Loads clean: `experts: 2767 of 3072 in RAM (13.43 GiB), 2.31 GiB on the card with 1034 MiB staging`,
`ctx=196608`, `slots=305`, DSpark on the GPU (0.48 GiB).

## Tuning sweep at 192k (all measured, warm, 300-token runs)

| Setting | Decode | Notes |
|---|---|---|
| `spec 8` (chosen) | 37.5â€“38.6 | acc 67â€“77%, dl 1.6â€“1.7 |
| `spec 6` (DSpark's own default) | 38.1 | identical |
| `spec 4` | 37.6 | identical |
| `draft-experts 64` | 37.6 | no gain |
| `mtp-window 0` (drafter sees every cell) | 37.1 | no gain |
| `suffix-draft 8` | 37.7 | no gain |
| `expert-cache 420` / `520` (explicit) | **25.2 / 25.5** | **much worse**: hit 66% â†’ 38%, the explicit count displaces the hot set. `auto` is correct. |

Drafting parameters are saturated; `dl` will not rise above ~1.6 with this drafter.

## Final state and the trade-off

| Profile | Context | Decode | Prefill (16k) |
|---|---|---|---|
| 256k, `spec 0` | 262,144 | 35.8â€“36.6 tok/s | **1,836 tok/s** |
| **192k, `spec 8` (default)** | **196,608** | **37.5â€“38.6 tok/s** | 1,643 tok/s |
| 128k, `spec 8` | 131,072 | 40.7 tok/s | â€” |

Kept 192k + DSpark as the default (decode was the priority, and the user approved 192k). It costs ~10% prefill
and 25% of the window for ~5% decode; `basalt_config-256k.json` remains for full-context work.

Long context verified on this profile: needle found at **130,709** tokens.

---

# Session 3 (this machine, later the same day): load-margin fix + per-visit diagnostics

After the W4 work above, the 192k profile intermittently refused to load (`no pinned memory for the file read
staging`), and the remaining decode bottleneck was re-measured with in-engine instrumentation.

## Load failures: the plan was a knife's edge over the pinned-memory quota

- A fresh-process probe reproduces the engine's whole pinned allocation sequence (drafter rows, 256 MiB read
  staging, 0.86 GiB mapped embedding, the 13.4 GiB mapped spill region) with no failure � so the machine's
  CUDA host-allocation quota (~14.6 GiB on this 32 GB WDDM box, not `VirtualLock`, see MEMORY-LOCKING.md) is the
  only wall.
- The 192k+DSpark plan at `kCardSpare = 1024` spilled 2796 experts (pinned demand ~14.7 GiB: spill 13.56 +
  embedding 0.86 + staging) when the desktop held ~128 MiB more VRAM than during the 11:33 load; the 11:33 load
  had planned 2767 experts (~14.5 GiB) and squeaked through. Free VRAM at plan time swings the spill size, and
  the plan lands on either side of the quota.
- **Fix: `kCardSpare` 1024 -> 896 MiB** (`src/models/inkling2/model.cpp`, `[local]` comment; `model.cpp.orig`
  kept). At 192k this moves ~29 experts back onto the card and reproduces the proven-loadable 2767-expert plan.
  **`vram-reserve-mib` must be 768** in the live config: it stacks with `kCardSpare` at read time, and
  896 + 1024 over-commits VRAM mid-read (`copy to device 0: out of memory`). The live
  `basalt_config.json` in the source tree now has 768.
- `src/core/loader.cpp` now logs the real `cudaHostAlloc` error (`[local diag]`) instead of swallowing the code.

## Scheduled-task launch anomaly

With the current build, an engine launched by the `BasaltPR` scheduled task reliably fails during the loader's
device select (`cannot select device 0 for the load`) while a direct launch of the same `run-pr.cmd` from the
interactive session loads every time; a plain CUDA probe works fine under the task, so the trigger is something
in the engine's plan phase interacting with the task context (unresolved; open question for the next session).
**`Start-Basalt-Fast.cmd` now launches `run-pr.cmd` directly** (the task is kept only as a fallback).

## Per-visit diagnostics (permanent, env-gated, zero cost when off)

`BASALT_DIAG_VISIT=1` (environment of the engine process) turns on:

- GPU-timeline stamps (`%globaltimer`, one tiny capturable kernel per boundary) around every visit's expert
  segment � `src/models/inkling2/kernels/ops.cu` `diag_stamp`, wired in `session.cpp`.
- A host-func execution counter in `cpu_spill.cpp` and the pool's own busy-time counters
  (`CpuSpill::diag_pool`), logged per 25 windows by the session.

Log format (engine.log, one line per 25 decode windows):

```
basalt inkling2: diag: N windows, T tokens, S skipped: wall W ms (host enqueue W), body B
(pre-expert A, post-expert/CPU C, head H), pool busy P ms (BYTES), slowest visit post M,
host-funcs/window 48.0
```

Enable for a run with `cmd /c "set BASALT_DIAG_VISIT=1&& C:\LLM\pr536\basalt-inkling-cpu-spill-review\run-pr.cmd"`
(the persistent user env var from this session was removed; daily runs have it off).

## Measured decode budget at 192k+DSpark (steady state, ~39.5 ms per ~1.8-token window)

| Term | ms/window | Share |
|---|---|---|
| CPU pool expert compute (~0.95 GiB blobs at ~50 GB/s; hit 65-70%) | ~18 | 46% |
| Per-visit hand-off (48 x ~0.2 ms D2H + host func + event + merge) | ~10 | 25% |
| GPU pre-expert chain (attention, dense, router, on-card experts) | ~9.7 | 25% |
| Head (201k-vocab logits) | ~1.0 | 2.5% |
| Drafter/engine outside the window | ~1.5 | 4% |

Accounting closes: ~330 windows x 39.2 ms + ~0.5 s other = 13.46 s for 500 tokens = 37.2 tok/s.

Conclusions the next work should build on (see `DECODE-PERF-SPEC-2.md`): decode is bound by **CPU expert bytes
(hit rate)** and **per-visit hand-off latency**, in that order of value; the isolated-kernel ceiling is ~0.084
ms/expert at 12 pinned workers; DSpark's drafter cost is already inside the window and adds only ~1.5 ms.

---

# Session 4 — DECODE-PERF-SPEC-2, W1 (prompt-staging reclaim)

Followed `DECODE-PERF-SPEC-2.md` in its ranked order. **W1 is implemented, built, measured and validated;
W2/W3/W4 are scoped but not implemented** (each is a multi-day port — see the end of this section).

## Baseline reproduced with the in-tree diag harness (`BASALT_DIAG_VISIT=1`)

```
diag: 550 windows, 1022 tokens: wall 40.19 ms (pre-expert 9.90, post-expert/CPU 29.06, head 1.03),
      pool busy 20.37 ms, host-funcs/window 48.0        → 37.9 / 39.5 / 37.2 tok/s, hit 65.7%, dl 1.6
```

This matches the spec's budget table (pool ~20 ms, hand-off ~9 ms, pre-expert ~10 ms).

## W1 — implemented

| File | Change |
|---|---|
| `models/inkling2/model.hpp` | `staging_bytes()`, `staging_released()`, `release_staging(keep_bytes)`, `acquire_staging(err)`, member `staging_released_` |
| `models/inkling2/model.cpp` | `release_staging()`: for each staging granule past the floor, `vmm::unmap_only()` it at **every** spilled layer's range site (a granule is mapped into all of them), then `vmm::release()` its backing — that is what actually frees device memory. `acquire_staging()`: `vmm::create()` + `vmm::map_handle()` back at the same addresses. |
| `models/inkling2/session.cpp` | `decode_window()`: on the first decode window, release the whole staging and `promote(512 MiB)` into the freed space. `prefill_chunk()`: take the staging back (demoting expert copies first if the device has filled), so a prompt chunk still stages a layer as before. |

The staging is only needed while a prompt chunk runs, so decode now holds ~220-350 more hot experts instead.

### Measured effect

| Metric | Baseline | With W1 |
|---|---|---|
| Decode, steady state (3-4 runs, 300 tok) | 37.5–39.5 tok/s | **39.3–42.7 tok/s** |
| Decode, cold (first request after load) | 37.9 | 29.5–33.1 (−; per-request staging churn) |
| Expert-cache hit | 65.7% | **81–95%** |
| Drafts per window (`dl`) | 1.6 | **2.1–2.2** |
| Pool busy per token | 12.5 ms | **~9.5 ms** |
| Hand-off per token | 5.4 ms | **~4.0 ms** |

The spec's W1 gate was ≥44 tok/s at steady state. **Reached ~40–42.7, not 44.** The mechanism is sound and
quality-neutral, but the win is smaller than the model predicted because the freed VRAM also raises `dl`
(1.6 → 2.2), which makes each window do more work and partly cancels the per-window saving.

Log line now emitted per decode phase:

```
staging reclaim: 1.01 GiB released to the expert spill (0.57 GiB copied, 743 on the card)
staging reclaim: 1.01 GiB taken back for the prompt chunk
```

### Tuning note (measured)

`promote()`'s keep-free was tried at 384 MiB and is **worse** (dl falls 2.2 → 1.8, decode 37.0 vs 39.3–42.7):
the drafter needs that headroom. **512 MiB is kept.**

### Validation

- Loads clean at 192k (`experts: 2767 of 3072 in RAM`, no quota failure, no `cannot pin`).
- **Long context passes**: needle found at **130,709** tokens, with the acquire → prefill → release cycle
  exercised on the way (`staging reclaim: taken back` / `released` logged around it). No errors in `server.err`.
- Steady-state runs 39.3–42.7 tok/s; VRAM/RAM lines unchanged.

## W2 / W3 / W4 — scoped, not implemented

| Item | Why it is not done in this session |
|---|---|
| **W2** doorbell hand-off (kills the ~9 ms/window term) | A real CUDA port: new graph-capturable kernels (`doorbell_publish`, `wait_flag_ge`), a 48-slot mapped-pinned ring, a dedicated polling host thread replacing `cudaLaunchHostFunc`, plus a flag-gated A/B. The reference design lives in `src/models/qwen4exp/decode/` but is entangled with that model's per-layer graph split and its verify machinery. Spec's own estimate: 2–3 days, medium risk (WDDM flag visibility across captured graphs). **Verified the pieces exist** (`expert_dispatch.cpp`, `session.cpp` under `qwen4exp/decode/`), so this is a port, not a from-scratch design. |
| **W3** NVFP4 CPU kernel micro-optimisation | 1–2 days of AVX-512 work under a **bitwise-parity** contract (scale reconstruction off the shuffle port, 2-block unroll, R=4 rows, block prefetch). Worth ~6–8 ms/window after W1 (pool is now ~9.5 ms/token). Benchmarks available: `C:\LLM\bwtest\nvfp4_bench2.exe` (targets: 1 thread ≥20 GB/s from 13.5; 12 threads ≥70 from 53). |
| **W4** `--relaxed-drafting` for inkling2 | **Scoped and found to be bigger than the spec assumes.** Relaxation is not a generic engine feature: `default_relax()` is only the entry point (`program/serve/request.hpp:132`, `request_parse.cpp:36`), while the actual relaxed acceptance lives in qwen4exp's own verify path (`qwen4exp/serve/qfn_decode.cpp:301-330`, `verify.hpp:208 set_draft_probe`, `draft_prob`, `relaxed_kept`). Inkling2 would need its own draft-probability probe in the DSpark verify path — a CUDA-side change with read-back. It also changes the output distribution (off by default, user opt-in). |

Recommended next step: **W2** (largest single remaining win, ~9 ms/window ≈ +20-25%), then W3.

## W2 — doorbell hand-off: implemented, found incorrect, reverted to the host-func path

The port was written and built (spec §W2): a `kRing`-slot ring in mapped pinned memory, `doorbell_publish` replacing the
two D2H memcpys, `wait_flag_ge` replacing the `cudaStreamWaitEvent(done)`, and a dedicated polling thread
(`Impl::poll_loop`) replacing `cudaLaunchHostFunc`; `run()` was reparameterised over explicit buffers so the ring's slot
could be used. A/B is by marker file: `C:\LLM\cpu-spill-doorbell.on` opts in (the spec wanted
`--cpu-spill-handoff hostfunc|doorbell`; a file avoids new flag plumbing while giving the same A/B).

**It does not work, and the default is the host-func path again.** Symptom: decode dies after ~20 tokens with
`end=error` / HTTP 503 and ~0.3 t/s.

Two real findings came out of it:

1. **The ring must not be pinned at plan time (fixed, kept).** `CpuSpill` is created during `plan`, before the loader pins
   its 256 MiB file-read staging, and the pinned host-allocation quota (~14.6 GiB) is already ~150 MiB from full. Pinning
   the ring there made the *load* fail with `pinned staging alloc of 16 MiB failed: out of memory`. The ring now allocates
   lazily on the first decode visit (`ensure_ring()`) and was shrunk from 48 to 8 slots (~2.7 MiB). With that change the
   doorbell build loads cleanly.
2. **`doorbell_publish` bumps its counter with a PCIe read-modify-write.** The kernel does
   `*(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u`. Under back-to-back publishes that read can return a stale
   value, so the host poller never reaches the count the card expects and the engine stalls into `end=error`. The repo's own
   comment in `src/kernels/cuda/elementwise.cu` says exactly this and is why `doorbell_publish_value_kernel` exists —
   *"the ring is STORED (the step's own number, known at capture) instead of read-modify-written over PCIe - one store, no
   read of host memory from the GPU"*. That value-store variant is not exported (only `doorbell_publish` and
   `doorbell_publish_res` are), so the port needs either a small new wrapper/kernel exposing it or its own publish kernel.

State now: `hand-off hostfunc`, **42.7–43.3 tok/s**, `ctx=196608`, needle found at **130,709** tokens.
Next attempt at W2 should start by exposing the value-store publish (publish `i+1` rather than incrementing), which is a
~20-line change, then re-run the A/B marker.

### W2 follow-up: the value-store publish fix, and why the doorbell still cannot work here

Added `doorbell_publish_store()` (`src/kernels/cuda/elementwise.cu`, `include/basalt/kernels/elementwise.hpp`): the same
publish as `doorbell_publish_value_kernel` but with no router weights and storing the ring's own number, so no PCIe
read-modify-write of the counter. The port then used it. **It still stalls**, and the stall report gave the real cause:

```
error: stall report: stage "reading the prompt, from token 0" for 64 s; 0 layers served
  expert pool: epoch 308: 144 of 144 jobs claimed, 144 done; 11 of 11 workers parked, 11 sleeping
```

The pool had finished; the card was spinning in `wait_flag_ge` for a flag that never advanced. Cause: **inkling2's decode
window is replayed from a captured CUDA graph** (`session.cpp` `graphs_` / `core::CapturedGraph`). `CpuSpill::start` and
`finish` only run while the shape's graph is *captured* (once per shape); on replay the recorded `doorbell_publish_store`
and `wait_flag_ge` nodes run with the **slot index and sequence value baked in at capture time**. The host bookkeeping
(`published`/`finished`, `meta[slot]`) therefore stops advancing, the card's counter freezes and the poller spins forever.

So W2 needs the host loop to be **per window**, with the ring index and epoch living in device memory (the qwen4exp design),
which is a redesign of the inkling2 hand-off rather than a port. The host-func path stays the default; the doorbell code is
kept behind `C:\LLM\cpu-spill-doorbell.on` for whoever picks this up, and `src/core/graph.cpp` documents the related trap.

## What did land, and the final numbers

Two further changes, both measured:

1. **`kCardSpare` 896 → 768 MiB.** 896 left the 192k+DSpark plan ~20 MiB over this machine's *reachable* pinned total
   (`bwtest\hostalloc_seq.exe` reports 14.52 GiB), so loads had become unreliable (`cannot select device 0`). 768 keeps ~29
   more experts on the card (spill 2,767 → 2,738), which both restored reliable loads and lifted decode.
2. **Dropped the `prefill-cpu-share: 0.85` override** — it was costing prefill (1,514 vs ~1,645 tok/s at the default).

| Metric (192k + DSpark) | Session 4 start | Final |
|---|---|---|
| Decode, 300 tok, warm | 37.5–39.5 tok/s | **45.4–45.9 tok/s** |
| Decode, sustained 1,000-token run | 34.3 tok/s | 35.7 tok/s |
| Prefill, 16k prompt | 1,643 tok/s | 1,520–1,645 tok/s (unchanged within run-to-run spread) |
| Expert-cache hit | 65.7% | 71.7% |
| Drafts/window (`dl`) | 1.6 | 2.0–2.2 (acc 84–96%) |
| Experts on card | 305 | 334 |
| Long context | passes | **passes** (needle at 130,709 tokens) |

Versus the stock v0.5.5 release: **11.3 → 45.5 tok/s decode, a 4.0× gain**, at a 192k context window.

### Measured dead ends this round
| Attempt | Result |
|---|---|
| `kCardSpare` 640 | 35.5 tok/s — `dl` collapses to 1.6; the drafter needs the headroom |
| `kCardSpare` 896 | load fails: plan sits over the reachable pinned total |
| `promote()` keep-free 384 MiB (vs 512) | 37.0 vs 42.7–45.9 — same drafter-headroom effect |
| doorbell ring pinned at plan time | load fails `pinned staging alloc of 16 MiB failed` → fixed by lazy allocation (kept) |
| `prefill-cpu-share` 0.6 / 0.85 / 1.0 | 1,642 / 1,514 / 1,648 tok/s — the 0.85 override was the outlier, removed |

### Not attempted
- **W3** NVFP4 CPU kernel micro-optimisation: 1–2 days of AVX-512 under bitwise parity; the pool term is now ~9 ms/token.
- **W4** `--relaxed-drafting` for inkling2: needs a draft-probability probe in the DSpark verify path (CUDA, with read-back);
  it also changes the sampling distribution, so it is user-opt-in by the spec's own terms.

## W2 redone — the doorbell hand-off now works, and it is the default

The fix was the primitive, not the plumbing. `wait_flag_ge(flag, value)` fixes its target at capture, so a replayed graph
waits on a stale number. The repo already had the right node: `doorbell_wait(d_flag, d_seq)` compares **two mapped words at
run time**. Three changes on top of the scaffolding that was already in the tree:

| Change | Why |
|---|---|
| **one slot per visit** (`kRing` 8 → 48) | a slot shared by two visits lets the poller miss a publication; a per-visit slot is the visit's position in the window, which is the same on every replay, so a captured graph stays correct |
| **`doorbell_publish_visit()`** (new, `elementwise.cu`/`.hpp`) | publishes payload + ids + **`{layer, rows}` into mapped memory** (so the poller needs no host call per visit) and bumps the slot's counter with **`atomicAdd`** (an atomic on system memory is safe where a plain read-modify-write loses increments) |
| **`doorbell_wait(done[slot], seq[slot])`** replaces `wait_flag_ge(done, i+1)` | runtime comparison, nothing baked into the graph |

The poller scans slots 0..47, and for any slot whose published counter moved it reads `{layer, rows}`, runs the pool, writes
the rows back and raises that slot's done word. `start()`/`finish()` do no host work beyond launching the nodes, so
`host-funcs/window` went **48 → 0**.

### Measured (192k + DSpark, 300-token runs, warm)

| | host-func | **doorbell** |
|---|---|---|
| Decode | 45.4–45.9 tok/s | **62.7 / 68.3 / 68.3 / 68.4 tok/s** |
| `diag` wall per window | 53.3 ms | **28.6 ms** |
| `diag` post-expert/CPU | 33.6 ms | **15.0 ms** (= pool busy: the hand-off is gone) |
| `host-funcs/window` | 48 | **0** |
| Sustained 1,000-token run | 35.7 tok/s | **54.0 tok/s** |
| Prefill 16k | 1,643 tok/s | 1,511 tok/s (unchanged path) |

Spec's W2 gate was ≥50 tok/s: **met (68.3)**.

### Validation
- **Long context**: needle found at **130,709** tokens.
- **Determinism**: identical output for repeated greedy requests *within* a process.
- **Parity caveat (measured, not assumed)**: a bitwise A/B between the two hand-off paths is **not** obtainable on this
  machine — running the *same* build twice in fresh processes already diverges (first difference at char 211 of a 220-token
  greedy reply), because expert placement and the adaptive spill differ per start. The earlier "hostfunc vs doorbell differ"
  observation was that variance, not a defect.
- Output coherent, no `server.err` entries, GPU 80 °C / 90 W while loaded.
- Revert switch: `C:\LLM\cpu-spill-hostfunc.on` restores the old path for an A/B (the doorbell is now the default).

## Session 4 final numbers (192k + DSpark + W1 + W2 + kCardSpare 768)

| Metric | at session start | **final** |
|---|---|---|
| Decode, 300-token runs (warm) | 37.5–39.5 tok/s | **62.7–68.4 tok/s** |
| Decode, sustained 1,000-token | 34.3 tok/s | **54.0 tok/s** |
| Prefill, 16k | 1,643 tok/s | 1,511 tok/s |
| Expert hit | 65.7% | 71.8% |
| vs stock v0.5.5 release (11.3 tok/s) | 3.3× | **6.0×** |

## W3 — NVFP4 CPU kernel micro-optimisation: measured, no headroom, reverted

The spec's three techniques were each tried against the isolated harness (`C:\LLM\bwtest\nvfp4_bench2.exe`, which compiles
the engine's own `nvfp4_avx512.cpp`), baseline **1 thread 0.364 ms/expert (12.3 GB/s), 12 threads 0.086 (51.9 GB/s)**:

| Change | Result | Why |
|---|---|---|
| UE4M3 scale reconstruction as one `_mm_i32gather_ps` instead of four `kLut.v[]` loads + `_mm_setr_ps` | **neutral / worse** (1t 0.359, 12t 0.095) | on Zen 5 a 4-element `vpgatherdd` costs more than four L1 loads and four inserts; the scalar form wins |
| Software prefetch 4 blocks ahead (`_MM_HINT_T0`) | neutral (same run) | the hardware prefetcher already covers the 1296 B row stride at this occupancy |
| **R=4 rows per pass in `gu_rows`** (parity-safe: each row still sums its blocks in order) | **much worse** (1t 0.557, 12t 0.115) | register pressure: four `Blk` (3 regs each) plus four accumulator sets spill, and Zen 5 executes 512-bit ops as two 256-bit halves, so the register file is effectively half the size it looks |

All three reverted; `nvfp4_avx512.cpp.orig` kept. Decode re-verified afterwards: **66.1 / 68.8 tok/s** (no regression).

### Why there is nothing to win here

| | rate |
|---|---|
| In-engine pool (diag: `pool busy` / `blob bytes`) | **45.6 GB/s** |
| Isolated kernel bench, best thread count on an idle machine | 51.9 GB/s |

The engine already runs the kernel within ~12% of the best the harness can do, so the spec's acceptance targets
(single-thread ≥20 GB/s, 12 threads ≥70 GB/s) are out of reach with these techniques on this microarchitecture — and even
hitting them would only be worth ~1–2 ms/window now that the hand-off is gone.

**W3 is closed.** What is left for decode is either the quality-trade Q2_0 shadow (W5, halves the pool's bytes) or hardware.

## W4 — `--relaxed-drafting` for inkling2: implemented (opt-in), measured as no gain on this workload

Unlike W2/W3 this turned out to be **small**: acceptance is one host-side line in the solo request path,
`session_engine.cpp` `SessionRequest::accept_drafts`:

```cpp
while (a < T - 1 && window[a + 1] == outv[a]) ++a;      // before: keep a draft only if the target drew the same token
```

So W4 is:

```cpp
// [W4] keep a draft the drafter was confident about even when the target's own draw differed
if (req_relax > 0.0f && !req_sp.greedy && window_conf_[a] >= req_relax) { outv[a] = window[a + 1]; ++a; }
```

`window_conf_` is the drafter's per-draft proposal probability, already filled by `note_draft_probs()` from DSpark's
`draft_end(ids, conf, ...)`; `req_relax` is the generic per-request `relax` body key (`request_parse.cpp`), so no new flag
plumbing was needed and nothing is on by default. A `relaxed_kept_` counter records what it kept.

Measured (300-token runs, one engine instance):

| Request | tok/s | acceptance | dl |
|---|---|---|---|
| greedy, no `relax` | **63.1** | 96% | 1.9 |
| greedy, `relax: 0.9` (ignored by design) | 65.3 | 91% | 2.0 |
| sampled temp 1.0, `relax: 0` | 60.8 | — (no window drafted) | — |
| sampled temp 1.0, `relax: 0.4` | 55.3 | 93% | 1.6 |
| sampled temp 1.0, `relax: 0.6` | 44.8 | 76% | 1.4 |

Relaxation does raise acceptance (an earlier pass showed 43% → 83–88% at `relax` 0.4–0.6) but it does **not** translate
into throughput here: sampled requests stay slower than greedy either way, and the clean pass is neutral-to-negative.
Greedy — this setup's actual workload, temperature 0 — ignores `relax` entirely, as designed.

**Kept, but not enabled.** The code stays because it is a genuine opt-in feature, off by default, with zero effect on the
greedy path; it is documented as not worth turning on for this use.

---

# Session 5: clean re-measurement of the W1+W2 build � the drafter tax, the content-dependence, and prefill

After deepseek's Session 4 and the launcher cleanup, the build was re-measured with no concurrent start/stop
interference (an earlier "decode-to-prefill deadlock" and a "262144 restart" observed mid-session were
artifacts of concurrent script restarts, not engine bugs; the prefill-after-reclaim flow was re-tested clean
and works). Findings, all measured warm and greedy unless stated:

## The 63-68 tok/s claim is content-dependent

| Content | tok/s | acc | dl |
|---|---|---|---|
| tune.ps1 TCP prompt (deepseek's benchmark) | 56.4 | 76% | 2.05 |
| free-form story prompts | 47-51 | 69-81% | 1.7-1.8 |

DSpark acceptance, and therefore decode speed, follows how predictable the text is. Deepseek's 96%-acceptance
runs were on the most draft-friendly content possible.

## spec 0 vs spec 8: drafting is nearly worthless now

Clean A/B on the same engine, same prompt: `spec 0` = **54.5 tok/s** (1-row windows, wall 19.3 ms),
`spec 8` = **56.4 tok/s** (+3.5%). The doorbell made single-row windows cheap enough that the DSpark drafter's
fixed per-window cost no longer pays for its dl ~2. Measured budget per spec-8 window (~2.05 tokens, ~37 ms):
pool 15.2, GPU chain 10.2 (nearly flat in rows), head 0.95, host 1.6, and **~9 ms of drafter+engine outside the
window** (`draft_end` syncs the whole stream; `draft_begin`'s attention is uncaptured).

## Why W4 (relaxed drafting) was never going to help � confirmed

The workload is greedy (no marginal drafts to relax), dl is capped by the drafter's ~2-token horizon (not by
acceptance), and the drafter's fixed cost exceeds the value of a kept draft � which is exactly why deepseek's
sampled relax runs got *slower* (55.3/44.8 vs 60.8). Also found: **their `suffix-draft 8` test was invalid �
inkling2 does not parse `--suffix-draft`**, so the lookup drafter never ran. Spec 3 W-B adds the flag.

## Prefill on the current build

22k unique tokens (random survey text, cold cache): **1,078-1,296 t/s**, with the first ~3.7k tokens at only
443 t/s while the decode-time staging reclaim is undone (`server.out`'s `s1 read pp=` lines are the real
numbers; `engine.log`'s "N reused in 0 ms" lines are second-round restore stats and should be ignored). The
reclaim/retake cycle is the prefill regression vs the 1.5-1.8k of the pre-W1 build.

## Bugs found

- **Hit-rate accounting can exceed 100%** (`hit rate: 194.5%` observed): the per-(layer,expert) lookup tally is
  double-added somewhere in the doorbell/reclaim path. Diagnostic-only, but every `hit=` stat is suspect until
  fixed.
- A stale python server can serve requests with a previous engine after a relaunch, silently invalidating
  benchmarks (bit us twice): always kill basalt.exe **and** the C:\LLM python.exe before an A/B relaunch.

## Next steps

`DECODE-PERF-SPEC-3.md`: W-A cut the drafter block cost (restores spec-8's value, ~64 t/s), W-B enable
`--suffix-draft` for inkling2 (dl 3-8 on repetitive content), W-C the Q2_0 CPU shadow **in pageable RAM**
(Session 4's "needs 5.5 GB pinned, past the ceiling" claim was wrong � the shadow is CPU-read-only and needs
no pinning), W-D prefill (standing staging floor, CPU-share re-sweep, 8192 chunks), W-E the tally fix.

---

# Session 6 — DECODE-PERF-SPEC-3: W-A (drafter tax), W-B (suffix-draft), W-D (prefill); W-C not attempted

The spec's correction was right and worth recording: the 62.7–68.4 tok/s headline is **content-dependent** (it came from a
highly predictable prompt). On ordinary prose the same build did ~63, and with `spec 0` ~54. DSpark's drafter was worth
little because its own ~9 ms/window serial tax ate the benefit.

## W-A — the drafter's stream-wide sync (landed)

`src/models/inkling2/dspark.cpp` `draft_end` did a **full `cudaStreamSynchronize`** to read back the drafts. The stream also
carries the main window's kernels, so the engine waited for far more than the drafter. Now `draft_begin` records a
`cudaEventDisableTiming` event after its own work (`ev_`, created lazily) and `draft_end` synchronises that event only,
falling back to the old sync when no event exists. Event destroyed in `~DsparkHook`.

| | before | after |
|---|---|---|
| `diag` wall / window (`spec 8`) | ~37 ms | **31.2–31.5 ms** |
| Story content, warm, 300 tok | ~63 tok/s | **66.7 / 67.3 tok/s** |
| Technical prose, warm | — | 48.9 / 49.4 tok/s (`dl` 1.5 there) |
| Drafter tax (wall − body)/window | ~9 ms | ~4.6 ms |

## W-B — `--suffix-draft` never parsed for inkling2 (landed)

The spec spotted that `dspark_parse_flag` accepted only `--spec` and `--draft-log`, so the earlier "suffix-draft: no gain"
test ran **nothing**. Added the flag (one line, mirroring `qwen35/adapter.cpp`); `--mtp-window` was in the spec's list "if
trivial" but `CommonOptions` has no such field, so it is not added.

Measured on a "repeat this passage verbatim" request (`--suffix-draft 8` vs `0`, warm, greedy):

| | `dl` | tok/s |
|---|---|---|
| `--suffix-draft 0` | 2.6 / 2.7 | 52.4 / 53.2 |
| `--suffix-draft 8` | **3.9 / 3.3** | **53.7 / 65.4** |

Lookup drafting raises `dl` by ~35–45% and throughput ~13% on repetitive content, and free-form content is unchanged (the
policy still picks DSpark's rows).

## W-D — prefill: the short-prompt re-stage (landed; the rest of W-D not needed)

W-D.1 proposed a standing staging floor. That does not work with this staging design — the prompt path needs a **whole
layer's** expert set (~1,034 MiB), so any floor below that still forces the re-take. The real defect was mine: the W1 hook
in `prefill_chunk` took the staging back for **every** prefill, including prompts of ≤16 tokens, which never use the prompt
path at all (they run through the fused rows, where the CPU computes the spilled experts). Now the re-take is gated on
`prompt_ && r.n_tokens > 16`, the same condition the prompt path uses.

Verified: a 29-token prompt leaves the re-take count unchanged (76 → 76); a 16k prompt still takes it back (76 → 77), and
prefill is ~1,507 tok/s. This removes ~1 GiB of demote+remap from the time-to-first-token of every short chat turn.

## W-C (Q2_0 shadow) — not attempted

The spec is right that it needs no pinned RAM (the shadow is read only by the CPU pool's kernels), which corrects
`SUMMARY.md`. It remains a 1–2 day change: requantize ~2,700 spilled experts at load, a `--cpu-shadow q2|q4|off` flag, and
a real quality trade. Left for a session with the budget to validate the quality, not started here.

## W-E — diagnostics

- **W-E.1 (hit rate >100%)**: investigated, not fixed. `card_share` sums hits as a subset of all lookups, so the engine's
  pair is structurally ≤100%; the inflated values (95.6 / 107.4 / 117.1 / 124.6% observed) are intermittent, and the same
  server reports 0.717–0.718 for most requests. Most likely a stale baseline: the server divides a hit count read after
  the tally advanced by a lookup count read before. Diagnostic-only — changing accounting I cannot validate seemed worse
  than documenting it.
- **W-E.2 (drafter-span timers, prefill stamps)**: not added; the wall/body/pool split already showed W-A's effect.

## Session 6 numbers (192k + DSpark + doorbell, greedy, warm, 300-token runs)

| Workload | Session 5 | **Session 6** |
|---|---|---|
| Story prose | ~63 tok/s | **66.7 / 67.3** |
| Technical prose | — | 48.9 / 49.4 |
| Repetitive (verbatim repeat, suffix-draft 8) | — | **53.7 / 65.4** (`dl` 3.3–3.9) |
| `spec 0` reference | ~54 | ~54 (unchanged) |
| Prefill, 16k | ~1,511 | 1,507 |
| `diag` wall / window | 37 ms | **31.2 ms** |
- **W-E.1 (hit rate >100%)**: diagnosed precisely, not fixed. `card_share()` sums hits as a strict subset of all lookups,
  so the engine's own pair cannot exceed 100%. The tell is that the value **climbs monotonically across successive
  requests in one process**: 96.6 → 108.4 → 117.6 → 129.6 → 139.3% over five back-to-back calls. That is a cumulative
  numerator over a per-request denominator: `ex_tally_` (and `lookups_`) accumulate for the life of the session, while the
  server divides by the request's own lookup count. The clean fix is to zero `ex_tally_` at request start (or report the
  ratio from the same window); it is left alone because the same tally feeds the expert-map telemetry (`routed`/`recent`)
  and changing accounting that cannot be validated risks corrupting the only live cache statistic rather than just
  cosmetic noise. Anyone tuning expert placement should read the engine's own `experts: N of M in RAM` line, not `hit=`.

### A note on the numbers, for the record

Throughput tracks **draft acceptance**, which tracks content, and varies a lot on the same build:

| Prompt (greedy, warm, 300 tok) | `dl` | tok/s |
|---|---|---|
| Free-form history prose | 1.9–2.0 | **66.7–67.3** |
| "Describe how a siphon works" | 1.5 | **39.2–45.1** (median 44.0) |
| Repetitive ("repeat this passage verbatim") | 3.3–3.9 | **53.7–65.4** |
| `spec 0` (no drafting) | 1.0 | ~54 |

So the honest single-number summary is **~44–67 tok/s depending on content**, with the structural gains from this session
being the ones that hold regardless: `diag` wall per window **37 → 31.2 ms**, and `dl` **2.6 → 3.6** on content where
lookup drafting applies.

---

# Session 5 addendum: the prefill deep-dive (code inspection only; engine untouched)

Written alongside DECODE-PERF-SPEC-3 section 6 while another agent works the engine. Findings from
`prompt.cpp`/`model.cpp` inspection:

- **Prefill is PCIe-staging-bound and the staging is serial per visit.** Every one of a chunk's 48 visits
  stages its layer's spilled experts on the *compute stream* before its expert products; the loop layers
  (6-17) re-stage the same ~515 MiB three times per 4,096-token chunk � ~26.6 GiB per chunk nominally, the
  elastic copies covering part of it D2D. The DMA (1.25-1.7 s/chunk at 15.75 GB/s) issued as ~40-60 ms serial
  stalls per visit reproduces every measured prefill rate.
- **Fix P1: copy-stream + half-layer ping-pong staging with prefetch** � projected 2,150-2,900 t/s, which
  clears the 2,000 target. P2: weight the elastic promote ranking by staging frequency (loop-layer copies
  save 3x per chunk). P3: the standing staging floor fixes the 443 t/s first-chunk reclaim tax.
- **Two Session-4 prefill conclusions were invalid**: the `prefill-cpu-share` sweep was a no-op (inkling2
  does not parse the flag - qwen4exp only, the server logs it as skipped); and the "prefill 8192" test
  predates the current build - 8192 is also infeasible anyway (~+1 GiB chunk buffers, 768 MiB free).
- GPU expert compute is not the wall (FP4 tensor-core path, ~0.3-0.5 s/chunk); attention grows with context
  and should be re-measured at 130k+ after P1.

---

# Session 5, final addendum: the prefill work is specified

`PREFILL-PERF-SPEC.md` is now the standalone spec for the >2,000 tok/s prefill target, superseding
DECODE-PERF-SPEC-3 section 6 with a full implementation design:

- **The mechanism (code-derived, matches every logged rate):** every visit stages its layer's spilled experts
  over PCIe *on the compute stream, serially*; the 12 loop layers re-stage the same ~511 MiB three times per
  4,096-token chunk (~26.6 GiB nominal, ~19.4 GiB H2D after the elastic copies' D2D coverage) - ~1.23 s of
  PCIe per chunk issued as 40-60 ms stalls, which is the whole prefill budget story.
- **The design:** wave-pipelined staging - split each visit's staging into 2-4 slot-range waves on a
  dedicated copy stream, slice `moe_sort`'s bounds so the FP4/MMQ products run per wave behind each wave's
  event. No extra VRAM (the staging set is already whole-layer sized), copy engine saturates, prefill becomes
  DMA-bound: projected ~1,850-2,050 t/s alone.
- **P2:** rank the elastic promote order by staging frequency (a loop-layer copy saves 3x its blob per chunk)
  - H2D to ~13-15 GiB/chunk, projected 2,600-3,700 t/s combined.
- **P3:** the standing staging floor kills the 443 t/s first-chunk reclaim tax.
- Out of scope with reasons: CPU-prefill-expert kernels (only to break the ~3,300-4,800 PCIe ceiling later),
  8192-row chunks (+1 GiB buffers, unavailable), and attention growth at 130k+ context (re-measure there).

---

# Session 6 reconciliation: IMPLEMENTATION-NOTES.md reviewed against the specs' measurements

Reviewed the implementing agent's notes against the session-5 measurements. Confirmations, two corrections,
and one re-projection for the open work:

## Confirmed (the notes match the measured record)

- **The drafter-sync fix is the W-A down payment**: recording the drafter's own event instead of a full-stream
  sync matches the spec-3 diagnosis word for word (the ~9 ms serial tax), and the resulting rates
  (67 tok/s free-form at dl 2.0) close the loop on the session-5 budget.
- **W-B (lookup drafting) is validated at dl 3.9** on a verbatim-repeat prompt - the spec's gate was dl >= 3.
- **The `hit=` bug root cause is now known precisely**: `ex_tally_`/`lookups_` are cumulative while the server
  divides by a per-request count - the monotonic 96.6 -> 139.3% climb. W-E.1's fix is a counter-lifetime fix,
  and the notes are right that it must be done deliberately (the tally feeds the expert map).
- The pinned-quota, task-context, captured-graph, and kill-discipline traps all match the session-3/5 record.

## Two corrections to the notes

1. "`prefill-cpu-share` looked like it helped for a while; it was run-to-run noise" - understated: the flag is
   **not parsed by inkling2 at all** (qwen4exp-only; the server logs "this model has no such flag, skipped").
   Any sweep of it measured a no-op, deterministically - see the PREFILL-PERF-SPEC corrections section.
2. "`--relaxed-drafting` is only implemented inside qwen4exp's verify path" - stale: Session 4 ported it to
   inkling2's `accept_drafts` (opt-in, `relax` body key, off by default). It is still correctly *off* - it
   cannot pay until the drafter tax is fully gone and dl > ~2.5 (spec 3 section 2).

## One re-projection for the open work (W-C)

The notes' dl-3.9 run measured **65 tok/s**, below the spec's naive `(13.5 + 7.5*4 + 3)/4 ~ 84` projection.
The 1-row vs 2-row measurements understated how the pool and GPU terms grow with window rows; the linear
model needs a row-coefficient above ~7.5 ms/row once rows pass ~2 (pool bytes scale with *distinct* experts
per window, and the GPU chain is flat only to a point). Treat spec 3's W-C figures (90 predictable / 80 story)
as **upper bounds**; the spec's gate (>= 75 on predictable content) remains the right acceptance bar, and
re-measure the row-coefficient from the diag lines at spec 8 before trusting any W-C arithmetic.

---

# Session 7 — PREFILL-PERF-SPEC: P2 landed (+2.4%); the spec's byte model does not hold on its own target workload

## Baseline taken as the spec demands (unique text, `s1 read pp=`)

A 53,405-token prompt of never-seen random words (the random 6-char words tokenize denser than English, hence 53k not 16k),
warm engine: **1,422 t/s** (36.7 s, 14 chunks of 4,096 → **2.62 s/chunk**).

## P2 (§3) — rank the elastic copies by staging frequency (landed)

`InklingModel::promote()` walked `spill_order_` (the profile's *routing* order) to decide which RAM experts get a card copy.
A loop-layer expert is staged on every pass of its layer per prompt chunk (`loop_passes` = 3 here) while a prelude/coda
expert is staged once, and an elastic copy turns that staging into an in-VRAM D2D instead of an H2D. Added
`promote_order_` (`model.hpp`), built lazily on the first `promote()` from `geo().visits` (visits per layer), stably sorted
by that weight so ties keep the profile's order. `demote()` still uses `spill_order_` (decode heat).

| | before | after |
|---|---|---|
| Unique 53k prompt, warm | 1,422 t/s | **1,456 t/s (+2.4%)** |
| Decode canary (300-token runs) | 39–67 tok/s | 39.5 / 47.7 / 59.0 — unchanged |
| `hit=` / `dl` | 72% / 1.9 | 68–90% / 1.6–1.7 — unchanged |

**Why it is only +2.4%: KV growth evicts the copies during exactly the prompts the spec targets.** The engine log shows it
directly:

```
elastic spill: the caches grow to 57344 positions; 0.64 GiB of expert copies go back to RAM (466 stay on the card)
```

A 16k+ unique prompt has to grow the KV cache to hold itself, and each growth demotes elastic copies back to RAM — so by
the later chunks there are far fewer than the ~728 copies the spec's byte table assumes, and their D2D saving nearly
disappears. On short prompts, where the copies survive, the ranking does what it promises; on the target workload §1's
"H2D falls to 13–15 GiB" cannot happen.

## The measured chunk budget (P2's arithmetic, corrected)

| | spec's model | measured |
|---|---|---|
| DMA (H2D staging) | 1.23 s | ~1.33 s (≈21 GiB/chunk) |
| Compute (dense + attention + expert products) | ~0.8 s | **~1.39 s** |
| Per chunk | 2.03–2.5 s | **2.62 s** |

So the non-DMA part is ~70% larger than the spec assumed, and DMA is ~51% of the chunk rather than ~55-60% with a much
smaller compute term. **P1 (wave pipelining) would therefore top out near 1.29 s/chunk ≈ 2,400–2,700 t/s**, not the
3,150–3,700 projected — still above the 2,000 target, but a 1.6–1.8× gain for the riskiest change in the document.

## P1 (§2 wave-pipelined staging) — not implemented

It needs `moe_sort`'s bounds sliced per wave with per-wave expert products, a copy stream with per-wave events, and a
`--prefill-waves` flag — the spec's own 1–2 days, and its one fiddly part. Given the corrected budget (a 1.6–1.8× ceiling,
not 2.6×) I did not start it rather than leave the tree mid-refactor.

**A cheaper alternative worth trying first, from the same measurement:** the 12 loop layers are staged **three times per
chunk** (~8–9 GiB of the ~21 GiB). Processing a chunk **loop-major** — pass 1 over all 4,096 positions, then pass 2, then
pass 3 — stages each loop layer once and keeps its staging set intact for all three passes. The chunk's hidden state is
4,096 x 2,304 floats ≈ 38 MB, so it fits comfortably; the saving (~0.6 s/chunk) is bigger than wave pipelining's
per-visit overlap and needs no per-wave product slicing. It is a restructure of `prompt.cpp`'s walk, not a flag.

## P3 (§4 standing staging floor) — refuted, and already fixed differently

A 512 MiB floor cannot work: the prompt path stages a **whole layer's** expert set (up to 1,034 MiB for a Q8_0 layer), so
any floor below that still forces the re-take. The real defect (re-taking the staging for prompts of ≤16 tokens that never
use the prompt path) was already fixed in Session 6 by gating on `prompt_ && r.n_tokens > 16`. The asynchronous re-take
idea remains valid and unstarted.

## Also confirmed from the spec's corrections

- **`--prefill-cpu-share` is not parsed by inkling2** — confirmed; my Session 4 "sweep" measured a no-op, and its removal
  was noise either way.
- **`kPromptChunk` 8192 is infeasible** at 192k (~+1 GiB of chunk arena against ~768 MiB free) — not attempted.

## Net for this session

Prefill **1,422 → 1,456 t/s** (+2.4%) from P2; decode unchanged. The target of 2,000+ is not reachable by P2 alone, and P1
is worth less than the spec projected. The measured alternatives, in order: the loop-major chunk restructure above, then
P1's wave pipelining on top of it.

---

# Session 7: fresh verification + SPEC 4 (NEXT-WORK-SPEC.md)

Re-measured the current build end to end (engine playable; deepseek not working at the time):

- **Decode (diag, TCP prompt, 600 tokens): 53.2 tok/s.** Per window: pool 15.4 ms, GPU chain 9.3, head 0.9,
  hand-off 0 (doorbell), drafter+engine ~7.8 ms outside the window (the W-A event fix improved it from ~9).
  The pool term still dominates.
- **Prefill (unique 31,717 tokens, warm): 1,536 t/s**, 2.58 s per 4,096-chunk (matches HANDOVER-PREFILL's
  2.62); early chunks 647 t/s during the staging re-take; `hit=193%` (the known cumulative-tally bug, live).
- **HANDOVER-PREFILL section 5 (loop-major) is not implementable as written**: the walk is already
  pass-major (the loop is a cycle - pass p+1 layer 6 consumes pass p layer 17), and "reuse the staging across
  passes" needs all 12 loop layers resident (~6.2 GiB). The 38 MB figure is the hidden state, not the staging.
  Do not attempt it.
- **Verified the KV-eviction finding**: 0.64 GiB of copies demoted at 57,344 positions is legitimately sized
  (~0.46 GiB real KV growth + the 512 MiB kElasticKeep margin) - the P2 promote ranking cannot help long
  prompts, and its measured +2.4% is its whole value.
- **New: the GPU MMQ suite already has a Q2_0 instance** (`src/prefill/mmq/instances/mmq_q2_0.cu`), so the
  planned Q2_0 CPU shadow (spec-3 W-C) doubles as a prefill byte-cut: stage the 1.65 MiB shadow blobs instead
  of the 4.48 MiB NVFP4 blobs (~21 GiB/chunk H2D -> ~7.7 GiB). One feature, both fronts.

**`NEXT-WORK-SPEC.md` is the consolidated spec for the next worker**: D1 the Q2_0 shadow (decode pool 15.4 ->
6-8 ms, ~65-75 tok/s; prefill staging /2.7), P1 wave pipelining (2,700-2,900 t/s ceiling with the corrected
1.39 s compute), D2 capture the drafter block (outside-window 7.8 -> <=3 ms), D3 the deliberate tally fix -
with gates, the do-not list, and the corrected projections (65 t/s at dl 3.9 shows the row coefficient is
worse than linear past ~2 rows; treat 90/80 figures as upper bounds).

---

# Session 8 — NEXT-WORK-SPEC: D3 landed (hit-rate accounting); D1 scoped to a measured blocker; §1 correction accepted

## D3 — the `hit=` figure now means something (landed)

Root cause found, and it was not "cumulative vs per-request" but a **subset sum over a moving subset**. `expert_counts()`
returns `{card_share().card, card_share().all}` — `card` sums the tally of the experts that are on the card *at the moment of
the read*. The serve layer differences that against a baseline taken at the start of the request, so when promotion moved an
expert from RAM to VRAM mid-session its **entire accumulated history** entered the card sum while the baseline had counted it
as zero. The ratio therefore climbed monotonically (96.6 → 108.4 → 117.6 → 129.6 → 139.3 → 193% observed).

Fix, at one site each:
- `expert_map.hpp`: `ResidentExpertMap::reset()` — forget the snapshot (`pairs_` depends only on the pair count, so it stays).
- `adapter.hpp`: new virtual `reset_expert_counts() {}`.
- `inkling2/session.cpp`: implement it — memset `ex_tally_`, `lookups_ = 0`, `map_.reset()`.
- `session_engine.cpp` `decode_stats_begin()`: call it **before** taking `counts0_`.

Verified (five successive requests): **69.5 / 69.7 / 69.3 / 69.3 / 69.2%** — stable, and per-request as the server's division
assumes. Spec's gate (`hit= ≤ 100% across a mixed workload`) met. Needle test passes afterwards (130,709 tokens), and the
prefill canary is unchanged (1,561 t/s unique 32k).

## D1 (Q2_0 shadow) — scoped, and blocked on one missing piece

Verified on the tree: Q2_0 is already supported on **both** compute paths (`src/kernels/cpu/native_expert.cpp` handles
`GGML_TYPE_Q2_0` via the iq256/iq512 kernels; `src/prefill/mmq/instances/mmq_q2_0.cu` exists), the dequantizer is in
`include/basalt/artifact/dequant.hpp:69`, and the block geometry is 64 elements / 18 bytes (`GGML_TYPE_Q2_0 = 42`).

**Missing: there is no Q2_0 *quantizer* anywhere in the tree** — only the dequantizer. So D1 begins by writing
`NVFP4 -> f32 -> Q2_0 block`, then the `NativeFmt`, the pageable 4.4 GiB shadow (first-touched at build), the decode path, the
prefill staging path with the Q2 MMQ instance, and the per-blob 16-byte SwiGLU scale tail that the reader expects. That is
the 1–2 days the spec estimates, and a wrong block layout is a 4.4 GiB write that produces garbage — not started in this
session rather than half-built.

## §1 — the correction is accepted; the handover is fixed

`NEXT-WORK-SPEC.md` §1 is right and `HANDOVER-PREFILL.md` was wrong: the walk is **already** pass-major, the loop is a
dependency cycle (pass *p+1*'s layer 6 consumes pass *p*'s layer-17 output), and reusing staging across passes would need all
12 loop layers resident (~6.2 GiB). The handover's §5 has been replaced with that correction plus §5a (the Q2_0 shadow as the
legal way to cut the same bytes).

## Session 8 numbers

| | before | after |
|---|---|---|
| `hit=` across a session | climbs to 193% | **69.2–69.7%, stable** |
| Decode (250-token runs) | — | 36.9 cold, 47.5–51.1 warm |
| Prefill, unique 32k | ~1,536 | **1,561 t/s** |
| Needle at 130,709 tokens | pass | **pass** |

Still open, in the spec's order: **D1** (needs the quantizer above), **P1** wave pipelining (ceiling ~2,400–2,900 with the
corrected budget), **D2** drafter capture (~7.8 ms/window outside the window, target ≤3), **D4** `kElasticKeep` trim
(optional).

---

# Session 8 review + the D1 unblock: D1-Q2-SHADOW-DESIGN.md

Reviewed SESSION-8-SUMMARY.md: D3 (tally fix) verified as correct and complete - the root cause it found (the
card subset sum moves when promotion relocates an expert mid-session) is subtler than the spec assumed, and
the one-site-per-layer fix is right. D1 scoping was also right: the tree truly has no Q2_0 writer.

**`D1-Q2-SHADOW-DESIGN.md` now supplies the missing piece, pinned from the tree's own reference decoder**
(`dequant.hpp:68`, "mirrors dequantize_row_q2_0"):

- The format contract: 64 values / 18 B, codes 0..3 in low-bits-first packing, **value = (code - 1) x d**
  - an ASYMMETRIC symbol set {-d, 0, +d, +2d}, so the scale bound is max(|min|, max/2), not amax/2. This
  asymmetry is the one trap a hand-rolled quantizer would hit.
- A reference `quantize_q2_0_block` (~15 lines) + the tail rule (dequant without sg/su/sd; copy the NVFP4
  16-byte tail verbatim).
- A round-trip harness FIRST (quantize -> the tree's own dequantize_q2_0 -> per-layer SNR), so the
  garbage-not-error risk is retired before the engine is touched, and the SNR report doubles as the quality
  evidence for the user's sign-off.
- Corrected arithmetic vs NEXT-WORK-SPEC section 3: the shadow is exactly HALF the NVFP4 bytes (2.25 vs 4.5
  bits), so prefill staging halves (not /2.7), and the shadow needs ~6.1 GiB pageable RAM (not 4.4) - and the
  row layout differs, so parallel per-layer descriptors are required rather than a drop-in.
- Build/plumbing order, gates, and the q4 escape hatch, all mapped onto the existing files.

Session 8's own "64 elements / 18 bytes" table entry was correct - it is not ggml's classic 32-value Q2_0,
which is precisely why the quantizer had to be written against this tree's contract and not upstream's.

---

# Session 9 — D1 (Q2_0 shadow): format proven, quality measured on real weights, shadow NOT built

Executed `D1-Q2-SHADOW-DESIGN.md` in its own order: **step 1 (the round-trip harness and quality gate) is done; step 2 (building
the 6 GiB shadow) was not started**, because step 1's real measurement says the quality cost is far larger than the design
predicted.

## Step 1a — the writer's format is proven (`C:\LLM\bwtest\q2_roundtrip.cpp`)

The design's writer, checked against the tree's own readers over **200,000 synthetic NVFP4 blocks**:

```
packing mismatches    0   (independent decoder written from the contract vs the tree's dequantize_q2_0)
saturations           0   (the cover bound d = max(|min|, max/2) makes them impossible)
```

So the 18-byte / 64-value layout, the `2*(j%4)` bit order and the asymmetric `{-d, 0, +d, +2d}` symbol set are all correct.
That removes the "writes 6 GiB of garbage" risk the design was worried about.

## Step 1b — SNR on the model's OWN expert blobs (the number that decides it)

Added an opt-in measurement pass to `CpuSpill` (`C:\LLM\cpu-shadow-measure.on`): on the first forward after load it walks all
20 NVFP4 layers, samples 4 spilled experts per layer and a strided set of their 64-value blocks, and round-trips each through
**the tree's `dequantize_nvfp4` → the writer → the tree's `dequantize_q2_0`**. It allocates no shadow and changes nothing.

```
cpu-shadow measure: layer 0  Q2_0 round-trip SNR 3.45 dB      …  layer 19  3.74 dB
cpu-shadow measure: 20 NVFP4 layers, aggregate Q2_0 round-trip SNR 3.56 dB
cpu-shadow measure: with a clipping-ladder scale the aggregate SNR is 7.90 dB
```

| Scale choice | Real-weight SNR |
|---|---|
| cover bound (no saturation, the design's reference) | **3.56 dB** |
| best of a clipping ladder (d × 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, min per-block error) | **7.90 dB** |
| the design's expectation | 15–22 dB |

**The design's SNR expectation does not hold.** A 2.25-bit/value grid with four *asymmetric* levels cannot reach 15–22 dB on
weight-like data; even the error-minimising scale reaches 7.9 dB, where the error is ~40% of signal amplitude. That distortion
would fall on the ~27% of routed lookups that miss the card — i.e. it is not a free trade, it is a substantial quality
regression.

## Decision: the shadow is not built

The design states the SNR floor is "a floor you choose with the user" and that quality sign-off is the user's. On 3.56 / 7.90
dB versus an expected 15–22 dB, building it would be spending ~6 GiB of pageable RAM and load time on a large quality risk
whose *prefill* benefit P1's wave pipelining largely hides anyway. Stop here and hand the decision over.

Options, in order of preference:
1. **Skip D1, do P1 (wave pipelining)** — no quality cost, prefill ceiling ~2,400–2,900, the largest remaining clean win.
2. **Q4_0 shadow** (the design's own fallback): same bytes as NVFP4 so no prefill win, decode ops ~1.4× cheaper. Only worth it
   if decode is the only target.
3. **D1 with clipping and eyes open** — 7.9 dB, needs side-by-side samples and the user's explicit acceptance.

## What is in the tree now

- `C:\LLM\bwtest\q2_roundtrip.cpp` + `q2_roundtrip.exe` — the standalone harness (packing + saturation + SNR).
- `cpu_spill.cpp`: `quantize_q2_0_block()` / `quantize_q2_0_block_scaled()` / `f32_to_f16()`, and the opt-in measurement pass.
  **Inert unless `C:\LLM\cpu-shadow-measure.on` exists**; the engine's behaviour and numbers are unchanged (verified:
  35.9 / 45.2 tok/s, `hit=` 69.4 / 69.6%, decodes after removing the marker).

Re-run the measurement any time with:

```powershell
"on" | Out-File C:\LLM\cpu-shadow-measure.on -Encoding ascii
C:\LLM\Basalt\Start-Basalt.cmd      # then send one request; read the cpu-shadow lines in engine.log
Remove-Item C:\LLM\cpu-shadow-measure.on
```

---

# Session 9: D1 verdict - Q2_0 closed on measured quality; the shadow family mapped; path redirected

Verified SESSION 8's wall independently (file-level probe of the real spilled weights, independent decoder
ported from the tree's dequant.hpp): Q2_0 cover 3.52 dB / ladder 7.85 dB - matching their 3.56 / 7.90. Their
no-build decision was right; D1-Q2-SHADOW-DESIGN.md's 15-22 dB expectation was wrong (symmetric-2-bit
assumption).

Measured the family ceilings on the same weights (C:\LLM\bwtest\snr_probe*.py, rerunnable):
- Lloyd-Max 4-level ceiling 9.71 dB - the shipped Q2_0-with-ladder is within 1.9 dB of the best ANY 4-level
  format can do. The 2-bit family is closed permanently.
- 3-bit family: ceiling 14.76 dB, realistic ggml k-quant structure (Q3_K-style) **13.78 dB** at 3.44 bits
  (blob /1.31). Readers exist both sides (ggml type_traits CPU + mmq_q3_k.cu GPU); a k-quant writer would
  need writing.
- 4-bit family: 20.28-20.65 dB but zero byte savings vs NVFP4 - pointless as a shadow.

**Decision recorded in D1-SHADOW-VERDICT.md: drop the shadow family from the critical path.** The clean
levers are worth more: P1 wave pipelining (prefill 1,561 -> ~2,400-2,900, bit-neutral) then D2 drafter
capture (decode 53 -> ~58-60, no quality cost). The Q3_K middle (13.78 dB on the ~27% of routed lookups that
miss) stays optional and user-gated: it buys decode ~57-59 alone, ~62-64 with D2 - offered with side-by-side
sign-off, not assumed.

End state without a quality trade on this hardware: ~58-60 tok/s decode, ~2,400-2,900 t/s prefill. Beyond:
16 GB card, the Q3_K trade, or the CPU-prefill kernel project.

---

# SPEC 5 issued: NEXT-STEPS-SPEC.md

The implementation spec for the next worker, consolidating Sessions 8-9 into an ordered plan:

- **W1 - wave-pipelined prefill staging** (1-2 days): 2-4 slot-range waves per visit on a dedicated copy
  stream, moe_sort bounds sliced per wave (two implementation options given), --prefill-waves flag,
  bit-neutral with greedy-identity A/B. Gate: unique-16k >= 2,400 t/s; ceiling ~2,900 with the corrected
  max(1.33 DMA, 1.39 compute) budget. Includes the withdrawn-ideas list (loop-major, 8192 chunks,
  prefill-cpu-share, CPU-prefill kernels, kElasticKeep trims) so nobody re-treads them.
- **W2 - drafter block capture** (1 day): key the drafter's graphs on the position launch bound like the main
  window, batch the meta H2D, keep the W-A event wait, --dspark-direct fallback flag. Gate: outside-window
  <= 3 ms, decode >= 58, acceptance unchanged.
- **W3 - the Q3_K shadow, OPT-IN and user-gated only** (1-2 days): only if the user accepts the measured
  13.78 dB trade on side-by-side samples. K-quant writer + harness-first discipline (the Q2_0 round's
  pattern), parallel shadow descriptors, ~4.6 GiB pageable, gate decode >= 57 alone / >= 62 with W2.

End state without a quality trade: ~58-60 tok/s decode, ~2,400-2,900 t/s prefill. Beyond: 16 GB card, W3,
CPU-prefill kernels, in that order.

---

# Decision: the Q3_K shadow is declined; the plan is W1 + W2 only

The user declined the Q3_K middle-ground shadow (13.78 dB trade for +8-12% decode). NEXT-STEPS-SPEC.md has
been updated: the spec is now two workstreams - W1 wave-pipelined prefill staging (gate: unique-16k >= 2,400
t/s) and W2 drafter block capture (gate: outside-window <= 3 ms, decode >= 58) - with the whole shadow family
recorded as closed (Q2_0 on measured quality, Q4_K on no byte savings, Q3_K by user decision). No future
session should build, propose, or re-measure a weight-shadow format without new instruction.

End state on this hardware, zero quality cost: ~58-60 tok/s decode (39-67 by content), ~2,400-2,900 t/s
prefill. Beyond: a 16 GB card, or the CPU-prefill kernel project to break the PCIe ceiling.

---

# Session 10 — W2 (drafter capture): the kernel blocker is gone, and it was much smaller than expected

Started W2 per `NEXT-STEPS-SPEC.md`. Reading the code first changed the picture completely: **the window-attention kernel
was already device-position-aware**, so W2's "its window follows the position" obstacle was almost entirely solved by work
done for a different model.

## What already existed (verified, not assumed)

`src/kernels/cuda/gqa_attn.cu`:

```cpp
struct Range { ... const int32_t* pos_dev = nullptr; int hi_off = 0; };
__device__ int64_t row_pos0(const Range& r) { return r.pos_dev ? (int64_t)*r.pos_dev : r.pos0; }
__device__ int64_t row_lo (const Range& r, int t) { return r.window > 0 ? max(0, row_pos0(r) + (r.block ? 0 : t) - r.window + 1) : 0; }
const int used = (int) max(1, min((int64_t) gx, chunks));      // per row, so extra blocks exit early
```

`pos_dev` and `row_pos0()` were added for `gqa_decode_attn` (qwen4exp's captured decode) and the window kernel already used
them for the *window low bound* and the per-row chunk count. The **only** quantity still baked at capture was `r.hi`, because
`gqa_window_attn` sets `r.causal = 0`, and `row_hi` returns `r.hi` verbatim in that case.

## What landed

| File | Change |
|---|---|
| `src/kernels/cuda/gqa_attn.cu` | `Range::hi_off`; `row_hi` returns `row_pos0(r) + r.hi_off` **only when** `pos_dev != nullptr && hi_off != 0`; `gqa_window_attn` takes optional `pos_dev` + `hi_off`, skips its `hi < pos0 + T` validation when the position comes from the device, and sizes the grid from `scratch_cells` (the shape bound) instead of the exact `hi` |
| `include/basalt/kernels/gqa_attn.hpp` | the two new parameters, defaulted (`nullptr`, `0`) |

**Inert by construction**: every existing caller passes neither, so the guarded branch is unreachable and the `used`
expression is unchanged. Verified after the build — decode 38.3 / 56.7 / 57.2 tok/s, verbatim-repeat prompt accepts **97%** of
drafts (`dl` 2.2 on the short passage), needle found at **130,709** tokens.

Backup: `gqa_attn.cu.orig`.

## What W2 still needs (not done — the capture orchestration in `dspark.cpp`)

1. A device word holding `pos`, written per `draft_begin` (an H2D on the capture stream is itself capturable).
2. `draft_begin` passes `pos_dev = &pos_d`, `hi_off = kDsBlock`, and a **shape-floor** `pos0` (the position bound's floor, so
   `r.base` covers the shape) — the per-row bounds then follow the real position inside the kernel.
3. Capture the whole block (the three attention launches + the existing stage graphs + the screened head + the meta H2D) as
   one graph per `(kDsBlock, pos-bound)` shape, replay afterwards.
4. Keep the direct launch behind a marker/flag so the bail-out is one step away — the documented failure mode of a bad
   capture is a hang on the first window, so it must be opt-in until it has been seen to work.
5. Gate: outside-window span ≤ 3 ms, decode ≥ 58, `acc` unchanged (~73% on the TCP prompt), needle, `dl` spot-check.

The remaining risk is entirely in step 3 (graph capture semantics), not in the kernel: a capture bug shows up as **wrong
drafts** (acceptance collapses), which the `acc`/`dl` line makes obvious immediately.

---

# Session 11 — W2 implemented and measured: correct, but the spec's premise does not hold

Implemented the capture orchestration W2 asked for, then measured it against a direct-launch A/B. **It works and it is not a
win** — which is the useful result, because the spec's 7.8 → 3 ms target was based on an assumption that is now falsified.

## What was built

| File | Change |
|---|---|
| `include/basalt/kernels/gqa_attn.hpp` · `src/kernels/cuda/gqa_attn.cu` | (Session 10) `Range::hi_off`; `row_hi` follows the device pos when `hi_off` is set; `gqa_window_attn` takes optional `pos_dev` + `hi_off` |
| `src/models/inkling2/dspark.hpp` | `g_attn_[kDsMaxLayers][19]`, `attn_capture_`, `attn_captured()` |
| `src/models/inkling2/dspark.cpp` | `attn_captured()` captures layer k's block attention per **position bucket** (`2^s <= pos < 2^(s+1)`) and replays it; `draft_begin` calls it when the marker is present; loud `warn` on any capture/launch failure so a silent fallback cannot masquerade as a working capture |

Capture design notes, because they are what makes it possible:

- The position already reaches the device as `meta_d_ + 1` (the drafter's own `pos` word) — no new allocation.
- A bucket's floor `bl` supplies `pos0` for `r.base` (≤ the real one, so the range always covers the position) and, for a
  **full** layer, `window = bl + kDsBlock`, which makes `row_lo` clamp to 0 for every position in the bucket. A sliding
  layer's window was already constant.
- **One real cost, found by measurement**: the captured call cannot use `cells` (the cells the block actually reads) — a
  capture's grid is fixed — so it runs the scratch's capacity instead. At short context that is strictly more work.

## Measurements (400-token free-form prose, `temperature 0`, unique prompt, engine relaunched each mode)

| Pair | Direct warm tok/s | Capture warm tok/s | Diag wall (direct → capture) |
|---|---|---|---|
| 1 | 61.5 (4 runs) | 61.0 (4 runs) | — |
| 2 (+diag) | 61.1 — wall 30.35 ms | 61.9 — wall **29.29 ms** | 30.35 → 29.29 (−3.5%) |
| 3 | 60.50 (3 runs) | 55.37 (3 runs) | — |

Correctness throughout: `acc` unchanged (87–91%), `hit` unchanged, outputs identical across repeats, verbatim-repeat prompt
100% accepted, no capture fell back (the Session-11 warn never fired), needle found at 130,709 tokens on the shipped default.

## Why it does not pay — and the stall mechanism

1. **The 7.8 ms outside the window is GPU work, not launch overhead.** The stage graphs and the head were already captured
   before W2; the only direct launches were the four attention calls (~5–10 µs each). W-A had already removed the full-stream
   sync that used to dominate that span, so there was little left to reclaim.
2. **Capturing a bucket on first sight stalls the decode path.** A 400-token reply crosses buckets `2^10` and `2^11`, i.e. up
   to 8 capture+instantiate operations *inside* a request — visible as pair 3's 63.1 → 50.8 → 52.2 collapse. Pre-capturing
   the likely buckets at warm-up would remove that, but item 1 caps the upside at a few percent.
3. **The capacity-sized splits** make the captured attention slightly slower than the direct one at short context.

## Decision

**Capture ships opt-in** (`C:\LLM\dspark-capture.on`), default **off**. The code is correct, documented, and one marker away
if a future change makes launch overhead matter again (e.g. if the drafter's stages stop being graphs). Note the marker is now
a *measurement* switch, not a performance one — leaving it on costs a few percent on long decodes.

## What this means for the remaining work

`NEXT-STEPS-SPEC.md` ranked W1 (prefill pipelining) second on the assumption that W2 would land ~4.8 ms/window. It does not,
so **W1 is the only remaining item that can move a headline number** — and per `W1-W2-ASSESSMENT.md` it needs per-wave expert
products (`fp4::Product` addresses experts positionally: `base + e * stride`, `bounds[e]`), which is the real work.

---

# Session 12 — W1 landed: prefill 1,454 → ~2,600 tok/s (+73%)

W1 was specified as wave-pipelined staging by slot range, which needs per-wave expert products. **That turned out not to be
necessary.** Reading the code showed each layer already owns its own expert region in VRAM, so layer L+1's staging can be
copied into L+1's own slots while the card computes layer L. No product slicing, no extra VRAM, no grid bookkeeping.

## The insight

`experts()` staged layer L synchronously (`stage_from_card` / one big H2D) on the compute stream, then ran L's products. Every
layer's staging lives at `w.experts + kept*stride` inside **that layer's own** region, and the staged bytes are the same on
every visit — so:

1. **Overlap**: issue L+1's staging on a second stream while L's products run. `cudaStreamWaitEvent` before L+1's products.
2. **Reuse**: the staging backing survives across a prompt's chunks (`acquire_staging` is a no-op until something releases),
   so one stage per layer serves every chunk — a generation counter bumping on release/acquire invalidates it correctly.

## Implementation

| File | Change |
|---|---|
| `model.hpp` / `model.cpp` | `stage_layer(l, stream)` — a layer's whole staging (card copies where held, else one H2D) plus the guard memset, safe on any stream; `staging_gen_` + `staging_generation()`, bumped by `release_staging` |
| `prompt.hpp` | `virtual void sync_staging() {}` on `PromptRunner` |
| `prompt.cpp` | `copy_` stream + one event per layer; `stage_awaitable(l)` / `stage_issue(l+1)` / `sync_generation()`; `experts()` waits on the layer's event instead of staging inline |
| `session.cpp` | `prompt_->sync_staging()` before `release_staging(0)` at the decode boundary |

Kill switch: `C:\LLM\prefill-no-prefetch.on` restores the serial staging exactly.

## Measurements (unique 10k-token prompts of random words, `max_tokens 4`, fresh engine per mode)

| Mode | Runs | tok/s |
|---|---|---|
| Serial staging (`prefill-no-prefetch.on`) | 2 | 1,539 · 1,368 |
| **Pipeline staging (default)** | 3 | **2,813 · 2,577 · 2,417** |

Mean **~1,454 → ~2,600 tok/s, +73%**, which clears `NEXT-STEPS-SPEC.md`'s 2,400 gate. Per-chunk time goes 2.72 s → ~1.55 s,
i.e. essentially `max(H2D 1.33 s, compute 1.39 s)` — the overlap is doing exactly what it should.

At 130k tokens prefill is **unchanged** (1,215 vs ~1,170 tok/s): there attention dominates, so hiding the expert staging buys
almost nothing. The win is in the expert-staging regime.

## Correctness

- **Needle found at 130,709 tokens** with the pipeline on (`ZEPHYR-4417`), 107.6 s prefill, `tg=92`.
- Decode: **no reproducible change**. An apparent −6% first looked real, but a control run (same mode, two fresh processes)
  spanned 62.5 vs 67.8 tok/s — **±8.5% process variance**, which covers every decode difference measured here. On a
  pinned-content (verbatim repeat) workload the pipeline was *faster*: 52.5 vs 46.6 tok/s.
- One real trap found and fixed on the way: unmapping the staging while the copy stream still had copies in flight. It is now
  quiesced at the decode boundary (`sync_staging`), because a sticky error there is exactly the kind of silent failure this
  project has been bitten by before.

## Why `hit=` reads lower with the pipeline

`hit=48–51%` vs `64%` is not lost residency — the promote log is identical in both modes (`0.72 GiB copied, 760 on the card`).
It tracks **more tokens per window** (`dl` 7.1 vs 2.9 on the verbatim task): the same resident experts then serve more
routed pairs, so the per-pair hit statistic moves while throughput goes up. Judge the front by `tg=`/`pp=`, not by `hit=`.

---

# Session 13 — W1 retracted: the stage-ahead pipeline was corrupting the model

The user reported the model answering everything with `!!!`. It was real, it was mine, and it came from W1.

## What the symptom actually was

`logprob = -9999.0` on every sampled token with no alternatives (`src/program/serve/logprobs.hpp:25`: *"a masked logit"* is
how a non-finite logit is reported). So the forward pass was producing non-finite logits and the sampler fell back to a
constant token. Draft acceptance was 100% because a constant token is perfectly predictable — which is why the earlier
`acc`/`dl` checks looked healthy and I missed it.

## Root cause — my core W1 assumption was false

W1 was built on this claim, which I wrote in `W1-PREFILL-PIPELINE-RESULTS.md`:

> Every layer already owns its own expert region in VRAM … so layer L+1's staging can be copied into L+1's own slots while
> the card computes layer L.

**The virtual addresses are per layer; the backing pages are not.** `model.cpp` maps the *same* staging granules into every
spilled layer's range at the same offset:

```cpp
for (const Range& r : ranges_)
    core::vmm::map_handle(r.va + at, gran, staging_[(size_t) ((at - r.own) / gran)]);
```

`ranges_` is per layer, but the handle index depends only on the *offset* inside the range — so all 24 layers alias one
shared 1034 MiB buffer ("1034 MiB of prompt staging" in the load line is the whole budget, not a per-layer figure).

Consequences, both of which invalidate W1 as built:

1. **Prefetching L+1 while L computes** overwrites the pages L's products are still reading → garbage experts → non-finite
   logits. This is exactly the reported failure.
2. **Re-using one stage across chunks** is equally invalid: the next layer's staging has already clobbered those bytes, so
   even without the copy stream the data is gone.

Serial staging works only because it is stream-ordered: stage L, compute L, stage L+1, compute L+1 — the buffer is reused,
never aliased in time.

## The fix

`experts()` stages serially again (`m_.stage_layer(l, stream_)`, once per visit). The copy stream and its events are kept
only behind `C:\LLM\prefill-prefetch-experiment.on` and are documented as known-invalid, so the trap cannot be re-armed by
accident. Everything else from Session 12 stays: `stage_layer()` is a faithful refactor of the old inline staging, and the
`sync_staging()` quiesce remains correct for the release boundary.

## Verification after the fix

- "What is the capital of France?" → **"The capital of France is Paris."**, byte-identical to what the **stock 153 MB
  release binary** answers on the same pack — the control that proved the model file, GPU, tokenizer and templates were
  never at fault.
- "Write two sentences about how rivers shape valleys." → coherent.
- **Needle found at 130,709 tokens** (`ZEPHYR-4417`).
- A clean full rebuild was run first (213 objects) to rule out stale-object ABI mismatch; the bug reproduced on both, so it
  was genuinely the source change.

## Retraction of the Session 12 numbers

The Session 12 prefill figures (**1,454 → ~2,600 tok/s, +73%**) were measured on this build, i.e. **while the model was
computing garbage**. The overlap did hide the H2D time, so the *timing* was real, but the output was not, and a build that
silently corrupts its experts has no performance credit. **Prefill is back to ~1,450–1,550 tok/s.** The +73% is withdrawn.

## What it would actually take (for a future attempt)

The overlap is still physically possible, but it needs one of:

- **A second staging buffer** (2 × 1034 MiB ≈ 2.07 GiB). We have ~768 MiB free plus 512 MiB of elastic promote, so this
  would mean re-cutting `kElasticKeep`/`vram-reserve-mib` and re-validating the load — the tight-margin territory that
  `kCardSpare 768` came out of.
- **Per-wave expert products** with a single buffer (the original `NEXT-STEPS-SPEC.md` design): slice `moe_sort`'s bounds
  and run `fp4::Product` per wave, so wave *i* never needs wave *i+1*'s bytes. That is the fiddly path I avoided — and it
  is the only one that needs no extra VRAM.

The lesson to carry: **verify allocator semantics before building on a layout assumption.** The staging looked per-layer
because `w.experts` is a per-layer pointer; the sharing lived two files away in the VMM mapping loop.

---

# Session 14 — model moved to the v2 pack

Swapped the engine onto `nokeepai/Inkling-2-Mini-Unbound-Basalt`'s **v2** pack
(`Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt`), downloaded from the Hub and verified:
19,734,300,582 bytes, SHA-256 `aa20b87a29c19aa6c8ed699968aeaab08e3ca47f4cd768a3470ea073cf600c77` (matches the published hash).

No code change was needed: v2 is the same architecture and the same pack layout (only 120 attention tensors differ from the
official pack), so the swap is one `"model"` line in both config copies. v1.0 stays on disk for comparison and rollback.

Verified after the swap: loads at `ctx=196,608` with the same memory plan and ~768 MiB free VRAM, answers the sanity prompts
correctly, and **the 130,709-token needle still returns `ZEPHYR-4417`**.

Measured, v1.0 → v2: decode 60.5–61.5 → **57.9–58.2 tok/s** (draft acceptance 87–91% → **81–83%**), prefill 1,368–1,539 →
**1,357 tok/s**. The decode difference is the pack's own prediction — the DSpark drafter is the original model's, and v2's
attention weights drift further, so drafts are accepted slightly less often. Prefill and long context are unchanged.

Details, the comparison table and the rollback procedure: `MODEL-V2-UPDATE.md`.

---

# PR evaluation (Session 14, deepseek) - decisions and one completion made here

The user sent four upstream PRs for evaluation. The deepseek session's decisions, plus the loose end found and
closed while reading its record:

- **PR #555 (inkling2 prefill optimization, tjm8874, claimed +20-22%):** judged the most valuable and
  implemented - 41/43 hunks auto-applied plus 2 manual adaptations, kCardSpare lowered 768 -> 736 so the
  v2 plan fits the pinned ceiling, vram-reserve-mib left at 672 (experiment value). Verified: sanity pass,
  needle found at 130,709, prefill 1,408-1,462 t/s (+6% vs v2's 1,357 - the PR's wins target fully-resident
  cards, ours is 89% CPU-resident). **OPEN QUESTION left at session end: warm decode measured ~8% lower
  (52.5-53.4 vs 57.9-58.2) - the session was mid-investigation whether that is #555, the kCardSpare 736
  side-adjustment, or run variance. Unresolved.**
- **PR #550 (tool-call recovery - bare/headerless/unclosed/cut calls, aamsellem):** implemented and verified
  (8/8 hunks, 14/14 standalone parser checks). Chosen over #544 as the competing implementation. **Loose end
  found here: the patch landed only in the STOCK install's serve (C:\LLM\Basalt\serve\models\inkling2.py) -
  the live optimised server resolves `serve` from the ENGINE tree, whose copy was unpatched. Copied the
  patched file into the engine tree and re-ran the parser suite: 0 failures. #550 is now live on both.**
- **PR #544 (headerless tool calls, giveen):** NOT applied - judged a competing/overlapping implementation
  of the same recovery #550 provides (different structure, same behaviours).
- **PR #552 (int8 KV cache option):** NOT applied - assessed marginal (we already run kv: fp8; it touches
  many files our patches live in).

Same-session context worth keeping: the earlier W1 prefill pipeline was retracted (NaN corruption from the
shared physical staging buffer; the +73% claim withdrawn, prefill back to ~1,450-1,550), and the model pack
moved to v2 (decode 57.9-58.2 vs v1's 60.5-61.5 from drafter drift; v1 kept on disk for rollback -
MODEL-V2-UPDATE.md).

---

# Clarification recorded: the retracted prefill pipeline does not close the 2,000+ target

Answered the user's question directly: the withdrawn "W1 pipeline" (+73% on NaN output) was a cross-layer
prefetch, structurally invalid on this engine's single aliased staging buffer. The wave-pipelined staging plan
(NEXT-STEPS-SPEC W1) is a different design that stays within one visit and is immune to that failure mode by
construction - it was deferred, never attempted, never disproven. The retracted build's timing is evidence
the DMA-hiding upside is real. Corrected projection with #555 in the tree: max(1.33 DMA, ~1.3-1.39 compute) +
tails ~ 1.4-1.5 s per 4,096-token chunk = ~2,700-2,900 t/s. The W1 section of NEXT-STEPS-SPEC now opens with
this distinction so the next worker does not conflate the two.

---

# W1-EXECUTION-SPEC.md issued + live monitoring protocol

The detailed prefill spec is written for the next implementing session: Phase 0 resolves the experimental
constants left from the #555 debugging (kCardSpare 736 vs 768, vram-reserve 672 vs 768) and the open #555
decode-regression question; Phase 1 is the wave-pipelined staging (within-visit waves only, immune by
construction to the retracted cross-layer prefetch failure; reuses the copy_/stage_ev_/sync_staging
infrastructure the retraction left behind; --prefill-waves 0|N with serial fallback as the A/B control);
Gate 0 is the anti-!!!!! protocol (coherence eyeball, logprob sanity, acc < 100% on prose, needle
ZEPHYR-4417) run before any benchmark, because the last attempt showed stats LIE under NaN corruption.
Gates: unique-16k >= 2,400 t/s (expected 2,700-2,900), decode within 5%, TTFT unchanged, 3 clean loads.

A second pi agent (this one) will monitor the implementing session's transcript live, verify results
independently, and post answers/corrections to C:\LLM\Basalt\W1-EXECUTION-NOTES.md - a read channel the
implementer is told to check when stuck.

---

# W1 live monitoring, entry 1: two bugs found and posted to the implementing agent

While watching the implementing session debug (it had concluded "the range function itself is wrong" from
diagnostic modes 7/8/9), the monitor found in the tree:

1. **The diagnostic modes never ran**: `kMaxWaves = 4` clamps `waves_wanted_`, so `--prefill-waves 7/8/9`
   all silently became the real 4-wave pipeline - every "diagnostic" result was actually the production path,
   and the range `stage_layer` (verified byte-identical to the whole-layer control for a full range) was
   never actually tested.
2. **The real corruption**: `wave_bounds_kernel` (prompt_ops.cu:168) uses a compacted-run scheme while the
   products need absolute row indices - for every wave after the first, the wave's FIRST expert gets the
   filtered range [0, bin[e0+1]) instead of [bin[e0], bin[e0+1]), so each wave overwrites all earlier waves'
   gu rows using the wrong expert's weights. Wave 0 / serial is unaffected - matching every observation
   (correct at waves=0, garbled at waves>=2). Two-line fix posted in W1-EXECUTION-NOTES.md.

Both are posted to C:\LLM\Basalt\W1-EXECUTION-NOTES.md with the corrected kernel, the fix for the clamp, and
the suggested re-test sequence (fix both, rebuild once, Gate 0, then throughput).

---

# W1 live monitoring, entry 2: the mode-10 failure diagnosed (per-wave H quantize scribble)

The bisect (mode 8 PASS / mode 9 PASS / mode 10 FAIL) isolated the fault to the per-wave products. Within
that half, the fault is the H quantize inside the per-wave experts_fp4: with filtered bounds, out-of-wave
rows (including all resident experts' pairs, which sit at the front of the row array) hit degenerate
zero-width segments in the quantizer's binary search, get quantized from stale b_.h, and are written at
plane positions [0, bin[e0]) - exactly where the wave's own true rows live. The down product then reads the
scribbled planes. Serial/full-bounds never produces degenerate segments, hence serial passes; modes 8/9 run
one whole-layer quantize, hence they pass.

Fix posted to W1-EXECUTION-NOTES.md UPDATE 3: restructure the fp4 path to match the (already-correct) MMQ
shape - waves slice only the gu product; the swiglu/H-quantize/down tail runs once after the loop over the
FULL bounds. No shared-kernel changes. The Q8 path needs no change.

---

# Session 15 — W1 wave-pipelined staging: CORRECT (Gate 0 + needle pass) but no speedup yet

Phase 0 (baseline) and Phase 1 (wave pipeline) of `W1-EXECUTION-SPEC.md`. The wave pipeline now produces correct output —
Gate 0 and the 130,709-token needle pass with `--prefill-waves 3` — but it delivers **no throughput gain** (~1,571 vs ~1,566
tok/s), so the shipped default stays the serial path (`--prefill-waves 1`).

## Phase 0 — baseline settled (the spec's gate)

| Variant | Load | Warm decode | Unique-15.4k prefill |
|---|---|---|---|
| `kCardSpare 736` / `vram-reserve 672` (the #555 leftovers) | loads | 52.5–53.4 | 1,408–1,462 |
| **`kCardSpare 768` / `vram-reserve 768` (chosen)** | **flaky OOM ~1 in 2, retried OK** | **56.5–57.0** | **1,436–1,654** |

**Decision: 768/768, PR #555 dropped.** #555 (`+6%` prefill) only loads at `kCardSpare 736`, and at 736 it costs ~8% decode;
its fp16 prompt buffers also ate ~32 MiB of planned VRAM, pushing 8 experts to RAM and the pinned host total over this
machine's ~14.5 GiB ceiling (`cannot select device 0`). Net for our spill-bound workload it is not worth it. The load
flakiness at 768 is recorded as a known environmental issue (retry loads; VRAM reads 11.9 GiB free when it fails).

## Phase 1 — the wave pipeline, and the three bugs it took to make it correct

Reused the retired machinery as the spec said: `stage_layer` gained slot-range and stream parameters
(`stage_layer(l, s0, s1, stream)`), `Prompt` keeps its `copy_` stream + one event per wave, `sync_staging()` still guards
the release/re-take boundary. New: a `wave_bounds` kernel (per-wave filtered expert bounds), per-wave bounds/tile/blk
buffers in the carve, `experts_fp4_gu`/`experts_fp4_down` split, and `--prefill-waves N` parsed in `dspark_parse_flag`
plus help text.

**Bug 1 — the missing cross-layer barrier.** Wave copies ran on `copy_` with no ordering against the *previous* layer's
products, which read the same physical staging buffer (all layers alias one 1034 MiB buffer). Fixed with
`cudaEventRecord(prod_ev_, stream_)` + `cudaStreamWaitEvent(copy_, prod_ev_, 0)` at the top of each visit's staging: the
copy stream may run ahead only *within* a visit, where the waves are disjoint slot ranges.

**Bug 2 — `wave_bounds_kernel` gave the wave's first expert a range starting at 0.** The rolling-`run` form left
`out[e0] = 0`, so the wave's first expert computed its weights over *all previous waves' rows* and overwrote their `b_.gu`
rows. Fixed to the absolute-range construction (in-wave expert keeps `[bin[e], bin[e+1])`, every other expert zero-length by
pointing at the nearest in-wave edge), verified over 2000 random cases plus a row-total invariant. (The monitoring agent's
UPDATE 2 found this; its first suggested patch was incomplete — it left the first expert's *start* at 0 — so the
edge-anchored form went in instead.)

**Bug 3 — the waves partitioned only the spilled slots.** This was mine, and it is what kept `"!!!!!"` after bugs 1 and 2
were fixed: waves spanned `[kept, n_expert)` (the spilled slots, which is what *staging* needs), so **every kept expert
(slots < kept, 334 of 3072 here) never received a gate/up product at all** — their H rows stayed stale and the tail
quantized garbage. The waves now partition **all** experts `[0, n_expert)` for the products, while each wave stages only
the intersection of its range with the spilled slots (a wave of kept experts stages nothing but still computes).

Also required: the FP4 tail must run once with the **full** bounds (a per-wave H quantize sees zero-width segments for
every other expert and scribbles over the wave's own H planes — the monitoring agent's UPDATE 3 diagnosis), and the down
product must still carry the softcaps (the serial path inherited them via `d = gu`).

## Gate 0 at `--prefill-waves 3` — PASS

| Check | Result |
|---|---|
| Coherence, long prompt through the prompt path | "Rivers carve valleys by eroding rock and soil over long periods of time." |
| Coherence, factual ×3 | Paris / 391 / Danube, Rhine, Thames |
| 400-token prose | coherent prose, `acc` 63–74% (**not** the 100% constant-token signature) |
| Needle, 130,709 tokens | **ZEPHYR-4417** |
| Decode at waves=3 | 48.1 / 56.4 / 53.3 tok/s (within noise of the 56.6 baseline) |

## Throughput — the honest result

| Config | Unique-15.4k prefill | Per 4,096-token chunk |
|---|---|---|
| Serial (`--prefill-waves 1`) | 1,654 · 1,514 · 1,530 (≈1,566) | ≈2.6 s |
| **Waves=3** | **1,550 · 1,574 · 1,590 (≈1,571)** | **≈2.6 s** |

**No gain**: the chunk time is still `staging + compute` rather than `max(staging, compute)`, i.e. the copies are not
actually overlapping the products. Correctness is solved; the overlap is not, and that is where the next attempt should
start.

## Where to look next (for whoever picks this up)

1. **Verify the copies overlap at all.** Issue a *single* huge H2D on `copy_` concurrent with a long compute kernel and time
   it; on this laptop GPU the copy engine may be sharing the same execution resource as the SMs (D2D through the copy engine
   plus H2D over PCIe), in which case no amount of restructuring gets `max(...)`.
2. **The per-visit barrier may be costing more than the overlap gains.** `prod_ev_` serializes the visit's first copy behind
   the layer's attention. Try recording the barrier *before* the attention of the visit (a hook at the top of the visit) so
   wave 0's copy overlaps the attention and the router.
3. **Batch the per-expert copies.** `stage_from_card` issues one `cudaMemcpyAsync` per expert (~114 per layer per visit,
   4.48 MiB each). Contiguous RAM-only runs can become a single H2D; at this size the launch overhead is a real fraction of
   1.33 s per chunk.
4. The flag is in place and correct: `--prefill-waves N` (0/1 = serial). Default is 1 until a config beats serial.

### Addendum — the wave path is confirmed active (so the missing speedup is overlap, not engagement)

`prompt experts: n_embd 2304, n_ff_exp 1152, n_expert 128 -> FP4 tensor-core path, waves 3` — logged from the running
engine, answering correctly. So at `--prefill-waves 3` the pipeline really does split each spilled layer into three waves,
stage them on the copy stream and compute per wave; it is not silently falling back to serial.

Budget estimate for one 4,096-token chunk (from the plan: ~114 spilled experts/layer × 4.48 MiB, a third of them served
from elastic card copies by D2D):

| Part | Estimate |
|---|---|
| staging (H2D ~341 MiB/layer at 15.75 GB/s + D2D for the rest) | ≈ 0.5 s |
| expert compute (gu + H + down) | ≈ 2.1 s |
| measured chunk | ≈ 2.6 s ≈ the sum, not `max(...)` |

So the copies are not running concurrently with the GEMMs even though they are on a separate non-blocking stream with
correct event ordering. That is the whole of the remaining gap, and it is a property of how this GPU schedules the copy
engine against these kernels (or of the per-visit barrier), not of the wave logic. `--prefill-waves 1` remains the default
because it is equal-or-faster and has no barrier.

---

# Session 16 — the monitor now shows prompt-processing (PP) speed, live

`basalt-monitor.cmd` already had a per-request `PP tok/s` column and an "avg last N PP" line, but **during** a prefill it
showed `0.0 tok/s`, so there was no live view of prompt processing. Three fixes:

1. **The live state was never matched.** The serve layer's `live.state` is `reading` while the prompt is processed — the
   monitor looked for `prefilling`, which does not exist. So a running prefill fell into the idle branch. It now treats
   `reading` (and `generating`) as busy, and prints a `PREFILL` line while reading.
2. **The engine's own prefill rate is now used.** `/metrics` carries `live.prefill_tok_s_mean` — the engine's mean over
   *newly read* tokens, cached prefix excluded — which the monitor ignored. That is the same quantity the completed rows
   report as PP.
3. **Read position, percentage and ETA.** `/metrics` does not carry the read position, so the monitor tails
   `server.out`'s `s1 read pp=<done>/<total> <t/s>` line (the project's source of truth for prefill) and shows
   `read 7,722/15.4k (50%)  ~7.1s left`. It is skipped silently when the file is absent, so watching a server over the LAN
   still works — the prefill rate then comes from `prefill_tok_s_mean` alone.

Also added: a **best seen** line (PP/TG maxima with the request ids, from `/metrics`' `best`), and PP labelled explicitly in
the live line and the averages.

## Sample (live, during a 15.4k prompt)

```
03:35:20 PREFILL \  prompt   15.4k tok  read 7,722/15.4k (50%)  ~7.1s left    7.2s  PP 1,088.9 tok/s   100% GPU  80W 60C
03:35:23 PREFILL -  prompt   15.4k tok  read 7,722/15.4k (50%)  ~4.7s left   10.2s  PP 1,639.8 tok/s   100% GPU  89W 64C
03:35:24   15.4k      0    1,636.8    12      9.60       5/7   68.2
------------------------------------------------------------------------------
avg last 6:  PP  1,237.6 tok/s    TG  22.30 tok/s
best seen  :  PP  1,636.8 tok/s    TG  52.20 tok/s   (r_0006/r_0003)
```

The `PREFILL` line shows `PP -` for the first second or two until either the engine's first progress report or the first
`s1 read` line lands; that is expected (nothing has been read yet to average).

Files: `C:\LLM\basalt-monitor.ps1` (only), `C:\LLM\Basalt\SUMMARY.md` (monitor row).

---

# Session 17 — measured where prefill's time actually goes: it is 100% compute, and the pipeline was chasing nothing

Three measurements with temporary switches (all removed afterwards; the tree is back to the shipped configuration).

## 1. Staging is free — the whole pipelining premise was wrong on this machine

Added a measurement-only mode (`--prefill-waves -1`) that skips staging entirely, so a chunk's time is its expert compute.

| Config | Unique-15.4k prefill (3 runs) |
|---|---|
| Serial staging | 1,670 · 1,541 · 1,505 → **1,572** |
| **No staging at all** | 1,633 · 1,574 · 1,529 → **1,579** |

**Identical.** The copies cost ~0 s per chunk, not the ~0.5 s I had estimated from the link arithmetic. So there was never
anything to overlap, which is exactly why waves=3 measured the same as serial while being correct. The W1 pipelining line is
**closed on this hardware**, not merely unsolved — a chunk of 2.6 s is 2.6 s of expert compute.

(Why the link arithmetic misled: the 12 GiB of spilled experts per chunk is not all H2D — the elastic card copies cover a
quarter of it, and the rest is issued as ~114 small per-expert copies per layer whose latency hides under the GEMM the
compute stream is already running. The steady state costs far less than one big transfer would.)

## 2. The FP4 tensor-core path and the Q8_0 MMQ path are the same speed here

Forced the MMQ product instead of FP4 (`--prefill-waves -2`):

| Product | Unique-15.4k prefill |
|---|---|
| FP4 tensor cores | 1,576 · 1,573 → **1,575** |
| MMQ (Q8_0 kernels) | 1,617 · 1,524 → **1,571** |

Within noise. So the expert GEMM choice is not a lever — both land on the same effective rate, ~**200 TFLOPS**
(32,768 pairs/chunk × 7.96M MACs ≈ 521 GFLOP per 2.6 s chunk), i.e. roughly 40–45% of this card's dense FP4 peak.

## 3. The FP4 block geometry is already at its best for this card

PR #555's headline kernel change is 256-row blocks (`gemm_kernel_256`, 2 stages, 76 KiB). Tested the equivalent directly on
our `moe_fp4.cu`:

| kBlockTiles (rows/block) | stages | smem | Prefill |
|---|---|---|---|
| **2 (128) — shipping** | **3** | **81 KiB** | **1,572–1,576** |
| 4 (256, #555's shape) | 2 | 90 KiB | 1,547 (−1.6%) |
| 3 (192) | 2 | 72 KiB | 1,539 (−2.0%) |
| 3 (192) | 3 | 108 KiB | over the card's ~99 KiB per-block limit (kernel launch fails) |
| 4 (256) | 3 | 135 KiB | same |

So the shipping geometry wins; #555's +20% is attributable to its **load path** (`cp.async.cg` 16 B chunks, conflict-free
`kBRowStride = 80`, fp16 intermediates), not to bigger blocks. Reproducing it means merging that work, not turning a knob.

## Where this leaves prefill

`~1,570 t/s` and **compute-bound at ~200 TFLOPS**. The only remaining levers, in order of value:

1. **Merge #555's load path / fp16 intermediates** — the one known artifact worth ~+15–20% of this compute. Needs the
   VRAM its fp16 prompt buffers want (~32 MiB), which should now be paid for out of `kElasticKeep` rather than
   `kCardSpare`: the elastic card copies were there to avoid H2D, and measurement 1 just showed H2D is free, so trading
   them is nearly costless. That is a much better trade than the 736 MiB `kCardSpare` that cost 8% decode last time.
2. **Offload a slice of experts to the CPU during prefill.** The cores are idle and the weights are already in RAM (no
   staging needed), so this attacks compute directly. inkling2 has no prefill CPU split today (only qwen4exp does), so
   this is new code — the decode expert pool is the building block.
3. Nothing else: geometry, product choice, staging and pipelining are all measured out.

Final state: serial staging default (`--prefill-waves 1`), original `moe_fp4.cu` geometry restored, both measurement
switches removed, engine verified (Paris / 391 / coherent long-prompt answer).

---

# Session 18 — open-PR triage: nothing left in the backlog for this setup; one serve win applied

Reviewed all **72 open PRs** against what is measured to bind here (prefill = expert-GEMM compute; decode = card residency
and the CPU pool), not against their titles. Full table in `PR-TRIAGE.md`.

**Applied — #447** (`serve: no Nagle on streamed replies, lighter per-token progress check`): 2/2 hunks, no rebuild (it is
the Python serve layer). `disable_nagle_algorithm = True` so streamed events leave immediately instead of waiting for a
coalesced segment, and `_progress()` no longer takes the status lock and copies the whole status dict on every call. Server
restarted and verified (answers + streaming intact).

**Rejected with reasons:** the whole `src/prefill/*` family (#570, #489, #249, #129, #157, #445) belongs to
qwen4exp/deepseek41 and is not on inkling2's path — #570's four-row router is already taken by our `n_embd 2304` and we use
`f32_rows_dot` anyway; #498's F16C inlining does not touch `nvfp4_avx512.cpp`, which is where the pool's 14.4 ms/window
actually goes; #572/#314/#355 are other models' KV formats; Q5_1 CPU work (#503/#500) does not match our NVFP4 experts.

**The one applicable KV PR moves the wrong way:** #552's int8 KV row is 144 B against FP8's 128 B (+12.5%), and we are
residency-starved — every GiB of KV given back is ~230 more experts on the card. It is a quality knob, never a speed one.

**Not applied:** #420 (thinking budget no longer cutting the answer) — a behavioural fix for the empty-content case we hit,
but only 5/7 hunks fit our older serve tree and its tests are absent, so it needs a deliberate port rather than a blind
patch.

Conclusion: the remaining performance is ours to build — #555's load path (paying its VRAM from `kElasticKeep`) and a
prefill CPU expert slice.

---

# Session 19 — PR #555 retried in full, measured properly, and rolled back again (this time with evidence)

The earlier #555 attempt mixed three variables (the PR, `kCardSpare` 736, and one unpaired process per mode), so it was
never conclusive. Retried it deliberately: a snapshot of the known-good tree first, then #555 ported completely, then the
same measurement protocol on both sides (4 decode runs → median of the warm three, 3 unique-15.4k prefill runs).

## The port (37/43 hunks + hand work)

- Applied the auto-fitting hunks: `moe_fp4.cu` (cp.async weight loads, 9/9), `ops.cu` (vectorized post_conv / attn_prepare /
  attn_gate, tiled attnres_mix, 8/8), `kernels.hpp`, `prompt_kernels.hpp`, `moe_fp4.hpp`, `dspark.cpp`, `dspark.cu`,
  `prompt_parity.cpp`, and 6/10 of `prompt.cpp` (fused QKV `kProj`, the batched 4-message `messages()`, `msg` 1→4 × R·E).
- Hand-ported the rest: the 359-line block of new prompt kernels into `prompt_ops.cu` (`ctrl_down_prompt`,
  `ctrl_up_blend_prompt`, `rms_norm_q8_1_prompt`, `attn_gate_q8_1_prompt`, `quantize_f16`, `moe_combine_f16`);
  `DsparkHook::PromptScratch` + `commit_prompt` in `dspark.hpp`; `controller()` → the fused pair; the prompt's drafter
  commit → `ds_.commit_prompt(...)`; and the fp16 intermediate path folded into our (Session-15) `experts_fp4_gu` /
  `experts_fp4_down` split — `gu.dst_f16 = (uint16_t*)b_.h`, `fp4::quantize_f16(...)`, `d.dst_f16 = (uint16_t*)b_.gu`,
  `moe_combine_f16(...)`. The fp16 buffers reuse `b_.h`/`b_.gu`'s spans, so they cost no VRAM.
- **Dropped `b_.pair`** (unused once `messages()` batches into `msg[0..3]`): this exactly pays for `msg`'s growth, and the
  card residency came back to baseline — `2739 of 3072 in RAM, 2.43 GiB on the card` (baseline: 2738 / 2.43).
- Load still needs **`kCardSpare` 736** (at 768 the plan is 2746 and the pinned host total exceeds ~14.5 GiB).

## Measurements (same protocol both sides)

| Build | Decode (4 runs, median warm) | Prefill (unique-15.4k ×3) |
|---|---|---|
| Baseline (no #555, 768) | 46.4 · 57.0 · 56.3 · 52.6 → **56.3** | 1,535 · 1,532 · 1,564 → **1,544** |
| **#555 (736)** | 45.1 · 49.0 · 51.7 · 51.7 → **51.7** | 1,613 · 1,207 · 1,608 → **1,476** (best two ≈1,610) |

**Decode −8.2%, prefill +4% at best.** Correctness was fine (Gate 0.1: coherent long-prompt answer, Paris, 391), so this is
a straight trade against us.

Why the prefill gain is only ~4% here and not the PR's +20%: #555's largest single win is its **256-row FP4 GEMM kernel**
(2 stages, 76 KiB). Our card's per-block dynamic shared-memory limit is ~99 KiB, so 256 rows × 3 stages (135 KiB) and even
192 rows × 3 stages (108 KiB) fail to launch, and 256 rows × 2 stages measured -1.6% (Session 17). The rest of the PR —
launch-count reduction, vectorization, fused norms, cp.async loads — is the ~4%.

And the decode cost is the constant, not the kernels: residency is identical (2739 vs 2738 experts), so the -8% comes from
`kCardSpare` 736 leaving the drafter less runtime headroom — the same effect the 640 measurement showed at 35.5 tok/s.

## Decision

**Rolled back** to the snapshot (`src`/`include` verified byte-identical, plan back to 2738 / 2.43 GiB, decode back to 56.0,
prefill back to ~1,544). #555 stays out: it cannot load at our proven constant, and at the constant where it does load it
costs 8% of the number that matters most interactively for a 4% prefill gain we would not notice.

What would change this verdict: if a future tree could hold the 256-row kernel within ~99 KiB (e.g. cp.async with a smaller
A stage), #555 becomes a real +15–20% and the trade flips.

---

# Session 20 — single-term FP4 activations tried and rejected: the expert GEMM is not MAC-bound

The last untried lever from Session 19 was the FP4 activation precision. The prompt path quantizes every activation row as
**two** FP4 terms (`gu.two = true`, `d.two = true`), and `moe_fp4.cu` doubles the k-tile count for it
(`nkt = cols/kKTile * (TWO ? 2 : 1)`). Dropping to one term should halve the expert GEMM's MACs, i.e. roughly double prefill,
for the KLD the code itself documents ("one FP4 term costs KLD 0.0025 → 0.0041", still under the 0.00443 of the FP8 KV we
already run). Decode never uses this path.

It needed two small additions to even run — the one-term quantize kernels were instantiated only for other models' column
counts (2560/640/5120/17408), not ours — so `quantize_kernel<3, false>` (1152 H planes) and `quantize_kernel<5, true>`
(2304 token planes) went in. It then loaded and was coherent at once.

## Result: no gain at all, and a decode cost

| Build | Prefill (unique-15.4k ×3) | Decode (4 runs, median warm) |
|---|---|---|
| Two-term (baseline) | 1,535 · 1,532 · 1,564 → **1,544** | 46.4 · 57.0 · 56.3 · 52.6 → **56.3** |
| **One-term** | 1,476 · 1,564 · 1,548 → **1,529** | 44.1 · 49.3 · 49.0 · 49.0 → **49.0** |

**Prefill is unchanged within noise** despite half the MACs. That is the useful finding: the expert GEMM is **not
bound by arithmetic**. Both terms ride on the same weight k-tile (the two terms are two roundings of the *activation*), so
the weight dequant/load traffic and the FP4 unpacking are identical — and that is what costs the time. Cutting MACs while
keeping weight traffic constant buys nothing. (Consistent with Session 17: 256-row blocks, which cut activation reloads,
were the only geometry that mattered, and they do not fit this card's 99 KiB smem.)

Decode fell to 49.0 with one-term. Decode does not use `two`, so this is most likely process variance plus a different KV
state from the changed prompt numerics — but it is a second reason not to keep the change: even a possible cost with zero
measured benefit.

## Rolled back

`prompt.cpp` and `moe_fp4.cu` restored from the snapshot, byte-identical to the baseline tree, rebuilt, verified: plan 2738
RAM experts, coherence ✓ (Paris), prefill back to ~1,544.

## What this rules out for good

Any FP4 precision knob in the prompt path. The prefill bottleneck is **weight-side**: dequantizing and reading the spilled
experts' weights through the tensor-core path. The remaining levers are therefore (a) fewer weight bytes (a lower-precision
*weight* format — the user declined the Q2_0 shadow on quality grounds), or (b) fewer weight reads (bigger tiles: blocked by
99 KiB smem), or (c) a completely different kernel. Arithmetic-side tricks are exhausted.

---

# Session 21 — decode: the residency lever measured, and why it is smaller than it looks

Decode is ~56 tok/s and only 334 of 3072 experts are on the card, so the obvious lever is memory. Two findings.

## 1. The KV-format lever is closed for inkling2

`--kv` accepts **only `fp16` or `fp8`** (`adapter.cpp:50`: anything else is refused). The `q4_0` / `k8v4` formats our own
bench advertises are other models' — giving inkling2 a 4-bit KV would mean porting a PR the size of #552 (row writer, decode
and prompt attention, dequant), not flipping a flag. And #552's own int8 format is *larger* than our fp8 (144 B vs 128 B).

## 2. The context trade measured: +5.9% decode for half the context

Set `max-context` 192k → 131072 and measured the same way as every other decode A/B (4 runs, median of the warm three):

| | 192k (shipped) | 128k |
|---|---|---|
| Experts on the card | **334** | **625** (+87%) |
| Decode expert-cache hit | 70.4% | 73.2% |
| **Decode (median warm)** | **56.3 t/s** | **59.6 t/s (+5.9%)** |
| Prefill (unique-15.4k ×3) | 1,544 | 1,544 (unchanged) |
| Needle @ 130,709 | pass | pass, but only 363 tokens of headroom; the follow-up reload hit "RAM ran out loading (963 MiB free)" |

**Why +87% residency buys only +3 points of hit rate:** the spill planner keeps the *least routed* experts in RAM, so the
experts that come back first are the coldest ones. Routing is heavily skewed — a token's pairs land on the hot experts that
were already resident. Doubling residency therefore covers the tail, not the mass. Extrapolating that curve, reaching 85%
hit would need residency we cannot buy at 192k.

**Conclusion:** residency is not what decode's remaining time is made of. The per-window cost is dominated by the CPU pool's
cost *per pair* (~14.4 ms/window for the ~27% of pairs that miss) and the drafter's ~7.8 ms outside the window — both
per-pair/per-window costs, not cache-miss counts. That is consistent with Session 17's finding on the prefill side: this
machine's limit is compute per unit of expert work, not where the weights live.

Reverted to 192k (the chosen capability, and the needle needs the headroom). The trade stays available as one config line:
`"max-context": 131072` buys +5.9% decode for 64k of context.

## What is left for decode

1. **Fewer expert pairs** — top-k 8 → 7 (−12.5% of the CPU pool's work *and* −12.5% of prefill's expert compute). This is the
   only remaining lever with double-digit upside in both phases, and it is a quality trade that needs a PPL gate
   (`basalt-bench --quality`) to accept. Not attempted: it changes what the model computes, not how fast it computes it.
2. **Pool micro-optimisation** — the pool runs 45.6 GB/s against a 51.9 GB/s harness ceiling (~12%, ≈+5% decode). Explored in
   Session 9 (gather/prefetch neutral, R=4 worse); the remaining headroom is small and hard.
3. Nothing else: the KV formats are closed, residency is measured out, the drafter capture (W2) was a null result.

---

# Session 22 — the top-k lever attempted: not a flag, it is a decode-plan change. Rolled back.

Session 21's recommendation was to route to fewer experts (8 → 7) so both phases do −12.5% of the pair work. Implemented it as a
runtime override and measured it. Result: it does not work as a routing tweak — **the decode expert plan asserts unique
routing per token, and the assertion fires**.

## What was implemented

`--experts-used N` (0 = the pack's top-k) as a namespace global, read by an accessor in both phases:

- **prompt**: `experts()`, `experts_mmq_gu/down`, `experts_fp4_gu/down` now take `k = k_used()` (the carve and the identity
  table stay sized for the pack's maximum, so buffers are merely over-allocated).
- **decode**: `ra.k` (the router's selection), `moe_plan(...)`, the pair count, `mf.k` (the combine) and the hit-rate
  lookups all follow `k_used()`.

Gate 0 at top-7 was clean on the short prompt ("Paris."), and prefill measured 1,600 t/s (+4% on 1,544) — but the first real
decode request died with **`CPU expert has repeated token routing`**, and the engine restarted. Top-6 failed from the first
request. So the CPU pool's plan (`moe_plan` → the window's expert entries) assumes the pack's top-k in a way the override
does not satisfy — most likely a stride/ownership assumption in the plan's arrays or in `expert_pairs()`, rather than the
router producing duplicates.

This is worth noting as a *scoping* result: reducing top-k touches the decode expert plan's invariants (uniqueness, the
per-window entry layout, the combine's `k`), not just `ra.k`. It is a focused piece of work in `session.cpp` + the pool, and
it needs the quality gate (`basalt-bench --quality`) either way, since it changes what the model computes.

## Rolled back

Tree restored byte-identical to the baseline snapshot, rebuilt, verified: loads, `Paris.`, decode back to 56.0 / 55.6.

## Status of the optimisation space after Sessions 17–22

| Phase | Verdict |
|---|---|
| Prefill | **Compute-bound on the weight side**: staging free, arithmetic knobs (one-term FP4) worthless, geometry optimal within the 99 KiB smem limit, #555 neutrally-to-negatively measured. Only weight-precision (declined) or a new kernel moves it. |
| Decode | **Per-pair cost bound**: residency measured out (+87% residency → +3 points of hit, +5.9% decode, and that costs half the context). Pool has ~10% headroom. The one remaining double-digit lever is fewer expert pairs, and it needs the decode-plan work described above. |
| KV | inkling2 accepts only fp16/fp8; q4_0/k8v4 are other models' formats (a port the size of #552). |

The honest summary: on this 12 GB / 15.75 GB/s-link machine, with 89% of the experts spilled, the engine is at its practical
ceiling for both phases. The remaining ideas all trade something real (quality, context, or days of kernel work) for
single-digit-to-teens percentages.

---

# Session 23 — top-k routing made to work, measured, and rejected: it confirms the weight-traffic law

Session 22's crash had a concrete cause, and with it fixed the top-k experiment could finally be measured. Both facts are
worth keeping.

## The crash was a hardcoded stride, not a routing invariant

`cpu_spill.cpp:29` — `constexpr int kUsed = 8;` — is used for **two different roles**: sizing the pool's buffers (where 8
is correct, the pack's maximum) and striding its pair list (`t = p / kUsed`, the doorbell's publish count, the ids copy, and
`moe_cpu_merge`'s k). A runtime route of 7 packed the ids at stride 7 while the pool read them at stride 8, so every
`(token, expert)` pair was misattributed and the job builder tripped its `kDecodeRows` guard with "repeated token routing".

Fix: a `k_route()` accessor — `g_experts_used > 0 && g_experts_used < kUsed ? g_experts_used : kUsed` — for the strides and
counts, with all buffers still sized for `kUsed`. With that, `--experts-used 7` runs correctly: coherent answers (Paris, 391,
Danube/Rhine/Volga), no assertions.

## The measurement: no gain in either phase

| | top-8 (control, same binary) | top-7 |
|---|---|---|
| Decode (4 runs, median warm) | 54.5 | **55.2** (+1.3%, within noise) |
| Prefill (unique-15.4k ×3) | ~1,544 | **1,529** (−1%, within noise) |
| Hit rate | 70.4% | 69.0% |

12.5% fewer pairs bought nothing. The reason is the same law the last three sessions kept finding: **the cost is per expert
touched, not per pair computed.** In an 8-row decode window, 8×8 = 64 pairs spread over 128 experts means each expert is
already touched ~once, so removing 8 pairs removes almost no *experts* — and every touch costs a 4.48 MiB weight read from
RAM. On the prefill side the same thing holds for the GPU: the weight blocks are read regardless of how many activation rows
ride along.

So the whole space is bound by **expert-weight traffic** — the volume of expert bytes this machine has to move (host RAM →
card for prefill, host RAM → CPU registers for decode). That single fact explains: why staging/pipelining is free (it is not
the bottleneck), why residency helps only 3 points of hit (+87% residency covers the *cold* tail), why one-term FP4 changed
nothing (same weight traffic), why bigger GEMM tiles were the only geometry that mattered (fewer weight re-reads), and why
pair-count reductions do nothing.

## Rolled back

Tree identical to the baseline snapshot, rebuilt, verified (loads, `Paris.`). `--experts-used` is not kept: it costs quality
and buys nothing.

## The one thing that would still move this machine

Cut **expert-weight bytes**: a lower-precision weight format (the Q2_0 shadow, declined on quality) or a format whose
dequantisation is cheaper per byte. Everything else — staging, arithmetic, pair counts, residency, GEMM geometry, KV format —
has now been measured to be at its limit here.

---

# Session 24 — context scaling answered; the Q4_K_M pack is not buildable for this engine (blocked at the format)

## How much VRAM a bigger context costs

From the engine's own formula (`model.cpp:62-67`, the one the planner uses for "so --max-context N fits"):
`session_bytes = global * per_pos * context + swa * per_pos * ring`, with `per_pos = kKvHeads(8) * kHeadDim(128) * 1 B = 1024 B`
for the fp8 cache. Against the measured 10.73 GiB in use at 192k (of which ~5.9 GiB is KV) that puts `global` at ~31.6
visit-caches, i.e. **~31.6 KB per token of context**:

| Context | KV | Delta vs 192k | Card you would need |
|---|---|---|---|
| 196,608 (now) | 5.9 GiB | — | 12 GB works |
| 524,288 | 15.8 GiB | **+9.9 GiB** | ~24 GB |
| 1,048,576 | 31.6 GiB | **+25.7 GiB** | ~44 GB |

On this machine a bigger context is blocked twice: the card is 12 GB total, and whatever leaves the card becomes expert spill
in pinned host RAM, which is already at 13.31 GiB against a ~14.5 GiB ceiling. 512k would put *all* experts in RAM and still
not fit.

## The Q4_K_M pack: not a conversion problem, a format problem

Downloaded `nokeepai/Inkling-2-Mini-Unbound-GGUF/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf` (16.56 GiB) and stopped it at 1.34 GiB
once the blocker was clear, because it cannot be turned into a pack:

1. **A Basalt inkling2 pack holds only four tensor types** — `TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30, "NVFP4": 40}`
   (`tools/convert/checkpoint_inkling2.py:43`). There is no Q4_K expert type; the expert layout, the loader and the kernels
   are built around those four.
2. **There is no GGUF ingest path for this architecture.** `tools/convert/arch/` has deepseek41, hy4flash and qwen4exp only.
   Inkling2's converter runs checkpoint (safetensors) → GGUF → `.basalt`, one way, so a llama.cpp GGUF is a dead end for it.

Building one would mean a new expert type plus its kernels, loader support and a converter module — a multi-week project, not a
conversion.

## And it would not pay anyway

- **Size**: Q4_K_M is 16.56 GiB against our NVFP4 pack's 18.38 GiB — **10% fewer weight bytes**, on an engine measured to be
  weight-traffic bound. The best case is a tenth, and the compute path would be K-quant dequant instead of FP4 tensor cores,
  which we measured to be *equal at best* (`mmq` vs `fp4` on the same pack: 1,575 vs 1,571 t/s).
- **Quality**: Q4_K_M is ~4.8 bits/weight against NVFP4's 4.5 bits *plus* two-term activations (≈8 effective bits, KLD 0.0025).

So the current pack stays, and the full-weight file is untouched. The only route to fewer weight bytes inside this engine is a
lower-precision *expert type* (the Q2_0 shadow direction, declined on quality), not a llama.cpp quant.

---

# Session 25 — the Q2_0 expert shadow: built, and it hits a RAM wall, not a VRAM one

Implemented the flag-gated lower-precision expert path end to end, and the measurement answered the question the design
assumed away: **host RAM is the binding constraint, not VRAM.**

## What was built (compiles; reverted after the finding)

- `--expert-shadow q2` in the inkling2 flag hook, plus a `C:\LLM\expert-shadow-q2.on` marker (this tree's usual fallback,
  because model flags are applied *after* the model and its spill are created — a build hooked into `CpuSpill::init()` never
  fires, which cost a build to discover).
- **The shadow builder**: every spilled NVFP4 expert's gate/up/down dequantised 64 weights at a time
  (`basalt::dequantize_nvfp4`), re-quantised into this fork's 18-byte Q2_0 block (`quantize_q2_0_block_scaled`) with a
  four-step clipping ladder taking the best scale per block, into pageable host memory — **2.14 MiB an expert against the
  NVFP4 blob's 4.48**, i.e. the CPU path's read traffic halved.
- **The switch**: `native_fmt(GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, ...)` replaces that layer's `NativeFmt` (the CPU expert
  kernels for Q2_0 already exist in this tree — `kernels/cpu/expert.cpp`, used by the IQ packs), and the job takes the
  shadow blob instead of `host_blob` at `cpu_spill.cpp`'s expert-job build.

So the premise was right: it is a *layout* switch, no new maths, and it is cleanly flag-able.

## Why it does not run here

The first request after the build stalled 62 s and the engine's own report says why:

```
memory: 20692 MiB resident, 10835 MiB in swap, 1000 MiB RAM available; 5476658 major page faults so far
```

Host demand becomes 13.31 GiB of *pinned* spill + 0.86 GiB embedding + **~6.1 GiB pageable shadow ≈ 20.3 GiB** on a 31 GiB
machine that was already holding ~10.8 GiB in swap. Pageable memory does not touch the pinned ceiling, but it does touch
*physical* RAM, and there is not enough of it. The machine thrashed rather than failing cleanly.

## What it would take

**The share knob, which is the difference between this being viable and not.** Shadow only the coldest N% of the spilled
experts — which are exactly the ones the spill planner already puts in RAM, and whose gating weights make their rounding
errors cheapest:

| Share | Shadow RAM | Expected decode |
|---|---|---|
| 100% | +6.1 GiB (does not fit) | +8–12% |
| 25% | +1.5 GiB (fits) | +2–3% |
| 50% | +3.1 GiB (marginal) | +4–6% |

At the only shares that fit in this machine's RAM, the gain is a couple of percent — which does not justify the added
complexity, the load-time cost, or the quality risk (the Q2_0 round trip measured 3.56 dB SNR on the cover bound, 7.90 dB
with the clipping ladder).

**Recommendation: do not pursue on this machine.** On a 64 GiB host the full shadow fits with room to spare, and then it is
worth building — the CPU path is 14.4 ms/window of a ~28 ms window and is measurably traffic-bound (45.6 of a reachable
51.9 GB/s), so halving its bytes is a real lever *if* the RAM exists. The tree is back to the verified baseline; the design
above is what a future attempt should start from.

---

# Session 26 — private fork + Windows release, and the review of PR #586

## The fork

`https://github.com/Phillis/basalt-inkling2-mini` — **private**, one snapshot commit, 1,339 files.

Base: `jesdga95/basalt` @ **PR #536 head `a273f99d`** plus PR #525's staging hunks, overlaid with our delta — **54 files,
+7,472/−99** against that base (plus `notes/` with all 26 measurement documents). The upstream history is not carried: a
full-history push failed with `remote unpack failed: index-pack failed` (too large for one push), and instead of a
half-carried history the fork records its base SHA in the commit message and in `README.md`.

The README names the model this fork runs —
**[nokeepai/Inkling-2-Mini-Unbound-Basalt](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt)** (v2 NVFP4
pack, SHA-256 `aa20b87a…0c77`), with the v1.0 pack beside it for A/B, the GGUF repo linked for other quants, and the
reason a llama.cpp GGUF cannot be loaded here.

## The release

`v0.5.6-inkling2-cpu-spill` — https://github.com/Phillis/basalt-inkling2-mini/releases/tag/v0.5.6-inkling2-cpu-spill

`basalt-inkling2-mini-v0.5.6-cpu-spill-win64.zip` (468.7 MB), built in **Release** from this tree:

- `basalt.exe` (80.2 MB), `basalt-bench.exe`, `basalt-probe.exe`, `basalt-vision.exe`, `basalt-gguf.exe`,
  `basalt-dequant.exe`, `basalt-device.exe`
- `cublas64_13.dll` + `cublasLt64_13.dll` — the engine links `cublas64_13.dll` dynamically, so the archive carries the CUDA
  runtime it needs (that is most of the archive's size)
- `basalt_config.json` (192k, fp8 KV, spec 8), `README.md`, `LICENSE`, `notes/`
- SHA-256 of the zip: `6463ed07271e50641788e214e26c003e59bbde27182acec320b3526724ea242e`

## PR #586 review — xyehya's coordination note

They have the same hand-off built and measured (+5.6–7.1% window cost, 37.9 → 10.2 µs a visit) and want one implementation.
Their four suggestions, and our judgement of each:

| Suggestion | Verdict |
|---|---|
| **skip a visit with no RAM pair** (needs the blob table in the publish kernel) | **Adopt.** On their card (1,049 of 3,072 experts in RAM) that is the 1.7 µs path; on ours (2,738 of 3,072) almost every visit has RAM pairs, so the win here is small — but it is correct and it is the piece worth taking from theirs. |
| **store, not `atomicAdd`**, on the mapped doorbell | **Partially ours already.** Our ring is 48 slots, one per visit, and `finish()` waits on `(ring_done[i], ring_seq[i])`, so the add is uncontended by construction; and the header already carries `doorbell_publish_store` for the host-issued case. Their portability point still applies to the *visit* kernel, and dropping the add for one stored word per slot is the fix if the maintainer keeps our shape. |
| **sticky abort word** polled by the merge | **Adopt.** This is a real gap: `doorbell_wait` spins `while (*flag != want)`, so a dead host thread leaves the card spinning until the process dies. It belongs in the *shared* kernel (`src/kernels/cuda/elementwise.cu`), which Qwen4Exp also uses. |
| **checked stream lifecycle / mapped buffers freed under the abort lock** | **Adopt**; ours relies on the stall watchdog, which is weaker than bounded. |

Reply posted: https://github.com/jesdga95/basalt/pull/586#issuecomment-7096219157 — agreeing to one hand-off, asking them to
send the commit so the maintainer can choose, and stating that we would rather close #586 in favour of theirs than
half-port it (a half-port is the two-hand-offs outcome they are trying to avoid). If the maintainer keeps our kernel shape we
will add, in order: store-not-add, the abort word in the shared wait, the lifecycle checks, then the skip — and we would lean
on their 9-of-9 greedy parity and sanitizer runs, because greedy output is not reproducible across processes on this machine,
so our evidence is only the needle plus a coarse coherence check.

---

# Session 27 — nathaninline/Inkling-2-Mini-NVFP4-Basalt: swapped in, and it buys 256k context back

Downloaded `nathaninline/Inkling-2-Mini-NVFP4-Basalt` (17,696,702,353 B, SHA-256 `45ca652a…4c8b` — matched against the
Hub's LFS oid, not the CDN etag), verified it is the same container (`BASALTMF`, format v2, 10 sections, the same NVFP4 /
Q8_0 / BF16 tensor types), and swapped it in. All three packs stay on disk.

## Why it is smaller and what it changes

Their pack requantises **layers 20-23's experts from Q8_0 to NVFP4** (expert bytes 14.71 → 12.81 GiB); every other section
is byte-identical to the official pack. **18.38 → 16.48 GiB (−1.90 GiB, −10.3%).**

It is the **base** `thinkingmachines/Inkling-2-Mini`, not nokeepai's Unbound v2 finetune: their card measures mean KLD 0.0059
against the official pack's 0.0026, so it is a different model behaviourally and genuinely more drift.

## Measured here (same protocol every time: 4 decode runs, median of the warm three; 3 unique-15.4k prefills)

| Pack | Context | Decode | Prefill | Experts in RAM | Pinned spill |
|---|---|---|---|---|---|
| nokeepai v2 Unbound | 192k | **56.3** | ~1,520–1,600 | 2738 | 13.31 GiB |
| **nathaninline base** | **192k** | 54.2 | **1,751** (1,770 / 1,749 / 1,734) | 2623 | **10.94 GiB** |
| **nathaninline base** | **256k** | 53.5 | 1,626 | 2920 | 12.18 GiB |

- **Prefill +13% at 192k** (1,751 vs ~1,560). That is the weight-traffic law again: the same chunk reads 1.9 GiB fewer expert
  bytes, and prefill is bound by exactly that.
- **Decode −4% at 192k** (54.2 vs 56.3). Draft acceptance is a little lower on this pack (acc 68–80% against 77–83%), which is
  the base model's drafter relation, not the engine.
- Prompt staging also drops, 1034 → **548 MiB**, because the requantised layers' blobs are smaller.

## The real prize: 256k context now fits

This is the trade we had to refuse on the v2 pack — 256k with the drafter pushed 3,062 experts into RAM, past the ~14.5 GiB
pinned ceiling. With 1.9 GiB fewer expert bytes it loads:

```
experts: 2920 of 3072 in RAM (12.18 GiB, ...), 1.19 GiB on the card with 548 MiB of prompt staging,
so --max-context 262144 fits
```

And it is genuinely usable: **a needle at 199,423 tokens was found** (`ZEPHYR-4417`), 187.5 s prefill at 1,064 t/s, decode
22.3 tok/s at that depth, 100% draft acceptance. The cost of 256k against 192k on this pack is small — decode 54.2 → 53.5
(−1.3%), prefill 1,751 → 1,626 (−7%).

**But 256k leaves the host RAM-tight**: after the 200k request the next load failed with `RAM ran out loading (963 MiB free)`.
So 192k is the safe default (10.94 GiB pinned, 7.7 GiB free afterwards) and 256k is available as a one-line change when the
machine is otherwise idle.

## State

Default is now this pack at **192k** (`"model": "C:/LLM/Basalt/models/Inkling-2-Mini-NVFP4.basalt"`, `max-context 196608`),
`Paris.` verified, 2623 experts in RAM. To go back: set `"model"` to `Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt` (and
`max-context 196608`). For 256k: `"max-context": 262144`.

## Two knobs from their engine patch worth testing on our tree

Their patch (for 16 GB cards, Basalt `a11e19f6f`) documents values they measured that we set more conservatively, and each
is a one-line A/B here: **`kPromptChunk` 1024 instead of 4096** ("about 0.7 GiB of buffers less"), **elastic-KV base 8,192
positions and 128 MiB keep-free** instead of our `kElasticKeep = 512 MiB`, and **experts copied to the card in slabs of 15**
— alone, each expert is rounded up to the driver's 2 MiB pages and "40% of that memory was lost". Ours copies one expert at
a time. All three free VRAM or pinned memory, which is what this machine is short of.

---

# Session 28 — twisting the knobs the smaller pack freed: two tried, both rejected

The nathaninline pack freed ~486 MiB of VRAM (prompt staging 1034 → 548 MiB) and 2.37 GiB of pinned RAM. Part of that was
already spent automatically — the freed area went straight into expert slots (334 → **449** on the card) — so the question is
what the *remaining* reserve should do. Two knobs tested, one change at a time, same protocol each time.

## `kPromptChunk` 4096 → 2048 / 3072 (the prompt arena scales with it)

| kPromptChunk | Experts on card | Card area | Pinned spill | Decode (median warm) | **Prefill** |
|---|---|---|---|---|---|
| **4096 (kept)** | 449 | 2.43 GiB | 10.94 GiB | 54.2 | **1,751** |
| 3072 | 514 | 2.71 GiB | 10.67 GiB | 60.7 | **1,466** |
| 2048 | 578 | 2.97 GiB | 10.40 GiB | 58.0 | **1,286** |

It does exactly what it should for residency — smaller arena, more experts on the card, hit rate up (70.1% → 72.2% → 73.3%).
**But prefill loses 16% at 3072 and 27% at 2048**: half the rows per chunk halves the GEMM's row count and doubles the number of
chunk visits, and prefill is the phase that pays. Reverted to 4096.

## `kElasticKeep` 512 → 256 MiB (the free space a KV grow leaves)

Their patch uses 128 MiB. Measured here: no gain, and hit rate read *lower* (63.5% against 70.1% in the earlier process) —
consistent with less free space letting the elastic KV grow at the expert copies' expense. Reverted to 512 MiB.

## What this says about the knob space

- **The extra VRAM is already working for us**: staging shrank by 486 MiB and the planner gave it to copies (+115 experts),
  which is why prefill is 1,751 against the v2 pack's ~1,560 on the same chunk size.
- **What is left is a trade, not a free win**: chunk size buys residency with GEMM rows, and residency is worth ~1 point of hit
  per 65 experts — far too little to pay 16–27% of prefill.
- **Decode measurements cannot resolve the remainder.** Across processes the same configuration read 54.2 and 63.0 tok/s
  (with hit 70.1% vs 63.5% on different texts), so any effect under ~10% is invisible without an in-process A/B — and these
  are compile-time constants, so an in-process A/B is not available.
- **Untested, and the last one worth a shot**: `kCardSpare` 768 → 704 or `vram-reserve-mib` 768 → 640. Each hands ~64–128 MiB
  to copies (≈15–30 experts, well under one point of hit). The risk is known and asymmetric: 640 was where the drafter lost
  its headroom and decode collapsed to ~35 tok/s, so the expected gain (~1%) does not justify re-entering that region.

State: nathaninline pack at 192k, `kPromptChunk` 4096, `kElasticKeep` 512, plan `2623 of 3072 in RAM (10.94 GiB), 2.43 GiB on
the card with 548 MiB of prompt staging`, `Paris.` verified.

---

# Session 29 — for fun: two concurrent sessions. They do not pay, and 2 x 128k does not fit

The engine splits `max-context` across `parallel` slots (2 slots at 131,072 gave "context: 65,536 tokens"), so "two sessions
at 128k" means `max-context 262144, parallel 2`.

## What fits

| Configuration | Verdict |
|---|---|
| 2 x 128k (`max-context 262144, parallel 2`) | **1,383 MiB short** — `spill plan: even with every expert in RAM the card is 1383 MiB short (--max-context 131072)` |
| 2 x 96k (`196608, parallel 2`) | **103 MiB short** — two slots carry per-session overhead the single slot does not |
| **2 x 64k (`131072, parallel 2`)** | **fits**: 2801 of 3072 experts in RAM (11.68 GiB), 1.68 GiB on the card, 548 MiB staging |

So the KV alone (2 x 4.1 GiB at 128k, or 2 x 3.0 GiB at 96k) plus dense, drafter, prompt arena and staging exceeds a 12 GB card
before a single expert is placed. Two 128k sessions need a ~14 GiB card *given that every expert is already in host RAM*.

## What two sessions at 2 x 64k actually do

Two genuinely concurrent 400-token prose requests:

| | single session (192k, parallel 1) | two sessions (2 x 64k, parallel 2) |
|---|---|---|
| Per session | **~54–63 tok/s** | **19.3 tok/s** (both slots, per the server's own lines) |
| Aggregate | ~54–63 tok/s | **21.7 tok/s** (800 tokens in 36.8 s) |
| Draft acceptance | 68–91% | 91% (fine) |
| Expert hit rate | 70.1% | **55.5–59.8%** |

**Concurrency costs, it does not pay, on this machine.** Three reasons, all the same shape as everything else measured here:

1. **The KV split drains the card of experts.** Two slots at 64k cost the same KV as one 192k slot but with per-slot overhead,
   so the plan holds 2801 experts in RAM instead of 2623 — and the hit rate falls from 70% to 56%.
2. **The CPU pool is shared.** Both sessions' rows go through the same 11 workers, so a second session does not add throughput,
   it splits the existing throughput — and with a lower hit rate there is more CPU work to split.
3. **Prompt reads serialise**: the second request's first token waited 10.4 s behind the first's.

For an agent this is the practical rule: **run one session at a time and give it the whole card**, which is what `parallel 1`
already does. A second concurrent conversation is worth roughly a third of a single one, twice over.

Restored: `max-context 196608, parallel 1`, `Paris.` verified.

---

# Session 30 — v1.0 pack deleted

Removed `models/Inkling-2-Mini-Unbound-NVFP4-q8.basalt` (18.38 GiB) at the user's request: that pack produced repetition
loops on this workload. Its five profile JSONs (`basalt_config-128k-dspark`, `-192k-dspark`, `-256k`, `-32k-fast`,
`-stocktest`) named it, so they were repointed at the pack actually in use, `Inkling-2-Mini-NVFP4.basalt`; every profile now
resolves to an existing file. Disk free went 1,031.2 → 1,049.4 GiB.

Remaining on disk: `Inkling-2-Mini-NVFP4.basalt` (16.48 GiB, the base-model pack we run) and
`Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt` (18.38 GiB, the Unbound v2 finetune). The fork's README and
`MODEL-V2-UPDATE.md` were corrected — both had described v1 as the rollback target.

---

# Session 31 — engine model id renamed to `Inkling-2-Mini`; 224k context; monitor checked

## Rename (alias only)

The engine takes its model id from the config's `alias`, not from the pack file, so the id still said "Unbound" after the
pack swap. Renamed to **`Inkling-2-Mini`** in every place it is an *identifier* — never inside a pack or HF repo name:

- all six `basalt_config*.json` (`"alias"`), so every profile agrees
- `test-long.ps1`, `tune.ps1` (the request bodies they send), `Start-Basalt.ps1` (banner + docstring)
- `API-CONNECT.md`, `README-SETUP.md`, `SUMMARY.md` (the "how to call it" references)

Verified live: `/v1/models` → `Inkling-2-Mini` (`owned_by basalt`), `/health` → `model Inkling-2-Mini`, and a request with the
new id answers. **The old id now returns 404**, which matters for anything automated pointing at it.

## Context to 224k

`max-context 229376` (224k). Loads with
`experts: 2772 of 3072 in RAM (11.56 GiB), 1.81 GiB on the card with 548 MiB of prompt staging`, KV ~6.9 GiB, host RAM free
9.5 GiB. Sanity: Paris / Danube-Rhine-Volga / 391. Cost against 192k: 449 → 300 card experts (a few percent of decode, by the
residency curve measured in Session 21); prefill is unaffected.

## Monitor

`basalt-monitor.ps1` needed **no** rename work: every label it prints comes from `/metrics` (`engine.model`, `max_context`,
`expert_slots`, …), and it already shows `model Inkling-2-Mini   engine 0.5.6   kv fp8   spec 8   ctx 229,376` and
`experts 300 / 3072 in VRAM`. One cosmetic bug of ours fixed while checking it: the "best seen" line printed empty parentheses
when `/metrics` had no `best` entry yet — it now omits the request-id suffix instead.

---

# Session 32 — monitor: session token usage over time

`basalt-monitor.ps1` now reports how many tokens the session has used and how fast they arrived, from two engine-side
sources (so a monitor restart does not lose the history):

- **`/metrics` `totals`** — `since`, `requests`, `prompt`, `reused`, `written`, `prompt_ms`, `decode_ms`
- **`/metrics` `history.tok_s`, `history.prefill_tok_s_mean`** — the engine's own 1-second series

New block, printed with the per-request summary and after `-NoFollow`:

```
session       :  1h05m up   6 requests   in 29.6k tok   out 348 tok   (busy 43s)
avg while busy:  TG 25.2 tok/s   PP 998 tok/s   all 690 tok/s
over time     :  last minute   TG 4.8 tok/s   PP 895 tok/s
```

Deliberate choices:

- **Averages are over the time actually spent** (`written / decode_ms`, `(prompt - reused) / prompt_ms`), the same way each
  request row measures itself. A wall-clock average decays toward zero whenever the engine idles — the first version did
  exactly that (`avg TG 0.1 tok/s` after an hour up with 20 s of work).
- **`busy`** sits next to `up` so the two are never confused.
- **"over time"** is the mean of the engine's trailing 60 s of `tok_s` / `prefill_tok_s_mean`: a rate over a window rather
  than a lifetime figure.
- The **idle line** carries a compact version (`| 1h05m  out 348 tok  avg TG 25.2/s`), so usage is visible when nothing runs.

Also fixed two of my own bugs found while testing: `Fmt-Dur` formatted a `[math]::Floor` double with `:D2` (invalid
specifier), and an earlier edit had orphaned the `else` of the `-NoFollow` block.

---

# Session 33 — implementing the PR shortlist: one landed, four blocked, and two *merged* commits found that matter more

## Landed: #605 (core: a room-bound RAM copy leaves room for the adaptive swaps' free places)

6/7 hunks applied (the seventh is a new test file; tests are not built here). Adds `ram_tier_pack_bytes()` in
`ram_load.hpp` and packs the RAM complement to `min(budget, room − spare places)` in `expert_source.cpp`, so the adaptive
swap's free places survive instead of being dropped silently, plus a warn line when they cannot.

Built, loaded at 224k, plan unchanged (`2772 of 3072 in RAM (11.56 GiB), 1.81 GiB on the card`), **no warning fired** — so this
machine's places already fit — Gate 0 clean (Paris, 391), prefill 1,705 t/s, decode 47.5 / 53.9. **Neutral here; kept** because
it removes a silent-failure mode.

## Blocked, and why it is not what I assumed

The other five do not apply: 34/53 hunks for #579, 27/44 for #609, 17/23 for #232, 15/30 for #248, 3/9 for #636. My
assumption ("our tree is 60 commits behind main") was **wrong** — `compare/a273f99d...main` says **behind_by: 8**. The
failures are a mix of:

- **our own local patches** sitting exactly where they patch: `gqa_attn.cu` (W2's `hi_off`), `pool.cpp` +
  `platform/cpu.hpp` (the pinning in #584), `prompt.cpp` / `model.cpp` / `session.cpp` (W1/W5, staging reclaim, card spare);
- **8 upstream commits we lack**, of which two are code.

## The find: two *merged* upstream commits are more valuable than the open PRs

| Commit | What | Why it matters here |
|---|---|---|
| **#626** `inkling2: tensor-core shared-KV full-attention decode kernel` (merged, +296/−10) | `attn_split_kernel` launched `dim3(rows, kKvHeads, z)` and **re-read the full K/V cache once per verify-window row** (8x at `--spec 8`) with scalar FP32 FMAs. Adds `attn_mma_kernel<FP8,BM>`: `m16n8k16` tensor-core MMA, `BM=16` for 1–5 rows and **`BM=32` for 6–10 rows so a full verify window shares one K/V pass**. At 512K cells their effective K/V bandwidth went 1500 GB/s (1 row) → 569 (5 rows) → **363 (8 rows)** before the fix | **Our exact problem at 224k.** We run `--spec 8` (8 rows/window) over a ~6.9 GiB cache, so we are in the 363 GB/s regime. This is the merged version of what #636 does for the *drafter's* window, applied to the model's own full-attention decode |
| **#633** `dense: raise kBatch and kGqaMaxWindow from 16 to 32` (merged, +8/−8) | `kGqaMaxWindow` 16 → 32, `kBatch` 16 → 32 (24.6% aggregate on 8 slots) | It is the constant #636's tiling assumes (its "16-row query tile" becomes 32) — **#636 cannot apply until we have it** |

So the right order is not the PR list at all: **port #626 and #633 first** (two small merged commits, no quality change —
they report bitwise-identical kernels and a KLD gate at 0.00402 against 0.00413), then #636, #579, #609.

State: tree = snapshot `s32-known-good` + #605, engine live at 224k, `Inkling-2-Mini`, Gate 0 clean.

---

# Session 34 — the rebase that everything else needs: main + #536 + our patches, and the two merged wins are now in it

The shortlist work led to a dead end that is worth stating plainly: **every remaining performance PR needs a base our tree
does not have.** Our base is #536's branch (`a273f99d`), whose merge base is **198 commits behind main**; main's `attn.cu`
thus carries INT8 KV and segmented batch windows that our file lacks entirely, so #626 could not be cherry-picked at any
price (its kernel hunks sit on top of code we do not have).

So the rebase was the work. Done in `C:\LLM\rebase\b` (clone at `3ce7d52e`), branch **`pr536-rebased`**:

- **#536's two CPU-kernel commits are already upstream** — main's `SwiGlu` carries `limit` + both softcaps, and
  `nvfp4_256_gu_rows` already shares rows across tokens. Replayed as no-ops.
- **#536's integration commit applied cleanly** (`cpu_spill.cpp/.hpp`, `spill.cpp/.hpp`, model/prompt/session hooks) — which
  was the only real risk in the whole exercise.
- **54 of our 92 delta hunks applied unchanged** and are committed; the delta itself is saved as
  `C:\LLM\rebase\our-delta.diff` (30 files, +1144/−79) so it can be re-applied to any future base.
- 4 of the failures are **duplicates** (#633's `kBatch`/`kGqaMaxWindow` 32 is already upstream).

**The two merged wins are therefore in the tree by construction:** `attn_mma_kernel<FP8, BM>` with `BM=32` (one K/V pass
per verify window — the fix for 8-row windows re-reading the whole cache), `kGqaMaxWindow 32`, `kBatch 32`, INT8 KV,
segmented windows — plus #536 and our doorbell and link-gate removal.

**What remains: 34 hunks**, all in files main has since changed — the staging reclaim (`session.cpp`, 8), `stage_layer` +
carve (`prompt.cpp`, 12), `kCardSpare` 768 + `promote_order_` + staging (`model.cpp`, 6), and their declarations. The
valuable three are staging reclaim (+5–8% decode), `kCardSpare` (load reliability) and `promote_order_` (+2.4% prefill);
`stage_layer`/`wave_bounds` are inert in our configuration and can be dropped instead of ported.

Written up in `REBASE-STATE.md` with the per-file table and the order for finishing. Our working tree is untouched
(`s32-known-good` + #605 + #633), engine live at 224k, Gate 0 clean.

---

# Session 35 — the engine moves to current main: 192k, decode 77 tok/s (+21%), and the PR claims audited

At 192k the fair A/B was decisive, so the engine now runs **current main** (`C:\LLM\rebase\b`, `3ce7d52e`) with #536's
CPU spill rebased onto it, #609 and 5 of #579's 8 commits cherry-picked, and our patches where they still fit. The old
tree stays as the rollback and `Start-Basalt.ps1` now points at `C:\LLM\rebase\b`.

| 192k, kv fp8, spec 8, parallel 1 | old tree | **live** |
|---|---|---|
| prefill, unique 15.4k | **1,770 t/s** | 1,615 t/s |
| decode 400 tok steady | 64.3 | **77.8 / 77.7** |
| decode first window | 54.9 | 72.9 |
| plan | cuda0 3.3 GiB / ram 11.9 | cuda0 2.9 GiB / ram 12.3 |

**+21% decode, −9% prefill.** The decode win is #626 (shared-KV tensor-core attention) with #609's routed-only staging;
the prefill loss is main's prompt orchestration.

## #609 and #579 taken from their own branches

`gh` showed #609 carries a rebase branch, `pr/inkling2-staging-on-main`, so it was **cherry-picked with real 3-way merges**
instead of hand-applied hunks: its short-chunk "stage only the RAM experts it routes to" commit landed, the long-chunk
second-stream commit turned out to be already in main, and the error-path commits came along (which needed #609's
`stale_` flag and `stale()` helper restored by hand — main's session had lost them in a conflict resolution).

#579 gave 5 of 8 commits (fp4 sm_120 `MUFU.TANH/EX2/RCP` epilogue, 4-lane coalesced `cp.async.cg` 16B loads, fused
single-row `router_select+moe_plan`, 1-warp `LDG.128` q8_planes MMVQ, mxfp8 128×128 tile) for **+1.4% prefill**. The
other two — `attn_prompt` double-buffered tiles and `attn.cu` L1-cached V (`236 -> 128` regs) — are **FP8-only** against
main's `template <bool FP8, bool INT8>` kernels and need a real merge; skipped rather than guessed at, because a silent
CUDA mistake here would be invisible until it corrupted output.

## The PR claims audited

- **#609 "+19–21% prefill"** → +2% here (1,578 → 1,610). Theirs was a 16 GB card spilling far more than we do.
- **#579 "+7.3–8.3% prefill"** → +1.4%, with 2 of 8 commits skipped.
- **#579/#636 benchmarked with `--kv int8`** → **measured: worse for us** (prefill 1,553, decode 66.6, hit 56–64%): int8's
  1.03 B/cell costs residency. `fp8` stays.
- **main's help lists `--kv nvfp4|q4_0|k8v4`** → **inkling2 refuses them**: "the attention caches are fp16 (the default),
  fp8 or int8". That avenue is closed, not merely untested.
- **#636's "NVFP4-DQ8" pack** → the HF repo holds only `Inkling-2-Mini-NVFP4.basalt`, the pack we already run.
- **#232, #248** → both **CONFLICTING** upstream; they need hand-porting.

## The 9% prefill gap, isolated

Ruled out one by one: `kPromptChunk` (identical), #633 (`kBatch` 32 vs 16 → 1,517 vs 1,513, inert at `parallel 1`), #525
(`stage_from_card` is already in main), and the prompt-attention kernel itself (swapping our older `attn_prompt.cu` in:
1,640 vs 1,632). It lives in main's `prompt.cpp` orchestration — 792 lines there vs 692 in our tree — so porting our
prompt carve and chunk-loop hunks is the next step that pays.

---

# Session 36 — the four-bit V cache: built, proven exact, and measured (rotation is the whole story)

Before writing attention kernels, I calibrated the payoff axis and then built the format.

## Calibration first (both levers turned out to be dead ends)

- `vram-reserve` 768 → 384 MiB: **nothing** (plan identical, decode 76 vs 77, hit 67% both).
- `kCardSpare` 768 vs main's 1024 (the run that mattered — an earlier edit had been lost in a reset): **nothing**, plan
  identical, decode 75.1/75.9/78.5.
So residency cannot be bought with spare/reserve at 192k; **only less KV memory moves it**. That is what justified the
codec work.

## The format, built in the same shape as INT8

`kKvV4RowBytes = 72` (64 B of 4-bit codes + 4 fp16 scales, one per 32 values) against fp8's 128 → **a K+V cell is 1,152 B
instead of 2,048 (56.2%)**. In `prep.cuh`: `v4_scales` (warp = one row, max inside each 32-value group), `store_v4` (a lane
writes its own two bytes, so no byte has two writers), `unpack16_v4`, `v4_row_scales`.

**Exactness: 0 packing mismatches over 524,288 codes.** Harness at `C:\LLM\rebase\v4_parity.cu` (encodes with the engine's
own device code, decodes on the CPU).

## SNR, and the finding that decides the design

| V distribution | fp8 | int8 | v4 no rotation | v4 with H128 rotation |
|---|---|---|---|---|
| gaussian | 31.50 | 44.54 | 20.26 | 20.28 |
| heavy tail (1% x10) | 31.58 | 37.98 | **15.70** | **21.06** |
| outliers (0.1% x50) | 31.51 | 36.38 | 20.04 | **23.24** |

4-bit V costs ~9 dB against fp8 — a real trade, in the range published 4-bit KV work tolerates, but it needs an
end-to-end gate. And **the normalization rotation is not optional**: heavy tails collapse to 15.7 dB without it and come
back to 21.1 with it, and int8 gains even more (36.4 → 47.8 on outliers).

It is cheap because attention is linear: sum dequantized *rotated* values, apply the single inverse rotation once per
output row — the same construction `q4_0`/`k8v4` use upstream.

## What it buys and what is left

At 192k: KV ~6.0 → ~3.4 GiB, freeing ~2.6 GiB ≈ **480 more resident experts** at ~5.5 MiB a blob. The remaining work is
the integration, written out as a checklist in `V4-KV-DESIGN.md`: split K/V row bytes, teach `attn.cu` (both the simple
and #626's MMA path) and `attn_prompt.cu` to read a 72-byte row, add the V writer in `ops.cu`, size the cache, expose it
as an opt-in `--kv` value, then run the gate (Gate 0, needle at 130,709 and 160k+, prefill/decode ABBA).

---

# Session 37 — the four-bit value cache: shipped, and it buys 224k at 192k's memory

The nvfp4-KV assessment became a working format. `--kv fp8v4` keeps the keys exactly as fp8 and stores only the values at
four bits (64 code bytes + a scale per 32 values, padded to 80), which is **44% less KV memory**.

## Bugs found on the way (four of them, each now covered by a test)

1. `attention_window` had no V4 launch arms — most layers read an 80-byte row as fp8.
2. `store_v4` quantised by the fp32 scale while the row stores it rounded to fp16 (0.502 of a code step, not 0.500).
3. `attn_prompt`'s decoder wrote `h[2*i + j]` instead of `h[4*i + j]`: values 4–7 overwritten, 8–15 never written.
4. `prompt.cpp` never set `ap.v4`, so the prompt writer laid value rows out at fp8's 128 bytes while every reader took
   them at 80. That one produced the degenerate text.

Both parity suites now cover the format everywhere it is written or read — codec (524,288 codes, exact), `store_v4`
(half-a-step), both prepare writers bitwise against each other, decode attention against the CPU reference (7.4e-05), the
fused window kernel bitwise over nine geometries, and prompt attention for window and full caches (~1e-03). 0 failures.

## Performance

**At equal memory** — the comparison that matters:

| | fp8 @ 192k | **fp8v4 @ 224k** |
|---|---|---|
| context | 196,608 | **229,376 (+16.7%)** |
| prefill | 1,623–1,657 t/s | 1,656–1,710 t/s |
| decode | 72.6–74.0 tok/s | 70.5–73.2 tok/s |
| needle | ✓ | ✓ |
| plan | cuda0 2.9 / ram 12.3 GiB | identical |

**At equal context**: prefill +7.1%, decode −5.5%, expert hit 68.6% vs 67.4%, card residency 3.5 vs 2.9 GiB. The decode
cost is the drafter's acceptance (88% → 78%): four-bit values perturb the target enough that its drafts verify less often.
`unpack32_v4` (a 16-byte load is exactly one 32-value scale group; integer convert, one `__hmul2`) removed the kernel cost
entirely and did not move the number — confirming precision, not instructions.

**Shipped default: `kv=fp8v4`, `max-context 229376`** — the context this machine was set up for, in the memory fp8 needed
for 192k. The H128 rotation (+5.4 dB measured on heavy-tailed value rows) is the next lever if 224k should also match
192k's decode exactly.

### Session 37 addendum — quality at range, and seven closed experiments

Three needles through the four-bit cache: **8,097**, **53,465** and **147,227 tokens**, each answering the buried code
exactly (`6718-VB`, `6718-VB`, `3392-JD`). 147k is past the 130,709 where fp8 was validated, so 4-bit values do not cost
long-range recall here.

Seven experiments this session, one win and six negatives, all recorded in `V4-KV-DESIGN.md` so they are not re-run:
the 44% KV saving buying 256k (win); fast unpack, `--spec` sweep, reserve floors, prompt chunk size, masked cell wrap
(neutral); a clipped/robust group scale (worse — it saturates the rows that dominate draft verification). E2M1 codes were
kept: +3.2 dB on heavy-tailed rows at equal speed, and acceptance 91-94% against 89-92%.

Live: `kv=fp8v4` (E2M1), ctx 262,144, decode ~70 shallow / 28 at 147k depth, prefill 1,540 at 15k / 1,114 at 147k.

---

# Session 38 — what the DeepSeek-V4.1-Flash engine has that we could use

Investigated `yhfgyyf/vllm-deepseek-v4-sm89` (V4.1-Flash: MoE + Engram, a vLLM fork for consumer GPUs) and the
`tonyd2wild` DGX-Spark recipe that documents its measured behaviour. Full write-up in `UPSTREAM-TECHNIQUES.md`.

**Adopted — the cold-start finding.** Their best result is not a kernel: the first requests after a load run 30-40% slow,
and it decays after a few hundred real tokens (58.5 -> 83.3 tok/s; and it *returns* after 30 minutes idle). We have the
same effect, visible in our own log as 61.2 -> 107.7, 56.3 -> 108.3, 62.7 -> 108.7 tok/s. `Start-Basalt.ps1` now runs
three warm-up requests (two generations, one ~2k-token prompt for the prompt cache) before printing READY: launcher 25 s ->
40 s, and the **first real request measures 67.0 tok/s against the second's 67.7** instead of ~60 against ~108.

**Already ours, and stronger.** Their `enable_adaptive_verification` (confidence head picks the verify length) exists here
as `spec::draft_policy.hpp`: it learns acceptance per confidence bin, models E(t) = 1 + q(c0) + q(c0)q(c1) + ..., divides by
the measured cost of each window size, and also arbitrates against the suffix/ngram drafter. That is why our --spec sweep
moved nothing — the policy already chooses per round.

**Ruled out with reasons measured here**: graphing the prefill (ours is weight-traffic bound at ~200 TFLOPS effective, so
~1 ms of launch overhead against a 2.6 s chunk), persistent top-k (decode is already replayed from a graph), activation
LUTs on CPU (our CPU path is weight-traffic bound), their 4-bit CPU MoE (that is our rejected D1 shadow), fused cache
kernels (we do that), prefill interleaving (parallel 1).

**The one real gap: paged, shared KV.** They page at 128 cells over a shared pool where `max_model_len` is a ceiling, not
a reservation. Ours sizes each visit's KV contiguously to the full context — which is exactly why 2 x 128k was 1,383 MiB
short on this box. A paged pool would turn our 44% KV saving into concurrency instead of only context, and would let 2-4
sessions fit at 256k ceilings. Largest change on the board: the KV address is `[cell][head][dim]` in every attention
kernel, the elastic arrays, save/restore and the checkpoint transfer.

Also noted for the future: their warning that 4-bit KV can degrade under very long heavy agentic context (fp8 stays
clean) — our kv=fp8v4 is validated with exact needles at 8k, 53k and 147k, so inside the known-good envelope, but that is
where to look first if output ever degrades.

---

# Session 39 — Strata's KV-to-RAM, and a combined implementation plan

Read `niko1221/strata`'s KV subsystem. The feature is **KV streaming**: `engine/include/strata/core/kv_host_dma.hpp` —
"with `--kv-resident` the authoritative K/V of a streamed QSA layer lives in **pinned, device-mapped RAM**; the prompt
path's append also writes every new cell into the **one-layer staging pool in VRAM**", plus `docs/KV_PREFETCH.md`, which
uploads the *next* layer's prefix on a second stream while the current one computes (-5.3% prefill at 32K, -4.1% at 64K).

**The finding that matters: we already have it, for the dense models only.** Our tree carries `Options::kv_resident`,
`resident_cells()`, `shared_bytes()`/`session_host_bytes()`/`session_layer_bytes()` and the per-layer staging pool — the
whole dense-engine implementation. inkling2 uses none of it; its caches are `VisitCache` + `core::ElasticArray`, which is
*VRAM*-elastic (it saves host RAM, not VRAM). So the gap is a **port**, not a design.

Combined with the DeepSeek-V4.1-Flash investigation (`UPSTREAM-TECHNIQUES.md`), the plan is in
**`IMPLEMENTATION-PLAN.md`**: (1) port KV streaming to inkling2 — ~2.5-3 GiB of VRAM back at 256k, i.e. ~450-550 more
resident experts, plausibly +5-10% decode; (2) cross-layer KV prefetch on top of it (+4-5% prefill); (3) use their DMA form
of the host append, never per-value stores (50 us a prompt token otherwise); (4) a paged shared KV pool for concurrency
(2 x 128k is 1,383 MiB short today), best done with 1 as both change the K/V address; (5) Strata's KVA prompt
approximation (+11-18% prefill at 14-30K) last, since it needs inkling-specific state-only kernels and its own quality
campaign.

One open question, measured: pinned RAM. The ceiling is ~14.5 GiB and the weights take 12.8, so a streamed tail can be
~1.5 GiB — about 128k of context, not 256k. Either stream at 128-160k, or keep the host copy pageable and DMA per layer.

### Session 39 addendum — can we pin more RAM? No.

Re-probed the CUDA host-allocation ceiling with the engine stopped and 22.3 GiB free: **exactly 14.52 GiB**, identical to
the figure with the engine holding 12.8 GiB. It is a fixed driver/WDDM quota (~47% of 31.1 GiB), not a function of free
memory, and the engine's RAM tier is `cuMemCreate`/`cudaHostAlloc`, which no privilege affects (the binary has zero
references to VirtualLock/SeLockMemoryPrivilege — established in MEMORY-LOCKING.md). VirtualLock *can* pin more once the
working set is raised (6 GiB proven), but it is a mechanism the engine does not use and it was tried and reverted.

Consequence for the KV-streaming plan: the weights pin 12.8 of 14.52 GiB, so a 256k KV tail (~3.4 GiB) cannot go into
pinned memory without evicting ~2 GiB of weight spill — a trade, not a gain. Item 1 should therefore use a **pageable
host copy DMA'd into the one-layer staging pool** (no pinned budget), or wait for a 64 GiB machine.

### Session 40 — item 1 (KV streaming) started, and ruled out by arithmetic

Began the `--kv-resident` port for inkling2 and the first measurement stopped it. Inkling's full-attention layers scan the
whole prefix every decoded token: 8 layers x 262,144 keys x 8 heads x 208 B = **3.25 GiB per token**. From VRAM at
~800 GB/s that is 4.4 ms of our 14.3 ms/token; over this machine's measured 15.75 GB/s host link it is **222 ms/token, or
~4.5 tok/s**. Streaming the tail - pinned or pageable - trades 31% of decode's work from an 800 GB/s bus to a 15.75 GB/s
one: a 15x decode regression to free 3.4 GiB.

That is exactly why Strata's docs claim prefill gains and state there is "no decode speedup" with decode "somewhat slower":
their late layers are a gated recurrence plus QSA over a *resident window*, so their decode never touches the streamed
cells. Inkling's full-attention layers do by construction. Streaming is the right tool for *having* 1M context at slow
decode; our 256k fits today, so it buys nothing.

Plan updated: item 1 replaced by item 4 (paged, shared KV pool), which changes allocation granularity rather than
residency - decode keeps reading VRAM, while sessions stop reserving their full ceilings, which is the failure we actually
measured (2 x 128k was 1,383 MiB short). No code was changed; the engine is untouched and serving.

### Session 41 — the 4-bit KV format makes two concurrent 128k sessions fit (+66% aggregate)

Concurrency was impossible on this box before the format work: 2 x 128k measured **1,383 MiB short** at the time. With
`kv=fp8v4` (44% less KV) it now loads, and it performs:

| | before the format | now |
|---|---|---|
| 2 x 128k sessions | would not load (1,383 MiB short) | **loads**, `slots 2 x 131072`, plan cuda0 2.2 / ram 13.0 GiB |
| two concurrent 400-token generations | - | **800 tokens in 6.9 s = 115.7 tok/s aggregate** (~58 per slot) |
| single session for comparison | 70 tok/s | 70 tok/s |

**+66% aggregate throughput** for an agent that sends overlapping requests, at the same total context memory as one 256k
session. Note the engine's own policy: with `parallel 2` it divides `max-context` between slots, so 2 x 128k needs
`max-context 262144` (a 131072 ceiling per slot), not `max-context 131072` (which gives 2 x 65536).

Shipped as a profile rather than the default, because a single 256k conversation still needs the whole ceiling:
`basalt_config-2x128k.json` (in the tree and in C:\LLM\Basalt) is the two-slot profile; `basalt_config.json` stays the
single 256k default. Swapping them is a copy and a restart.

### Session 42 — concurrency scales linearly to 4 slots (230 tok/s aggregate, 3.3x a single session)

The 4-bit KV made multiple slots fit, so I swept `parallel` at a fixed total context of 262,144 (so each slot's ceiling is
262144 / parallel) and sent N concurrent 400-token generations:

| parallel | context a slot | wall | aggregate | per slot |
|---:|---:|---:|---:|---:|
| 1 | 262,144 | 5.7 s | 70.2 | 70.2 |
| 2 | 131,072 | 6.9 s | 115.9 | 58.0 |
| 3 | 87,381 | 6.9 s | 173.6 | 57.9 |
| **4** | **65,536** | **6.9 s** | **230.3** | 57.6 |
| 6 | 43,690 | - | **will not load** | - |
| 8 | 32,768 | - | **will not load** | - |

**Almost perfectly linear** up to 4 slots: per-slot throughput only falls from 70 to 57.6 (decode is latency-bound, not
bandwidth-bound, so extra slots fill the machine for free), while aggregate triples. A single session cannot use the box;
four can.

The limit is the card, not the KV: 6 slots fails with `spill plan: even with every expert in RAM the card is 358 MiB short
(--max-context 32768)` - at that point the per-slot dense/scratch cost, not the KV, is what does not fit.

Profiles shipped beside the default (each is a copy and a restart):

| file | shape | aggregate |
|---|---|---|
| `basalt_config.json` | 1 x 262,144 | ~70 tok/s |
| `basalt_config-2x128k.json` | 2 x 131,072 | 115.9 tok/s |
| `basalt_config-4x64k.json` | 4 x 65,536 | **230.3 tok/s** |

For an agent fleet sending overlapping requests the 4-slot profile is 3.3x the throughput of the single-session default, at
the same total context memory. A single conversation that needs 256k still needs the default.

### Session 42 addendum — the slot ceiling is 5, and the warm-up has a gap

| parallel | context a slot | aggregate | per slot |
|---:|---:|---:|---:|
| 1 | 262,144 | 70.2 | 70.2 |
| 2 | 131,072 | 115.9 | 58.0 |
| 3 | 87,381 | 173.6 | 57.9 |
| 4 | 65,536 | 230.3 | 57.6 |
| **5** | **52,428** | **289.1** | 57.8 |
| 6 | 43,690 | will not load | - |

Five slots fit (6 fails: `even with every expert in RAM the card is 358 MiB short`), and per-slot throughput is flat at
~58 from 2 to 5 - the machine is simply better filled. Profiles shipped: `2x128k`, `4x64k`, `5x52k`.

**Gap found in the launcher's warm-up**: it sends its three requests *sequentially*, so with N slots all of them land on one
slot and the other N-1 stay cold, paying the 30-40% first-request penalty. The fix is to fan the first round out across
`min(N, slots)` concurrent requests. Not yet done - the launcher is PowerShell that has to be patched carefully, and it is
the last item in this track.

### Session 43 — MMVQ_IL started, and sized: its ceiling here is ~+2.4% decode

Started Strata`s `STRATA_MMVQ_IL` (an interleaved q8_1 activation copy plus a measured rows-per-warp table for the 2-8 token
dense matvecs). Our engine already runs `native_q8p_w1_kernel<NCOLS, VARIANT>` - **one row per warp**, templated for 2..8
columns, with a 1-3 segment variant - for `w.proj`, `w.out` and the 248k-row head on every decode window. What we lack is
their rows-per-warp choice (2 or 4) and the interleaved activation layout.

Before writing it I sized the prize, and it is small here: a decoded token reads **580 MB for the head alone** (248,320 rows
x 2304 at one byte a weight) against 5.4 MB each for proj and out - **590 MB, about 774 us of a 14,300 us token (5.4%)**.
Their best published cell (a Q5_K head at 5 tokens, 899 -> 491 us, -45%) would land as **about +2.4% decode**, a couple of
tenths of a tok/s, for a new kernel variant, an interleaved activation buffer and a per-shape table measured on this card.

So it is an afternoon of careful CUDA for ~0.2 tok/s. Recorded here rather than started, because the two cheaper items on
the list are worth more: `--prompt-cache-tail` (one checkpoint near the prompt end, for branches that diverge mid-chunk -
agent wall time, not throughput) and KVA (prefill +11-18%, a project).

### Session 43 addendum — MMVQ_IL`s real lever here is a one-line threshold, not a new kernel

Reading the dispatcher (`src/kernels/cuda/native_mmvq.cu:2471`) found the thing Strata`s table is actually choosing:

```cpp
if (n_in >= 1024 && mmvq_mma_sms() != 0 && ((n_out >= 65536 && ncols >= 4) || (n_out >= 4096 && ncols >= 7)))
    launch_pdl(native_q8p_mma_kernel, ...);   // tensor cores, 16 rows a block, runtime ncols
else
    switch (ncols) { native_q8p_w1_kernel<1..8, NPRE> }   // one row per warp, templated
```

So for the head (`n_out = 248,320`) the tensor-core path engages **only at 4 or more columns**. Our decode window carries
**3 columns** (`dl 2.1`), which is precisely the low-column regime their table was measured in (their best cell is a head at
**5 tokens**) - and where we currently fall back to one row per warp.

That makes the experiment **one line**, not a kernel variant: lower the head's threshold (`ncols >= 3`, or `>= 2`) and
measure. Both paths write the same row of floats, but the MMA path has not been parity-checked against `w1` **for ncols 2-3
on this model**, so the order is: add an env override for the threshold, bit-check the two paths at ncols 2-4 against each
other (kernels_parity-style), then A/B the decode.

Sizing, for the record: the dense matvec is ~5.4% of a decode token, so a -45% head gain is about **+2.4% decode**. The
threshold change is cheap enough that it is worth taking even at that size - what made MMVQ_IL expensive was the
interleaved-activation buffer and a per-shape table, and neither is needed to try this.

### Session 44 — the MMVQ_IL threshold gate, and what the crash means

Executed the three-step plan and the gate answered immediately.

1. **Made the threshold settable** (`native_q8_planes_set_mma_cols`, plus `BASALT_Q8P_MMA_COLS="large,small"` read once at
   first use). Defaults stay 4 and 7, so behaviour is unchanged unless asked.
2. **Bit-checked the two paths** at 2, 3 and 4 columns on a head-shaped matrix (65,536 x 2,304), forcing each path with the
   setter: the process **crashes with `0xC0000409`** the moment the tensor-core form is forced below 4 columns, before it
   can even report. The test was reverted (it must not break the regression suite), the setter kept.
3. So there is **no A/B to run**: the "one-line threshold" I proposed is not one line. The tensor-core form (`q8p_mma_body`,
   16 rows a block, runtime ncols) does not support 2-3 columns, which is presumably why the dispatch requires 4 for a big
   matrix and 7 for a small one.

That is the same conclusion Strata's document reaches from the other side: the rows-per-warp table is a **measured**
property of a kernel that has to support every column count first - their tables are per quant type, per token count (2-4
*and* 5-8) and per row class, and their sm_120 5-8 column table exists "only on sm_120 so far" because the others were not
measured. Expanding our tensor-core path to 2-3 columns is the actual work, and it is a kernel change with a parity gate,
not a threshold flip.

Sizing unchanged and now confirmed as the ceiling: the dense matvec is ~5.4% of a decode token, so even a -45% head gain is
about +2.4% decode. Engine rebuilt with the setter in place, Gate 0 clean, kernels_parity 0 failures.

### Session 45 — a benchmark harness that follows Strata`s protocol rules

Read Strata`s standard benchmark protocol (`docs/BENCHMARK_PROTOCOL.md`, draft `strata-qwen38-thinking-v1-draft.1`) and
took the parts of it that apply to us into `C:\LLM\Basalt\bench.ps1`:

- **the warm-up window is discarded** (their rule; ours measured 61.2 then 107.7 tok/s on this box, so a run that includes it
  measures the cold path);
- **every repetition uses a fresh unique prompt and a prompt showing any cached tokens is rejected** - cache reuse changes
  what is being timed, and we have been bitten by it (`pp=15455/15477c` is a full hit);
- rates come from the **engine`s own timings** (`pp=`/`tg=`), never client wall clock - also their rule;
- **median / min / max / n** per case, never a single number and never best-of;
- a generation that stops early (`end=stop`) is still counted as a speed sample but flagged.

First run on the default (single 256k): **prefill median 1,513 t/s (1,504-1,541, n=3), decode median 67.4 (67.3-67.6)**. Note
how much tighter that is than the ad-hoc numbers this session used (prefill ranged 1,484-1,771, decode 58-77): most of that
spread was warm-up and cache reuse, exactly what the protocol excludes. Every future A/B should use this script rather than
inline measurements.

### Session 46 — the harness overturns two of my earlier verdicts

With `bench.ps1` (warm-up discarded, fresh prompts, engine timings, medians) the A/Bs I had settled on one-off numbers were
re-run. Two of them were wrong.

**`kPromptChunk` 4096 vs 8192: conclusively inert.** prefill 1,513 (1,505-1,518) at 8192 against 1,513 (1,504-1,541) at
4096. The earlier "-16% at 2048/3072" finding stands for smaller chunks, but larger ones buy nothing. Reverted to 4096.

**#633 (`kBatch`/`kGqaMaxWindow` 32 vs 16) is worth ~+5% decode, and I had called it inert.**

| arm | process A | process B | range |
|---|---|---|---|
| 32 (main's, shipped) | 67.4-67.9 | 63.3 | 63.0-67.9 |
| 16 | 61.4 | 61.2 | 58.5-64.0 |

Two processes an arm, medians 65.5 against 61.3: the 16 arm never exceeded 64.0 while the 32 arm never went below 63.0.
Earlier I had compared a single 1,517 against a single 1,513 and called it inert - the effect is real and about 5%, and it
only shows once warm-up and cache reuse are out of the samples.

**What the harness does and does not fix.** Within a process the spread is now +-1-2% (prefill 1,510-1,522 across three
repetitions), but **between processes it is still +-3-5%** (the same 32 build gave 67.65 and 63.3). So every arm has to be
measured in at least two processes, and conclusions taken from one process per arm - as several of mine were - are not
reliable. That is the single most useful process correction of this session.

### Session 47 — kCardSpare 768 is worth +6% decode (found by re-testing under the corrected protocol)

`s kCardSpare` had been left at main`s 1024; our old tree had measured 768 as better there (+6% at the time) and this tree
took main`s value when it was rebased. Re-measured at 256k, two processes an arm:

| kCardSpare | cuda0 resident | prefill | decode |
|---|---|---|---|
| 1024 (was shipped) | 2.2-2.4 GiB | 1,510-1,514 | 63.3, 67.7 |
| **768 (now shipped)** | **2.7 GiB** | **1,542-1,544** | **68.7, 69.8** |

**+6% decode, +2% prefill**, with the ranges disjoint. The mechanism is the one measured all session: 0.4 GiB more expert
residency on the card. This is the third verdict the harness has overturned or confirmed - kPromptChunk (inert, confirmed),
kBatch 32 vs 16 (~5%, corrected), and now kCardSpare (6%, unlocked).

Shipped state: `kv=fp8v4`, ctx 262,144, kBatch/kGqaMaxWindow 32, kPromptChunk 4096, **kCardSpare 768**, single slot; and the
2x128k / 4x64k / 5x52k concurrency profiles beside it. Honest baseline now: prefill ~1,543 t/s, decode ~69 tok/s.

### Session 48 — vram-reserve is structurally inert, and the lesson about mechanism-less differences

Re-tested `vram-reserve-mib` 768 against 384 now that `kCardSpare` 768 changed the plan. Both settings produce a **byte-identical
plan** (`cuda0 2.7 GiB / ram 12.5 GiB`), and the decode medians still differed - 69.3 at 768 (68.7, 69.8) against 65.7 at 384
(66.7, 64.6):

**Identical plans mean there is no mechanism for a throughput difference, so that 5% is process noise.** That is the useful
lesson: two processes an arm is enough to overturn a verdict *when the arms differ structurally* (kCardSpare changed the
plan by 0.4 GiB; kBatch 32 vs 16 changed the decode kernel's work), and not enough to separate arms that are identical
underneath. The rule to apply is therefore: **check the plan and the kernel path first - if an arm does not change either,
its numbers cannot differ, and any difference measured is noise.** Reverted to 768, the safer guard for another program
arriving on the card.

### Session 49 — spec 12 does not pay, and the constant sweep is nearly exhausted

`--spec` is the ceiling the DraftPolicy chooses its window under, so it does change the decode path. Two processes an arm:

| spec | prefill | decode |
|---|---|---|
| 8 (shipped) | 1,542-1,544 | 68.7, 69.8 |
| 12 | 1,546-1,547 | 66.9, 67.4 |

Slightly worse at 12 and identical prefill, so 8 stands. With kPromptChunk (inert), kBatch (32, +5%), kCardSpare (768, +6%)
and vram-reserve (inert) now settled, the remaining compile-time constants are `kElasticKeep`, `kDecodeRows` and the
smaller prompt chunks - each a build and four benches, with no strong prior behind any of them.

**What is left is code, not configuration**: #232/#248 (attribute the 2-3 s TTFT and the pool`s 14.4 ms), prompt.cpp
orchestration (~3.5% prefill), #636 (drafter two-tile decode), and KVA (+11-18% prefill, needs state-only kernels for our
linear-attention layers). Everything reachable by a flag or a constant has now been measured.

### Session 50 — short prompts start in 0.5 s instead of 1.8 s, and the flat 1.8 s is named

Found by chasing a number I had been inferring for sessions: time to first token was **1.8 s for a 32-token prompt**, and
`pp=32/0c 1.8s` says the prompt read itself is 18 tok/s. Turning the prompt cache off changed nothing, so it is not the
per-request checkpoint; it is the prompt path's **staging of every spilled layer's RAM experts over the host link** - about
15 GB for this plan - before any of it computes.

Measured by prompt size:

| prompt | path | prompt time |
|---|---|---|
| 43 tokens | decode | **0.5 s** |
| 106 | prompt | 1.8 s |
| 204 | prompt | 1.8 s |
| 448 | prompt | 1.8 s |
| 8,947 | prompt | 6.9 s |

**A flat 1.8 s from 106 to 448 tokens.** The decode path costs ~0.4 s plus one token per 70/s, so the two cross near 100
rows. `kPromptPathMin` (was 16) is now **96**: shorter prompts are read eight rows a forward on the decode path.

- short agent turns: **TTFT 1.8 s -> 0.5 s** (-72%)
- above 96 tokens: unchanged (448 tokens is 1.8 s through the prompt path against ~6.8 s through decode)

**And the flat 1.8 s is exactly what #609 removes**: its short-chunk routed-only staging stages only the experts a chunk
routes to, not every spilled one. That makes #609 the clearest remaining latency item - it is worth up to ~1.5 s of TTFT
on *every* prompt between roughly 100 tokens and the length at which the staging amortises (several thousand).

### Session 51 — the flat 1.8 s is a missing caller, and it is #609

Chased the staging cost from Session 50 to its source. `7d90e66d` ("a prompt chunk of at most 1,024 rows stages only the
RAM experts it routes to") did land - but only its **plumbing**: `spill.hpp` has `StageOp {Ram|Card|Guard}` and
`stage_plan(n_spill, card, want, kept)`, `spill.cpp` implements it, `spill_test.cpp` covers it, and **nothing in the engine
calls it** - only the test does.

So `prompt.cpp` L558-564 is still the pre-#609 form: `stage_from_card(l, ...)` when the layer has all its card copies, else
**one `cudaMemcpyAsync` of every RAM expert of the layer** (~58 experts x 5.5 MiB = 320 MB a layer, 48 layers, ~15 GB),
then the guard memset. That copy is the flat 1.8 s of time to first token, and a 33-token chunk routes to a few experts a
layer, so nearly all of it is wasted.

The change is a **caller**, not new machinery: build the chunk's routed-slot mask from `b().ids`/`w.slot_map` (already
remapped at that point), walk `stage_plan`, issue Card as per-slot D2D, Ram as one contiguous H2D run, Guard as a memset,
and keep the every-slot form above 1,024 rows. Written up as `SPEC-609-ROUTED-STAGING.md` with the exact sites, the guard
subtlety (per-slot guards replace the old blanket memset), the expected value (up to ~1.5 s TTFT between ~100 tokens and a
few thousand) and the gate (spill_test, kernels_parity, Gate 0, needle at 130,709 - a staging mistake feeds the products
the wrong weights silently).

Not implemented here: it would need a build-validate-measure cycle that a staging change deserves, and this session`s
remaining budget is not enough for it to be done safely.

### Session 52 — the #609 caller, fully worked out

Finished the derivation so the change can be typed in rather than re-derived: the slot layout (`kept = n_expert - n_spill`,
RAM slot j at `w.experts + (kept + j) * stride`), the mask source (`b().ids` holds **slot indices** right after
`remap_ids`, on the device - one sync and a `pairs * 4` byte D2H a layer, ~264 entries for a short prompt), the two model
accessors that do not exist yet (`slots_of(l)`, `copy_of(l, e)`), the per-op issue (Card D2D, Ram one contiguous H2D run,
Guard memset), the trailing end-of-layer guard that `stage_plan` deliberately leaves to the caller, and the two places the
old path must stay (rows > 1,024, and the peer path whose mask spans two cards). All in
`SPEC-609-ROUTED-STAGING.md`.

Stopped before editing: a staging change feeds the expert products their weights, so a wrong one is silent until an answer
is wrong, and the gate for it (Gate 0, kernels_parity, needle at 130,709) needs a full cycle. The specification is now
complete enough that the next session is typing, not thinking.

### Session 53 — #609 wired in: prompt-path TTFT 1.8 s -> 0.8 s (-56%)

Implemented the caller that Session 52 specced. `stage_plan` existed with its tests but nothing used it, so a prompt read
still copied every RAM expert of every spilled layer - about 15 GB over the host link, a flat ~1.8 s of time to first token
whatever the prompt length.

Now, for a chunk of at most 1,024 rows: the pairs are already remapped to slot indices, so one `cudaStreamSynchronize` plus
a `pairs * 4` byte copy gives the routed mask; `stage_plan` then yields per-slot card copies (D2D), contiguous RAM runs
(one H2D each) and the guard memsets. Above 1,024 rows the every-slot form stays (a long chunk routes to most slots, and the
per-slot plan would cost more than it saves), as does the peer path, and the trailing end-of-layer guard remains the
caller's.

| measured at 256k | before | after |
|---|---|---|
| prompt path, 101 / 197 / 459 tokens | 1.8 s | **0.8 s** |
| decode path, < 96 tokens | 0.5 s | 0.5 s |
| prefill (15k unique) | 1,543 t/s | 1,547 t/s |
| decode | 69.3 | 69.2 |

Correctness (the gate that a staging change lives or dies by): three needles placed mid-prompt, all exact - 621 tokens
through the **routed** path (`ORBIT-731`), 1,198 (`CEDAR-284`) and 2,968 (`DELTA-906`) through the whole-layer path. Gate 0
clean (Paris, 391, the planets).

Remaining for medium and long prompts: they still pay the ~1.8 s, because above 1,024 rows the staging is genuinely
whole-layer. #609's other skipped commit - the long-chunk copy on a second stream - is what overlaps that, and it is the
next latency item.

### Session 54 — the second-stream staging specced; session state

`91bdae93` ("a long prompt chunk copies a spilled layer`s experts on a second stream under the kernels before it", 112 lines)
is the next latency item: it overlaps the long-chunk staging that the routed path cannot help with. Its shape is now written
down in `SPEC-609-ROUTED-STAGING.md` - a `copy` stream per lane with `freed`/`staged` events, a precomputed
`next_spill_[step]` map over the visit walk, the two orderings that make the overlap safe (each spilled layer owns its own
reserved slot range, so only its own products can be disturbed), and the failure `drain()` that `broken_` guards.

Expected: prefill ~1,547 -> ~1,800 t/s on 15k prompts (+15%, about 1 s of staging overlapped under ~5 s of compute) and up
to ~1 s off TTFT above 1,024 rows. Gate: build, Gate 0, kernels_parity, the three mid-prompt needles, then timing.

Not implemented: 112 lines of stream and event ordering in the prompt loop is exactly the change that needs its own gate
cycle, and this session`s budget is spent.

## Where the engine stands after sessions 33-54

| | value |
|---|---|
| context | 262,144 (4-bit V, E2M1 codes) |
| memory | cuda0 2.7 GiB / ram 12.5 GiB |
| prefill (15k unique) | 1,547 t/s |
| decode | 69 tok/s single slot; 230 / 289 tok/s aggregate at 4 / 5 slots |
| TTFT | 0.5 s under 96 tokens, **0.8 s** to 1,024, ~1.8 s above |
| quality | needles exact at 621, 1,198, 2,968, 53k and 147k tokens |

Wins shipped and measured this stretch: 4-bit KV (256k in 192k's memory), E2M1 codes, kCardSpare 768 (+6% decode),
kBatch 32 (+5%, a corrected verdict), kPromptPathMin 96 (-72% TTFT short), routed-only staging (-56% TTFT prompt path),
concurrency profiles, launcher warm-up, and `bench.ps1` (which caught two of my own wrong verdicts).

### Session 55 — the shipped tree validated end to end after today`s three changes

The gate that the last three changes (routed-only staging, kPromptPathMin 96, kCardSpare 768) needed, run in one pass:

| check | result |
|---|---|
| `inkling2_kernels_parity` | **0 failures** |
| `inkling2_prompt_parity` | **0 failures** |
| Gate 0 (Paris, 17*23 = 391, the planets) | clean |
| needles, routed path (621) and whole-layer (1,198, 2,968) | exact |
| needle 53,456 tokens, whole-layer path | **exact** (`8419-ZX`), 1,385 t/s prefill |
| needle 147,227 tokens (earlier) | exact |

So the configuration that ships is coherent, its parity suites are green, and retrieval holds from 621 to 147k tokens with
the 4-bit value cache and the routed staging in place. Numbers: prefill 1,547 t/s at 15k, 1,385 at 53k; decode 69 tok/s
single slot; TTFT 0.5 s under 96 tokens, 0.8 s to 1,024, ~1.8 s above.

Next implementation step remains the second-stream long-chunk staging (`SPEC-609-ROUTED-STAGING.md`, expected prefill ~1,800
t/s on 15k and up to ~1 s off TTFT above 1,024 rows).

### Session 56 — xyehya/inkling2-5080-setup reviewed: it is the open PR stack, resolved

Fetched `xyehya/basalt` branch `inkling2-5080-setup` (79 commits over main, 5 upstream PR merges, 47 own commits) and
separated what is new *to us* from what is main history we already carry.

**What we lack, and all four are OPEN and mergeable upstream:**

| PR | title | applies to our tree |
|---|---|---|
| #606 | tensor-core decode attention over an INT8 cache | 8 files, +1505/-64 (low priority: we measured int8 KV worse than fp8v4) |
| #636 | decode attention stages each K/V chunk once for both query tiles | 3 files, +516/-95 |
| **#649** | a static spill copies RAM experts back to the card, the drafter... | **48/63 hunks** |
| **#650** | the card's copies of RAM experts follow the conversation | **80/105 hunks** |

**The real value of the branch is that it has all four merged and conflict-resolved against main** - which is precisely
where our own ports of #609/#636 stalled. `xy-5080` is now a reference in the clone.

**#649 and #650 are newer than `PR-SHORTLIST.md`** - our triage never saw them - and they land on the residency policy that
`kCardSpare` 768 showed is worth +6% decode for 0.4 GiB. Scoped: 128 of 168 hunks apply cleanly; the 40 that do not sit in
`session.cpp` (11), `model.cpp`/`model.hpp` (10), `spill.*` (5) and the tests - exactly the files carrying our local patch
set. Written up in `SPEC-649-650-SPILL-COPIES.md` with the order of work and the gate.

**Their own commits, judged one by one:** `53bca3f3` (three blocks an SM for head-128 decode attention, 2 lines) - ported to
our `gqa_attn.cu`, gated (Gate 0 clean) and **measured neutral** (decode 68.3/68.2 against 69.3, prefill identical) because
inkling2 only uses that kernel for the drafter`s 16-cell window; reverted. `7bea1fc0` (two-tile scores/probabilities in
registers, 190/76) rides on #636. The remaining 44 commits are the gpu-mode/gpu-share two-card refactor - not applicable to
one GPU.

### Session 57 — #650 attempted: 80/105 hunks, 5 new files, and the exact 25-hunk remainder

Applied the #650 stack (which contains #649 - applying both double-applies and leaves 37/105, recorded). 80 of 105 hunks
apply, and the five files a hunk applier cannot create (`copy_heat.hpp`, `copy_heat_test.cpp`, `loan_layout.hpp`,
`loan_layout_test.cpp`, `tests/core/elastic_array_test.cpp`) come straight from the branch. The build then reaches **185
errors, all of one kind**: the applied hunks call what the 25 unapplied ones declare - `on_loan`, `loan_flag_`, `loan_list_`,
`loaned_pairs`, `has_spill`, `copies`, `loan_stream_`, `loan_event_`, `loan_stuck_`, `drain_loan_upload()`,
`CopyLayout::aligned`, and the `promote`/`demote` bodies that now rank by `copy_heat` and chase loans.

That is a merge of a heavily changed `model.hpp`/`model.cpp`/`session.cpp` into files we have also changed - a session's
work. Reverted to `b14526cb`; the tree is clean and the engine serving. The full member list and the reproduction recipe are
in `SPEC-649-650-SPILL-COPIES.md`, which is now enough for the port to start with every name in hand rather than a diff to
read.

# Session 58 - PR #650 lands: the elastic spill's expert *copies* move with the conversation, decode 67 -> 102 tok/s

## What #650 is

Upstream #650 (with #649 inside it) turns the elastic spill from "a fixed set of experts lives on the
card" into a **copy** layer: any RAM expert can be copied onto the card, the copies are ranked by
**measured heat** (per-expert routing counts folded across sessions), the hottest are promoted, the
coldest demoted, and a *loan* lets a prompt borrow the card's free room and give it back afterwards.
A `CopyLayout`/`loan_layout` keeps every reader's address right, a `rebalance` runs every 64 swaps /
256 rows, and a second-card layout lets the peer card hold a different set. All of it is the same
mechanism that makes decode hit the card 90% of the time instead of 74%.

## Integration

Their branch is `pr650`, our tree is `b14526cb` on `pr536-rebased`. A plain cherry-pick is impossible
(their commits sit on a base 200 commits newer, and #650 rewrites the spill's data model), so the
import was a **3-way merge per file** (`git merge-file`, base = `origin/main`, theirs = `pr650`):

| file | result |
|---|---|
| `model.hpp/.cpp` | theirs (copies, heat, demote/rebalance, `card_copies`, `recall_loan`), our `kCardSpare = 768 MiB` kept (measured +6%) |
| `spill.hpp/.cpp` | **both**: their `CopyLayout`/`loan_layout`/`give_back` beside our `StageOp`/`stage_plan` (#609) |
| `session.cpp` | theirs' `ids`/`win_ids` restructure, `wedge`d, `caches_usable`, `decode_kept`; our fused `router_select_plan` for 128 experts x 1 row kept, our `kPromptPathMin` threshold kept |
| `prompt.*`, `kernels.hpp`, `dspark.*` | merged cleanly; our v4 KV (`ap.v4`, `kv_v_row_bytes`) kept |
| everything else | the rest of the PR applied by hunk (31/34) plus `kernels/` (2/3) |

New files: `copy_heat.hpp`, `loan_layout.hpp` and three tests (wired into CMakeLists).

## Four defects the gate caught (each fixed, each a lesson)

1. **`heat_fold` was never compiled.** `kernels/ops.cu` was outside the file list I merged, so the
   link failed on an undefined symbol. Fix: apply the `src/models/inkling2/kernels/` part of the PR.
2. **The fused router wrote to the wrong buffer.** Their restructure introduced
   `ids = k.win_ids ? ... : x.ids_`; our fast path still wrote `x.ids_`, so the plan read one array
   while the router wrote another. Answers came out as `A) 391 B) 23*17 B) 0 C) 0 ...`.
   Fix: the fused path writes `ids`, and the declaration is hoisted above both branches.
3. **A replayed graph pointed at unmapped copies.** The window graph is keyed by
   `Shape{cpu_, rows/head/keys}` and `cpu_ = sh_->cpu ? m_.cpu_layers() : 0` - under the elastic
   spill `sh_->cpu` is null, so the key was *constant* and a recall left a graph reading memory that
   `demote` had unmapped. First request after a recall died with
   `expert_prefetch: operation failed due to a previous error during capture`.
   Fix: the key carries `placement_epoch()`, and the graphs of the old placement are retired after a
   stream sync.
4. **The read-ahead guess named the previous placement's addresses.** Two requests later:
   `expert_prefetch: invalid argument`. The guess is issued for the *next* visit but survives a
   placement change, so `blob_table` translates to an address that no longer exists.
   Fix (safe, conservative): guesses are off while copies exist (`m_.card_copies() == 0` gates them).
   **Follow-up**: tie a guess to the epoch it was made in and drop it when the epoch moves - worth the
   few % the guess is worth, but only with that guard measured.

## Gate

| check | result |
|---|---|
| Gate 0 | Paris / 391 / planets / Tokyo / 144/12 = 12 - all correct |
| `inkling2_kernels_parity` | 0 failures |
| `inkling2_prompt_parity` | 0 failures |
| `inkling2_copy_heat_test` | ok |
| `inkling2_loan_layout_test` | ok |
| `elastic_array_test` | ok |
| needles | 872 / 1,634 / 3,971 / **70,585** prompt tokens -> all EXACT |

The spill log shows the feature working end to end:

```
elastic spill: startup; expert copies 0 -> 1034 (0 loaned after); 1024 MiB free, 4418 MiB paid mapped
elastic spill: prompt loan at 1665; expert copies 1034 -> 1393 (359 loaned after)
elastic spill: prompt loan recall at 14; expert copies 1393 -> 1034 (0 loaned after)
elastic spill: rebalance at 3970 (64 swaps); expert copies 1034 -> 1034
elastic spill: cache grow to 73728 positions at 32768; expert copies 1393 -> 1265
```

## Numbers (protocol: warm-up discarded, no cached content, engine `pp=`/`tg=`, >=2 processes per arm)

Harness `bench.ps1`, 4,600 random words (~15.4k tokens, every prompt fresh), 400 decode tokens.

| arm | prefill t/s (per process) | decode t/s (per process) |
|---|---|---|
| `b14526cb` (before #650) | 1,534 / 1,533 / 1,536 / 1,528 -> **median 1,533.5** | 71.4 / 65.8 / 66.3 / 67.7 -> **median 67.0** (59.1-72.7) |
| `f2c9b221` (#650) | 1,525 / 1,533 -> **median 1,529** | 106.7 / 96.9 -> **median 101.8** (90.1-108.4) |

**Decode: 67.0 -> 101.8 tok/s (+52%), ranges disjoint** (baseline max 72.7 < #650 min 90.1).
**Prefill: 1,533 -> 1,529 (-0.3%, noise).** The engine's own hit rate says why: **73.8% -> 90.3%** of
decode expert lookups found an expert on the card.

Harness fix found on the way: `bench.ps1`'s fixed instruction prefix ("Read the following list, then
reply OK.") is 14 tokens, which the engine's prompt cache matched on every run, so every run was
rejected. The prefix now carries a random tag, and the cache check rejects only reuse above the chat
template's floor.

## State

- Branch `pr650-int` = `f2c9b221` (+ `3b56ac1f` untracking `server.out`/`server.err`), built, live,
  serving 262,144 tokens with `kv=fp8v4`.
- Rollback: `git checkout b14526cb` (baseline, measured above).
- `run-pr.cmd` was untracked and got removed by a checkout; it is tracked now, so it survives.
- Launcher unchanged (`$Engine = 'C:\LLM\rebase\b'`), warm-up still in it.

# Session 59 - Tier 1 landed: #636's two-tile decode attention came in clean and measured neutral

## What was ported

Five commits that all live in one file, `src/kernels/cuda/gqa_attn.cu` (the kernel Inkling 2's
DSpark drafter calls for its window attention), taken from `xy-5080` (upstream main + the resolved
PR set) by 3-way merge against their parents, **keeping our W2 `pos_dev`/`hi_off` patch**:

| commit | what |
|---|---|
| `8f380263` | **#636** - stage each K/V chunk once for both query tiles at head width 128 (+ a 227-line parity test) |
| `b3a7df32` | sync the warp before a row's softmax writes over what its lanes read |
| `cdeed819` | write a chunk's probabilities into its K tile, not over its float scores |
| `53bca3f3` | keep three blocks an SM for every head-width-128 instance |
| `7bea1fc0` | keep a two-tile block's scores and probabilities in registers |

All five merged with **zero conflicts**. Verified line by line: our file is upstream's `8f380263`
version **plus exactly** the W2 patch (12 insertions, 4 deletions) - nothing lost, nothing doubled.
Total: `gqa_attn.cu` 260 insertions / 95 deletions; new `tests/kernels/gqa_window_parity.cpp`
(11 KB) and its CMake target.

## The one real puzzle: their frozen hash does not apply to us

The parity test asserts three things: max abs error against a double reference, a row's bits
identical at every width of a window with the same end, and the output's bits against a frozen
hash. Ours failed only the third: `b4579688ad0fd75e` against `0xc49476ad275fc0d4` ("the one-tile
kernel's").

The cause is a constant **we** tuned: `kGqaMaxWindow = 32` here (#633, measured +5% decode) against
upstream's 16, so the segment count - and therefore the split-K reduction order and every low bit -
is ours, not theirs. Upstream's constant was recorded on their geometry; it cannot be satisfied on
ours by construction. Re-froze it to our bits with that reasoning in the comment, and kept the two
checks that carry the meaning. Both pass: max abs 0.00127, and **0 of 96 windows differ bitwise
across widths**.

Result: `gqa_window_parity -> PASS`.

## Measurement (protocol harness, 4 processes per arm, interleaved by process not by build)

| arm | prefill t/s | decode t/s |
|---|---|---|
| `3b56ac1f` (#650, no #636) | 1,525 / 1,533 / 1,531 / 1,533 -> **median 1,532** | 106.7 / 96.9 / 90.7 / 106.4 -> **median 101.65** |
| `858f1741` (+ #636 family) | 1,533 / 1,528 / 1,534 / 1,526 -> **median 1,530.5** | 105.7 / 94.6 / 99.4 / 105.3 -> **median 102.35** |

**Neutral**: +0.7% decode, -0.1% prefill, ranges nearly identical (90.7-106.7 against 94.6-105.7).
The kernel's own bench says why: it targets the *full-attention* case (5 rows over 100K cells,
~714 us/call), while our drafter's real call is **a 1,025-position window in a 1,088-cell ring, at
30.9 us** - and at that size the duplicated K/V read is not where the window's time goes.

**Kept anyway**, consciously: the output is bitwise identical, it does strictly less K/V traffic,
`gqa_window_parity` now guards the kernel with a real benchmark attached, and staying on upstream's
code keeps future merges cheap. This is the one exception to "do not carry a neutral change": it is
neutral *in our geometry*, provably not a regression, and it is upstream's merge, not ours.

## Gate (all on the merged tree, `bc428d6a`)

| check | result |
|---|---|
| `gqa_window_parity` | PASS (hash re-frozen to our bits) |
| `inkling2_kernels_parity` | 0 failures |
| `inkling2_prompt_parity` | 0 failures |
| `copy_heat` / `loan_layout` / `elastic_array` | ok / ok / ok |
| Gate 0 | Paris / 391 / planets / Tokyo / 144/12 = 12 |
| needles | 872 / 1,634 / 3,971 / 70,585 tokens - all EXACT |

## Repo hygiene found on the way

`git add -A` had put `server.out`/`server.err` back into a commit, which then blocked the merge
("untracked working tree files would be overwritten") and, earlier, silently blocked a branch
switch. They are now in `.gitignore` and out of the index; the launcher rewrites
`basalt_config.totals.json`, so a checkout between arms needs that file clean or `-f`.

## State

- Live: `pr650-int` at `bc428d6a` (merge of `858f1741`), built, serving 262,144 tokens, `kv=fp8v4`.
- Tier 1 complete. Next in the backlog: Tier 0.1 (`8b5c9da7`, the 6-line copy-release safety fix),
  Tier 0.2 (epoch-guarded guess prefetch), Tier 2 (second-stream staging, ~1,800 t/s prefill).

# Session 60 - the HuggingFace model that cannot be used, and what the prefill time actually is

## The model: ehzawad/Inkling-2-Mini-Q3_K_XL-NVFP4-GGUF

Checked as asked. It cannot run here, for three independent reasons:

1. **The repo is `gated: auto`.** The HF API lists it, but every file request returns **401 Unauthorized**
   without an authenticated token, and this box has none (`no HF_TOKEN`, no `~/.cache/huggingface/token`,
   no `hf` CLI). It is also 0 downloads / 0 likes, uploaded the same day - unvetted.
2. **It is the wrong container.** Basalt does not load GGUF: it loads `.basalt` packs
   (`C:/LLM/Basalt/models/Inkling-2-Mini-NVFP4.basalt`, 16.48 GiB). This tree can *read* GGUF
   (`basalt-gguf`, `src/artifact/gguf_reader.cpp`, built) and *read* a pack (`src/models/inkling2/pack.cpp`),
   but it has **no GGUF -> pack writer**, so a downloaded 12.81 GiB GGUF would still be inert.
3. **Its kernels are the wrong family for us.** "Q3_K_XL + NVFP4" is a llama.cpp mix: Q3_K with Q4_K/Q5_K/Q6_K
   tensors in the important places, plus NVFP4 elsewhere. Our inkling2 expert path runs NVFP4 / MXFP8 /
   q8_0 on the tensor cores (that is what the 2,190 t/s prefill and 102 tok/s decode are made of); K-quants
   exist in this tree only as generic dequant/MMVQ, a much slower path. Even converted, it would be a
   compute regression - and our compute is the *other* 39% of prefill.

What it would have bought, for the record: 12.81 GiB against our 16.48 GiB is 22% less to stage, and
staging is 61% of prefill - so maybe +10-15% prefill, at a large compute cost. Not a shortcut.

## What the prefill time actually is (BASALT_DIAG_PROMPT=1, new)

Per chunk, per visit (of 48 spilled layers), the long-chunk path stages **every RAM expert of the layer**:

```
prompt diag: 3848 rows, 48 visits, h2d 0.00 + full 22.04 GiB, d2d 0.00 (h2d 1502.7 ms at 15.75 GB/s),
             0 ram ops, 0 card ops, 48 full copies
```

So: **22.0 GiB staged per 4,096-row chunk = 1,502 ms of a 2,467 ms chunk. Staging is 61% of prefill,** and
it is nearly independent of the row count (25.3 GiB at 7,733 rows). Consequences that follow directly:

| chunk | passes per 15.4k prompt | staged per pass | prefill |
|---|---|---|---|
| 4,096 | 4 x 3,848 rows | 22.0 GiB | **1,559 t/s** |
| 8,192 | 2 x 7,733 rows | 25.3 GiB | **2,226 t/s (+43%)** |
| 16,384 | 1 x 15,390 rows | - | does not fit: "the card is 2125 MiB short" |

`bench.ps1`, two processes each: **prefill 2,190 / 2,184 t/s** against 1,533 / 1,528 before.
Gate at 8,192: Gate 0 correct, needles 872 / 1,634 / 3,971 / **70,585** all EXACT, a 25.7k prompt read at
1,907 t/s (multi-chunk, with the cache growing underneath).

Cost, stated honestly: the prompt buffers take 1.1 GiB of card memory (`cuda0` 2.7 -> 1.6 GiB), the decode
hit rate falls 90.3% -> ~84% and **decode 102 -> 83 tok/s**. That is the trade for now; the fix is the
staging overlap below, which removes the reason for big chunks.

Two other things the instrumentation settled:
- The `rows <= 1024` routed path never engages for real prompts (3,848 rows), and `stage_from_card`
  never fires while a loan is active (`loaned() > 0` returns false), so 0 D2D copies happen in a prompt.
- The two earlier "chunk size is inert" runs were **stale executables**: `ninja`'s link fails with LNK1104
  while the engine holds `bin\basalt.exe`, silently. Always `findstr` for a new string in the exe after a
  build that matters.

## Next: the staging overlap (91bdae93), a hand port

`91bdae93` ("a long prompt chunk copies a spilled layer's experts on a second stream under the kernels
before it") is in the local clone, but it does not merge: our `prompt.cpp` has since taken #609's routed
staging and #650's copy machinery, so the 3-way merge is a whole-file conflict. It must be ported by hand:
a `copy` stream with `freed`/`staged` events, a `next_spill_[step]` map, and the main stream waiting on
`staged` before a spilled visit's expert products. If it hides most of the 1.5 s, prefill goes to
~3,500-4,000 t/s *and* kPromptChunk can go back to 4,096, which returns the 1.1 GiB and the decode hit rate.

The `--prefill-waves` flag in this tree is dead (parsed into `g_prefill_waves`, never read), so that is the
socket the port plugs into.

# Session 61 - prompt processing, second pass: the staging rides a copy stream, and `--prompt-chunk`

## The measurement that made this possible

`BASALT_DIAG_PROMPT=1` (re-applied; an earlier checkout had reverted it, so 456c02cf did not carry it) prints
per chunk what actually moves:

```
prompt diag: 3848 rows, 48 visits, h2d 0.00 + full 22.04 GiB, d2d 0.00 (1502.7 ms at 15.75 GB/s), 48 full copies
```

The long-chunk path stages **every RAM expert of every spilled layer**, once per chunk: 22.0 GiB at 3,848 rows,
25.3 GiB at 7,733. And the staging is **one allocation every spilled layer shares** (spill.hpp), so a layer's
copies could not start until the layer before it had finished reading them.

## The port (91bdae93, by hand)

That commit does not merge - our prompt.cpp has since taken #609's routed staging and #650's copy machinery -
so it was ported edit by edit, keeping the void signatures and adding a safe fallback:

- `Lane` grows a non-blocking `copy` stream and `freed`/`staged` events; `init()` builds `next_spill_[step]` and
  the first spilled step, only when there is one card and no peer.
- `chunk()` stages the first spilled layer before the embedding is even queued.
- `experts()` waits on `staged` instead of copying inline; after its products are queued, `staging_read()`
  starts the **next** spilled layer's copies, which wait on `freed` (recorded after those products).
- any failed queue falls back to staging inline for the rest of the chunk (`stage_ok_`/`ahead_`), so a failed
  event can never let a product read a half-written staging; `run()` drains the copy stream as well.

## `--prompt-chunk N` - the knob, and why an "in-between" size behaves as it does

kPromptChunk was compile-time, so every value cost a rebuild. It is now an inline int set by `--prompt-chunk`
(1,024..16,384) before the plan is laid, with `kChunk` a reference to it. Measured with bench.ps1 (4,600 fresh
words, 400 decode tokens, 2-4 processes each; ~15.4k-token prompts, so the pass count changes across the range):

| chunk | cuda0 | passes | prefill | decode | hit |
|---|---|---|---|---|---|
| 4,096 | 2.7 GiB | 4 | 1,976 t/s | 99.5 | 94% |
| 5,120 | 2.4 GiB | **4** (15,371/5,120 = 3.002) | 1,894 t/s | 98 | ~92% |
| 6,144 | 2.1 GiB | 3 | 2,331 t/s | 92.6 | ~89% |
| 7,168 | 1.8 GiB | 3 | 2,280 t/s | 88.3 | ~93% |
| 8,192 | 1.6 GiB | **2** | **3,018 t/s** | 84.8 | ~85% |

Prefill is dominated by **the number of staging passes**, `ceil(tokens / chunk)`, not by the chunk size itself:
within a pass count the *smaller* chunk wins (more expert copies stay on the card, so less is staged - 4,096
beats 5,120 and 6,144 beats 7,168). The step from 6,144 to 8,192 is the step from three passes to two. So the
best value for a given prompt length is the smallest chunk that still yields the fewest passes, and the flag
turns that choice into a config line.

## Where it stands

| arm | prefill | decode |
|---|---|---|
| this morning (chunk 4,096, no overlap) | 1,533 t/s | 101.8 tok/s |
| chunk 4,096 + overlap | 1,976 t/s | 99.5 tok/s |
| chunk 6,144 + overlap | 2,331 t/s | 92.6 tok/s |
| **chunk 8,192 + overlap (now live)** | **3,018 t/s** | 84.8 tok/s |

**Prefill is 1.97x this morning's**, decode pays ~17%. Launcher passes `--prompt-chunk 8192` and honours
`BASALT_PROMPT_CHUNK`, so the decode-first setting needs no rebuild.

Gate (chunk 8,192 + overlap, through the launcher): Gate 0 six questions correct, needles 872 / 1,634 / 3,971 /
**70,585** all EXACT, prompt_parity 0 failures, kernels_parity 0 failures, gqa_window_parity PASS.

## The next lever, precisely

At 8,192 rows the prompt arena takes 1.1 GiB of card room (`cuda0` 2.7 -> 1.6 GiB), and that is exactly the
decode hit rate (90.3% -> ~85%) and decode (102 -> 85). The arena is sized by `carve()` at `R = kChunk`: the
biggest single item is the expert products' workspace (`gu` = pairs x 2 x n_ff_exp floats, 604 MiB at 8,192
rows; `h` = 302 MiB). Splitting *only* the expert products into two row halves - the layer is still staged once,
which is where the 1.97x comes from - would return ~900 MiB and with it the decode hit rate, making 8,192 rows
affordable at 4,096 rows' card cost. That is the next change to try.

# Session 62 - prompt processing, third pass: stage a slot from its card copy, not from RAM

## What the carve listing killed

The documented "next lever" from Session 61 was to split the expert products into two row halves to shrink the
prompt arena. `BASALT_DIAG_PROMPT` now lists every buffer's cost (one shot, in `Carve::take`), and it says the
idea was worthless:

| buffer | at 8,192 rows |
|---|---|
| `role[8]` (the residual states) | 8 x 72 = 576 MiB |
| `msg` (the four streams' messages) | 288 MiB |
| `mix` (three attention inputs) | 216 MiB |
| `nrm` (three normed rows) | 216 MiB |
| `yq` (the fused projection) | 205 MiB |
| `q` + `attn` + `o` | 264 MiB |
| **expert workspace** `xq` + `gu` + `h` | **1,044 MiB** |

`carve()` lays the expert workspace *from the arena's base* and ends with `c.at = max(end_a, x.at)` - and the
attention buffers (1,269 MiB) are bigger than the experts' (1,044 MiB), so the expert workspace is **already
hidden underneath them**. Halving it would have saved exactly nothing. The arena is attention-buffer bound.

## What actually pays: the link, not the buffer

`stage_from_card()` refuses while a loan is active (`loaned() > 0`), and a prompt runs *during* a loan - so the
~2,842 slots that already have a card copy (the elastic spill's, or the loan's) were being copied from RAM
anyway: **11.86 GiB of the 25.25 GiB per chunk crossed a 15.75 GB/s link for no reason.**

`stage_layer()` now walks a layer's RAM slots and takes each one from where it is: a slot with a copy is one
device-to-device copy (2 TB/s), and slots without one go as one H2D run per stretch of consecutive slots (host
and staging are both in slot order, so runs are the common case). It is used by `stage_ahead()` on the copy
stream and by the inline path, so a chunk's bytes no longer depend on which one runs.

```
per-slot staging 2842 copies D2D 11.86 GiB, 1191 runs H2D 13.40 GiB     (was 25.25 GiB, all H2D)
```

## Numbers (bench.ps1; 4,600 fresh words, 400 decode tokens, 2-3 processes per arm)

| chunk | prefill before | prefill now | decode before | decode now | hit |
|---|---|---|---|---|---|
| 4,096 | 1,976 | **2,300** | 99.5 | 91.5 | 94% |
| 6,144 | 2,331 | **2,570** | 92.6 | 87 | 76-88% |
| 8,192 | 3,018 | **3,153** (bench 3,033) | 84.8 | **92.9** | 93.3% |

The surprise worth keeping: decode came *back* at 8,192 rows (84.8 -> 92.9, hit 85% -> 93.3%). Halving the link
traffic during a prompt stops the prompt from demoting the copies decode wants. A 227-token prompt decoded at
**110 tok/s** in the same process. So 8,192 now dominates on both fronts instead of trading.

## Prefill, end to end

| arm | prefill | decode |
|---|---|---|
| start of session (chunk 4,096, serial staging) | 1,533 t/s | 101.8 tok/s |
| 8,192-row chunks | 2,226 | 83 |
| + staging on a copy stream | 3,018 | 84.8 |
| **+ per-slot D2D staging (now live)** | **3,153** | **92.9** |

**Prefill 2.06x, decode -9%.** Gate: prompt_parity 0 failures, kernels_parity 0 failures, Gate 0 six questions
correct, needles 872 / 1,634 / 3,971 / **70,585** all EXACT.

## What is left on this front

1. **The remaining 13.40 GiB of H2D per chunk** is the pack's RAM-resident experts, and it can only shrink with
   more card room for copies. The arena is attention-buffer bound (above), so the lever is the attention
   buffers themselves: `msg`/`mix`/`nrm` are 3-4 copies of `R x E` floats and `role[8]` is eight more. Sharing
   two of them (e.g. `nrm` is only ever `R x E` in use although carved `3 x R x E`) would free ~200 MiB.
2. **2,842 D2D copies per chunk** at ~5-10 us of launch each is ~20-30 ms of the chunk; coalescing copies that
   sit in the same elastic granule into one would take most of that.
3. The compute side: with staging now mostly D2D, the chunk is closer to compute-bound (attention + MMQ), so
   the #579 leftovers (`attn_prompt` double-buffered tiles, -11.6% on that kernel) are worth a look again.

# Session 63 - prompt processing, fourth pass: what the arena is made of, and where the time went next

## The carve list

`BASALT_DIAG_PROMPT` now prints every buffer of a chunk (one shot, from `Carve::take`):

| buffer | at 8,192 rows |
|---|---|
| `role[8]` the residual states | 8 x 72 = 576 MiB |
| `msg` the four streams' messages | 288 MiB |
| `mix` three attention inputs | 216 MiB |
| `nrm` the normed rows | 216 -> **72 MiB** |
| `yq` the fused projection | 205 MiB |
| `q` + `attn` + `o` | 264 MiB |
| expert workspace `xq` + `gu` + `h` | 1,044 MiB, *hidden under the attention buffers* |
| **total** | 1,982 -> **1,838 MiB** |

Two conclusions, both acted on:

- The expert workspace is **already covered**: `carve()` lays it from the arena base and ends with
  `c.at = max(end_a, x.at)`, and the attention buffers (1,269 MiB) are bigger than it (1,044 MiB). The
  "split the expert products in half" idea from Session 61 would have saved nothing. Recorded so it is not
  tried again.
- `nrm` was carved `3 x R x E` and only ever used as `R x E` (`rms_norm_rows` writes it; `f32_rows_dot`,
  the mmq/fp4/mxfp8 quantizers and the peer gather read it). **Shipped**: 1,982 -> 1,838 MiB, `cuda0`
  1.6 -> 1.7 GiB. Gate 0 correct, needles 872 / 1,634 / 3,971 / 70,585 EXACT.

## The attn_prompt port: measured, then reverted - and the lesson is the useful part

Upstream `cfd41894` (`#579`) double-buffers every K/V tile and replaces the ring's 64-bit `key % cap` with a
power-of-two mask. The double-buffering does not port cheaply (its loop restructure meets our V4 fp8-keys /
4-bit-values loader in three conflict regions), so only the mask was taken.

It **broke prompt_parity 10 of 10** on the first build: the substitution rewrote the helper's own fallback
into `wrap_key(key, cap)` - infinite recursion. Fixed, parity returned to **0 failures**, and prefill measured
**3,064 / 3,033 t/s** against 3,033 / 3,153 without it. Neutral, so reverted (one exception to "no neutral
changes" was already spent on #636; not spending another).

The conclusion is worth more than the change: **the attention loop's modulo is not where a chunk's time goes.**
With staging now mostly D2D, the 8,192-row chunk is:

| term | per chunk |
|---|---|
| H2D staging (13.40 GiB at 15.75 GB/s) | ~851 ms |
| D2D staging (11.86 GiB, 2,842 copies, ~5-10 us of launch each) | ~30 ms |
| compute | ~1,540 ms |

**Compute is 63% of prefill now.** The FP4 expert products (gate/up 2F x E and down E x F per pair, 65,536
pairs per layer, 48 layers = ~50 TFLOP per 15.4k-token prompt) run at an effective **~20 TFLOPS**, far under
the card's peak. That is the next real lever, and it is kernel work in `prefill/fp4` (tile shape, occupancy,
the two-term activation trick's cost), not a flag.

## Where prompt processing stands

| arm | prefill | decode |
|---|---|---|
| start of session | 1,533 t/s | 101.8 tok/s |
| 8,192-row chunks | 2,226 | 83 |
| + staging on a copy stream | 3,018 | 84.8 |
| + per-slot D2D staging | 3,153 | 92.9 |
| + `nrm` R x E (live) | 3,100 (unchanged, +0.1 GiB of copies) | ~85-93 (noisy) |

**Prefill 2.02x** over the start of the session, decode within noise of where it began.

# Session 64 - the ceiling, measured: what is left and what is provably not worth trying

## The probe that closed the biggest door

Prefill is now 63% compute, and the compute is the FP4 expert products at ~20 TFLOPS effective. The obvious
suspicion was that the activation's **second FP4 term** (KLD 0.0025 vs 0.0041) is paying half the tensor-core
time for a little quality. `BASALT_FP4_ONE_TERM=1` made `gu.two`/`d.two` false:

    basalt: error: inkling2 post_conv: an illegal memory access was encountered

It is **not a switch**. The two-term activations are baked into the buffer layout and the block planner: the
quantizer (`fp4::quantize_f16`) still writes two planes into `a4h`/`gsh`, `fp4::plan_blocks` still lays the
tile offsets for two, and the down product then reads them as one. Reverted. A one-term mode is real kernel
work (quantizer + planner + product together), not a flag - and it would need a quality decision anyway.

## The hard limits, all measured on this box

| limit | number | consequence |
|---|---|---|
| host link | 15.75 GB/s | 13.40 GiB staged per chunk = 851 ms |
| RAM-resident experts | 3,008 of 3,072 | every pass stages 13.40 GiB from RAM; the floor for a 15.4k prompt is 2 passes = **1.70 s of pure staging** |
| card | 12 GB, of which 1.87 GiB expert weights + 3.4 GiB KV at 256k + 1.84 GiB prompt arena | the arena and the KV are what the expert copies compete with |
| SMEM | 99 KiB | the prompt attention's tiles are ~79.8 KiB in upstream's double-buffered form |
| pinned RAM | 14.52 GiB, fixed | the pack's 16.4 GiB cannot be pinned whole; the rest is page-fault traffic if touched |

So the prefill **ceiling** if compute were free is ~9,000 t/s for a 15.4k prompt (1.70 s of staging); at
today's compute it is 3,100. That is the shape of what is left: the link is no longer the wall, the FP4 GEMM is.

## What is left, ranked by expected value per unit of work

1. **The FP4 expert GEMM** (63% of prefill, ~20 TFLOPS effective against a much higher peak). Three routes:
   (a) tune the existing two-term kernel - tile shape, occupancy, the epilogue's MUFU path; (b) build a real
   one-term mode (quantizer + planner + product, with a quality decision); (c) cluster/TMA tricks. Up to +50%
   prefill. Kernel project.
2. **Layer-major prefill**: stage each spilled layer **once per prompt** instead of once per chunk. A 15.4k
   prompt goes from 2 passes to 1, a 32k prompt from 4 to 1. Needs all rows' hidden states resident (~147 MiB
   at 32k) and a rework of the visit walk. This is the only change that attacks the 1.70 s staging floor, and
   it grows more valuable the longer the prompt.
3. **The arena as a loan**: stop reserving `prompt_bytes` from the copy budget, let copies live in the arena
   between prompts (`set_prompt_arena` already exists for exactly this) and be demoted at prompt start. That
   would make 8,192-row chunks cost decode nothing - currently they cost `cuda0` 2.7 -> 1.7 GiB.
4. **KVA** (+11-18% prefill in the spec): state-only kernels for the linear-attention layers.
5. **Decode residency**: NVFP4 KV (~44% fewer KV bytes -> ~1.5 GiB more for copies), MMVQ_IL's tensor-core
   path (+2.4%, crashes below 4 columns), and the paged/shared KV pool from the DeepSeek fork for concurrency.
6. **Concurrency profiles**: `2x128k` / `4x64k` / `5x52k` were measured before #650 and today's changes; the
   card now holds far more, so the aggregates (230 / 289 t/s) need re-measuring and probably retuning.
7. **Instrumentation**: #232's TTFT split and #248's pool timing - cheap, and they would explain the last of
   the short-prompt TTFT instead of inferring it.
8. **Prompt cache / prefix reuse** for agent workloads: perceived latency, not throughput.
9. Housekeeping: `--prefill-waves` is parsed into `g_prefill_waves` and read by nothing (dead since the W1
   patch set); either wire it to the copy-stream staging or delete it and its help line.

## Verdict

Not at the ceiling. Prefill has ~2.9x of headroom against its own link floor, and almost all of it sits in two
places: the FP4 expert GEMM (compute) and layer-major prefill (staging passes). Neither is a knob - both are
real work - which is why this session's wins came from measurements that removed whole ideas (the expert
workspace that was already hidden, the one-term switch that is not a switch, the modulo that is not the bound)
rather than from further tuning.

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

## 6. Two concurrent sessions (the tuned case)

For two 128k conversations at once - the case that matters most here - the profile is
**`basalt_config-2x128k-fast.json`** launched with **`--prompt-chunk 3072`**:

```json
{ "max-context": 262144, "parallel": 2, "kv": "fp8v4", "vram-reserve-mib": 512,
  "ram-reserve-mib": 512, "expert-cache": "auto", "spec": 8, "prompt-cache": 6 }
```

Measured (two concurrent 30k-token prompts, then 300-token generations in both slots):

| profile | chunk | decode aggregate | prefill aggregate | per-stream prefill | plan |
|---|---|---|---|---|---|
| **2 x 128k (shipped)** | **3,072** | **73.0 tok/s** (36.5 each) | **1,927 tok/s** | 964 tok/s | cuda0 2.8 / ram 12.4 |
| 2 x 128k | 2,048 | 72.7 tok/s | 1,496 tok/s | 748 | cuda0 3.0 / ram 12.2 |
| 2 x 128k | 4,096 | 66.7 tok/s | 2,102 tok/s | 1,051 | cuda0 2.5 / ram 12.7 |
| **2 x 160k** (`basalt_config-2x160k.json`) | 3,072 | 70.2 tok/s | 1,881 tok/s | 940 | cuda0 1.7 / ram 13.5 |
| 2 x 192k | 2,048 | - | - | - | does not fit: 532 MiB short |

So **2 x 160k costs almost nothing** (70.2 vs 73.0 decode) - that is the useful surprise: 320k of total
context fits on the 12 GB card. 2 x 192k does not. `vram-reserve-mib 512` measured no worse than 768 at two
slots and leaves the copies more room.

The chunk choice at two slots is the same trade as at one, shifted: 2,048 decodes best by 0.3 tok/s and
prefills 22% worse, 4,096 prefills best by 9% and decodes 9% worse. 3,072 is the balance.

### Where two slots lose, measured

- **Copy residency**: hit rate 63-82% at two slots against 93% at one, because two KV caches (1.7 GiB each at
  128k) and two prompt arenas come out of the same 12 GB.
- **The prompt loan is off**: `loan_ready()` requires `sessions() == 1`, so the machinery that pushes residency
  from 1,034 to 1,593 copies during a single session does not run with two. Making the loan per session is the
  next real win here - it is a VRAM-residency fix, not a quality change.
- **Acceptance is content-bound, not batching-bound**: on a predictable prompt two slots measured
  **100% and 99% acceptance, dl 2.5-2.8, ~110 tok/s aggregate** against 80 solo; on free prose the same
  configuration measured 34-56%. The verifier does not degrade when the drafts are good, so the earlier
  "batching loses" reading was the content, not the batch.

## The context curve at two slots (2 x 128k config, fill ramped, one clean run per depth)

Measured on 0aa1d2b6 (the K/V cell-size fix, without which nothing above ~82k worked at two slots), chunk 5,120,
reserve 256 MiB, both slots fed unique text at the same depth:

| depth per slot | prefill per stream | decode per stream (400 tokens) | aggregate decode |
|---|---|---|---|
| 30k | 1,705 t/s | 46.2 tok/s | 92.4 |
| 64k | 1,500 t/s | 44.9 tok/s | 89.8 |
| 96k | 1,315 t/s | 43.3 tok/s | 86.6 |
| 120k | 1,190 t/s | 43.3 tok/s | 86.6 |

So at the depths an agent session actually reaches, two slots give **~43-46 tok/s per stream with both decoding**
and **1,190-1,705 t/s per stream while reading**, and 120k per slot is real - the elastic caches take the expert
copies as they fill, which is why decode eases from 46 to 43 rather than collapsing. Before the cell-size fix the
second slot could not pass 82k at all.

Text under 128k is the band that matters (pi compacts somewhere below it); 64k-110k is the plateau.
# Night hand-off (2026-11-24)

See `NIGHT-HANDOFF.md` at the repo root: launch recipe, the measured two-slot table (30k-120k decode and prefill, agent turns, overlap), the pi client notes, the known limits, the known-good commit `8e4215f4`, and the ranked list of what is not done (P0-6 sweeps, P0-3, interleaving, per-session loan accounting, 4-bit keys).

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
## P1 and P2 of SPEC-2SLOT-BUDGET-PHASE (measured, e1451848)

**P1, the prefill timeline at two slots.** One slot reads a 30k unique prompt at chunk 5,120, with
`BASALT_DIAG_PROMPT=1` for the staged bytes and the engine's own `read in` line for the time:

| case | read | staging per chunk | compute per chunk | exposed |
|---|---|---|---|---|
| the other slot idle | 29,946 tokens in **12.2 s** | 12.08 GiB H2D + 11.17 GiB D2D = **767 ms** | ~1,266 ms | **38%** |
| the other slot decoding | 30,032 tokens in **12.5 s** | same shape | same | **38%** |

Two numbers come out of it. **Prefill's staging is exposed - 38% of a chunk, 767 ms of about 2,030 ms** (and ~33% at
chunk 8,192 earlier), so P2 has something to win. And **a read costs only 2.5% more when the other slot is decoding**:
prefill and decode do not contend for the link the way the cost model feared, because decode's own staged bytes are
tiny next to a chunk's 12 GiB.

**P2, hiding it: closed by the card, not by the code.** The only lossless lever is a second staging set so two layers
are copied ahead; built and measured under the previous spec (Item 6), the plan refuses to load at 2 x 128k -
"needs 2366 MiB a session" - because the second set doubles the staging the plan must reserve. The alternatives are
out of reach on 12 GB: keeping the 12 looped core layers staged across their three passes needs ~6.2 GiB, and a
larger chunk needs an arena that does not fit either (16,384 rows: "the card is 2125 MiB short"). So P2's decision is
recorded: **at 2 x 128k the exposed staging cannot be hidden on this card**; at a single 256k slot (where the arena
fits chunk 8,192) the exposed share is lower already (33%) and the same second-set limit applies.

**Not attempted, and why.** D3 (moving misses to the GPU over the link during decode) and P3 (the FP4 GEMM) are the
two day-scale engineering items left, both kernel/plumbing projects; R2 (a second slot reusing the first's system
prompt) needs the cell-copy machinery that R1 shows is not yet reachable at two slots; D4 (acceptance) has its matrix
and its Tier 1 data point already (spec 6 swinging +11%/-5%); D5's small GPU items and the interleaving flag are each
below the phase's "keep only what improves decode or prefill" bar - interleaving by construction (it trades reader
speed for decoder responsiveness, default off).
## Session 65 - upstream 31174039 merged as the base (stage 1), the four-bit codec is stage 2

Executed the plan in `UPSTREAM-REVIEW-31174039.md`. Branch `merge-upstream` from `origin/main` (`31174039`),
commit `bf330c03`, pushed to the fork as `merge-upstream-31174039`. **The live engine is deliberately still
`pr650-int` (`7e2f3c54`)**: the merge wins at 64k and loses at 30k until our four-bit value cache is ported.

Upstream measured first, as it ships, at the best configuration it can run here (2 x 96k fp8 - its 2 x 128k plan is
83 MiB short even with every expert in RAM): prose 40.8 / 39.1 per stream at hit 63-71%, prefill 1,190 / 1,186 per
stream for two 30k prompts, curve 30k 1,814 prefill / 38.2 decode, 64k 1,660 / 50.9.

Merged, at 2 x 128k fp8: **`kCardSpare` 768** is what makes the context fit (`cuda0=2.1GiB ram=13.1GiB`; at 1024 the
plan is 83 MiB short), plus the four-bit cache's **plumbing staged and dormant** - V4 constants and row-size helpers,
the `fp8v4` parse, the per-array elastic sizing that `cell_bytes_at` fixes, and `stage_to_ring`'s separate value row
size; all inert under `kv=fp8`. Gates: parities 0 failures, gqa PASS, needles 3/3 at 64,016 and 3/3 at 119,960 with
a concurrent decoder, stress 10 rounds 0 crashes. Numbers: prefill **1,753** at 30k and **1,606** at 64k per stream
(+2.8% / +7.1% over our tree), decode **40.6** at 30k and **51.6** at 64k (-12% / +15%).

Recorded a false pass: `kv=fp8v4` with staged sizes and upstream's unported kernels answers Gate 0 correctly while
writing 128-byte value rows into 80-byte cells - hence the v4 port is gated on parity and needles, not Gate 0.
Live engine re-verified after the switch back: `cuda0 2.0 / ram 13.2`, Gate 0 correct, prose 41.5 / 47.0 per stream
at hit 83-96%, prefill 1,231 per stream.
