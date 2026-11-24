# Implementing the decode-performance specs — what I learned

Practical notes from working through `DECODE-PERF-SPEC.md`, `-2.md` and `-3.md` on this machine (Inkling 2 Mini,
Basalt fork, RTX 5070 Ti Laptop 12 GB, Ryzen AI 9 HX 370). Not a changelog — that is `CHANGES.md`. This is the
"what will bite you next time" document.

Headline for context: stock **11.3 tok/s → ~44–67 tok/s** (content dependent), ~1.5k tok/s prefill, 192k window.

---

## 1. Work discipline that actually paid off

**Kill everything between measurements.** A leftover `python.exe` from `C:\LLM` keeps serving requests with a *stale*
engine, so the new build is never exercised and the numbers look unchanged. Before every A/B:

```powershell
Get-Process basalt -ErrorAction SilentlyContinue | Stop-Process -Force
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.ExecutablePath -like 'C:\LLM\*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

This silently ruined two A/Bs before I caught it — the giveaway was a request succeeding instantly after a "restart".

**Change one thing, then measure.** Every real win here came from a single isolated change with a number attached;
every wild goose chase came from changing two things and guessing which mattered.

**Measure the *engine*, not the flags.** Two of the biggest findings were that a feature was silently not running:
PR #536's link gate (CPU path disabled at 15.75 GB/s, CPU sat at 15% during decode) and `--suffix-draft` (never
parsed, so the earlier test measured nothing). Before concluding a knob "does nothing", confirm it engaged — look for
its log line, CPU/GPU utilisation, or the `diag` split.

**Use the in-tree diag harness, and trust its structure over its tok/s.** `BASALT_DIAG_VISIT=1` gives per-window
wall/body/pre-expert/post-expert/pool-busy. The *shape* of that split told me the hand-off was the problem; tok/s alone
would not have. It turned "the GPU is at 31%" into "28.6 ms of a 53 ms window is the hand-off".

**Throughput is content-dependent — quote ranges, not one number.** Same build, same process: 67 tok/s on free-form
prose (`dl` 2.0), 44 on "describe how a siphon works" (`dl` 1.5), 65 on a verbatim-repeat prompt with lookup drafting
(`dl` 3.9). Any single figure is a claim about your prompt, not the build.

**Greedy output is not reproducible across processes here.** Two fresh runs of the *same* build diverged at character
211 of a 220-token reply, because expert placement and the adaptive spill differ per start. So a bitwise A/B between
two builds is unobtainable — validate with within-process determinism plus a retrieval test (`test-long.ps1`), and say
so rather than claiming parity you cannot show.

---

## 2. Engine-specific traps

### Captured graphs bake their arguments — a "wait" must compare at runtime
Inkling 2's decode window is replayed from a `core::CapturedGraph`. My first doorbell port hung the card: it used
`wait_flag_ge(flag, value)` where `value` is fixed at capture, so every replay waited on a stale number while the host
poller's bookkeeping froze at capture time. The fix is `doorbell_wait(done, seq)` — it compares **two mapped words at
run time**, so nothing depends on a captured constant. If you add anything host-synchronised to a captured window,
check this first.

### Never read-modify-write a mapped host counter from the GPU
`doorbell_publish` does `*(volatile uint32_t*)seq = *(volatile uint32_t*)seq + 1`. Over PCIe that loses increments as
soon as publishes are back-to-back, and the host then waits forever. The repo's own comment points at the answer
(`doorbell_publish_value_kernel`: *one store, no read of host memory from the GPU*). I added
`doorbell_publish_store()` / `doorbell_publish_visit()` using `atomicAdd` / a stored value.

### One slot per visit, not a shared ring
A ring slot reused by two visits lets the poller miss a publication (there is no host back-pressure once the graph
replays without host calls). One slot per visit — the visit's index in the window — is stable across replays and
therefore safe to bake into the captured node.

### Plan-time pinned allocations starve the loader
`CpuSpill` is constructed during `plan`, *before* the loader pins its 256 MiB read staging, and the pinned-host quota
is already ~150 MiB from full. Pinning my ring there produced `pinned staging alloc of 16 MiB failed: out of memory`
and `cannot select device 0` at load. Allocate such buffers **lazily, on first use**, never in `open()`.

### The real wall is the pinned-host quota (~14.5 GiB), and it looks like other things
Symptom was `cannot select device 0 for the load` — which reads like a GPU problem and is nothing of the sort. It was
a sticky CUDA error from a failed pinned allocation surfacing at the first `cudaDeviceSynchronize`. Check
`C:\LLM\bwtest\hostalloc_seq.exe` for the machine's reachable total, and watch the `share ram` / `experts: N of M in
RAM` lines. Reducing `kCardSpare` (fewer experts on the card → fewer pinned) is what fixed it — and it also *raised*
decode, because more experts stayed resident.

### Flags that exist but do nothing for this model
`route-prior`, `route-tail-skip` are accepted by the generic parser and never referenced by `src/models/inkling2` — no
effect, and no error. `prefill-cpu-share` looked like it helped for a while; it was run-to-run noise. `--relaxed-drafting`
is only implemented inside qwen4exp's verify path.

### "Sync the stream" is a trap in a windowed decode
`dspark.cpp`'s `draft_end` synchronised the whole stream to read back the drafts, so it waited on the main window's
kernels too — a ~9 ms serial tax between windows. Record an event after the drafter's own work and wait on that
instead. This single change was worth 37 → 31.2 ms per window.

### Custom `k` values are held together by hand
`kCardSpare` 896 → 768 MiB fixed load reliability *and* added decode; 640 MiB collapsed `dl` 2.2 → 1.6 (the drafter
needs free VRAM). The optimum is narrow and specific to this card — data, not a midpoint.

---

## 3. Hardware-specific traps

**Zen 5 executes 512-bit ops as two 256-bit halves.** Register pressure arrives twice as fast as it looks. Widening the
NVFP4 kernel to R=4 rows was *much* worse (0.364 → 0.557 ms/expert) from spills alone.

**A gather is not automatically faster than four L1 hits.** `_mm_i32gather_ps` for the UE4M3 scale bytes was neutral at
best; four scalar table loads plus inserts win at this size.

**A kernel can already be at the machine's limit.** The engine's pool runs at ~45.6 GB/s against the isolated harness's
best 51.9 on an idle machine — ≤12% headroom. Measure the harness before investing days in SIMD.

**Two "not pinned" surprises.** `SeLockMemoryPrivilege` grants the right but does not lift the `VirtualLock` cap (that
is the process's *minimum working set*, raised with `SetProcessWorkingSetSizeEx`) — and this engine doesn't use
`VirtualLock` at all, it uses CUDA host VMM. Hours went into a privilege that turned out inert. `MEMORY-LOCKING.md` has
the full post-mortem.

---

## 4. Windows / tooling traps

| Trap | Symptom | Fix |
|---|---|---|
| Scheduled task context | `cannot select device 0 for the load` (device select fails only under the task) | launch directly in the interactive session; the task is a trap with this build |
| UTF-8 BOM | `--config: not valid JSON: Unexpected UTF-8 BOM` | `[System.IO.File]::WriteAllText(path, s, New-Object System.Text.UTF8Encoding($false))` |
| PowerShell single-quoted `\n` | literal `\n` pasted into C++ source, turning the next statement into a comment (build error on one occasion, silently dead code on another) | use the editor, or `"`n"` in a double-quoted string |
| Killing by command-line pattern | `Command aborted` — my cleanup matched and killed my own shell | match on `ExecutablePath`, never on a command line that contains your own arguments |
| Killing the process tree | the engine dies when the tool call that started it is aborted | start it detached (`Start-Process` / `schtasks`), and re-check it is alive before reporting |
| `--quiet` on `secedit /export` | exit 1, no file, no message | `/quiet` is only valid for `/configure` |

---

## 5. The order I would do it in, next time

1. **Confirm the feature is running.** Log line / utilisation / `diag` split. Two of three big wins were "the feature
   was off".
2. **Get the `diag` budget for the current build.** wall / pre-expert / post-expert / pool-busy per window. Everything
   is decided by which term dominates.
3. **Attack the largest term only.** At the start the hand-off and the CPU pool each looked small until measured
   properly; then the hand-off was 28.6 of 53 ms.
4. **Re-budget after every change** — the dominant term moves. Cutting the hand-off made the drafter's fixed tax the
   bottleneck, which is why drafting "suddenly" became worthless and W-A mattered.
5. **Only then tune constants** (`kCardSpare`, `vram-reserve`, `spec`, pool sizing). They are narrow optima, and every
   one I set by intuition was worse than the measured value.
6. **Validate with the retrieval test, not tok/s**: `test-long.ps1 -TargetTokens 131072` must find the needle.

---

## 6. Quick reference

```powershell
# build                        (MSVC 19.44 + CUDA 13.5, -DBASALT_MODELS=inkling2)
C:\LLM\rebuild.bat

# run                           (canonical config: C:\LLM\Basalt\basalt_config.json)
C:\LLM\Basalt\Start-Basalt.cmd     ; # stop
C:\LLM\Basalt\Stop-Basalt.cmd      ; # live decode/prefill rates
C:\LLM\basalt-monitor.cmd

# with the diag harness on
cmd /c "set BASALT_DIAG_VISIT=1&&call C:\LLM\pr536\basalt-inkling-cpu-spill-review\run-pr.cmd"

# isolated kernel harness (compiles the engine's own nvfp4_avx512.cpp)
C:\LLM\bwtest\nvfp4_bench2.exe
C:\LLM\bwtest\hostalloc_seq.exe    ; the reachable pinned-host total

# long-context correctness
powershell -File C:\LLM\Basalt\test-long.ps1 -TargetTokens 131072
```

| Where | What |
|---|---|
| `C:\LLM\pr536\basalt-inkling-cpu-spill-review` | the engine tree; `server.out`, `server.err` there |
| `~\.cache\basalt\55cfc8c42932d132\engine.log` | engine log: `experts:`, `staging reclaim`, `CPU spill:`, `diag:` |
| `C:\LLM\Basalt\basalt_config.json` | the config to edit (the launcher copies it into the tree) |
| `C:\LLM\Basalt\*.md` | `SUMMARY.md` → results, `CHANGES.md` → every change, specs, `MEMORY-LOCKING.md`, `API-CONNECT.md` |
| `C:\LLM\cpu-spill-hostfunc.on` | marker file: restores the old hand-off for an A/B |
| `*.orig` beside each edited source | the pre-edit file, for reverting a single change |

---

## 7. What is genuinely left

- **W-C**: the Q2_0 shadow of the spilled experts in **pageable** RAM (the spec is right that it needs no pinned
  memory). Halves the pool's bytes, ~1–2 days, and a real quality trade.
- **W-E.1**: the `hit=` figure climbs monotonically (96.6 → 139.3% over five requests) because `ex_tally_`/`lookups_`
  are cumulative while the server divides by a per-request count. Fixing it means touching the tally that also feeds
  the expert map; do it deliberately, not as a side effect.
- **Prefill (~1.5k tok/s)** is host-link bound (staging experts over 15.75 GB/s), not CPU or kernel bound.
- **Hardware**: a 16 GB+ card removes the spill entirely and would put this model well past 100 tok/s.
