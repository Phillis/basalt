# Prefill performance â€” handover

State of the prompt-processing work on **Inkling 2 Mini under Basalt** (fork at `C:\LLM\pr536\basalt-inkling-cpu-spill-review`).
Written to be handed to whoever picks this up next: what landed, what the measurements actually say, what to do next, and how
to reproduce every number.

Context you need before reading on: decode was taken from 11.3 â†’ ~44â€“67 tok/s in earlier sessions (see `SUMMARY.md`,
`CHANGES.md`). **Prefill is the unfinished half.** Current: **~1,456 tok/s** on a unique 53k-token prompt. Target: **â‰¥2,000**.

---

## 1. Current state

| | |
|---|---|
| Engine | `C:\LLM\pr536\basalt-inkling-cpu-spill-review`, build with `C:\LLM\rebuild.bat` |
| Launch | `C:\LLM\Basalt\Start-Basalt.cmd` (copies `C:\LLM\Basalt\basalt_config.json` into the tree, then launches **directly** â€” the scheduled task breaks device selection with this build) |
| Config | 196,608 context, `kv fp8`, `spec 8` (DSpark), `expert-cache auto`, `vram-reserve 768`, `ram-reserve 512`, `host 0.0.0.0` |
| Decode | ~44â€“67 tok/s depending on content (draft acceptance varies) |
| Prefill | **1,456 t/s** unique 53k prompt, warm Â· ~1.5k fresh Â· first chunks after a decode burst are slower (staging re-take) |
| Hardware | RTX 5070 Ti Laptop 12 GB (11.9 GiB usable), Ryzen AI 9 HX 370, 31 GB RAM, PCIe 4.0 **x8** = 15.75 GB/s measured |
| Model | 18.38 GiB NVFP4 pack, 3,072 experts, 24 layers / 48 visits per token, ~2,738 experts spilled to pinned RAM |

---

## 2. What landed this round (P2)

`InklingModel::promote()` (~`src/models/inkling2/model.cpp:517`) decided which RAM-resident experts get an elastic card copy
by walking `spill_order_` â€” the *routing* order from the profile. But a prompt chunk stages a spilled layer's experts on
**every visit of that layer**, and the loop layers (6â€“17) are visited **3Ã—** per chunk while prelude/coda layers are visited
once. An elastic copy turns that staging into an in-VRAM **D2D** instead of an **H2D** across the host link.

So promotion is now ranked by **staging frequency** (`promote_order_`, added to `model.hpp`; built lazily on first
`promote()` from `geo().visits`, `std::stable_sort` by visits-per-layer so ties keep the routing order). `demote()` still
uses `spill_order_` (decode heat).

| | before | after |
|---|---|---|
| Unique 53k prompt, warm | 1,422 t/s | **1,456 t/s (+2.4%)** |
| Decode canary, 300-token runs | 39â€“67 tok/s | 39.5 / 47.7 / 59.0 â€” unchanged |
| `hit=` / `dl` | 72% / 1.9 | 68â€“90% / 1.6â€“1.7 â€” unchanged |

**+2.4%, not the projected ~25%.** Â§3 below is why.

---

## 3. The key finding: KV growth evicts the elastic copies during the very prompts prefill targets

The spec's byte table assumed ~728 elastic copies stay on the card while a chunk runs, so ~26% of staging would come for
free (D2D). That is true for a **short** prompt. It is not true for a 16k+ **unique** prompt, because the prompt must grow
the KV cache to hold itself, and every growth demotes copies back to RAM. The engine log states it outright:

```
elastic spill: the caches grow to 57344 positions; 0.64 GiB of expert copies go back to RAM (466 stay on the card)
staging reclaim: 1.01 GiB released to the expert spill (0.58 GiB copied, 758 on the card)
```

By the later chunks of a long prompt there are far fewer copies than the model assumes, so the D2D saving nearly vanishes.
Any prefill plan that counts on elastic copies must first deal with KV growth (or accept that it only helps short prompts).

---

## 4. The corrected chunk budget â€” and what it does to the wave-pipelining plan

Measured: 53,405 tokens, 36.7 s, 14 chunks of 4,096 â†’ **2.62 s/chunk**.

| Term | The spec's model | **Measured / corrected** |
|---|---|---|
| DMA (H2D staging) | 1.23 s | **~1.33 s** (~21 GiB/chunk at 15.75 GB/s) |
| Compute (dense + attention + expert products) | ~0.8 s | **~1.39 s** |
| Per chunk | 2.03â€“2.5 s | **2.62 s** |

Consequences:

- The **non-DMA part is ~70% larger** than assumed. It is *not* negligible.
- **Wave pipelining (P1 in the spec) therefore tops out near 1.29 s/chunk â‰ˆ 2,400â€“2,700 t/s** â€” a 1.6â€“1.8Ã— gain, not the
  projected 3,150â€“3,700. Still above the 2,000 target, but it does not buy as much as advertised.
- Byte-cutting work (the P2 ranking, or a Q2 shadow) is worth less than advertised for the same reason: the copies it
  optimises are the ones KV growth takes away.

---

## 5. CORRECTION: "loop-major" is not implementable — do not attempt it

> An earlier version of this document recommended processing a chunk pass-major so each loop layer's staging set is built
> once and reused for all three passes. **That is wrong** and has been withdrawn (`NEXT-WORK-SPEC.md` §1 caught it):
>
> - **The walk is already pass-major.** `walk.hpp` runs prelude → [loop 6-17] × 3 passes → coda over all 4,096 rows per
>   visit. The loop is a *cycle*: pass *p+1*'s layer 6 consumes pass *p*'s layer-17 output and the bank messages, so no
>   other order (e.g. a layer's three passes adjacent) is legal.
> - **Reusing the staging across passes needs all 12 loop layers resident at once**: 12 × ~511 MiB ≈ **6.2 GiB**, which does
>   not exist. The 38 MB figure in the draft was the hidden-state size, not the staging.
>
> The ~8–9 GiB/chunk of loop-layer re-staging is real, but the only legal ways to avoid it are to **make the staged bytes
> smaller** (§5a) or to **hide the copy behind compute** (§6). Do those instead.

## 5a. The lever that moves both fronts: a Q2_0 shadow of the spilled experts

Staging the *shadow* blobs instead of the NVFP4 originals cuts H2D per chunk from ~21 GiB to ~7.7 GiB (~1.33 s → ~0.49 s),
and the same shadow cuts the decode pool's bytes (15.4 → ~6-8 ms/window). One artifact, two wins.

**What already exists (verified on this tree):**

| Piece | Where |
|---|---|
| Q2_0 on the CPU pool path | `src/kernels/cpu/native_expert.cpp` handles `GGML_TYPE_Q2_0` (iq256/iq512 paths), `q2_avx2_rows.inc` |
| Q2_0 on the GPU prefill path | `src/prefill/mmq/instances/mmq_q2_0.cu` |
| Q2_0 dequantizer | `include/basalt/artifact/dequant.hpp:69` (`dequantize_q2_0`) |
| Block geometry | **64 elements per 18-byte block**, `GGML_TYPE_Q2_0 = 42` |

**What is missing — this is the work:** there is **no Q2_0 *quantizer* anywhere in the tree** (only the dequantizer). So the
job starts by writing `NVFP4 blob -> dequantised f32 -> Q2_0 block` (18-byte blocks: fp16 scale + 2-bit codes), then:

1. build the shadow at load for the spilled experts of layers 0-19 (~2,738 × 1.65 MiB ≈ **4.4 GiB of pageable RAM** — no
   pinning; re-verify with `hostalloc_seq.exe` that the pinned total is unchanged) and **first-touch all of it at build
   time**, or the first decode after load pages in GiBs and stalls the pool;
2. a `NativeFmt` for Q2_0 blobs for the decode path (`CpuSpill::run()`), flag-gated;
3. prefill staging from the shadow plus the Q2_0 MMQ instance. The per-blob 16-byte `{s_gate, s_up, s_down}` tail the SwiGLU
   form reads is NVFP4-specific — Q2_0 blobs need the equivalent scales stored at the blob's end for the same reader;
4. `--cpu-shadow q2|off` (default **off** — it is a real quality trade: ~27% of routed lookups go from NVFP4 to Q2_0).

Gates: pool busy ≤ 8 ms, decode ≥ 65 on the TCP prompt, unique-16k prefill ≥ 1,900, needle, and side-by-side samples for
quality sign-off.
## 6. If you'd rather do wave pipelining first (the spec's P1)

Split each visit's staging into 2â€“4 **waves** and pipeline copy against compute *within* the visit:

- a dedicated copy stream (`cudaStreamNonBlocking`; precedent: `ra_st_`/`pf_st_` in `session.cpp`) plus one event per wave;
- wave *i*: issue the copy for that slot range on the copy stream, record its event, have the compute stream wait on it and
  launch **wave *i*'s expert products**; the copy stream proceeds to wave *i+1*;
- this needs `moe_sort`'s `bounds`/`tok`/`dst` sliced per wave (contiguous expert range, no re-sort) â€” **this slicing is the
  fiddly part** and the reason I stopped;
- start wave 0 right after the router so it also overlaps the visit's own attention/dense (~5â€“15 ms);
- `stage_from_card` (`model.cpp`, ~L562) needs a stream parameter and a slot range; today it copies expert-by-expert on the
  caller's stream;
- keep it behind a flag (`--prefill-waves 0|N`) â€” inkling2's parser is `dspark_parse_flag` (`dspark.cpp:722`), and note it is
  *not* a config-JSON key;
- data movement only, so it is bit-neutral: any output difference is a slicing bug, not a trade-off.

Ceiling with the corrected budget: **~2,400â€“2,700 t/s**.

---

## 7. Refuted / do not bother

| Idea | Why not |
|---|---|
| **Standing staging floor** (spec P3) to avoid the decodeâ†’prompt re-take | The prompt path stages a **whole layer's** expert set (up to 1,034 MiB for a Q8_0 layer), so any floor below that still forces the re-take. Already fixed differently: Session 6 gates the re-take on `prompt_ && r.n_tokens > 16`, so short prompts never pay it. An **asynchronous** re-take during idle remains valid and unstarted. |
| `--prefill-cpu-share` | **Not parsed by inkling2** (qwen4exp-only; the server logs "this model has no such flag, skipped"). Every earlier sweep of 0.6/0.85/1.0 measured a no-op. There is no CPU share in this model's prefill. |
| `kPromptChunk` 4096 â†’ 8192 | The chunk arena roughly doubles (~+1 GiB VRAM) against ~768 MiB of headroom at 192k. |
| A CPU-prefill-expert path (qwen4exp-style pool assist) | The pool's multi-token kernels cap at nt=8 per expert, so prefill-scale CPU compute is a new kernel project. Only if the PCIe wall must be broken after the above. |
| GPU expert-kernel work | The FP4 products are ~0.3â€“0.5 s/chunk â€” not the wall. |

---

## 8. How to measure (do not skip this)

**Unique text only.** Repetition is exploited by the suffix drafter and the prompt cache and inflates everything. Random
6-character words tokenize densely â€” 87 kB of them â‰ˆ **53k tokens**, not 16k.

```powershell
# kill everything first: a stale python server serves A/Bs with a PREVIOUS engine and silently ruins the result
Get-Process basalt -ErrorAction SilentlyContinue | Stop-Process -Force
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.ExecutablePath -like 'C:\LLM\*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

- **The truth is the `s1 read pp=<new>/<total> <t/s>` line** in `...\basalt-inkling-cpu-spill-review\server.out`.
  Ignore `engine.log`'s "N reused in 0 ms" lines â€” those are conversation-restore stats.
- Warm the engine with a short decode first (that triggers the reclaim/promote cycle) before timing a large prompt.
- After any change, re-check **decode** too (`diag`'s `pool busy` line, or just tok/s): the promote ranking feeds both.
- Regression set: `powershell -File C:\LLM\Basalt\test-long.ps1 -TargetTokens 131072` (must find the needle), a 16k+ unique
  prefill, and a sustained decode run.

**Useful instrumentation that is already in the tree:**

```powershell
# per-window decode budget (wall / pre-expert / post-expert+CPU / pool busy)
cmd /c "set BASALT_DIAG_VISIT=1&&call C:\LLM\pr536\basalt-inkling-cpu-spill-review\run-pr.cmd"
```

Engine log (`~\.cache\basalt\55cfc8c42932d132\engine.log`) lines worth reading: `experts: N of M in RAM â€¦ with Y MiB of
prompt staging`, `staging reclaim: â€¦`, `elastic spill: the caches grow to â€¦`, `CPU spill: â€¦ hand-off doorbell`.

---

## 9. Known measurement/accounting quirks (don't chase them)

- **`hit=` can read over 100%** and climbs monotonically across a session (96.6 â†’ 139.3% over five requests):
  `ex_tally_`/`lookups_` are cumulative while the server divides by a per-request count. `card_share()` is structurally
  â‰¤100%, so the cache is fine. Read the engine's `experts: N of M in RAM` line instead. Fixing it means touching the tally
  that also feeds the expert map â€” do it deliberately, not in passing.
- **Greedy output is not reproducible across processes here** (expert placement differs per start: two runs of the *same*
  build diverged at char 211 of a 220-token reply). So a bitwise A/B across builds is not obtainable â€” use within-process
  determinism plus the needle test, and say so rather than claiming parity you cannot show.
- **Throughput is content-dependent** on decode (draft acceptance 1.5â€“2.0 `dl` â†’ 39â€“67 tok/s). Quote ranges and name the
  prompt.

---

## 10. Pointers

| Path | What |
|---|---|
| `src/models/inkling2/prompt.cpp` | the prompt chunk walk and `experts()` staging â€” **where the loop-major change goes** |
| `src/models/inkling2/walk.hpp` | the 48-visit walk |
| `src/models/inkling2/model.cpp` | `promote`/`demote`/`stage_from_card` (~L517 / ~L581 / ~L562), `kCardSpare`, staging release/acquire (W1) |
| `src/models/inkling2/session.cpp` | staging reclaim hooks, elastic KV grow/trim, the session |
| `src/models/inkling2/dspark.cpp` | drafter (and `dspark_parse_flag` at ~L722 for new flags) |
| `C:\LLM\Basalt\*.md` | `SUMMARY.md` (results), `CHANGES.md` (Sessions 1-7, every change), `IMPLEMENTATION-NOTES.md` (traps), the three specs |
| `C:\LLM\bwtest\` | isolated harnesses: `nvfp4_bench2.exe` (kernel), `hostalloc_seq.exe` (reachable pinned-host total) |

**One standing constraint to respect:** the machine sits ~150 MiB under its reachable pinned-host ceiling (~14.5 GiB, probe
with `hostalloc_seq.exe`). Any change that increases spill or pins more memory at plan time fails the load with a
misleading `cannot select device 0` (a sticky error from a failed pinned allocation). `kCardSpare = 768 MiB` and
`vram-reserve-mib = 768` are tuned to that margin â€” treat both as measured constants, not knobs.
