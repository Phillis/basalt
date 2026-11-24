# PR shortlist — what we could borrow to improve Basalt on this machine

Two scans of the open PR list (`jesdga95/basalt`, 73–79 open): one filtered to inkling2, one by *file path* for code we
actually execute (`src/core`, `src/kernels`, `src/platform`, `src/program/serve`, `serve/`, and the shared mmq). This
supersedes `PR-TRIAGE.md`, whose conclusion ("nothing left in the backlog") was written before the pack swap and the CPU
identification.

**This machine, for context:** AMD Ryzen AI 9 HX 370 — 4 Zen 5 + 8 Zen 5c, 24 threads, AVX-512 on both core types;
RTX 5070 Ti Laptop 12 GB (`sm_120`); 31 GiB RAM with a ~14.5 GiB pinned ceiling. Inkling 2 Mini **base** pack (16.48 GiB,
2772 of 3072 experts in RAM, 300 on card) at **224k context**. Prefill **~1,750 tok/s** (weight-traffic bound), decode
**~54–63 tok/s** (CPU-pool bound), drafter ~7.8 ms/window outside the main window.

---

## Tier 1 — take these

### #636 — drafter window attention reads each K/V chunk twice
`kernels: decode attention stages each K/V chunk once for both query tiles of a block drafter's window` · +366/−59 ·
`src/kernels/cuda/gqa_attn.cu`

`gqa_window_attn` is the kernel Inkling 2's DSpark drafter calls. A drafter block is 8 rows × **3 query heads per KV head**
at head width 128, so it spans **two 16-row tiles**, and the launcher builds one block per tile — so both tiles **read the
same K/V**, and the drafter's full-attention layer reads its whole K/V twice per window. At 100K cells they measured ~490
GB/s of read-once and **1.8× the time of a 5-row window**. The fix: at head width 128 one block takes both tiles of a
segment, staging each K/V chunk once (`n_chunks ≥ 4×splits` gates it; below that the one-tile launch stays).

- **Why us:** we run at **224k**, so the duplicated read is bigger than their case, and the drafter is already the largest
  span outside the main decode window.
- **Risk:** low-to-moderate — new code path in one kernel, one-tile path untouched. We have touched this file (W2's
  `hi_off`, inert).
- **Gate:** Gate 0 coherence, needle, decode ABBA.

### #579 — Blackwell `sm_120` prefill and decode kernels
`perf(inkling2): Blackwell sm_120 prefill & decode kernel optimizations` · +538/−280 · 17 files · claims **+7.3–8.3%
prefill, +1.2–4.1% decode** with zero quality/decode regression

Seven commits, several directly applicable:
- native `MUFU.TANH/EX2/RCP` in the softcapped SwiGLU epilogue (`CALL.REL.NOINC` 384 → 2)
- `attn_prompt`: double-buffered KV tiles + fast cap wrap instead of `p % cap` (**−11.6%** on that kernel, bitwise identical)
- 1-warp `LDG.128` q8 MMVQ for decode `w.proj` / `w.out` / `lm_head`
- **fused `router_select` + `moe_plan` for single-row decode** (7.40 → 4.31 µs, −41.7%, drops 57,600 launches on `--spec 1`)
- decode attention register pressure **236 → 128** (2× occupancy, bitwise identical)

- **Why us:** same model, same architecture; and our decode runs 48 visits/window, each with a router.
- **Caveat:** our prefill is bound by expert *weight traffic*, so their prefill percentage may transfer only partly; the
  attention and launch-count items should.
- **Note:** it touches `attn.cu` like #606 — they say whichever lands first rebases.

### #609 — short chunks stage only their routed RAM experts
`inkling2: prompt staging: short chunks stage only their routed RAM experts, long chunks copy on a second stream` ·
+338/−42 · depends on **#525**

Three ideas in one: **one copy per run** (adjacent RAM slots in a single H2D instead of per expert), **routed-only
staging** for chunks ≤1,024 rows (read the pairs back after the router, stage only those slots, zero the guard), and a
**second stream** for long chunks with the visit-event barrier.

- **Why us:** we stage **548 MiB per spilled visit regardless of chunk length**, so a 20-token agent turn stages ~13 GiB
  across 24 layers to compute 20 rows — that is our **2–3 s short-prompt TTFT**. Their 1,024-row figure (301.85 → 276.95 ms)
  is the conservative end; at 20 rows staging is nearly all of it.
- **Also ours to gain:** the batched "one copy per run" is the item flagged in Sessions 21–22 as the remaining staging
  optimisation.
- **Effort:** depends on #525, which we hand-applied; they publish a rebase branch (`pr/inkling2-525-on-main`). Needs
  hunk-level application against our tree.
- **Interesting overlap:** its second stream is the W1 pipeline I withdrew — they made it safe with the same cross-layer
  event barrier, and gated it to long chunks where staging all experts needs no routing.

## Tier 2 — cheap probes (instrumentation or small fixes, no big win by themselves)

| PR | What | Why run it |
|---|---|---|
| **#605** `core: a room-bound RAM copy leaves room for the adaptive swaps' free places` (+120/−4) | The RAM complement packs to the last byte, so the adaptive tier's spare places are dropped **silently** and the tier refuses every swap forever | Our spill is packed against a ~14.5 GiB ceiling and we have seen `RAM ran out loading`. If our elastic tier is silently dead, this restores expert promotion — and card residency is our decode lever |
| **#232** `serve: log where a request's time to its first token goes` (+72/−5) | One log line splitting TTFT into `wait/match/ram/disk/kv/park/restore/adapt/windows/batched/refill/other` | Measure **our** 2–3 s short-prompt TTFT before committing to #609; tells us whether it is staging, KV fit, or a restore |
| **#248** `pool: --pool-timing` (+347/−6) | Per-batch span, tail after the median thread, task time by core class, per-window park/spin | Tells us where our 14.4 ms pool window goes, and settles the P/E placement question in our #584 with data instead of argument |

**#248 also contains the finding that should drive our #584 reply:** on their hybrid box *"an E-core ends last in 97.5% of
batches (14.5 µs per task against 8.7 on a P-core), but that tail costs at most ~0.2 ms per window, so we dropped a P/E
placement change before building it"* — i.e. the "skip the siblings, P then E-cores" change we were asked for is worth
**<1% of a window**, and the real pool lever is overlapping the CPU and GPU phases (which our doorbell already does).

## Tier 3 — idea sources, not borrowable code

- **#604** (qwen35 `--kv nvfp4`): an NVFP4 K/V at ~0.56 B/value would cut our ~6.9 GiB cache by ~44% ≈ **+3 GiB for card
  experts** — the most direct residency win available *if* upstream generalises it to inkling2.
- **#606 / #552** (int8 KV + tensor-core decode attention): for us a loss — INT8 is 144 B/row against FP8's 128 (+0.86 GiB at
  224k), and attention is not our decode bound.
- **#591** hybrid CPU lists: Linux `cpu_capacity`/PMU path (we use `EfficiencyClass` on Windows) — but the same
  classification question behind our #584.
- **#442 / #589** peer expert cards, **#388** layer splits: we have one CUDA GPU.
- **#503** Q5_1 unpack, **#580** `tool_choice: none` correctness, community rows **#601 / #602**: not our format / not perf /
  data only.
- Our own open PRs: **#584** (needs the sibling-order answer + an ABBA), **#585**, **#586** (waiting on xyehya's commit so
  there is only one hand-off in the tree).

## Order I would take

1. **#636** — highest value per unit of risk: one kernel, the duplication is provably present at 224k.
2. **#579** — the only drop-in for our exact architecture, and it also carries the router fusion.
3. **#232**, then **#609** — measure our TTFT split first, then take the short-chunk staging; it is the biggest single win for
   agent-style use.
4. **#605** — quick probe of whether the elastic tier is silently disabled.

Every one of these needs hunk-level application against our tree (it carries #525, #536 and our local patches) and the standing
protocol: kill both the engine and the `C:\LLM` python server between A/B runs, Gate 0 first (coherence, acc < 100%, needle),
unique text for prefill, and judge by `pp=` / `tg=` rather than `hit=`.
