# Open PR triage — is there anything left for In­kling-2-Mini? (72 open PRs)

Reviewed the full open-PR list on `jesdga95/basalt` against **this** machine's measured bottlenecks, not against the
titles. The filter was deliberately narrow:

1. **Does the PR touch code inkling2 actually executes?** Our prompt path is `src/models/inkling2/prompt.cpp` +
   `src/prefill/moe_fp4.cu` / `moe_mmq.cu`, and our decode path is `src/kernels/cpu/nvfp4_avx512.cpp` + the pool +
   `src/models/inkling2/{session,dspark}.cpp`. The `src/prefill/*` framework (with `qsa`, `gdn`, `handoff`, `expert_stream`,
   `layer`, `routing`) belongs to the **qwen4exp / deepseek41** families and is not on our path at all.
2. **Does it attack one of the two things that are actually binding?** Measured, not assumed:
   - **Prefill** is 100% expert-GEMM compute (~200 TFLOPS, ~2.6 s/chunk). Staging is free (Session 17), so I/O, staging
     and pipelining PRs are worth zero here.
   - **Decode** is bounded by card residency (334 of 3072 experts) and the CPU expert pool. So anything that frees VRAM
     is worth more than anything that frees I/O.

## Verdict: nothing in the set improves our two bottlenecks

No open PR touches inkling2's expert GEMM, its routing, its drafter, or its KV format in a way that helps a
residency-starved, compute-bound 12 GB setup.

### Applied — #447 `serve: no Nagle on streamed replies, lighter per-token progress check`
Two real, tiny wins on the serve layer we run: `disable_nagle_algorithm = True` (streamed events leave immediately
instead of waiting for a coalesced segment — latency to each streamed chunk, which matters for an agent consuming
tokens), and `_progress()` no longer takes the status lock and copies the whole status dict on every call (that runs
per progress tick, per token). Both hunks applied cleanly; server restarted, answers and streaming verified.

### Considered and rejected, with the reason

| PR | Why not |
|---|---|
| #570 prefill: router four-row path at n_embd 5120 | Our `n_embd` is 2304, which **already** takes the four-row path (`4*3*192`), and inkling2's router uses `f32_rows_dot`, not this kernel. The PR adds the 5120 case. Zero effect. |
| #489 prefill: skip unused pinned expert copies | Staging is **free** here (Session 17: identical throughput with staging removed), and the files are qwen4exp-only. |
| #249 prefill: `--prefill-timing` | Instruments `src/prefill/*` (the qwen/deepseek framework). Our prompt path would need its own timers. Good idea, wrong code. |
| #129 prefill: opt-in `--prefill-fp8` for dense projections | Same framework — not inkling2's projections (ours run through `mmq`). |
| #157 prefill: cuBLASLt algorithm per weight shape | Only FP16/BF16 products; our experts are NVFP4/Q8_0. |
| #445 prefill: two-pass QSA block scores | QSA is qwen4exp's attention/indexer. |
| #572 KV: K4V8, #314 NVFP4 KV (QSA), #355 KV pool | All qwen4exp / deepseek41 conversation and attention formats. |
| **#552 inkling2: `--kv int8`** | **Applicable, but it is a loss for us**: an INT8 row is 144 B vs FP8's 128 B, i.e. +12.5% KV, and we are residency-starved — every GiB of KV saved is ~230 more experts on the card. Its quality is slightly better (KLD 0.00344 vs 0.00443), so it is a *quality* knob, never a speed one. |
| #503 / #500 CPU Q5_1 kernels | Our CPU experts are NVFP4 (and Q8_0 on four layers), not Q5_1. |
| #498 cpu: inline F16C scale conversions | Applies (we are MSVC and have the shim), but our hot CPU loop is `nvfp4_avx512.cpp`, not ggml's dot products — `ggml_cpu_dots.c` is not where our 14.4 ms/window of pool time goes. Cheap to try, expected ~0. |
| #415, #571, #547, #546, #543, #469, #509, #501 | qwen4exp / hy4flash decode or prefill specifics. |
| #13, #392, #311, #183, #481 | Expert quant formats / PLE tables for other model families. |
| #213, #212, #388, #442, #473 | Layer-split / peer-card features; we run one card. |
| #574, #576, #461, #458, #457, #447*, #337, #315, #96, #184 | Serve/admin/robustness (no throughput). |
| **#420 serve: thinking budget no longer cuts the answer** | *Behavioral* fix for exactly the empty-content failure we hit (thinking consumes the budget, no answer text). Only 5/7 hunks fit our older serve tree and its tests are absent, so patching it blind risks the reply path. Worth porting deliberately if empty answers recur — not a performance item. |

`*` applied.

## What this means

The PR backlog is not where our remaining performance is. The two levers identified in Session 17 remain the only ones,
and both are ours to build:

1. **#555's load path** (`cp.async.cg` 16 B, conflict-free `kBRowStride = 80`, fp16 intermediates) — the one known
   artifact worth ~15–20% of prefill compute. Pay for its ~32 MiB of VRAM out of **`kElasticKeep`, not `kCardSpare`**:
   the elastic copies existed to avoid H2D, and H2D is measurably free, so that trade is nearly costless.
2. **Offload a slice of prefill experts to the idle CPU cores** — attacks compute directly, no VRAM needed; inkling2 has
   no prefill CPU split today (only qwen4exp does).

For decode, `#552`'s int8 KV is the only applicable KV work and it moves the wrong direction; the real decode lever is
still residency, i.e. finding VRAM to keep more of the 3072 experts on the card.
