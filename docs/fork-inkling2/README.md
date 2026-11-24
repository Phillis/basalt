# Inkling-2 fork notes

Everything measured on this fork, kept with the code so the reasoning travels with it. Started from
`jesdga95/basalt` and this fork's engine tree; hardware is one **RTX 5070 Ti Laptop (12 GB, sm_120)**,
**Ryzen AI 9 HX 370** (24 threads, AVX-512), 31 GiB RAM with a fixed ~14.52 GiB pinned ceiling, on Windows 11.

Read them in this order.

## The record

| file | what it is |
|---|---|
| `CHANGES.md` | the session log, 1 to 64: every change, every measurement, every negative result |
| `CEILING-ASSESSMENT.md` | what is left, what it is worth, and what is provably not worth trying |
| `REBASE-STATE.md` | the live tree, the commit, the constants, and the traps |
| `PR-SHORTLIST.md` | which upstream PRs were worth taking and which were not, with reasons |

## Prompt processing (prefill), this fork's main work

| file | what it is |
|---|---|
| `PREFILL-CHUNK-RESULTS.md` | where prefill time actually goes (61% staging), and the chunk-size step function |
| `PREFILL-OVERLAP-RESULTS.md` | staging on a copy stream (`91bdae93`, hand ported): 1,533 -> 3,018 t/s |
| `PREFILL-D2D-RESULTS.md` | per-slot D2D staging: 13.40 GiB H2D + 11.86 GiB D2D instead of 25.25 GiB H2D |
| `PREFILL-ARENA-RESULTS.md` | the arena's contents, and the compute wall that was left |
| `PREFILL-PERF-SPEC.md`, `HANDOVER-PREFILL.md`, `SPEC-609-ROUTED-STAGING.md` | the specs behind those changes |

## Decode and memory

| file | what it is |
|---|---|
| `PERFORMANCE-RESULTS.md` | the decode/throughput record across the session |
| `V4-KV-DESIGN.md` | the four-bit value cache (`kv=fp8v4`) that made 262,144 tokens fit |
| `NVFP4-KV-ASSESSMENT.md` | whether an NVFP4 cache is worth building |
| `MEMORY-LOCKING.md` | the pinned-RAM ceiling and what it allows |
| `DECODE-PERF-SPEC*.md`, `NEXT-WORK-SPEC.md`, `NEXT-STEPS-SPEC.md` | the decode specs and their measured budgets |
| `W1-*.md`, `W2-W1-PASS-SUMMARY.md` | the wave-pipelining and Windows-side investigations |
| `IMPLEMENTATION-PLAN.md`, `UPSTREAM-TECHNIQUES.md` | what Strata and the DeepSeek-V4.1-Flash vLLM fork had that we could use |
| `D1-Q2-SHADOW-*.md`, `IMPLEMENTATION-NOTES.md`, `PR-TRIAGE.md`, `SUMMARY.md` | closed experiments and the measurement discipline |

## Tools

`tools/bench.ps1` is the measurement harness every number here came from: unique random words per run,
warm-up discarded, prompts with cached content rejected, engine `pp=`/`tg=` as the truth, median/min/max/n,
and at least two processes per arm. `tools/Start-Basalt.ps1` is the launcher (warm-up before `READY`,
`--prompt-chunk` from `BASALT_PROMPT_CHUNK`).

## The numbers this fork is about

| | before | after |
|---|---|---|
| decode, 15.4k prompt | 67.0 tok/s | 101.8 tok/s (+52%, PR #650's expert copies) |
| decode hit rate | 73.8% | 90.3% |
| prefill | 1,533 t/s | 3,153 t/s (2.06x) |
| context | 192k (fp8 KV) | 262,144 (four-bit V) |

Correctness on every step: Gate 0 coherence, `inkling2_kernels_parity` and `inkling2_prompt_parity` at
0 failures, and needles exact at 872 / 1,634 / 3,971 / 70,585 prompt tokens.
