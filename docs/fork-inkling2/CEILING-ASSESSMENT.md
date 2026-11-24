# What is left, and what is provably not worth trying
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
