# Is the NVFP4 K/V cache viable for Inkling-2-Mini? — assessment

Short answer: **the format is the best untapped memory lever we have, but it is kernel work, not a flag.** And one
correction first, because it changes what "from Qwen3.8-Flash-Next" refers to.

## What that run actually used

`docs/community/runs/qwen3.8-flash-next/02e31cdf2536.json` (#639, the tip of main) records:

```
"quant": "NVFP4",  "kv": "int8",  "engine_args": "--gpu 0 --max-context 66661 --kv int8"
```

So the **NVFP4 is the weights, not the cache** — and our pack is already NVFP4 weights. Its K/V cache is **int8**, which
we tested on our engine and it is *worse for us*: prefill 1,553 (vs 1,615), decode 66.6 (vs 77.8), hit 56–64% (vs 67%),
because int8 rows cost more bytes per cell than fp8 (below).

The separate thing that does exist is **`--kv nvfp4` for the qwen35 arch** (`src/models/qwen35/adapter.cpp:687`), i.e. a
4-bit K/V cache in the dense stack. That is the idea worth assessing, and it is not what #639's numbers came from.

## The numbers, from `include/basalt/models/inkling2/kernels.hpp`

Inkling-2-Mini: `kHeadDim = 128`, `kKvHeads = 8`. One K+V cell per token per attention layer:

| format | row bytes | K+V cell | vs fp8 | at 192k (≈6.0 GiB fp8) |
|---|---|---|---|---|
| fp16 | 256 | 4,096 B | 2.0× | 12.0 GiB |
| **fp8 (ours)** | **128** | **2,048 B** | 1.00× | **6.0 GiB** |
| int8 | 144 (128 + 4 scales + 12 pad) | 2,304 B | 1.125× | 6.75 GiB |
| **nvfp4** | 72 (64 E2M1 codes + 8 E4M3 scales) | **1,152 B** | **0.5625×** | **3.4 GiB** |

**−44% KV memory ⇒ ~2.6 GiB back for expert copies.** At ~5.5 MiB per expert blob in our pack that is roughly **480 more
experts resident**. Residency is worth real throughput here: going 192k → 128k (far more residency) measured **+5.9%
decode**, and our hit rate is 67%. This is plausibly a **+8–15% decode** change, and it would also let 224k–256k fit again
at today's VRAM.

## Why it cannot just be switched on for inkling2

Three concrete blockers, in order of size:

1. **`include/basalt/kernels/kv_fp4.cuh` is written for 256-wide heads.** Its first line: "normalized **256-point**
   Hadamard transform the keys are turned by". That is also why `src/dense/model.cpp:527` refuses non-256 heads:
   `if (o.kv_fp4 && (g.head_dim != 256 || !o.kv_int8 || o.kv_resident > 0))`. Inkling2 is 128-wide, so the rotation
   butterfly needs a 128-point variant.
2. **inkling2 does not share the dense attention kernels at all.** It has its own `attn.cu` / `attn_prompt.cu` written as
   `template <bool FP8, bool INT8>` with `kv_cache_row_bytes(fp8, int8)` in `kernels.hpp`. A third arm — `FP4` — has to be
   added to the decode kernel (including #626's `attn_mma_kernel`), the prompt kernel, and the codec/scale plumbing.
3. **The adapter refuses it**: `src/models/inkling2/adapter.cpp:63` allows only `fp16`, `fp8`, `int8`.

The codec itself is width-agnostic (`kv_code_bytes(values, fp4) = values / 2`, `kv_scale_bytes = values / 16`), so the
format costs nothing for 128-wide heads — only the Hadamard and the kernels do.

## A smaller, safer first step: 4-bit V only (`k8v4` analogue)

`--kv k8v4` (dense, INT8 K + rotated Q4_0 V, 816 B/cell vs int8's 1,056) is the lower-risk shape of the same idea: the
**keys stay exact**, so needle retrieval and attention scores are untouched, and only V — which is averaged, not matched —
is compressed. For inkling2 the analogue is fp8 K + 4-bit rotated V = 1,024 + 576 = **1,600 B/cell (78% of fp8, ~1.3 GiB
freed)**. It needs the same kernel arms but no K-side rotation and no K-precision question, so the quality gate is a
perplexity check rather than a needle hunt.

## Verdict and order

Viable, and the largest single remaining lever — larger than **#636** (drafter two-tile window, decode compute) and
comparable to the 9% prefill gap. But it is a multi-hour CUDA job with a quality gate, so it should come **after** the
cheap wins we have already isolated:

1. **prompt.cpp orchestration port** — the missing 9% prefill, no quality risk (~1,615 → ~1,770).
2. **#636** — decode compute, upstream-tested, moderate merge risk.
3. **nvfp4 (`k8v4`-first) K/V for inkling2** — ~44% KV memory (or ~22% for V-only), the biggest lever, needs the 128-point
   Hadamard and a third template arm in three kernels, gated on KLD/perplexity.
4. #579's two skipped commits, #232/#248 instrumentation.
