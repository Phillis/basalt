# What we can take from the DeepSeek-V4.1-Flash engine (yhfgyyf/vllm-deepseek-v4-sm89)

Investigated the vLLM fork that runs DeepSeek-V4.1-Flash (MoE + **Engram**, its n-gram table) on consumer GPUs
(SM89 Ada / SM120 Blackwell), plus the `tonyd2wild/…-1M-NVFP4-KV-2x-DGX-Spark` deployment recipe that documents its
measured behaviour. Inkling-2-Mini shares the MoE half and has no Engram, so everything below is judged against our
engine, not theirs.

## Adopted: the cold-start finding (their most transferable result)

Their best-documented finding is not a kernel — it is that **the first requests after a load run 30–40% slow**, that the
penalty decays after a few hundred real tokens, and that it *returns after an idle period* (their 58.5 → 83.3 tok/s;
84 → 60.4 after 30 minutes idle → 83.5 after heavy traffic).

We have the same effect, and it is visible in our own log as a first-vs-second pattern: **61.2 → 107.7**,
**56.3 → 108.3**, **62.7 → 108.7** tok/s.

**Done**: `Start-Basalt.ps1` now runs three warm-up requests (one 320-token generation, one 300-word generation, one
~2k-token prompt for the prompt cache) *before* it prints `READY`, and says `warmed up`. Launcher time went from ~25 s to
40 s; the first real request then measures **67.0 tok/s against the second's 67.7** — the penalty is gone. They also warn
it comes back after idle, so a future option is a trickle of traffic or a re-warm on first request after N minutes idle.

## Already in Basalt, and in a stronger form: adaptive verification

Their speculative config is the headline tuning knob:

```json
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic",
 "rejection_sample_method":"block","enable_adaptive_verification":true}
```

`enable_adaptive_verification` uses DSpark's **confidence head** to decide how many drafts to verify. Basalt already does
this, and does more with it: `include/basalt/spec/draft_policy.hpp` learns the acceptance rate `q(c)` per **confidence
bin**, models the window's expected committed tokens as `E(t) = 1 + q(c₀) + q(c₀)q(c₁) + …`, divides by the *measured*
cost of that window size, and picks the best `E(t)/ms` — and it also arbitrates between the MTP drafter and the suffix
(ngram-lookup) drafter on the same expected-value basis. That is why our `--spec` sweep (4/8/12/16) moved nothing: the
policy is already choosing per round. **No action.**

Their `draft_sample_method: probabilistic` is a *quality/robustness* choice (it fixed a cold-start garble where a greedy
draft diverged from the target); ours is greedy argmax and our output is clean, so this is an option to keep in mind, not
a win to take.

## Ruled out with measurement, not opinion

| their technique | why it does not pay here |
|---|---|
| `cudagraph_mode: FULL_AND_PIECEWISE` (graph the prefill too) | we graph decode windows by shape (`graphs_`, `CapturedGraph`) and leave prefill eager. Prefill is **weight-traffic bound** at ~200 TFLOPS effective: a 4,096-row chunk takes ~2.6 s, against maybe 1 ms of total launch overhead. <0.1%. |
| `csrc/cooperative_topk.cu` (persistent top-k for routing) | decode is already replayed from a CUDA graph, so per-window launches are amortised; #579 already fused single-row `router_select + moe_plan`. Nothing left to save. |
| `activation_lut_bf16.cpp` (LUT activations on CPU) | our CPU expert path is also weight-traffic bound (~14 ms/window, 100% of it weight traffic); the SwiGLU tanh is not the cost. |
| `dynamic_4bit_int_moe_cpu.cpp` (4-bit CPU MoE) | that is our D1 Q2_0 shadow, implemented and **rejected** on measured quality (3.56/7.90 dB) and host RAM. |
| `cache_kernels_fused.cu` (fused cache quantise+store) | our v4 writer already quantises and stores a row in one pass. |
| `--prefill-schedule-interval`, chunked-prefill interleaving | we run `parallel 1`; there is no second request to interleave. |

## The one real architectural gap: paged, shared KV

They run `--block-size 128` paged attention over a **shared KV pool**: `max_model_len` and `max_num_seqs` are *ceilings,
not reservations*, so `sum(live tokens) <= pool` is the only constraint, and 6 sessions share ~1.9M tokens.

Ours is the opposite: each visit's KV is a **contiguous range sized to the full context** at session creation. That is
exactly why we measured **2 × 128k is 1,383 MiB short** and 2 × 96k is 103 MiB short, and why 256k costs 12.8 GiB of host
RAM even for a 2-token conversation. A paged pool would:

- let N sessions share one pool, sized by live tokens rather than by `max-context × sessions`;
- make 2–4 concurrent sessions fit at 256k **ceilings** instead of failing to load;
- and turn `kv fp8v4`'s 44% saving into concurrency rather than only context.

Cost: this is the largest change in the list — the KV is addressed per visit (`[cell][head][dim]`) throughout the
attention kernels, the elastic-KV arrays, `save_cells/restore_cells` and the checkpoint transfer. It needs a block table
and an indirection in the K/V row address. **The highest-value item, and a project rather than a patch.**

## Their mojibake-level details that are model-specific (not for us)

`VLLM_DSV41_INDEXER_TP4`, `FLASHINFER_DSV41_PAIRED_SWA`, `VLLM_DSV41_INLINE_WOB_QUANT`, `ced_prefill`,
`--engram-config {"cpu_offload":true}` (the n-gram table pinned in host RAM), `--enable-expert-parallel`,
`mla/sm100_cutlass_mla_kernel.cu`, `octave_*.cu`. Inkling has no Engram, no sparse indexer and no MLA; the *spirit* of
"paired SWA" — share one K/V pass across the window layers that read it — is what #636 already does for the drafter.

## And their caution, which we have already tested past

Their recipe is emphatic that **4-bit KV can collapse into salad** under long, heavy agentic context and that fp8 stays
clean, and that the 4-bit path is only kept for the 1.5M context it buys. That is the failure mode to watch for our
`kv=fp8v4`: we have validated it at 8k, 53k and **147,227** tokens with exact needle retrieval, so we are inside the
known-good envelope — but a very long, very heavy session is where I would look first if output ever degrades.

## Summary

- **Adopted**: warm-up before `READY` (real, measured, done).
- **Already better than theirs**: confidence-driven adaptive verification.
- **Worth a project**: paged/shared KV pool — the reason concurrency fails on this box.
- **Everything else**: ruled out by our own measurements, with the reasons above.
