# Session 8 — executing `NEXT-WORK-SPEC.md`

A summary of just this batch of work: what the spec called for, what I implemented, what I deliberately did not, and the
measurements. Full log in `CHANGES.md` ("Session 8"); prefill context in `HANDOVER-PREFILL.md`.

**One item landed (D3, verified). One item scoped down to a measured blocker (D1). One correction accepted and applied
(§1). Two items not started, with reasons.**

---

## 1. What the spec ranked

| Order | Item | Spec's gate |
|---|---|---|
| 1 | **D1/P-shadow** — Q2_0 shadow of the spilled experts (moves decode *and* prefill) | pool busy ≤ 8 ms; decode ≥ 65; prefill ≥ 1,900; needle; quality sign-off |
| 2 | **P1** — wave-pipelined staging | unique-16k ≥ 2,400 |
| 3 | **D2** — capture the drafter block (finish W-A) | outside-window ≤ 3 ms; decode ≥ 70 |
| 4 | **D3** — the `hit=` tally fix | `hit=` ≤ 100% across a mixed workload |

Plus §1: a correction withdrawing the "loop-major chunk" advice from `HANDOVER-PREFILL.md`.

---

## 2. ✅ D3 — the `hit=` figure now means something (landed and verified)

### The root cause was not what the spec assumed

The spec described it as "cumulative `ex_tally_`/`lookups_` divided by per-request counts". It is subtler, and the difference
matters: **`hits` is a subset sum over a subset that moves.**

`expert_counts()` returns `{card_share().card, card_share().all}`, and `card` sums the tally of the experts that are on the
card *at the instant of the read*. The serve layer differences that against a snapshot taken at the start of the request. When
promotion moves an expert from RAM to VRAM mid-session, its **entire accumulated history** enters the card sum while the
baseline counted it as zero — so the ratio climbs monotonically. That is exactly what the logs showed:

```
96.6 → 108.4 → 117.6 → 129.6 → 139.3%   (five successive requests, one process)
```

### The fix (one site each, no accounting rewrite)

| File | Change |
|---|---|
| `src/models/inkling2/expert_map.hpp` | `ResidentExpertMap::reset()` — forget the snapshot (`pairs_` depends only on the pair count, so it stays) |
| `include/basalt/model/adapter.hpp` | new virtual `reset_expert_counts() {}` |
| `src/models/inkling2/session.cpp` | implement it: memset `ex_tally_`, `lookups_ = 0`, `map_.reset()` |
| `src/program/serve/session_engine.cpp` | `decode_stats_begin()` calls it **before** taking `counts0_` |

Zeroing at request start also keeps the expert map honest: its `take()` reports "since the last call", and it would otherwise
underflow against a reset device tally.

### Verified

| Check | Before | After |
|---|---|---|
| `hit=` across five successive requests | climbs to 193% | **69.5 / 69.7 / 69.3 / 69.3 / 69.2%** |
| Needle at 130,709 tokens | pass | **pass** |
| Prefill, unique 32k prompt | ~1,536 t/s | 1,561 t/s |
| Decode, 250-token runs | — | 36.9 cold / 47.5–51.1 warm |

Spec's gate met. The cache statistic is now trustworthy enough to tune against — which was the point.

---

## 3. ⏸️ D1 — scoped down to one missing piece

The spec's premise is right, and I verified the supporting pieces exist:

| Piece | Status |
|---|---|
| Q2_0 on the CPU pool path | exists — `native_expert.cpp` handles `GGML_TYPE_Q2_0` (iq256/iq512 kernels), `q2_avx2_rows.inc` |
| Q2_0 on the GPU prefill path | exists — `src/prefill/mmq/instances/mmq_q2_0.cu` |
| Q2_0 dequantizer | exists — `include/basalt/artifact/dequant.hpp:69` |
| Block geometry | **64 elements / 18 bytes**, `GGML_TYPE_Q2_0 = 42` |
| **Q2_0 quantizer** | **does not exist anywhere in the tree** |

So D1 starts by writing `NVFP4 blob → dequantised f32 → Q2_0 block`, then the `NativeFmt` for the decode path, the pageable
4.4 GiB shadow (first-touched at build), the prefill staging from the shadow plus the Q2 MMQ instance, the per-blob 16-byte
`{s_gate, s_up, s_down}` SwiGLU tail the reader expects, and the `--cpu-shadow q2|off` flag.

That is the 1–2 days the spec estimates, and a wrong block layout writes 4.4 GiB that produces garbage rather than an error.
**Not started rather than half-built.** This is the single highest-value item left: it is the only lever in the plan that
moves decode (pool 15.4 → ~6–8 ms) *and* prefill (H2D 21 → 7.7 GiB per chunk) together.

---

## 4. ✅ §1 — the correction is right; my handover was wrong

`NEXT-WORK-SPEC.md` §1 correctly withdrew the "loop-major chunk" recommendation I had written in `HANDOVER-PREFILL.md`. The
walk is **already** pass-major; the loop is a dependency cycle (pass *p+1*'s layer 6 consumes pass *p*'s layer-17 output), so
no reordering is legal; and reusing staging across passes needs all 12 loop layers resident (~6.2 GiB), not the 38 MB I had
quoted (that was the hidden state).

`HANDOVER-PREFILL.md` §5 has been replaced with that correction plus **§5a**, which redirects the same goal (fewer staged
bytes) onto the Q2_0 shadow — so the bad advice cannot mislead whoever picks this up.

---

## 5. Not started, and why

| Item | Why |
|---|---|
| **D1** Q2_0 shadow | Needs a quantizer written from scratch (see §3) — multi-day, garbage-output risk |
| **P1** wave pipelining | `moe_sort` bound-slicing per wave is the fiddly part; 1–2 days; ceiling ~2,400–2,900 with the corrected budget |
| **D2** drafter capture | ~7.8 ms/window sits outside the window; touches the drafter's capture scheme (position-dependent attention) |
| **D4** `kElasticKeep` trim | The spec itself makes it conditional on D1+P1 leaving prefill short |

Also respected from the spec's "do not": `--relaxed-drafting` stays off; `kCardSpare` 768 and `vram-reserve` 768 untouched
(measured constants); no SIMD kernel work; no mapped-RAM GPU compute.

---

## 6. State now

```
engine   C:\LLM\pr536\basalt-inkling-cpu-spill-review   (rebuilt 15:42, 4 files changed this session)
config   196,608 ctx · spec 8 · kv fp8 · expert-cache auto · vram-reserve 768 · ram-reserve 512
live     loaded=True · ctx=196608 · model=Inkling-2-Mini-Unbound
decode   ~37–51 tok/s on the prompt used for the canary (content dependent, 39–67 overall)
prefill  1,561 t/s unique 32k · needle found at 130,709 tokens
```

Files touched: `include/basalt/model/adapter.hpp`, `src/models/inkling2/expert_map.hpp`,
`src/models/inkling2/session.cpp`, `src/program/serve/session_engine.cpp`.

---

## 7. The next session should start here

1. **Write the Q2_0 quantizer** (the missing piece in §3), then build the shadow behind `--cpu-shadow q2|off`. Off by
   default — it is a real quality trade, so it needs side-by-side samples for sign-off, not just a tok/s number.
2. Validate against D1's gates: pool busy ≤ 8 ms, decode ≥ 65 on the TCP prompt, unique-16k prefill ≥ 1,900, needle.
3. **Then** P1 wave pipelining on top, with `--prefill-waves 0|N` for the A/B.
4. Read the `hit=` line for cache truth again — it is fixed now, and it should stay ≤ 100%.
