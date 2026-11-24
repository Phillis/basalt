# D1 verdict — the Q2_0 shadow is dead; the shadow family's real options, measured

**For the user and the next worker.** Follows `D1-Q2-SHADOW-RESULTS.md`. Everything here was measured on the
model's **real spilled weights**, read directly from the pack file (no engine contact), with an independent
decoder ported from the tree's own reference (`dequant.hpp`). Probe scripts kept at
`C:\LLM\bwtest\snr_probe.py`, `snr_probe2.py` (rerun anytime; ~1 min).

## 1. Session 8's wall is real — and independently verified

| Measurement | Session 8 (engine sampling pass) | **This probe (file-level, independent decoder)** |
|---|---|---|
| Q2_0, cover bound | 3.56 dB | **3.52 dB** |
| Q2_0, best-of-ladder | 7.90 dB | **7.85 dB** |

Two methods, two paths, same numbers. The decision **not to build the 6 GiB shadow was correct**, and the
D1-Q2-SHADOW-DESIGN.md 15-22 dB expectation was wrong (carried from symmetric 2-bit assumptions; see §2).

## 2. Why it cannot be rescued — the 4-level ceiling, measured

An optimal **Lloyd-Max 4-level** quantizer (per-block optimal codebooks — better than any 4-level format can
ever ship, including codebook schemes) reaches **9.71 dB** on these weights. The shipped Q2_0-with-ladder is
7.85 dB — within 1.9 dB of the family's mathematical ceiling. **No scale strategy, codebook, or refinement can
make a 2-bit/4-level shadow acceptable.** Q2_0 is closed permanently on quality grounds.

## 3. The shadow family's real decision table (all measured on this model's weights)

| Format | bits/value | blob vs NVFP4 | **Real-weight SNR** | Ceiling (Lloyd) | Readers in-tree | Quantizer |
|---|---|---|---|---|---|---|
| Q2_0 (this fork) | 2.25 | **÷2.0** | **7.85 dB** ✗ | 9.71 dB | CPU ✓ GPU ✓ | done, moot |
| **Q3_K-style** (k-quant, 3-bit + 6-bit sub-scales) | 3.44 | **÷1.31** | **13.78 dB** ? | 14.76 dB | CPU ✓ (ggml traits) GPU ✓ (`mmq_q3_k.cu`) | **must be written** (~1-2 days, k-quant super-block packing is the work) |
| Q4_K-style | 4.5 | 1.0 (no byte win) | 20.65 dB ✓ | 20.28 dB | CPU ✓ GPU ✓ | must be written; **pointless** (no bytes saved; op-cost gain marginal and unproven) |
| IQ3_XS-style (codebooks + imatrix) | ~3.4 | ÷1.3 | likely +3-5 dB over Q3_K | — | GPU ✓ CPU ✓ | major project; not recommended |

What 13.78 dB means in practice: error power ≈ 4% of signal power, applied to the ~27% of routed lookups that
miss the card (decode) — aggregate residual contamination around −19 dB. **Probably imperceptible in casual
use, likely detectable on close reading.** That judgement is the user's, made on side-by-side samples — the
same sign-off gate as before.

## 4. What the Q3_K shadow would actually buy (be clear-eyed before choosing it)

- **Decode:** pool term 15.4 → ~11-12 ms (bytes ÷1.31; instruction cost similar) → **~57-59 tok/s** on the
  benchmark prompt — **+8-12% over today's 53**, not the 65-70 the Q2_0 arithmetic promised.
- **Prefill:** staged bytes ÷1.31 — but **P1 wave pipelining hides the DMA behind the 1.39 s compute anyway**,
  so with P1 landed the Q3_K prefill benefit is ~zero. Its prefill value exists only in a world *without* P1.
- **Cost:** ~6 → ~4.6 GiB pageable RAM, +10-20 s load, a new k-quant writer (super-block layout, 6-bit packed
  scales — harder than Q2_0's writer), the same harness-first discipline, and the quality sign-off.

## 5. The recommendation

**Drop the shadow family from the critical path.** The two clean, zero-quality-cost levers are worth more
than any of it:

| Order | Work | Payoff | Quality cost |
|---|---|---|---|
| 1 | **P1 wave pipelining** (prefill staging behind compute; design in `PREFILL-PERF-SPEC.md` §2, corrected budget in `HANDOVER-PREFILL.md` §4) | prefill **1,561 → ~2,400-2,900** | none — bit-neutral data movement |
| 2 | **D2 drafter capture** (the remaining ~7.8 ms/window outside the decode window) | decode **53 → ~58-60** | none |
| 3 | *(optional, only if the user wants more decode and accepts the trade)* | Q3_K shadow | decode ~57-59 → combined with D2 **~62-64** | 13.78 dB on the ~27% of routed lookups that miss — user sign-off required |

If the Q3_K option is taken later, everything needed is already in place: the writer spec pattern and
harness discipline from the Q2_0 round (`D1-Q2-SHADOW-RESULTS.md` §5-6, substituting the k-quant block
layout), the measured quality expectation (13.78 dB), and the readers exist on both sides
(`native_expert.cpp` routes through ggml's `type_traits_cpu`; `mmq_q3_k.cu` for prefill).

**The end state on this hardware, without a quality trade, is ~58-60 tok/s decode and ~2,400-2,900 t/s
prefill.** Beyond that sits the 16 GB card (removes the spill; >100 tok/s decode), the Q3_K trade, or the
CPU-prefill-expert kernel project to break the PCIe ceiling — in that order of value.
