# Prefill: where the time goes, and the chunk-size win
# Session 60 - the HuggingFace model that cannot be used, and what the prefill time actually is

## The model: ehzawad/Inkling-2-Mini-Q3_K_XL-NVFP4-GGUF

Checked as asked. It cannot run here, for three independent reasons:

1. **The repo is `gated: auto`.** The HF API lists it, but every file request returns **401 Unauthorized**
   without an authenticated token, and this box has none (`no HF_TOKEN`, no `~/.cache/huggingface/token`,
   no `hf` CLI). It is also 0 downloads / 0 likes, uploaded the same day - unvetted.
2. **It is the wrong container.** Basalt does not load GGUF: it loads `.basalt` packs
   (`C:/LLM/Basalt/models/Inkling-2-Mini-NVFP4.basalt`, 16.48 GiB). This tree can *read* GGUF
   (`basalt-gguf`, `src/artifact/gguf_reader.cpp`, built) and *read* a pack (`src/models/inkling2/pack.cpp`),
   but it has **no GGUF -> pack writer**, so a downloaded 12.81 GiB GGUF would still be inert.
3. **Its kernels are the wrong family for us.** "Q3_K_XL + NVFP4" is a llama.cpp mix: Q3_K with Q4_K/Q5_K/Q6_K
   tensors in the important places, plus NVFP4 elsewhere. Our inkling2 expert path runs NVFP4 / MXFP8 /
   q8_0 on the tensor cores (that is what the 2,190 t/s prefill and 102 tok/s decode are made of); K-quants
   exist in this tree only as generic dequant/MMVQ, a much slower path. Even converted, it would be a
   compute regression - and our compute is the *other* 39% of prefill.

What it would have bought, for the record: 12.81 GiB against our 16.48 GiB is 22% less to stage, and
staging is 61% of prefill - so maybe +10-15% prefill, at a large compute cost. Not a shortcut.

## What the prefill time actually is (BASALT_DIAG_PROMPT=1, new)

Per chunk, per visit (of 48 spilled layers), the long-chunk path stages **every RAM expert of the layer**:

```
prompt diag: 3848 rows, 48 visits, h2d 0.00 + full 22.04 GiB, d2d 0.00 (h2d 1502.7 ms at 15.75 GB/s),
             0 ram ops, 0 card ops, 48 full copies
```

So: **22.0 GiB staged per 4,096-row chunk = 1,502 ms of a 2,467 ms chunk. Staging is 61% of prefill,** and
it is nearly independent of the row count (25.3 GiB at 7,733 rows). Consequences that follow directly:

| chunk | passes per 15.4k prompt | staged per pass | prefill |
|---|---|---|---|
| 4,096 | 4 x 3,848 rows | 22.0 GiB | **1,559 t/s** |
| 8,192 | 2 x 7,733 rows | 25.3 GiB | **2,226 t/s (+43%)** |
| 16,384 | 1 x 15,390 rows | - | does not fit: "the card is 2125 MiB short" |

`bench.ps1`, two processes each: **prefill 2,190 / 2,184 t/s** against 1,533 / 1,528 before.
Gate at 8,192: Gate 0 correct, needles 872 / 1,634 / 3,971 / **70,585** all EXACT, a 25.7k prompt read at
1,907 t/s (multi-chunk, with the cache growing underneath).

Cost, stated honestly: the prompt buffers take 1.1 GiB of card memory (`cuda0` 2.7 -> 1.6 GiB), the decode
hit rate falls 90.3% -> ~84% and **decode 102 -> 83 tok/s**. That is the trade for now; the fix is the
staging overlap below, which removes the reason for big chunks.

Two other things the instrumentation settled:
- The `rows <= 1024` routed path never engages for real prompts (3,848 rows), and `stage_from_card`
  never fires while a loan is active (`loaned() > 0` returns false), so 0 D2D copies happen in a prompt.
- The two earlier "chunk size is inert" runs were **stale executables**: `ninja`'s link fails with LNK1104
  while the engine holds `bin\basalt.exe`, silently. Always `findstr` for a new string in the exe after a
  build that matters.

## Next: the staging overlap (91bdae93), a hand port

`91bdae93` ("a long prompt chunk copies a spilled layer's experts on a second stream under the kernels
before it") is in the local clone, but it does not merge: our `prompt.cpp` has since taken #609's routed
staging and #650's copy machinery, so the 3-way merge is a whole-file conflict. It must be ported by hand:
a `copy` stream with `freed`/`staged` events, a `next_spill_[step]` map, and the main stream waiting on
`staged` before a spilled visit's expert products. If it hides most of the 1.5 s, prefill goes to
~3,500-4,000 t/s *and* kPromptChunk can go back to 4,096, which returns the 1.1 GiB and the decode hit rate.

The `--prefill-waves` flag in this tree is dead (parsed into `g_prefill_waves`, never read), so that is the
socket the port plugs into.
