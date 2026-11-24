# v0.5.8-basalt - PR #650 integrated, and prompt processing at 2.06x

A fork build of `jesdga95/basalt` for **Inkling-2-Mini** on a single RTX 5070 Ti Laptop (12 GB, sm_120)
with 31 GiB of RAM, at **262,144 tokens of context** with the four-bit value cache (`kv=fp8v4`).

## Highlights

**PR #650 (with #649) is in** - the elastic spill's expert *copies*: any RAM expert can be copied onto the
card, ranked by measured heat, promoted/demoted between windows, and *loaned* to a prompt and given back.
It is the single biggest decode win in this tree so far.

| | before | after |
|---|---|---|
| decode, 15.4k-token prompt, 400 tokens | 67.0 tok/s | **101.8 tok/s (+52%)** |
| decode expert hit rate | 73.8% | **90.3%** |

Ranges were disjoint across 4 processes per arm (baseline max 72.7 < #650 min 90.1).

**Prompt processing, 2.06x** (this is the bulk of the fork's work):

| change | prefill | decode |
|---|---|---|
| start | 1,533 t/s | 101.8 tok/s |
| 8,192-row prompt chunks | 2,226 | 83 |
| + staging on a copy stream (`91bdae93`, hand ported) | 3,018 | 84.8 |
| + per-slot D2D staging | 3,153 | 92.9 |
| + `nrm` arena fix (live) | ~3,100 | 85-93 |

- **`--prompt-chunk N`** makes the chunk size a runtime knob; prefill is governed by the *number of staging
  passes* `ceil(tokens / chunk)`, not the chunk size, so 4,096 (4 passes) / 6,144 (3) / 8,192 (2) are a real
  choice: 8,192 is fastest, 4,096 decodes best.
- **Per-slot D2D staging**: a long chunk used to re-copy experts from RAM even when the card already had a
  copy (11.86 GiB of every 25.25 per chunk). It now takes each slot from where it is: 13.40 GiB H2D + 11.86 GiB
  D2D instead of 25.25 GiB H2D.
- **`BASALT_DIAG_PROMPT=1`** logs the staged bytes, the per-slot split and every buffer of a chunk's arena.

## Also in

- **#636 + its four follow-ups** (decode attention stages each K/V chunk once for two query tiles). Measured
  neutral on this geometry and kept: bitwise identical output, strictly less K/V traffic, and a new
  `gqa_window_parity` test guards it (hash re-frozen for this tree's `kGqaMaxWindow = 32`).
- **#609 routed prompt staging** wired in (short prompts read only the experts they route to).
- Measured wins kept from the local patch set: `kCardSpare 768`, `kBatch`/`kGqaMaxWindow 32`,
  `kPromptPathMin 96`, the four-bit value cache, warm-up in the launcher.

## Gate

Gate 0 (coherence) correct; `inkling2_kernels_parity` and `inkling2_prompt_parity` **0 failures**;
`copy_heat`, `loan_layout`, `elastic_array`, `gqa_window_parity` pass; needles exact at 872 / 1,634 / 3,971 /
**70,585** prompt tokens.

## Measured limits (so the next step is not guesswork)

- Prefill is now **63% compute**: the FP4 expert products run at ~20 TFLOPS effective. The activation's second
  FP4 term is *not* a switch - forcing one term crashes (it is baked into the quantizer and block planner).
- Staging floor: 3,008 RAM-resident experts, 13.40 GiB per pass, 2 passes for a 15.4k prompt = **1.70 s**,
  i.e. a ~9,000 t/s ceiling for that prompt length. **Layer-major prefill** is the only change that attacks it.
- The prompt arena (1.84 GiB at 8,192 rows) is attention-buffer bound; the expert workspace is already hidden
  underneath it, so splitting it saves nothing.

## Build

Windows 11, CUDA 13.5, VS 2022 BuildTools, Ninja:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=inkling2 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j 12
```

Tests: add `-DBUILD_TESTING=ON` in a second build directory and build the `inkling2_*_parity` targets.
