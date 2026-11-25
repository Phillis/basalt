
# BEST-CONFIG.md - the best measured configuration on this machine

Updated Session 68 (AMD Ryzen AI 9 HX 370, RTX 5070 Ti Laptop 12 GB, 31 GiB RAM, Windows 11).

## Decode and prefill per stream, 2 concurrent 128k sessions (the default workload)

Measured with `curve2.py` (one 30k/64k/96k/120k prompt per slot, both slots decoding), `sweep2.py B`
(two 30k prompts at once), `needles3.py` for correctness. t/s per stream; aggregate is 2x.

| build | kv | chunk | KLD mean / top-1 | prefill 30k / 64k / 96k / 120k | decode 30k / 64k / 96k / 120k |
|---|---|---|---|---|---|
| **merged** (`nvfp4-kv`) | **fp8** | 4096 | **0.00586 / 96.80%** | 1,770 / 1,623 / 1,398 / 1,284 | 41.0 / **53.1** / **57.6** / **51.6** |
| merged | fp8 | 5120 | 0.00586 / 96.80% | **2,006 / 1,781 / 1,563 / 1,435** | 37.7 / 51.3 / 49.6 / 48.5 |
| merged | nvfp4 | 4096 | 0.01374 / 95.04% | 1,890 / 1,756 / - / - | 40.1 / 46.5 / 50.0 / - |
| ours (`pr650-int`, live now) | fp8v4 | 5120 | 0.01193 / 95.33% | 1,705 / 1,500 / 1,315 / 1,190 | 46.2 / 44.9 / 43.3 / 43.3 |

**Recommendation: the merged base with `kv fp8` and the default 4,096-row chunk.** Against the configuration live now it
is **+18% / +33% / +19% decode** at 64k / 96k / 120k, **+6-8% prefill** at those depths, and **better quality** (KLD
0.00586 against 0.01193, top-1 96.80% against 95.33%). It loses only at 30k decode (41.0 against 46.2). Use
`--prompt-chunk 5120` instead when a workload is prefill-dominated (agent turns that add a lot of new context): it buys
**+10-13% prefill at every depth** for **-3 to -8% decode**.

The chunk is a real trade on this base: prefill is governed by the staging passes `ceil(tokens / chunk)`, and the chunk's
row buffers take the card room the expert copies want. 8,192 does not fit at two 128k slots with fp8 (the plan is 450 MiB
short); the four-bit formats could use it because their KV staging is half the size.

## Quality, measured on the public base (100 chunks, T=3, `tools/ppl_gate.py`)

| kv | KLD mean | median | p99 | top-1 |
|---|---|---|---|---|
| fp16 | 0.00454 | 0.00217 | 0.0391 | 97.22% |
| **fp8** | **0.00586** | 0.00282 | 0.0502 | **96.80%** |
| fp8v4 (ours) | 0.01193 | 0.00581 | 0.0998 | 95.33% |
| nvfp4 | 0.01374 | 0.00674 | 0.1153 | 95.04% |

Keys at four bits are the expensive half on this model: nvfp4 (both planes 4-bit) costs +0.0079 KLD over fp8, more than
our values-only fp8v4 (+0.0061). nvfp4 is therefore implemented, correct (needles 3/3 at 64k and 120k) and **not
recommended**: it is worse on quality and no faster on decode than fp8 on this base.

## Profiles

- `basalt_config-2x128k-fp8-merged.json` - the recommendation above (merged base, fp8, 2 x 128k, reserve 256).
- `basalt_config-merge-upstream.json` - the same plan, kv swapped by the launcher.
- `--prompt-chunk 5120` on the command line for prefill-first use.

## Correctness gates a profile must pass (all passed by the recommendation)

Gate 0 (391, Tokyo, planets); `needles3.py` 3/3 at 64,000 and 3/3 at 120,000 with a concurrent decoder; `stress_two_slot.py`
20 rounds with no crash; `tools/ppl_gate.py` KLD within 0.0010 of fp8's.

## Knobs measured and NOT worth changing on this base

- `--spec` 0 / 2 / 4 / 8: all within run-to-run noise on decode (35.8-40.8 at 30k, 48.5-50.7 at 64k); the draft length
  does not move the needle, so the default stands. Turning drafting off also re-plans memory (the drafter's tap goes back
  to expert copies) and does not help.
- `--vram-reserve-mib` 256 is the floor the plan accepts; more reserve only takes room from the copies.
- `kv nvfp4`: see above - correct but dominated.

## Session 68 addendum: the placement pump, and the numbers it settles on

With `Session::pump_placement` called from the batch path (`decode_kept`), warm medians at 2 x 128k, fp8:
decode 30k ~55.8, 64k ~55.2, 96k ~51.7, 120k ~51.1; prefill 30k 1,634 / 64k 1,540 / 96k 1,378 / 120k 1,280.
The pump is kept (it also restores the expert-loan cycling the batch path skipped).

Methodology note that matters for every number in this file: **the first `curve2.py` run after an engine start is 10-30%
slower than the steady state** (30k decode 38.2 against 55-57). Every figure quoted as a comparison here is from the second
run onward.
