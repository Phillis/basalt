
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

## One session at 256k (the second workload)

The merged base fits a full 256k context with **fp8** on one slot: plan `cuda0=2.3 GiB ram=12.9 GiB`, profile
`basalt_config-1x256k-fp8-merged.json`. Measured with a 172,001-token prompt (the passphrase buried at note 2):

| | result |
|---|---|
| correctness | **ZEBRA-42 recalled** from position ~120 of 172,001 |
| prefill | **~1,660 t/s** (172,001 tokens in 103.6 s end to end) |
| decode | **~58 t/s** (9 tokens in 154 ms on a warm window, 6 of 6 drafts accepted) |
| hit rate | 68% |

`kv nvfp4` also fits at 256k (plan `cuda0=3.9 / ram=11.3`) but is not used: its KLD is 2.3x fp8's. `kv fp8v4` does not
exist on this base (nvfp4 replaced it there).

## The fair comparison (both builds, warm runs, same harnesses, 2 x 128k)

Re-measured on the same day with the warm-up rule applied to both (first curve after a start discarded):

| build | kv | chunk | prefill 30k / 64k / 96k / 120k | decode 30k / 64k / 96k / 120k |
|---|---|---|---|---|
| **merged** | fp8 | 4096 | 1,632 / 1,540 / 1,377 / 1,281 | **55.8 / 55.2 / 51.7 / 51.1** |
| **merged** | fp8 | 5120 | **2,006 / 1,781 / 1,563 / 1,435** | 37.7 / 51.3 / 49.6 / 48.5 |
| ours (live) | fp8v4 | 5120 | 1,840 / 1,558 / 1,375 / 1,245 | 46.8 / 44.3 / 43.1 / 44.8 |

**The merged base with `kv fp8` wins decode at every depth (+14% to +25%)**, is level on prefill at 96k/120k, and its quality
is 2x better (KLD 0.00586 against 0.01193). Our tree keeps two things: prefill at 30k (1,840 against 1,632) and decode at no
depth. With `--prompt-chunk 5120` the merged base also wins prefill at **every** depth (2,006 / 1,781 / 1,563 / 1,435) while
keeping decode at 64k+ (51.3 / 49.6 / 48.5); the only thing it gives up is 30k decode (37.7 against 46.8).

So: **decode-first -> merged + fp8 + chunk 4096. Prefill-first (agent turns that add a lot of context) -> merged + fp8 +
chunk 5120.** Both are better than the live configuration wherever 128k sessions actually spend their time.

## Session 68 final: the two real options at 2 x 128k (warm, both slots decoding)

| kv | KLD / top-1 | prefill 30k/64k/96k/120k | decode 30k/64k/96k/120k | KV at 2 x 128k |
|---|---|---|---|---|
| **fp8** | **0.00586 / 96.80%** | 1,632 / 1,540 / 1,377 / 1,281 | 55.8 / 55.2 / 51.7 / 51.1 | 3.9 GiB |
| **nvfp4** (group-8 + MSE) | 0.01132 / 95.42% | **1,768 / 1,678 / 1,503 / 1,415** | **58.6 / 55.3 / 52.9 / 53.6** | 2.43 GiB |

fp8 is the quality choice; nvfp4 is +8-10% prefill with equal-or-better decode and quality better than the configuration
live today (fp8v4: 0.01193 / 95.33%). Both pass Gate 0, needles 3/3 at 64k and 120k with a concurrent decoder, and stress
20/20. One session at 256k: fp8 fits and recalled a fact at 172,001 tokens (~1,660 t/s prefill, ~58 t/s decode).

## The chunk matrix, warm, 2 x 128k (this is the prefill lever)

Prefill is governed by the number of staging passes, `ceil(tokens / chunk)`; a bigger chunk amortises them but its row
buffers take the card room the expert copies want. The four-bit cache's staging is half fp8's, which is what lets 8,192
fit here at all (it does not with fp8: "the card is 450 MiB short").

| kv | chunk | prefill 30k / 64k / 96k / 120k | decode 30k / 64k / 96k / 120k | plan |
|---|---|---|---|---|
| fp8 | 4096 | 1,632 / 1,540 / 1,377 / 1,281 | 55.8 / 55.2 / 51.7 / 51.1 | cuda0 2.1 |
| fp8 | 5120 | 2,006 / 1,781 / 1,563 / 1,435 | 37.7 / 51.3 / 49.6 / 48.5 | cuda0 1.8 |
| **nvfp4** | 4096 | 1,768 / 1,678 / 1,503 / 1,415 | **58.6 / 55.3 / 52.9 / 53.6** | cuda0 3.6 |
| **nvfp4** | **5120** | **2,047 / 1,834 / 1,685 / 1,556** | 55.8 / 53.3 / 51.6 / 49.0 | cuda0 3.4 |
| **nvfp4** | **8192** | **2,340 / 2,117 / 1,883 / 1,762** | 55.4 / 48.7 / 46.1 / 47.0 | cuda0 2.5 |

Three configurations worth naming:

- **Best mix - nvfp4 + chunk 5120**: prefill **+25% / +19% / +22% / +22%** over the quality-first row, decode within 4% of
  it at 30-96k (49.0 at 120k, -9%). KLD 0.01132, top-1 95.42%.
- **Prefill-first - nvfp4 + chunk 8192**: prefill **2,340 / 2,117 / 1,883 / 1,762 t/s**, i.e. **+43% / +37% / +37% / +38%**
  over fp8 at 4,096, paying about -12% decode at depth.
- **Quality-first - fp8 + chunk 4096**: KLD 0.00586, top-1 96.80%, decode unchanged.

The chunk does not change numerics (a row's forward does not depend on which rows share its chunk), so these are purely
speed/space trades; the KLD figures above are the format's, measured at the default chunk.
