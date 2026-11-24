# The best configuration

Everything below is what this fork actually runs, measured on the box in `README.md` (RTX 5070 Ti Laptop
12 GB, Ryzen AI 9 HX 370, 31 GiB RAM, Windows 11, CUDA 13.5). The tree's `basalt_config.json` is this file's
first block verbatim; the constants in the third block are compile-time and are what the numbers below were
measured with.

## 1. `basalt_config.json`

```json
{
    "version": 2,
    "models": [
        {
            "alias": "Inkling-2-Mini",
            "model": "C:/LLM/Basalt/models/Inkling-2-Mini-NVFP4.basalt",
            "gpu": "0",
            "max-context": 262144,
            "kv": "fp8v4",
            "parallel": 1,
            "vision": "off",
            "vram-reserve-mib": 768,
            "ram-reserve-mib": 512,
            "expert-cache": "auto",
            "spec": 8,
            "prompt-cache": 6,
            "reasoning-effort": "medium"
        }
    ],
    "server": { "host": "0.0.0.0", "port": 8080, "open": false, "allowed-host": "*" }
}
```

Why each line, from the measurements:

| key | value | why |
|---|---|---|
| `max-context` | **262144** | the four-bit value cache (`kv=fp8v4`) gave 256k in the memory fp8 needed for 192k |
| `kv` | **fp8v4** | fp8 K + 4-bit (E2M1) V with a per-32-value fp16 scale: 1,664 B/token/layer instead of 2,048. ~9 dB SNR cost, needles hold to 147k and beyond |
| `vram-reserve-mib` | **768** | 384 measured identical (same plan, so noise); 768 keeps the card from being carved to the bone |
| `ram-reserve-mib` | **512** | the pack is 16.4 GiB against a fixed 14.52 GiB pinned ceiling; the rest is paged as needed |
| `expert-cache` | **auto** | **do not set this explicitly.** A number (`expert-cache N`) was catastrophic: hit rate 65% -> 33%, decode 65 -> 37 tok/s |
| `spec` | **8** | the adaptive policy in `spec/draft_policy.hpp` chooses per round anyway; 12 measured worse (67.2 vs 69.3) |
| `prompt-cache` | **6** | keeps the last six prompts; agent-style reuse |
| `parallel` | **1** | one 262k conversation uses the whole ceiling. Multi-slot profiles exist (`basalt_config-4x64k.json` etc.) and were measured at 230 / 289 t/s aggregate before the copy machinery landed - retune them if you want concurrency |
| `vision` | **off** | no image workloads |

## 2. The launcher's flag

```
basalt.exe serve --config basalt_config.json --python <venv>\Scripts\python.exe \
                 --api-key-file <path>\api-key.txt --prompt-chunk 8192
```

**`--prompt-chunk 8192`** is the single most valuable runtime knob (prefill 2,226 -> 3,018 t/s on its own,
before the staging work). It is not "bigger is better": prefill is dominated by the number of staging passes,
`ceil(tokens / chunk)`, so the value that matters is the smallest chunk that yields the fewest passes for the
prompt lengths you care about.

| chunk | passes for a 15.4k prompt | prefill | decode | card left for experts |
|---|---|---|---|---|
| 4,096 | 4 | 1,976 -> 2,300 t/s | 99.5 -> 91.5 | 2.7 GiB |
| 6,144 | 3 | 2,331 -> 2,570 t/s | 92.6 -> 87 | 2.1 GiB |
| **8,192** | **2** | 3,018 -> **3,153 t/s** | 84.8 -> **92.9** | 1.7 GiB |

**Decode-first alternative:** `--prompt-chunk 4096`. Same engine, no rebuild; the launcher reads
`BASALT_PROMPT_CHUNK` so you can switch per launch. Use 8,192 for long documents and agent prompts, 4,096 when
you are chatting and latency per token matters.

The launcher also sends three warm-up requests before it prints `READY` (a 320-token generation, a 300-word
generation and a ~2k-token prompt). Without them the first real request runs 30-40% slow; with them the first
request measures 67.0 tok/s against the second's 67.7.

## 3. The constants (compile-time)

| constant | file | value | measured worth |
|---|---|---|---|
| `kCardSpare` | `src/models/inkling2/model.cpp` | **768 MiB** | +6% decode, +2% prefill over 1024 |
| `kElasticKeep` | `src/models/inkling2/session.cpp` | **512 MiB** | free room a cache grow leaves; not swept further |
| `kBatch` / `kGqaMaxWindow` | `include/basalt/kernels/gqa_attn.hpp` | **32** | +5% decode over 16 |
| `kPromptPathMin` | `src/models/inkling2/session.cpp` | **96 rows** | prompts under 96 rows read on the decode path: short-prompt TTFT 1.8 s -> 0.5 s |
| `kPromptChunk` | `src/models/inkling2/prompt.hpp` | 6144 (default) | the `--prompt-chunk` default; 8192 is what the launcher passes |
| `kSlotGuard` | `src/models/inkling2/spill.hpp` | 64 KiB | the zero guard after a layer's last staging slot |
| `kGqaMaxWindow`-driven graph key | `src/models/inkling2/session.cpp` | carries `placement_epoch()` | without it a replayed CUDA graph reads unmapped copies after a recall (crash) |

## 4. What this configuration measures

| | |
|---|---|
| plan at load | `cuda0 1.7 GiB / ram 13.5 GiB`, 1,034-1,593 expert copies on the card |
| decode hit rate | 92-95% |
| prefill, 15.4k unique tokens | **~3,100 t/s** (3,033-3,153 over processes) |
| decode, 400 tokens | **~85-93 tok/s** (variance across processes is real; the 15.4k prefill itself decodes at 110 in the same process) |
| TTFT | 0.5 s under 96 tokens, 0.8 s to 1,024, a 15.4k prompt reads in ~4.9 s |
| correctness | needles exact at 872 / 1,634 / 3,971 / **70,585** prompt tokens |

## 5. Environment assumed

- Model pack: `Inkling-2-Mini-NVFP4.basalt` (16.48 GiB), 3,072 experts of which 3,008 live in RAM.
- Python venv for the tokenizer: `--python <venv>\Scripts\python.exe`; API key in a file, not the environment.
- Pinned RAM ceiling is a fixed **14.52 GiB** on this machine (WDDM quota, ~47% of 31 GiB) and cannot be
  raised; the config is sized around it, not against it.
- Measured with `tools/bench.ps1` (unique random words every run, warm-up discarded, prompts with cached
  content rejected, engine `pp=`/`tg=` as the truth, median/min/max/n, >=2 processes per arm).
## 6. Two concurrent sessions (the tuned case)

For two 128k conversations at once - the case that matters most here - the profile is
**`basalt_config-2x128k-fast.json`** launched with **`--prompt-chunk 3072`**:

```json
{ "max-context": 262144, "parallel": 2, "kv": "fp8v4", "vram-reserve-mib": 512,
  "ram-reserve-mib": 512, "expert-cache": "auto", "spec": 8, "prompt-cache": 6 }
```

Measured (two concurrent 30k-token prompts, then 300-token generations in both slots):

| profile | chunk | decode aggregate | prefill aggregate | per-stream prefill | plan |
|---|---|---|---|---|---|
| **2 x 128k (shipped)** | **3,072** | **73.0 tok/s** (36.5 each) | **1,927 tok/s** | 964 tok/s | cuda0 2.8 / ram 12.4 |
| 2 x 128k | 2,048 | 72.7 tok/s | 1,496 tok/s | 748 | cuda0 3.0 / ram 12.2 |
| 2 x 128k | 4,096 | 66.7 tok/s | 2,102 tok/s | 1,051 | cuda0 2.5 / ram 12.7 |
| **2 x 160k** (`basalt_config-2x160k.json`) | 3,072 | 70.2 tok/s | 1,881 tok/s | 940 | cuda0 1.7 / ram 13.5 |
| 2 x 192k | 2,048 | - | - | - | does not fit: 532 MiB short |

So **2 x 160k costs almost nothing** (70.2 vs 73.0 decode) - that is the useful surprise: 320k of total
context fits on the 12 GB card. 2 x 192k does not. `vram-reserve-mib 512` measured no worse than 768 at two
slots and leaves the copies more room.

The chunk choice at two slots is the same trade as at one, shifted: 2,048 decodes best by 0.3 tok/s and
prefills 22% worse, 4,096 prefills best by 9% and decodes 9% worse. 3,072 is the balance.

### Where two slots lose, measured

- **Copy residency**: hit rate 63-82% at two slots against 93% at one, because two KV caches (1.7 GiB each at
  128k) and two prompt arenas come out of the same 12 GB.
- **The prompt loan is off**: `loan_ready()` requires `sessions() == 1`, so the machinery that pushes residency
  from 1,034 to 1,593 copies during a single session does not run with two. Making the loan per session is the
  next real win here - it is a VRAM-residency fix, not a quality change.
- **Acceptance is content-bound, not batching-bound**: on a predictable prompt two slots measured
  **100% and 99% acceptance, dl 2.5-2.8, ~110 tok/s aggregate** against 80 solo; on free prose the same
  configuration measured 34-56%. The verifier does not degrade when the drafts are good, so the earlier
  "batching loses" reading was the content, not the batch.
