# Basalt + Inkling-2-Mini on Windows 11

**Basalt v0.5.5** serving **nokeepai/Inkling-2-Mini** (NVFP4 q8 pack, 18.38 GiB)
at a **262,144-token (256k) context window** on an RTX 5070 Ti Laptop (12 GB, sm_120) + 31 GB RAM.

## Use it

| | |
|---|---|
| Start | double-click **`Start-Basalt.cmd`** |
| Stop | double-click **`Stop-Basalt.cmd`** |
| Chat UI | http://127.0.0.1:8080/ |
| OpenAI API | http://127.0.0.1:8080/v1 — model `Inkling-2-Mini` |
| Anthropic API | same port |

The server runs under the **`Basalt` scheduled task**, so closing the console window does
not stop it, and it is not tied to whatever shell launched it.

## Measured performance (this machine)

| Context | Drafting | Decode (short prompt) | Experts in VRAM | Cache hit |
|---|---|---|---|---|
| 262,144 | off (cannot fit) | **11.3 tok/s** | 374 / 3,072 | 72.5% |
| 131,072 | off | 12.0 tok/s | 842 / 3,072 | 76.8% |
| **131,072** | **DSpark on** | **12.8 tok/s** | 565 / 3,072 | 68.0% |
| 32,768 | DSpark on | **14.4 tok/s** | 1,002 / 3,072 | 66.4% |

| Workload | Result |
|---|---|
| Prefill, 16k-token prompt | **1,452 tok/s** |
| Prefill, 131k-token prompt | 1,196 tok/s |
| Prefill, 261k-token prompt | 981 tok/s (266 s) |
| Long-context retrieval | needle found at **261,416 tokens** and at **130,709 tokens** |

### DSpark at 128k: it fits, but gains little

DSpark needs ~2.6 GiB of VRAM at 256k, which the card does not have
(`even with every expert in RAM the card is 74 MiB short`). At **128k the KV halves to 2 GiB**, and
the drafter fits — 3.8 GiB VRAM, 13.3 GiB RAM, 565 expert slots.

But it only nets **+7%** (12.8 vs 12.0 tok/s, 45% drafts accepted): the drafter's VRAM comes out of
the expert cache (842 → 565 slots), so the cache hit rate drops 76.8% → 68.0% and cancels most of
the drafting win. Drafting pays off properly only where VRAM is not the binding constraint.

## Profiles

`Start-Profile.cmd [config]` switches profile and restarts:

| File | Context | Drafting | Speed |
|---|---|---|---|
| `basalt_config-256k.json` (default) | 262,144 | off | 11.3 tok/s |
| `basalt_config-128k-dspark.json` | 131,072 | DSpark | 12.8 tok/s |
| `basalt_config-32k-fast.json` | 32,768 | DSpark | 14.4 tok/s |


## Configuration

```json
"max-context": 262144,  "kv": "fp8",  "spec": 0,  "parallel": 1,
"expert-cache": "auto", "vram-reserve-mib": 1024, "ram-reserve-mib": 512,
"vision": "off"
```

Plus `engine.expert-profile-save`, which records what the adaptive tier learns so later
starts place the hot experts correctly from the first token.

### Why each value

- **`kv: fp8`** — Inkling 2 Mini accepts only `fp16` or `fp8` (there is **no** q8/int8 option
  for this model). `fp8` halves the full-attention caches: 16 KiB/position, so 4 GiB at 256k
  instead of 8 GiB. This is the single biggest VRAM saving.
- **`spec: 0`** — the DSpark drafter is **impossible at 256k** on this card: with the drafter
  loaded the engine reports `spill plan: even with every expert in RAM the card is 74 MiB short`.
  At 32k the drafter fits and gives **14.4 tok/s vs 11.3** (+27%), so it is a context trade-off.
- **`expert-cache: auto`** — must stay `auto`. A fixed smaller value (e.g. `128`) forces
  2,944 of 3,072 experts into RAM (14.18 GiB) and the load dies with
  `cannot pin 14515 MiB of RAM for the experts that leave the card`.
- **`vram-reserve-mib: 1024`** — headroom for graphs/scratch; 512 makes `auto` over-commit.
- **`vision: off`** — frees VRAM. `"dynamic"` also works and costs no VRAM at rest, but
  needs ~0.85 GiB extra pinned RAM, which is tight here.

## Levers that were tested (and did not help)

| Lever | Result |
|---|---|
| `spec 8` drafting at 256k | **impossible** — 74 MiB short on VRAM (works at 128k, see above) |
| `route-prior 2.0` (bias routing to VRAM experts) | **no gain**: 10.9 vs 11.3 tok/s |
| `prefill-cpu-share 0.9` / `0.2` | **no gain**: 1348 / 1443 vs 1452 tok/s |
| `expert-cache 128` | fails (RAM pin wall) |
| `--mmap-experts` | rejected for this model |

**DFlash is not in this model** — that is the Qwen3.8-27B drafter. Inkling 2 Mini ships with
**DSpark** (tested above). `kv q8` does not exist for Inkling (fp16/fp8 only).

## Why it is not faster

The pack needs **16.3 GiB** of weights; the card has **11.9 GiB**. At 256k the KV cache takes
4 GiB and prompt staging ~1 GiB, leaving only **2.8 GiB for experts = 374 of 3,072**. The
remaining **13.1 GiB** is streamed from system RAM over the host link, so 28–58% of expert
lookups leave the GPU. That bandwidth — not decode arithmetic — sets the speed.

What would actually help: a card with 32 GB (Basalt's own docs measure 333–357 tok/s on an
RTX 5090), or a smaller context so more experts fit. At 32k context with drafting the same
machine reaches 14.4 tok/s.

## Layout

```
C:\LLM\Basalt\
  bin\basalt.exe               engine v0.5.5 + CUDA 13 runtime DLLs beside it
  .venv\                       Python 3.11 server environment
  models\Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt   18.38 GiB, SHA-256 verified
  basalt_config.json           the tuned configuration
  run-server.cmd               the command the scheduled task runs
  Start-Basalt.cmd / .ps1      launcher
  Stop-Basalt.cmd              stopper
  server.out                   console log of the last run
  data\profiles\               learned expert placement
```

## Environment notes

- CUDA 13.5 toolkit installed (the release ships no CUDA libraries). `cublas64_13.dll`,
  `cublasLt64_13.dll`, `cudart64_13.dll` and `nvcudart_hybrid64.dll` are also copied next to
  `bin\basalt.exe`.
- Windows privilege **"Lock pages in memory"** is granted to `Phil` and `Administrators`
  (`SeLockMemoryPrivilege`). It turned out **not** to be what blocked the 256k load — the real
  blocker was the explicit `expert-cache: 128` — but it is harmless and is needed if a
  configuration ever has to page-lock the spill.
- **Strata** auto-starts its own engine (`strata.exe`) and holds ~10.8 GB of VRAM. Basalt and
  Strata cannot share the card; the launcher offers to stop it.
- Long prefills on a display-attached GPU can hit the driver's TDR timeout. Raising `TdrDelay`
  under `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (admin) is the documented fix;
  it was not needed for the 261k-token test above.
