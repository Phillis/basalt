> **Update (later session):** the v1.0 pack described below as the rollback target has been **deleted** from this
> machine (it produced repetition loops on this workload). Rollback today means the v2 pack or the base-model pack
> Inkling-2-Mini-NVFP4.basalt; the rest of this document is kept as the record of that swap.

# Model update: Inkling-2-Mini-Unbound **v2** (Session 14)

The engine now runs the **v2** pack from `nokeepai/Inkling-2-Mini-Unbound-Basalt` instead of the v1.0 pack the install
shipped with. v2 is the same model architecture and the same Basalt pack layout — only the attention projections changed —
so nothing in this tree needed a code change: the swap is a file and one line of config.

## What changed

| | Before | After |
|---|---|---|
| File | `Inkling-2-Mini-Unbound-NVFP4-q8.basalt` | **`Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt`** |
| Size | 18.38 GiB (19,734,300,582 B) | 18.38 GiB (19,734,300,582 B) — identical layout |
| SHA-256 | `8f2e1c2a…79b5` | **`aa20b87a29c19aa6c8ed699968aeaab08e3ca47f4cd768a3470ea073cf600c77`** |
| Model id (API) | `Inkling-2-Mini-Unbound` | `Inkling-2-Mini-Unbound` (unchanged — the alias is ours) |

Both packs are kept on disk, so switching back is one line (see *Rolling back*).

## Why v2

From the pack's own model card, measured on their sealed split with the Q8_0 GGUF (not with this pack):

|  | Original | v1.0 | **v2** |
|---|---|---|---|
| Useful answers to 277 held-out harmful requests, thinking off | 0.7% | 51.6% | **78.7%** |
| Same, thinking on (effort 0.9) | 0.7% | 62.1% | **83.4%** |
| Drift from the original (KLD, lower = closer) | — | 0.0102 | 0.0318 |
| MMLU-300 / GSM8K-100 | 76.3 / 97 | 75.7 / 99 | 76.0 / 99 |

In practice, the difference that matters for everyday use is that v2 stops *dodging*: fewer disclaimers, fewer
"conceptual" rewrites, more answers that do the thing asked. The cost is more drift in wording on ordinary requests.

## Verification performed on this box

1. **Size and checksum** — 19,734,300,582 bytes and SHA-256 `aa20b87a…0c77`, matching the published values exactly.
2. **Load at 192k** — `Start-Basalt.cmd` reports READY with `ctx=196,608`, `spec=8`, hand-off `doorbell`, same 18.4 GiB load
   and ~768 MiB of free VRAM as v1.0 (identical layout, identical memory plan).
3. **Basic correctness** — "capital of France" → *"The capital of France is Paris."*; "rivers shape valleys" → two coherent
   sentences; `17 * 23` → `391`.
4. **Long context** — the 130,709-token needle test still returns `ZEPHYR-4417`: **NEEDLE FOUND**.

## Measured on this box (RTX PRO 6000-class 12 GB-usable spill setup, 192k context, `spec 8`)

| Metric | v1.0 | **v2** |
|---|---|---|
| Decode, 400-token prose (warm) | 60.5–61.5 tok/s | **57.9 / 58.2 tok/s** |
| Draft acceptance | 87–91% | **81–83%** |
| Drafts per window (`dl`) | 1.6–1.7 | **1.5** |
| Prefill, unique 10k prompt | 1,368–1,539 tok/s | **1,357 tok/s** |
| Needle @ 130,709 | found | **found** |

The decode gap is the one the pack's card predicted: the DSpark drafter is the *original* model's, and v2's attention
weights drift further from the original than v1.0's did, so drafts are accepted slightly less often (`acc` 87→82%, `dl` 1.7→1.5).
Prefill and long-context behaviour are unchanged, as expected for a pack with identical layout and only 120 changed tensors.

## Rolling back to v1.0

Both files are in `C:\LLM\Basalt\models`. Edit `"model"` in **both** copies of the config (the launcher copies the canonical
one into the tree), then restart:

```powershell
# C:\LLM\Basalt\basalt_config.json  and  C:\LLM\pr536\basalt-inkling-cpu-spill-review\basalt_config.json
"model": "C:/LLM/Basalt/models/Inkling-2-Mini-Unbound-NVFP4-q8.basalt"      # v1.0
"model": "C:/LLM/Basalt/models/Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt"   # v2 (current)
```

```powershell
C:\LLM\Basalt\Stop-Basalt.cmd
C:\LLM\Basalt\Start-Basalt.cmd
```

## Notes carried over from the pack's card

- **Thinking effort** is per request (`reasoning_effort`). Even at `none` the model writes a short thinking block, so leave
  room in `max_tokens`; a 4,096-token thinking cap is a sane default.
- **Drafter acceptance** is a little lower than v1.0's — that is the weights, not this tree. If you would rather trade drafts
  for raw speed, `"spec": 0` still works.
- The pack is uncensored by design. The card is explicit that it follows harmful instructions and ships no safety training;
  deployments that can reach other people need their own safeguards.
