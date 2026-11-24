# Night hand-off - two concurrent 128k sessions, what it measures and how to run it

Known good: **8e4215f4** (this commit adds the config default, the launcher's two-slot warm-up and this note).
Engine up at 2 x 128k. Tree clean.

## Launch

    C:\LLM\Basalt\Start-Basalt.cmd

It copies `C:\LLM\Basalt\basalt_config.json` into the engine tree, launches
`run-pr.cmd --prompt-chunk 5120`, and warms up: three single requests, then two concurrent ones and one overlap
(a prompt in one slot while the other decodes), so the batch graph and the loan path are warm before the first real
request. `BASALT_PROMPT_CHUNK` overrides the chunk; `BASALT_DIAG_SLOT=1` adds the restore/boundary diagnostics.

Profiles, all `parallel 2`, `kv fp8v4`, `vram-reserve-mib 256`:

| file | what |
|---|---|
| `basalt_config.json` | **the default**: 2 x 128k, no `reasoning-effort` (the pack template's own default, high, applies; a client-sent value still wins) |
| `basalt_config-2x128k-fast.json` | the same with `reasoning-effort: medium` - a speed profile, explicitly not the default |
| `basalt_config-2x160k.json` | 2 x 160k, the largest that fits (2 x 192k does not) |

## Measured (all on this box, engine timings, one clean run per point)

Two slots, both decoding 400 tokens at once, by context depth:

| depth per slot | prefill per stream | decode per stream | aggregate decode |
|---|---|---|---|
| 30k | 1,705 t/s | 46.2 tok/s | 92.4 |
| 64k | 1,500 t/s | 44.9 tok/s | 89.8 |
| 96k | 1,315 t/s | 43.3 tok/s | 86.6 |
| 120k | 1,190 t/s | 43.3 tok/s | 86.6 |

Overlap (one slot reads a 20k prompt while the other decodes): 23-25 tok/s per stream, the read at 960-1,050 t/s.
Scenario A (both decoding, no prompt in flight): 43.8-46.3 tok/s per stream, hit 83%.

Agent turns, real OpenAI roles (each turn sends the previous reply back as an assistant message), two conversations
alternating, affinity header on:

| depth | turn 1 (first read) | turns 2-4 |
|---|---|---|
| ~40k | 21 s | **0.9-3.7 s** |
| ~100k | 63-65 s | **3.4-5.4 s** |

Hit rate at two slots: 93% with the loan and the batch pump live (it was 60-70% before P0-2).

## pi client notes

- context window 131,072 per slot (`max-context 262144`, `parallel 2`);
- send **no** `max_tokens`, or at least 16,384: with `answer-room: auto` a small `max_tokens` caps the thinking budget;
- send the **session-affinity header** (`x-session-affinity`) so each conversation keeps its slot - without it the
  server alternates slots and nothing can reuse the prefix;
- `reasoning_effort` per request wins over the config's.

## Known limits

- 2 x 192k does not fit; 2 x 160k does.
- Context per slot is now the full 128k (the K/V cell-size fix), and filling it costs decode: the caches take the
  expert copies (~2.5 GiB when both slots are full).
- Pinned RAM is a fixed 14.52 GiB; at two slots the plan sits on it (`ram 13.0 GiB`).
- Prefill is ~63% compute now (the FP4 expert GEMM at ~20 TFLOPS effective); the staging floor is 13.40 GiB a pass.
- The loan is single-session-safe only in the sense that it is now correct at two sessions; its per-session heat
  accounting is still the simple version (the publisher's baseline), so a loaned copy's hits are not credited to both
  conversations yet.

## Not done, in the order I would take it

1. **P0-6 sweeps** (cheap): `--spec 4/5/6/8`, `--suffix-draft 3/8/16`, `--pool-workers 5/7/11`, `--prompt-cache 6/2`.
2. **P0-3** (diagnosis first): per-slot drafts offered/accepted, the round's cost, `--spec 3/5` forced, greedy vs
   sampled - rejected drafts cost the pool the same bytes as kept ones, ~40% at prose acceptance.
3. **P1-A interleaving**: a decode round between prompt chunks so the other slot keeps producing during a long read
   (it stalls today; `--interleave-ms` flag planned).
4. Proper per-session loan accounting (both sessions' tallies as the baseline).
5. 4-bit keys (`--kv nvfp4v4`, off by default): ~27% fewer KV bytes, needs the rotation for 128-wide heads and a
   needle hunt.
6. The tool-call and reasoning-echo harness variants once each (the code is in `tools/agent_turns.py`).