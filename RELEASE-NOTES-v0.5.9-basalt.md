# v0.5.9-basalt - two concurrent 128k sessions: working, and 6-20x faster per agent turn

The build after `v0.5.8-basalt`. Everything here is measured on one RTX 5070 Ti Laptop (12 GB, sm_120), Ryzen AI 9
HX 370, 31 GiB RAM, Windows 11, CUDA 13.5 - two sessions of 128k context at once, quality unchanged (no lossy switch
is on by default).

## What was broken at v0.5.8, and is not now

1. **The prompt loan crashed with two sessions.** `Session::recall_loan()` settled the loaned copies' heat from a
   **per-session** baseline while the loan is model-level; the session that did not publish recalled with an empty
   vector and read `NULL + 4*pair` (pairs 1464/1474 - layer 11's hottest RAM experts, first in the rank-ordered loan
   list). It is guarded now, the publication is synchronised where a second session's reader can see it, and the
   loan is allowed with more than one session: the stress harness runs **20/20 rounds in one process with zero
   crashes and zero restarts** (it died on the second round before).
2. **A slot's context was ~82k, not 128k.** `ensure_positions` charged every elastic array the KEY row size
   (1,024 B) while a value row is 80 B a head under `kv=fp8v4`, so each V array reserved only 0.625 of the context.
   With the per-array cell size, 131,072 cells back fully: an **83,516-token** prompt works at two slots and a
   **180,594-token** one works at one slot (both were refused before).
3. **The copy placement was off during concurrent decode.** `submit_loan`/`publish_loan` were only called from the
   solo window path; with two sessions both decode through the batch path, so after the first recall the loan stayed
   off and the hit rate sat at 60-70%. The batch round now pumps each session's placement at its boundary, and the
   rebalance no longer requires a single session: **hit rate 93%** at two sessions.
4. **Every agent turn re-read its whole context** - in the *harness*, not the engine: it sent one growing user
   message with no assistant blocks, so no reply-role token was ever in the prompt. The real client path works, and
   to be sure the pack's reply role is now resolved by its literal (`<|message_model|>`) when the pack does not
   name it, so the turn checkpoint sits at the generation header rather than at the last token.

## Measured, two slots (engine timings, one clean run per point)

Both slots decoding 400 tokens at once, by depth:

| depth per slot | prefill per stream | decode per stream | aggregate |
|---|---|---|---|
| 30k | 1,705 t/s | 46.2 tok/s | 92.4 |
| 64k | 1,500 t/s | 44.9 tok/s | 89.8 |
| 96k | 1,315 t/s | 43.3 tok/s | 86.6 |
| 120k | 1,190 t/s | 43.3 tok/s | 86.6 |

Overlap (one slot reads 20k while the other decodes): 23-25 tok/s per stream, the read 960-1,050 t/s.
Agent turns, real OpenAI roles, replies sent back, two conversations, affinity header on:

| depth | first read | turns 2-4 |
|---|---|---|
| ~40k | 21 s | **0.9-3.7 s** |
| ~100k | 63-65 s | **3.4-5.4 s** |

Correctness on the shipped build: Gate 0 correct, needles exact at 872 / 1,634 / 3,971 / **70,585** tokens, a
17,348-token needle exact while the other slot decoded, `inkling2_prompt_parity` and `inkling2_kernels_parity` at
0 failures, `gqa_window_parity` PASS, and `copy_heat` / `loan_layout` / `elastic_array` ok.

## Configuration

`basalt_config.json` - **2 x 128k**, `parallel 2`, `kv fp8v4`, `vram-reserve-mib 256`, chunk **5,120**, and **no
`reasoning-effort` key** so the pack template's own default (high) applies; a client-sent value still wins.

| profile | what |
|---|---|
| `basalt_config-2x128k-fast.json` | the same with `reasoning-effort: medium` - the speed profile, explicitly not the default |
| `basalt_config-2x160k.json` | 2 x 160k, the largest that fits (2 x 192k does not) |

The launcher warms three single requests, then **two concurrent ones and one overlap**, so the batch graph and the
loan path are warm before the first real request.

## pi client notes

- context window **131,072** per slot;
- send **no** `max_tokens`, or at least 16,384 - a small value caps the thinking budget under `answer-room: auto`;
- send the **session-affinity header** (`x-session-affinity`) so each conversation keeps its slot; without it the
  server alternates slots and prefix reuse cannot happen;
- `reasoning_effort` per request overrides the config.

## Known limits, and what is not done

Prefill is ~63% compute now (the FP4 expert GEMM at ~20 TFLOPS effective) with a 13.40 GiB-per-pass staging floor;
filling both contexts costs ~2.5 GiB of expert copies and decode eases from 46 to 43 tok/s; pinned RAM is a fixed
14.52 GiB and the two-slot plan sits on it. Not done, in the order I would take it: **P0-6** sweeps (`spec`,
`suffix-draft`, `pool-workers`, `prompt-cache`), **P0-3** (the cost of rejected drafts, diagnosis first), **P1-A**
interleaving a decode round between prompt chunks, proper per-session loan heat accounting, and 4-bit keys behind a
flag. `NIGHT-HANDOFF.md` at the repo root carries the launch recipe, the tables and this list in one place, and
`docs/fork-inkling2/` carries every session's measurements including the negative results.