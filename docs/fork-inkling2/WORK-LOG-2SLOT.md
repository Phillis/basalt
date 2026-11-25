# What was done - two concurrent 128k Inkling-2-Mini sessions on one 12 GB card

A record of this working session: every change, every measurement, every bug and its root cause, and what is left.
Engine tree `C:\LLM\rebase\b`, branch `pr650-int`, latest commit `274ba7bb` (pushed to `Phillis/basalt`).
Hardware: RTX 5070 Ti Laptop (12 GB, sm_120), Ryzen AI 9 HX 370, 31 GiB RAM (14.52 GiB pinned ceiling), Windows 11,
CUDA 13.5. Nothing here degrades model quality: every speed item is bit-neutral and the lossy switches are off.

## The headline numbers

| | start of session | now |
|---|---|---|
| single-slot decode | 67.0 tok/s | **101.8** with PR #650's expert copies |
| single-slot prefill | 1,533 t/s | **3,153** (2.06x) |
| two-slot decode, per stream (30k-120k) | did not work reliably | **43-46 tok/s** |
| two-slot prefill, per stream (30k-120k) | - | **1,190-1,705 t/s** |
| two-slot overlap: decoder while the other reads | 23-25 tok/s | **38.8-39.6 tok/s** |
| two-slot overlap: reader | ~1,000 t/s | **1,525-1,629 t/s** |
| context per slot | ~82k (a bug) | **128k**, 2 x 160k available |
| agent turn at 40k / 100k | ~21 s / 63-65 s (a harness artefact, see below) | **0.9-3.7 s / 3.4-5.4 s** |
| decode hit rate at two sessions | 60-82% | **91-96%** |
| needles, both slots | unverified | **3/3 at 64k and 3/3 at 120k**, exact |
| stress gate | crashed on the 2nd round | **20/20 rounds, 0 crashes, 0 restarts** |

## 1. PR #650 - the elastic spill's expert copies

The big decode change. Three-way merged from `pr650` (base `origin/main`, ours the rebased `#536`), keeping our
measured wins (`kCardSpare 768`, `kPromptPathMin 96`, the fused router, the four-bit value cache). Four defects the
gate caught:

1. **`heat_fold` never compiled** - `kernels/ops.cu` was outside my file list; the link error named it.
2. **The fused router wrote `x.ids_` while their plan read `ids`** (their restructure introduced `win_ids`), so
   answers came out as `A) 391 B) 23*17 B) 0 ...`. Fixed by writing `ids` and hoisting its declaration.
3. **A replayed CUDA graph pointed at unmapped copies.** The window graph key was `Shape{cpu_, rows/head/keys}`, and
   `cpu_ = sh_->cpu ? ... : 0` is always 0 under the elastic spill, so the key never changed when a recall moved
   copies: the first request after a recall died with `expert_prefetch: previous error during capture`. The key now
   carries `placement_epoch()`, and old placements' graphs are retired.
4. **The read-ahead guess named the previous placement's addresses** (`expert_prefetch: invalid argument`). Guesses
   are off while copies exist; the proper epoch-guard is on the list.

Result: **decode 67.0 -> 101.8 tok/s (+52%)**, hit 73.8% -> 90.3%, ranges disjoint over 4 processes per arm.

## 2. #636 - two-tile decode attention

Ported `8f380263` + its four follow-ups (stage each K/V chunk once for both query tiles at head width 128) with our
W2 `pos_dev`/`hi_off` patch intact. Their frozen-hash assertion does not apply to us (our `kGqaMaxWindow` is 32
against upstream's 16, so the reduction order differs); re-froze it to our bits and kept the checks that matter
(max abs 0.00127, 0 of 96 windows differing bitwise). **Measured neutral** on our geometry (the drafter's real call
is a 1,025-position window in a 1,088-cell ring at 30.9 us) and kept anyway: bitwise identical, strictly less K/V
traffic, and a new `gqa_window_parity` test guards it.

## 3. Prompt processing: 1,533 -> 3,153 t/s (2.06x)

Measured with a new `BASALT_DIAG_PROMPT=1`, which logs per chunk the staged bytes, the per-slot D2D split and every
buffer of the arena. That instrumentation decided all four steps and killed two wrong ideas:

| change | prefill | decode |
|---|---|---|
| baseline (chunk 4,096, staging on the compute stream) | 1,533 | 101.8 |
| **chunk 8,192** (staging is a fixed cost per chunk: 22.0 GiB at 3,848 rows, 25.3 at 7,733) | 2,226 | 83 |
| **staging on a copy stream** (`91bdae93`, hand ported: the staging is one allocation every layer shares, so layer L+1's copies wait on L's products) | 3,018 | 84.8 |
| **per-slot D2D staging** (a slot with a card copy is copied from it, not from RAM: 13.40 GiB H2D + 11.86 GiB D2D instead of 25.25 GiB H2D) | 3,153 | 92.9 |
| `nrm` is `R x E`, not `3 x R x E` | ~3,100 | 85-93 |

`--prompt-chunk N` became a runtime flag. Prefill is governed by the number of staging passes ``ceil(tokens/chunk)``,
not the chunk size, so 4,096 (4 passes) / 6,144 (3) / 8,192 (2) is a real choice.

Two ideas were measured and dropped: the expert workspace is **already hidden** under the attention buffers
(`carve()` ends with `max()`), so splitting it saves nothing; and the FP4 activation's second term is **not a
switch** (forcing one term crashes - it is baked into the quantizer and block planner).

## 4. Two concurrent sessions: the four bugs that made it unusable

### 4.1 The prompt loan crashed with two sessions (fixed)
`Session::recall_loan()` settled the loaned copies' heat from a **per-session** baseline while the loan is
model-level. The session that did not publish recalled with an empty vector and read `NULL + 4*pair` - the crash read
`0x16E0` and `0x1708`, i.e. pairs 1464/1474, layer 11's hottest RAM experts, first in the rank-ordered loan list.
It was found by reading the code after a map-based symbolization attempt gave only public symbols. Fixed: the
accounting runs only when the baseline matches, the publication is synchronised before another session can read it,
and the loan is allowed with more than one session at all.

### 4.2 A slot's context was ~82k, not 128k (fixed)
`ensure_positions` charged **every** elastic array the key row size (1,024 B) while a value row is 80 B a head under
`kv=fp8v4`, so each V array reserved only 0.625 of the context: 131,072 cells backed 81,920 and the next 8,192 step
failed with "the array is full". `cell_bytes_at(i)` gives each array its own size (used in the map lambda,
`count_backed()` and `grow_bytes()`). Now an **83,516-token** prompt works at two slots and a **180,594-token** one at
one slot.

### 4.3 The loan was off during concurrent decode (fixed)
`submit_loan`/`publish_loan` were only called from the solo window path; with two sessions both decode through
`decode_batch()`, so after the first recall the loan stayed off. `decode_kept()` now pumps each session's placement
at its round boundary, and `rebalances()` no longer requires a single session -> **hit 93%** at two sessions.

### 4.4 "Every agent turn re-reads its whole context" - a harness artefact, not a bug
My first agent harness sent one growing **user** message with no assistant blocks, so no reply-role token was ever in
the prompt and no checkpoint could match. The engine was correct. Two real improvements came out of chasing it: the
turn checkpoint now stays at the conversation boundary (`n-1` is kept as a *second* checkpoint), and the pack's reply
role is resolved by its literal `<|message_model|>` when the pack does not name one. With a real client shape
(`[user, assistant, user, ...]`) an agent turn costs **0.9-3.7 s at 40k** and **3.4-5.4 s at 100k**.

## 5. Tier 0 and Tier 1 of the follow-up spec

**Tier 0 (gates nothing had covered):** three needles at 10/50/90% of depth in one prompt, read into one slot while
the other decodes - **3/3 exact at 64,049 tokens and 3/3 at 119,985**; cancel mid-generation five times while the
other slot decoded - decoder intact, next request fine, 0 crashes.

**Tier 1 (flags):** `spec` 8/6/5/4 (`6` measured +11% then -5% across two processes), `suffix-draft` 3/8/16 (neutral
on prose, as predicted), `prompt-chunk` 5,120/6,144/7,168 (6,144 +3.2% median prefill, inside noise, and costs a
little decode). **No flag beats the base outside the 3-5% noise band, so the defaults stand** - which is itself the
signal that the remaining decode gains have to come from engineering, not tuning.

## 6. Tier 2A: the loan's heat counts every session (landed, large)

The loan is model-level but its accounting was per session: `publish_loan` took the baseline from the publishing
session's tally and `recall_loan` banked only its own hits, so a copy the other conversation used looked cold to
`feed_heat` and was demoted first. Now every session in `sh_->heat_sessions` takes the baseline at publication and
settles its own hits at recall.

| | before | after |
|---|---|---|
| overlap: decoder per stream | 23-25 tok/s | **38.8-39.6** |
| overlap: the reader | 960-1,050 t/s | **1,525-1,629** |
| overlap: wall per round | 23.5-25 s | **15.1-15.5 s** |
| scenario-A hit | 82.8% | **91-96%** |

## 7. How to run it

```
C:\LLM\Basalt\Start-Basalt.cmd
```

Copies `basalt_config.json` into the engine tree, launches `run-pr.cmd --prompt-chunk 5120`, and warms up three
single requests plus two concurrent ones and one overlap. Config: 2 x 128k, `parallel 2`, `kv fp8v4`,
`vram-reserve-mib 256`, `spec 8`, no `reasoning-effort` (the template default applies; a client value wins).
Profiles: `-2x128k-fast.json` (`reasoning-effort: medium`), `-2x160k.json` (2 x 160k, the largest that fits).

Client notes: context 131,072 per slot; send no `max_tokens` or at least 16,384; send `x-session-affinity` so each
conversation keeps its slot (without it the server alternates slots and prefix reuse cannot happen).

## 8. What is not done

1. **2B rejected drafts** - diagnosis first: per-slot drafts offered/accepted, the round's cost, `--spec 3/5` forced.
   Tier 1's `spec` swing (+11% then -5%) is already a hint that the batch policy offers drafts that cannot pay.
2. **2C interleaving** - a decode round between prompt chunks so the other slot keeps producing during a long read
   (`--interleave-ms`, default 0). The decoder is still at zero while the other slot reads.
3. **2D prefill timeline** - per-visit events to see whether any staging is exposed (currently inferred, not measured).
4. `--diag-batch-check` and `tools/batch_identity.py` at slots 1,2; the tool-call and reasoning-echo harness variants.
5. 4-bit keys behind a flag (~27% fewer KV bytes), the latent lookahead prefetch idea, and the VP4 GEMM project.

## 9. Timing, and what each thing was worth

| item | wall-clock | result |
|---|---|---|
| PR #650 import + 4 bugs | a long day | decode +52% |
| #636 + 4 follow-ups | ~1 h | neutral, kept (bitwise identical, less traffic, new test) |
| prompt processing | ~3 h | prefill 2.06x |
| two-slot: loan crash, cell size, batch pump | ~5 h | two-slot usable; hit 60->93%; 128k per slot |
| agent-turn investigation | ~2 h | found my own harness bug; turn 21 s -> 0.9-3.7 s |
| Tier 0 + Tier 1 | ~1.5 h | gates green; flags neutral |
| Tier 2A | ~40 min | overlap +58% decode, +50% reader |

The pattern worth keeping: **three of the biggest wins came from measurements that killed an idea** (the expert
workspace was already hidden, the one-term FP4 is not a switch, the "modulo" is not the prefill bound), and **two of
the worst hours came from trusting a harness** (the chunk tests that ran a stale exe behind a silent LNK1104, and the
agent harness that never sent an assistant message).