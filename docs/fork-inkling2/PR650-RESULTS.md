# PR #650 integration - results and state
# Session 58 - PR #650 lands: the elastic spill's expert *copies* move with the conversation, decode 67 -> 102 tok/s

## What #650 is

Upstream #650 (with #649 inside it) turns the elastic spill from "a fixed set of experts lives on the
card" into a **copy** layer: any RAM expert can be copied onto the card, the copies are ranked by
**measured heat** (per-expert routing counts folded across sessions), the hottest are promoted, the
coldest demoted, and a *loan* lets a prompt borrow the card's free room and give it back afterwards.
A `CopyLayout`/`loan_layout` keeps every reader's address right, a `rebalance` runs every 64 swaps /
256 rows, and a second-card layout lets the peer card hold a different set. All of it is the same
mechanism that makes decode hit the card 90% of the time instead of 74%.

## Integration

Their branch is `pr650`, our tree is `b14526cb` on `pr536-rebased`. A plain cherry-pick is impossible
(their commits sit on a base 200 commits newer, and #650 rewrites the spill's data model), so the
import was a **3-way merge per file** (`git merge-file`, base = `origin/main`, theirs = `pr650`):

| file | result |
|---|---|
| `model.hpp/.cpp` | theirs (copies, heat, demote/rebalance, `card_copies`, `recall_loan`), our `kCardSpare = 768 MiB` kept (measured +6%) |
| `spill.hpp/.cpp` | **both**: their `CopyLayout`/`loan_layout`/`give_back` beside our `StageOp`/`stage_plan` (#609) |
| `session.cpp` | theirs' `ids`/`win_ids` restructure, `wedge`d, `caches_usable`, `decode_kept`; our fused `router_select_plan` for 128 experts x 1 row kept, our `kPromptPathMin` threshold kept |
| `prompt.*`, `kernels.hpp`, `dspark.*` | merged cleanly; our v4 KV (`ap.v4`, `kv_v_row_bytes`) kept |
| everything else | the rest of the PR applied by hunk (31/34) plus `kernels/` (2/3) |

New files: `copy_heat.hpp`, `loan_layout.hpp` and three tests (wired into CMakeLists).

## Four defects the gate caught (each fixed, each a lesson)

1. **`heat_fold` was never compiled.** `kernels/ops.cu` was outside the file list I merged, so the
   link failed on an undefined symbol. Fix: apply the `src/models/inkling2/kernels/` part of the PR.
2. **The fused router wrote to the wrong buffer.** Their restructure introduced
   `ids = k.win_ids ? ... : x.ids_`; our fast path still wrote `x.ids_`, so the plan read one array
   while the router wrote another. Answers came out as `A) 391 B) 23*17 B) 0 C) 0 ...`.
   Fix: the fused path writes `ids`, and the declaration is hoisted above both branches.
3. **A replayed graph pointed at unmapped copies.** The window graph is keyed by
   `Shape{cpu_, rows/head/keys}` and `cpu_ = sh_->cpu ? m_.cpu_layers() : 0` - under the elastic
   spill `sh_->cpu` is null, so the key was *constant* and a recall left a graph reading memory that
   `demote` had unmapped. First request after a recall died with
   `expert_prefetch: operation failed due to a previous error during capture`.
   Fix: the key carries `placement_epoch()`, and the graphs of the old placement are retired after a
   stream sync.
4. **The read-ahead guess named the previous placement's addresses.** Two requests later:
   `expert_prefetch: invalid argument`. The guess is issued for the *next* visit but survives a
   placement change, so `blob_table` translates to an address that no longer exists.
   Fix (safe, conservative): guesses are off while copies exist (`m_.card_copies() == 0` gates them).
   **Follow-up**: tie a guess to the epoch it was made in and drop it when the epoch moves - worth the
   few % the guess is worth, but only with that guard measured.

## Gate

| check | result |
|---|---|
| Gate 0 | Paris / 391 / planets / Tokyo / 144/12 = 12 - all correct |
| `inkling2_kernels_parity` | 0 failures |
| `inkling2_prompt_parity` | 0 failures |
| `inkling2_copy_heat_test` | ok |
| `inkling2_loan_layout_test` | ok |
| `elastic_array_test` | ok |
| needles | 872 / 1,634 / 3,971 / **70,585** prompt tokens -> all EXACT |

The spill log shows the feature working end to end:

```
elastic spill: startup; expert copies 0 -> 1034 (0 loaned after); 1024 MiB free, 4418 MiB paid mapped
elastic spill: prompt loan at 1665; expert copies 1034 -> 1393 (359 loaned after)
elastic spill: prompt loan recall at 14; expert copies 1393 -> 1034 (0 loaned after)
elastic spill: rebalance at 3970 (64 swaps); expert copies 1034 -> 1034
elastic spill: cache grow to 73728 positions at 32768; expert copies 1393 -> 1265
```

## Numbers (protocol: warm-up discarded, no cached content, engine `pp=`/`tg=`, >=2 processes per arm)

Harness `bench.ps1`, 4,600 random words (~15.4k tokens, every prompt fresh), 400 decode tokens.

| arm | prefill t/s (per process) | decode t/s (per process) |
|---|---|---|
| `b14526cb` (before #650) | 1,534 / 1,533 / 1,536 / 1,528 -> **median 1,533.5** | 71.4 / 65.8 / 66.3 / 67.7 -> **median 67.0** (59.1-72.7) |
| `f2c9b221` (#650) | 1,525 / 1,533 -> **median 1,529** | 106.7 / 96.9 -> **median 101.8** (90.1-108.4) |

**Decode: 67.0 -> 101.8 tok/s (+52%), ranges disjoint** (baseline max 72.7 < #650 min 90.1).
**Prefill: 1,533 -> 1,529 (-0.3%, noise).** The engine's own hit rate says why: **73.8% -> 90.3%** of
decode expert lookups found an expert on the card.

Harness fix found on the way: `bench.ps1`'s fixed instruction prefix ("Read the following list, then
reply OK.") is 14 tokens, which the engine's prompt cache matched on every run, so every run was
rejected. The prefix now carries a random tag, and the cache check rejects only reuse above the chat
template's floor.

## State

- Branch `pr650-int` = `f2c9b221` (+ `3b56ac1f` untracking `server.out`/`server.err`), built, live,
  serving 262,144 tokens with `kv=fp8v4`.
- Rollback: `git checkout b14526cb` (baseline, measured above).
- `run-pr.cmd` was untracked and got removed by a checkout; it is tracked now, so it survives.
- Launcher unchanged (`$Engine = 'C:\LLM\rebase\b'`), warm-up still in it.
