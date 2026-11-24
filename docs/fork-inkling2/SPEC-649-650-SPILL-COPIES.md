# SPEC: #649 + #650 — spill copies and the copies that follow the conversation

## What they are

| PR | title | upstream | applies to our tree |
|---|---|---|---|
| **#649** | "inkling2: a **static spill** copies RAM experts back to the card, the drafter…" | OPEN, mergeable | **48 of 63 hunks** |
| **#650** | "inkling2: the card's copies of RAM experts **follow the conversation**, changing…" | OPEN, mergeable | **80 of 105 hunks** |

Both are about **which RAM experts hold a card copy and when** — the elastic spill's residency policy, the same mechanism that
`kCardSpare` 768 bought +6% decode with by moving 0.4 GiB. They are newer than `PR-SHORTLIST.md`, so our triage never saw
them, and they land on the code this session has been working in.

## Why they are the next thing to do

`xy-5080` (branch `xy-5080` in the clone) has both **merged and conflict-resolved against main**, which is the part our own
ports keep getting stuck on. #650 is 489 changed lines of `session.cpp` alone; re-deriving that from the PR diff is the
expensive path, and their merge is the reference for it.

## The scoping, measured

168 hunks in total; **128 apply cleanly to our tree (76%)**. The 40 that do not are concentrated where our own changes live:

| file | #649 | #650 |
|---|---|---|
| `src/models/inkling2/session.cpp` | 4 | 7 |
| `src/models/inkling2/model.cpp` | 3 | 3 |
| `src/models/inkling2/model.hpp` | 2 | 2 |
| `src/models/inkling2/spill.cpp` / `spill.hpp` | 1 + 1 | 1 + 1 |
| `src/models/inkling2/spill_test.cpp` | 1 | 2 |
| `kernels/kernels_parity.cpp`, `kernels.hpp` | - | 1 + 1 |
| docs (`inkling2.md`, `precision/inkling2.md`) | 2 | 2 |

Those are exactly the files carrying our local patch set (the staging reclaim / doorbell / `kCardSpare` / `kPromptPathMin`
work) and the four-bit KV plumbing, so the conflicts are expected rather than surprising.

## Order of work

1. `spill.hpp` + `spill.cpp` first - they are the smallest conflicts and they define what the rest calls (the stage plan's
   neighbours: which slots a spill copies back, and how the copies follow a conversation).
2. `model.cpp` / `model.hpp` next (the copies' lifecycle).
3. `session.cpp` last and most carefully (11 of the 40 hunks; it also holds our reclaim and prompt-path decisions).
4. Docs last, they carry no risk.

Build after each file group rather than at the end; a spill change that compiles but is semantically wrong reads a stale
expert, which is a wrong answer rather than a crash.

## Gate

`spill_test` -> build -> Gate 0 (Paris, 17*23 = 391, the planets) -> `kernels_parity` and `prompt_parity` -> the mid-prompt
needles at 621 / 1,198 / 2,968 -> a 53k needle -> then `bench.ps1`, **two processes an arm**, against the current baseline
(prefill 1,547 t/s, decode 69 tok/s, TTFT 0.8 s to 1,024 rows).

Expected: a few percent of decode if the copies' policy is what makes residency sticky across a conversation - the same
mechanism as `kCardSpare` 768 (+6% for 0.4 GiB). Worth measuring per-slot residency (`hit=`) alongside tok/s, since that is
what these PRs move.

## Attempted: the recipe, and where it stops (measured, so it does not have to be re-derived)

**Apply only `pr650.diff`** - the #650 stack already contains #649 (`origin/main...pr650`). Applying #649 and then #650
double-applies and leaves 37/105.

```
git -C C:\LLM\rebase\b fetch origin +refs/pull/650/head:pr650
git -C ... diff origin/main...pr650 --output=C:\LLM\rebase\pr650.diff
apply_hunks.py C:\LLM\rebase\pr650.diff C:\LLM\rebase\b        # 80 of 105 hunks
```

Then the **five new files** the stack creates, which a hunk applier cannot make (it was the first build error -
`copy_heat.hpp` not found):

```
src/models/inkling2/copy_heat.hpp          (86 lines)
src/models/inkling2/copy_heat_test.cpp     (81)
src/models/inkling2/loan_layout.hpp        (31)
src/models/inkling2/loan_layout_test.cpp   (69)
tests/core/elastic_array_test.cpp          (84)
```

`git show pr650:<path> > <path>` for each; after that the build reaches **185 errors**, all of one kind: the applied hunks
call what the 25 unapplied hunks declare. `loan_stream_`, `loan_event_`, `loan_stuck_`, `drain_loan_upload()` (model.cpp),
`CopyLayout::aligned` (loan_layout.hpp), and the members behind them in `model.hpp`. So the remaining work is a **hand-port of
25 hunks** - the declarations in `model.hpp` (2), `model.cpp` (3), `session.cpp` (7), `spill.hpp`/`spill.cpp` (2), plus
`kernels.hpp` (1), `kernels_parity` (1), `spill_test` (2) and the docs (3) - with `xy-5080` open as the resolved reference.

The tree was reverted to `b14526cb` after the attempt: the 80 hunks are reproducible in two minutes from the recipe above,
and leaving a tree with 185 errors would have been worse than leaving none.

## The exact member list to hand-port (from the 185 errors and the unapplied hunks)

`model.hpp` - the new members the applied hunks call:

```cpp
#include "basalt/core/progress.hpp"
#include "models/inkling2/copy_heat.hpp"
bool on_loan(int l, int e) const;                     // loan_flag_[l * n_expert + e]
const std::vector<size_t>& loaned_pairs() const;      // loan_list_
bool has_spill() const { return has_spill_; }
bool copies() const { return has_spill_ && flags_.expert_cache <= 0; }
bool on_card(...) must also exclude a loaned copy: `&& !on_loan(l, e)`
```

plus the private side: `loan_flag_`, `loan_list_`, `loan_stream_`, `loan_event_`, `loan_stuck_`, `has_spill_`, and
`drain_loan_upload()` in `model.cpp`; `CopyLayout::aligned` in `loan_layout.hpp` (it takes `CopyLayout` from
`copy_heat.hpp`, so the include order matters); and the `promote`/`demote` bodies now rank by `copy_heat` and chase loans.

`session.cpp` - 7 hunks (the loan lifecycle around a window and a prompt chunk); `spill.hpp`/`spill.cpp` - 2 (the static
spill's own copy pass); `kernels.hpp`, `kernels_parity.cpp`, `spill_test.cpp` - the new fields' plumbing and tests.

So the port is: declarations and members first (`model.hpp` + `copy_heat.hpp` + `loan_layout.hpp`), then `model.cpp`'s
promote/demote/loan bodies, then `session.cpp`, then the tests. Build between each; the errors are all "identifier not found"
until the header is complete, which makes the order self-checking.
