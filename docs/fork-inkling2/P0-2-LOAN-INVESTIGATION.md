# P0-2 investigation: the prompt loan at two sessions

The spec's `P0-2` calls the loan "the largest decode lever": alone, copies rebalance every 256 kept rows and a
prompt's loan lifts them from 1,034 to about 1,400-1,600, and the hit rate is 93%; at two slots both are switched
off (`rebalances()` wants `sh_.use_count() == 1`, `loan_ready()` wants `sessions() == 1`) and the hit rate is
58-82%. This is what was tried, what it proved, and exactly where it stopped.

## The mechanism, as read from the code

- `loan_ready()` maps the **shared staging's granules** a second time into `loan_stage_`, and points `loan_arena_` at
  the prompt arena (set by `set_prompt_arena` from `Prompt::init_lane`). Both are one pointer each, on the model.
- `submit_loan(publish_at)` builds `loan_plan_` (the highest-ranked RAM experts without a paid copy) with
  destinations `loan_arena_ + offset` or `loan_stage_ + offset`, and `queue_loan()` uploads them on `loan_stream_`.
- `publish_loan()` waits for the upload event and writes the addresses into each layer's `blob_table`, so decode
  reads them in place; `loan_list_`/`loan_flag_`/`copy_` record them.
- `recall_loan()` clears those table entries, joins every reader (`join_placement`, a `cudaDeviceSynchronize`),
  and drops the records. `recall_loan` is what a prompt calls before it writes the arena.

So a loan is only safe while the arena it lives in is not being written - which is exactly the prompt's buffer.

## What was implemented (and measured)

1. **Publication gated during a read**: `publish_loan()` returns early while a read owns the arena. (`submit_loan`
   was already gated by the earlier `prompt_busy_` attempt.)
2. **In-flight uploads drained at the read's start**: `PromptGuard`'s constructor calls `drain_loan_upload()` before
   a prompt byte is written, so an upload queued by the *other* session's window cannot land mid-read.
3. **Recall made unconditional**: the `prepare` lambda that recalls was installed only `if (heat_tally_)`
   (`session.cpp`), so a read could start with the loan still published. Now every read recalls.
4. **Rebalance bisect**: `rebalances()` opened to several sessions (`!m_.prompt_busy_now()`), with a
   shared/exclusive staging lock (prompt exclusive, decode windows shared) so a read and a window never share the
   one staging allocation.

## What it proved

With 1-4 in place the loan **does run at two sessions** - the log shows `prompt loan at 366; expert copies
1015 -> 1411 (396 loaned after)` and `rebalance at 591 (64 swaps)` - but the stress test still fails on a
**deterministic alternation**: rounds 1, 3, 5, 7 die and 2, 4, 6, 8 pass (or the reverse, depending on the
starting state), each death `EXCEPTION_ACCESS_VIOLATION ... in reading the prompt, from token (0)`, read from a
small offset (`0x1708`, `0x200C`) - a null base plus an offset.

**The bisect is decisive**: with the rebalance reverted to one session and only the loan opened (1-3 kept), the
crash is unchanged. It is the loan, not the rebalance, and it is not a race (alternation, not intermittency).

The committed build - loan back to `sessions() == 1` - runs the same stress **4/4 with 0 crashes**, decode
23.5-24.1 tok/s per stream while the other slot reads a 20k prompt at ~1,040 t/s. That is what ships.

## Where it stopped, and the next step

The addresses could not be symbolized: the release build ships **no PDB** and no `llvm-symbolizer` is installed.
The next step is therefore mechanical, not speculative:

1. Build with debug information (`-DCMAKE_BUILD_TYPE=RelWithDebInfo`, or `/Zi` + `/DEBUG`), keep `bin\*.pdb`, and
   symbolize `0x7FF6...` from the crash line with `llvm-symbolizer` or `winaddr2line`. The frame that reads
   `null + 0x1708` at token 0 of a read names the field (a struct member offset that size is diagnostic: about
   5.9 KiB into some object).
2. Failing that, the spec's own prescription: per-transition logging (states Idle -> Publishing -> Lent ->
   Recalling, the session, the position, the pair count, the registered readers) - added *carefully* (the first
   attempt at that broke the source and was reverted), reproduced under `tools/stress_two_slot.py`, and read
   backwards from the crash.

## What is safe today

- The loan stays single-session. **Do not relax `loan_ready()` without `tools/stress_two_slot.py` passing 8 rounds.**
- The rebalance at two sessions was *not* proven unsafe (the bisect cleared it); it is simply not enabled, because
  the loan it depends on is not.
- Everything else in this session's work is independent of the loan and is in the shipped build.

## Update: five hypotheses tested, all negative

After the bisect, three more candidate causes were implemented and each was judged by `tools/stress_two_slot.py`:

| hypothesis | change | result |
|---|---|---|
| the loan's publication races a read | `publish_loan()` returns early while `prompt_busy_` | still alternates |
| an upload queued by the other slot lands mid-read | `PromptGuard` drains `drain_loan_upload()` before the first byte | still alternates |
| the batch graph key lacks the placement | `key.push_back(m_.placement_epoch())` in `batch_stage` (the spec's P0-1 item, and what the solo key needed after #650) | still alternates |
| the prompt cache's pinned checkpoints interact with a loaned arena | `prompt-cache 0` | every prompt succeeds, every decode session fails, 23 crashes - different, not better |
| the rebalance is the culprit | bisect: rebalance back to one session, loan on | crash unchanged - the rebalance is cleared |

**Sharper characterization, and the most useful result here**: failures are not per round but per process. Round 1
in a fresh engine always works; the *second* round in the same process dies. A crash restarts the engine, so the
round after it works again - which is what made it look like an alternation. The state that breaks it is therefore
created by a *completed* round, i.e. at a request's end, exactly as the spec suspected ("Suspect this first: the
alternation began after a teardown"). With the loan on, the arena is the loan's home between prompts; a request's
end is when `recall_loan(-1)` ("prompt loan recall at sleep") runs, and the next request's read is the first thing
to touch the arena afterwards.

That narrows the search to the sleep recall and the state it leaves (`lent_`, `loan_list_`, `copy_`, `loan_flag_`,
each layer's `blob_table`), with another session's request able to start in between. Instrument exactly that
transition - but from a build that can be symbolized (`RelWithDebInfo` + PDB first): five blind attempts have now
been spent on hypotheses the log alone could not rank.
## Symbolization attempt (per the review session), and where it stopped

Cheapest route tried - relink with `/MAP:basalt.map`, print `GetModuleHandleW(nullptr)` from the crash handler,
symbolize `frame - base` against the map:

- the handler now prints `basalt crash: module base: 0x7ff747290000` and 24 frames;
- the map the linker produced is **871 KB with 4,991 symbols** - public symbols only, no file-local statics - and
  the nearest symbol for the faulting RVA came out as a header artefact (`__guard_flags`), so the top frame is
  not resolvable this way. The `/MAP` flag also truncated `bin\basalt.exe` once (LNK1104 on the map) and had to
  be removed from the cache and relinked.

The working route is the fallback: a `RelWithDebInfo` tree in its own build directory plus
`SymInitialize`/`SymFromAddr` in the handler (dbghelp ships with Windows).

## The strongest lead, from the review session's reading of the fault

`0x16E0 = 732 * 8` and `0x1708 = 737 * 8`: an **8-byte array indexed from a NULL base**. In the loan code that
shape is `loan_addr_[loan_list_.size()]` in `publish_loan()` (an `unsigned long long*` from `cudaHostAlloc`) or
`copy_[pair]` (a `std::vector<const void*>`, also 8 bytes an entry). 732-737 is a plausible count of loaned pairs.
So the fault most likely is **`publish_loan()` running with `loan_addr_` (or `copy_`) null**.

Two ways that happens, both worth checking first next time:

1. `loan_ready()` allocates `loan_addr_` and is only ever entered when `loan_state_ == 0`; after any path that
   leaves `loan_state_ == 1` with `loan_addr_` freed or never set (the `~Prompt`/`set_prompt_arena(nullptr, 0)`
   teardown of the *other* session, and the model's own release path), `submit_loan` still passes its guard
   (`loan_state_ == 1 && loan_arena_`) while `loan_addr_` is null.
2. `copy_` is sized by the copy machinery; `publish_loan` indexes it before its first `if (copy_[pair])`.

Also from that reading, a real ordering bug independent of the crash: **`publish_loan` writes the blob-table
entries on the publishing session's stream and returns without syncing**, while `recall_loan` zeroes them on the
recalling session's stream; nothing orders a *reader* after a publish. `join_placement` covers the recall only.
That should be fixed regardless (`cudaStreamSynchronize` after the publish, or a publication event the readers
wait on).

**P0-2 is parked with the loan single-session**, per the review session's rule: the trace did not name the fault
inside the time box.