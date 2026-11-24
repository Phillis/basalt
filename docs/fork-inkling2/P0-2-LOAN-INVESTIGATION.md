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
