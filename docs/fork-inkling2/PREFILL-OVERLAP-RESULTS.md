# Prefill: staged-ahead copies + --prompt-chunk
# Session 61 - prompt processing, second pass: the staging rides a copy stream, and `--prompt-chunk`

## The measurement that made this possible

`BASALT_DIAG_PROMPT=1` (re-applied; an earlier checkout had reverted it, so 456c02cf did not carry it) prints
per chunk what actually moves:

```
prompt diag: 3848 rows, 48 visits, h2d 0.00 + full 22.04 GiB, d2d 0.00 (1502.7 ms at 15.75 GB/s), 48 full copies
```

The long-chunk path stages **every RAM expert of every spilled layer**, once per chunk: 22.0 GiB at 3,848 rows,
25.3 GiB at 7,733. And the staging is **one allocation every spilled layer shares** (spill.hpp), so a layer's
copies could not start until the layer before it had finished reading them.

## The port (91bdae93, by hand)

That commit does not merge - our prompt.cpp has since taken #609's routed staging and #650's copy machinery -
so it was ported edit by edit, keeping the void signatures and adding a safe fallback:

- `Lane` grows a non-blocking `copy` stream and `freed`/`staged` events; `init()` builds `next_spill_[step]` and
  the first spilled step, only when there is one card and no peer.
- `chunk()` stages the first spilled layer before the embedding is even queued.
- `experts()` waits on `staged` instead of copying inline; after its products are queued, `staging_read()`
  starts the **next** spilled layer's copies, which wait on `freed` (recorded after those products).
- any failed queue falls back to staging inline for the rest of the chunk (`stage_ok_`/`ahead_`), so a failed
  event can never let a product read a half-written staging; `run()` drains the copy stream as well.

## `--prompt-chunk N` - the knob, and why an "in-between" size behaves as it does

kPromptChunk was compile-time, so every value cost a rebuild. It is now an inline int set by `--prompt-chunk`
(1,024..16,384) before the plan is laid, with `kChunk` a reference to it. Measured with bench.ps1 (4,600 fresh
words, 400 decode tokens, 2-4 processes each; ~15.4k-token prompts, so the pass count changes across the range):

| chunk | cuda0 | passes | prefill | decode | hit |
|---|---|---|---|---|---|
| 4,096 | 2.7 GiB | 4 | 1,976 t/s | 99.5 | 94% |
| 5,120 | 2.4 GiB | **4** (15,371/5,120 = 3.002) | 1,894 t/s | 98 | ~92% |
| 6,144 | 2.1 GiB | 3 | 2,331 t/s | 92.6 | ~89% |
| 7,168 | 1.8 GiB | 3 | 2,280 t/s | 88.3 | ~93% |
| 8,192 | 1.6 GiB | **2** | **3,018 t/s** | 84.8 | ~85% |

Prefill is dominated by **the number of staging passes**, `ceil(tokens / chunk)`, not by the chunk size itself:
within a pass count the *smaller* chunk wins (more expert copies stay on the card, so less is staged - 4,096
beats 5,120 and 6,144 beats 7,168). The step from 6,144 to 8,192 is the step from three passes to two. So the
best value for a given prompt length is the smallest chunk that still yields the fewest passes, and the flag
turns that choice into a config line.

## Where it stands

| arm | prefill | decode |
|---|---|---|
| this morning (chunk 4,096, no overlap) | 1,533 t/s | 101.8 tok/s |
| chunk 4,096 + overlap | 1,976 t/s | 99.5 tok/s |
| chunk 6,144 + overlap | 2,331 t/s | 92.6 tok/s |
| **chunk 8,192 + overlap (now live)** | **3,018 t/s** | 84.8 tok/s |

**Prefill is 1.97x this morning's**, decode pays ~17%. Launcher passes `--prompt-chunk 8192` and honours
`BASALT_PROMPT_CHUNK`, so the decode-first setting needs no rebuild.

Gate (chunk 8,192 + overlap, through the launcher): Gate 0 six questions correct, needles 872 / 1,634 / 3,971 /
**70,585** all EXACT, prompt_parity 0 failures, kernels_parity 0 failures, gqa_window_parity PASS.

## The next lever, precisely

At 8,192 rows the prompt arena takes 1.1 GiB of card room (`cuda0` 2.7 -> 1.6 GiB), and that is exactly the
decode hit rate (90.3% -> ~85%) and decode (102 -> 85). The arena is sized by `carve()` at `R = kChunk`: the
biggest single item is the expert products' workspace (`gu` = pairs x 2 x n_ff_exp floats, 604 MiB at 8,192
rows; `h` = 302 MiB). Splitting *only* the expert products into two row halves - the layer is still staged once,
which is where the 1.97x comes from - would return ~900 MiB and with it the decode hit rate, making 8,192 rows
affordable at 4,096 rows' card cost. That is the next change to try.
