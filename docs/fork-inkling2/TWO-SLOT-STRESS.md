# Two-slot stress: a prompt in one slot while the other decodes

This is the test that found the bug which kept two 128k slots from being usable together with prompts. Run it
against a `parallel 2` engine; each round sends a 600-token generation in one session and, 0.8 s later, a
~20k-token prompt plus a short generation in the other - the exact overlap a real agent client produces.

    python tools/stress_two_slot.py 6

## What it caught

**Enabling the prompt loan with more than one session crashes.** `loan_ready()` requires `sessions() == 1`,
and relaxing that - even with a `prompt_busy_` flag that stops `submit_loan()` during a read - crashed with
`EXCEPTION_ACCESS_VIOLATION ... in reading the prompt` on the *second* round, every time, because the loan's
VA range is the prompt arena and one session's decode keeps copies loaned in it across the other's read.

The same test at the committed state runs 6/6 clean:

    round 1..4: decode-session 23.7 / 23.2 / 23.6 / 23.4 tok/s   prompt-session ~20k tokens at 1,017-1,057 t/s

So: a prompt in one slot while the other decodes is safe on the shipped build, with the loan restricted to a
single session. Anything that relaxes that gate has to be tested with this script first.

## What it measures, and why the numbers are lower than the isolated ones

| scenario | decode per stream | prefill per stream |
|---|---|---|
| isolated (two prompts, no decode) | - | 1,252 t/s |
| isolated (two generations) | 36.5 tok/s | - |
| **this overlap (prompt + decode at once)** | **23.5 tok/s** | **~1,030 t/s** |

They contend for the one host link and the one GPU, which is the honest number for agent-style use.
