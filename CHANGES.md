
# Tier 0 and Tier 1 of SPEC-2SLOT-TIER1-TIER2.md (2026-11-25)

Reference commit `7abd762e`. Engine 2 x 128k, chunk 5,120, reserve 256, prompt-cache 6, spec 8 unless stated.
Every arm is one engine process from a cold stop, warm-up, then the scenario; truth is the engine's own
`done pp=... tg=... acc=... dl=... hit=...` lines.

## Tier 0: the two gates nothing had covered

**Needles at depth, both slots** (`needles3.py`: three passphrases at 10 / 50 / 90% of the depth in one prompt,
read into one slot while the other decodes 400 tokens):

| depth | result |
|---|---|
| 64,049 tokens | **3/3 exact** |
| 119,985 tokens | **3/3 exact** |

(The first attempt reported 0/3 at "120k" because the harness built only 66,871 tokens - a harness undershoot,
not a quality miss; with the depth fixed it is 3/3 and the answers list all three passphrases.)

**Cancel mid-generation** (client socket closed one second into a streamed generation, five times, while the
other slot decoded): the decoding slot's stream finished intact, the next request in the cancelled slot worked,
**0 crash lines, 0 restarts**.

## Tier 1: the flags, and the answer is that the defaults stand

**`spec`** (scenario A: both slots decoding 400 tokens of prose at once, engine `tg` per stream):

| arm | per stream | aggregate | acc | hit |
|---|---|---|---|---|
| **8 (base)** | **34.0** | 68.1 | 34/51 | 93.8% / 73.4% |
| 6 | 37.8, then **32.2** in a second process | 75.6 / 64.4 | 33-38 | 90.5-94.9% |
| 5 | 36.9 | 73.9 | 67 | 74.2% |
| 4 | 32.9 | 65.8 | 64 | 93.7% |

`spec 6` measured +11% and then -5% across two processes: not reproducible **outside the 3-5% run-to-run noise**
this box shows on decode. `spec 5` is the same shape. **The default stays 8** - the spec's own prior (+5-15% from
fewer wasted drafts) is not visible here, which is itself the datum 2B needs.

**`suffix-draft`** (same scenario, with `spec 6`): 3 (base 34.0), 8 -> 33.8, 16 -> 34.9. Neutral on fresh prose,
exactly as the spec predicted ("0 on fresh prose"); it is agent-like text that was never in this scenario, so the
arm does not test it. Default stays 3.

**`prompt-chunk`** (scenario B: two 30k prompts read at once, prefill per stream):

| chunk | prefill per stream | plan |
|---|---|---|
| **5,120 (base)** | **1,217** | cuda0 2.2 GiB |
| 6,144 | 1,283, then 1,230 in a second process | 2.0 GiB |
| 7,168 | 1,271 | 1.7 GiB |

6,144 is +5.4% then +1.1% (median +3.2%) - inside the noise band, and scenario A at 6,144 measured 32.4 tok/s
against 34.0 at 5,120, so the larger arena does still cost a little decode. **The default stays 5,120.**

`pool-workers` was not run: the spec's own prior is neutral (one core streams ~13 GB/s, the box is at its 52 GB/s
ceiling by 4-5 workers) and the earlier single-slot sweep measured it neutral. `prompt-cache` 6 vs 2 was measured
neutral earlier tonight (identical `ram 13.0 GiB`).

**Conclusion of Tier 1: no flag beats the base outside noise on this box.** The canonical config is unchanged
(`basalt_config.json`: 2 x 128k, `spec 8`, `suffix-draft` default, chunk 5,120, reserve 256, no `reasoning-effort`).
That is a result, not a failure: it says the remaining decode gains have to come from Tier 2 (the loan's heat
accounting, the rejected-draft policy, interleaving), not from tuning.

## Not done: Tier 2

2A (loan accounting across sessions), 2B (rejected drafts), 2C (interleaving), 2D (prefill timeline) are **not
started** - the session's context budget was spent on Tier 0 and Tier 1. The spec's order stands, and 2A is the
one it calls "a sure gain": the loan's heat still counts only the publishing conversation (hit 82.8% with both
slots decoding against 93% with one), and the fix is the four steps in section 5.2A.