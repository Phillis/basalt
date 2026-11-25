# Review: upstream `31174039` against our fork (`f0e377ff`) - analysis only, nothing merged

## 1. What "the new upstream build" actually is

There is **no new GitHub release**: the latest is still **v0.5.7** (`400b411c`, 25 Nov 02:18 UTC). The
`v0.5.8/9/10-basalt` tags on that releases page are *ours*, not upstream's.

What did move is **`origin/main`**, now `31174039` (25 Nov 04:06 UTC) - **79 commits past v0.5.7**, unreleased. That
is the new code, and it is a large body of work: **76 commits since our merge base `3ce7d52e`**, by area:

| area | commits |
|---|---|
| docs / tooling / ci / arm64 | 21 |
| other (community, factory, spec) | 19 |
| **inkling2** | **16** |
| **kernels / core / prefill** | **10** |
| **serve** | **10** |

## 2. What upstream brings that lands on exactly our bottlenecks

| commit | what | why it matters here |
|---|---|---|
| `6450a411` #652 | **16-row batch windows, multi-session batched DSpark**, bitwise multi-tile Q8/GQA MMA | the two-slot drafter problem (our acceptance/dl loss) |
| `56473ab5` #674 | **16-row single-pass verify window**, hybrid DSpark/SuffixDrafter continuation | fewer verify rows = less wasted pool traffic (our Item 2's target) |
| `f9e689df` #673 | shared-KV MMA decode attention extended to **sliding-window layers**, fused batch segments | decode attention |
| `071018a6` #669 | staging wait deferred to the first expert product, **cross-chunk staging overlap** | our Item 6 / P2 - done properly |
| `5cb82de4` #657 | **128x128 TMA Q8 GEMM**, SMEM-coalesced controller epilogue, cp.async `f32_rows_dot` | prefill compute |
| `a035d98c` #651 | **4-stage cp.async pipeline + LPT scheduling** in prompt attention | prompt attention |
| `01762a7a` #579 | the Blackwell sm_120 prefill/decode kernels (+7.3-8.3% prefill, +1.2-4.1% decode, upstream's claim) | we took only 5 of its 8 commits |
| `95024237` #606 | INT8-cache decode attention | we skipped it |
| `#605`, `#697` | room-bound RAM copy; `--vram-reserve-mib` honoured at load | our VRAM findings |
| `fae1575d` #654 | a read of up to 64 tokens runs as decode windows | our `kPromptPathMin` |
| `b843142d` #642, `130a6d76` | park chats on a shared prefix; disk tier room check | our R1/R2 area |

## 3. What of ours upstream already has (redundant, no need to carry)

- **#650 / #649** - the same hunks appear on both sides with identical sizes (e.g. `core/elastic_array.cpp` +42/-10
  on each, `progress.hpp` +13/0 on each): upstream merged the PR we had ported by hand.
- **#609 routed staging, the staged-ahead copy stream, per-slot D2D staging** - upstream's `b78aadaa`, `72961b05`,
  `071018a6`.
- **#636** - upstream's `9acd845f`.
- **#579** - upstream has all 8 commits (`01762a7a`); we took 5.
- **The reply-role token and the `n-1` turn checkpoint** - present on both sides.
- Constants that match: `kGqaMaxWindow 32`, `kElasticKeep 512`, `kTasksPerThread 12`, `kBatch 32`.

## 4. What is uniquely ours and must be re-ported

1. **The four-bit value cache (`kv=fp8v4`)** - seven markers in our tree, **zero upstream**. It is the reason 262,144
   tokens fit; without it our pack's 256k config is not possible. Touches `kernels.hpp`, `kernels/attn.cu` (ours +230
   lines), `kernels/attn_prompt.cu` (+83), `session.cpp`, `prompt.cpp`, the config schema and the two parity suites.
2. **`cell_bytes_at()` - the K/V cell-size fix.** Upstream charges `cell_bytes()` to *every* elastic array
   (`a->ensure(cell_bytes() * target)`, `a->growth(cell_bytes() * positions)`). Harmless for their fp8-only KV where a
   V row equals a K row; **wrong under our v4**, where a V row is 80 B against 128 - a slot would stop at 62.5% of its
   context exactly as ours did before the fix. So the v4 port *must* carry this.
3. **`kCardSpare` 768 MiB** (upstream 1024): our measured +6% decode / +2% prefill, taken before #697/#605 rewrote
   the plan - it must be re-measured on their plan, not assumed.
4. **`--prompt-chunk`** (upstream: no such flag, a fixed 4096): our runtime knob and its 6144/5120 default. Their
   `071018a6` cross-chunk staging overlap changes that trade, so the value needs re-measuring.
5. **Item 2 - the round's cost charged per slot by rows** (`ms_share` in `batch_round.cpp`). Upstream barely touched
   that file (12/6), so the idea still applies, but #652/#674 replace the window-shape logic it sits next to: re-derive
   and re-measure rather than patch blind.
6. **R1's yardstick** (`restore_image` comparing against what `running_state()` produces): upstream still computes
   `running_bytes()` in that path, so their mount may carry the same bug - check before porting.
7. Our harnesses (`scenA.py`, `curve.py`, `stress_two_slot.py`, `agent_turns.py`, `needle*`), the configs and these docs.

## 5. The conflict surface, and why a naive merge is dangerous

- **64 files changed on both sides.** The heavy ones, upstream vs ours (added/removed lines):
  `session.cpp` +574/122 vs +534/72 · `model.cpp` +677/117 vs +636/100 · `attn.cu` +957/141 vs +230/54 ·
  `attn_prompt.cu` +480/222 vs +83/11 · `prompt.cpp` +189/28 vs +198/20 · `kernels.hpp`, `batch_round.cpp`,
  `session_engine.cpp`, `request_*.cpp`, `spill.*`, `dspark*`.
- Net of the two: **201 files, +4,582 / -12,588** - we are *missing* ~12.6k lines of their work and carry ~4.6k of
  ours, a third of which duplicates theirs.
- The overlap is exactly the machinery where a wrong resolution silently feeds wrong weights - the failure mode this
  session hit twice (the router buffer, and the stale graph key). A merge must be gated, not eyeballed.

## 6. Recommended plan (for when you want it done)

**Take upstream as the base, then re-port our four unique items** (their new work is the superset, and it targets the
two bottlenecks we measured: pool-bound rounds and 38%-exposed staging):

1. Branch from `origin/main`; build and gate it *as upstream ships it* first, and measure (that number is the bar).
2. Port, one commit each, each with its own gate and measurement:
   a. **v4 KV + `cell_bytes_at`** - the biggest and most invasive (6 source files + parity tests). Gate:
      `kernels_parity`, `prompt_parity`, needles 3/3 at 64k and 120k, 262,144 still loads, and the 83k-token prompt.
   b. **`kCardSpare` 768** vs their 1024 on the merged plan - keep only if it wins outside 3-5%.
   c. **`--prompt-chunk`** and its value, at their new staging overlap.
   d. **Item 2 re-derived** on their batch path.
   e. **R1's yardstick** if their mount still rejects with "another size".
3. Full gate after each: parity suites (theirs and ours), needles, stress 20/20, both decode harnesses, the 30k-120k
   curve. Only then choose the config defaults and cut a release.

The alternative - `git merge origin/main` into our branch, preferring theirs wherever they overlap - reaches the same
place with more conflict work and less control over which of their rewrites lands.

## 7. What to expect

Their #652/#674/#673/#657/#651/#579 work attacks precisely our two measured bottlenecks. The numbers to beat are ours:
**43-46 tok/s per stream** at two slots (prose: 35-47 by depth), **1,190-1,705 t/s prefill per stream**, 38% of a
prefill chunk exposed to staging, and the round budget's 42.5% CPU-pool merge wait. If upstream's batch windows and
batched DSpark are what they claim, that wait is the first thing that should move.

Nothing in this review has been merged. The tree is clean at `f0e377ff`; `origin/main` is fetched and available as
`31174039`.