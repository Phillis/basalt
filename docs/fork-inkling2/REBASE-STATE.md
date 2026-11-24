# REBASE-STATE (latest)

| item | value |
|---|---|
| live engine tree | `C:\LLM\rebase\b` |
| branch / commit | `pr650-int` = `a6b5bc0d` (per-slot D2D prompt staging) (staged-ahead prompt staging + `--prompt-chunk`) (merge of `858f1741`, which is #636 + its 4 follow-ups on `f2c9b221`) |
| rollback commit | `b14526cb` (pre-#650, measured) |
| build | `C:\LLM\rebase\rebuild-rebased.bat` (log `build-650.log`); tests `rebuild-test.bat` |
| config | `basalt_config.json`: `kv=fp8v4`, `max-context 262144`, `spec 8`, `kCardSpare 768`, `kBatch`/`kGqaMaxWindow` 32, `kPromptChunk 4096`, `kPromptPathMin 96` |

## Measured (protocol harness, >=2 processes per arm)

| arm | prefill | decode |
|---|---|---|
| `b14526cb` | 1,533 t/s median | 67.0 tok/s median (59.1-72.7) |
| 2c9b221 (#650) | 1,529 t/s median | **101.8 tok/s median (90.1-108.4)** |
| 858f1741 (+ #636 family) | 1,530.5 t/s median | 102.35 tok/s median (94.6-105.7) - neutral, kept |

Decode +52%, prefill neutral. Expert cache hit rate 73.8% -> 90.3%.

## Open follow-ups (in priority order)

1. **Guess prefetch with an epoch guard** (#650 disabled it while copies exist): a guess must be tied to
   the placement epoch it was made in; worth the few % the L2 warm-up was worth. Needs the same
   discipline as the graph key fix: the epoch, not the address, decides whether a guess is still valid.
2. **Second-stream long-chunk staging** (`91bdae93`, 112 lines): expected prefill ~1,800 t/s and up to
   ~1 s off TTFT above 1,024 rows. Spec: `SPEC-609-ROUTED-STAGING.md`.
3. **#636** (decode attention, 516 lines, resolved in `xy-5080`), then **#606**.
4. `prompt.cpp` orchestration (~3.5% prefill), **KVA** (+11-18% prefill, project).
5. Constants with no strong prior: `kElasticKeep`, `kDecodeRows`, prompt chunks 2048/3072.

## Known traps

- `serve` ignores `--config` relative paths; it reads the tree's own `basalt_config.json`.
- A restore brings older mtimes: touch `src include` before rebuilding.
- `git checkout` between arms needs `-f`: `server.out` used to be tracked, and untracked files block it.
- Engine log per model hash: `C:\Users\Phil\.cache\basalt\<hash>\engine.log`.
