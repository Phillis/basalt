# Memory locking ("Lock pages in memory") — what was done, and why it doesn't matter in the final setup

Record of the page-locking work done while bringing up Basalt / Inkling 2 Mini on Windows 11, including the
final correction: **the privilege that was granted is inert on the configuration now in use.**

Related: [`PERFORMANCE-RESULTS.md`](PERFORMANCE-RESULTS.md) (what actually made it fast),
[`CHANGES.md`](CHANGES.md) (full chronological log, §2.1).

---

## 1. What "locking" is needed for

Inkling 2 Mini is 16.3 GiB of weights on an 11.9 GiB card, so most MoE experts live in system RAM and must be
readable by the GPU (or, with the CPU-spill path, readable by the CPU). Holding that memory in a fixed,
non-pageable state is what "locking" refers to. Windows has two distinct mechanisms, and confusing them
costs hours:

| Mechanism | Capped by | Privilege |
|---|---|---|
| `VirtualLock` (user-mode API) | the process's **minimum working set** | `SeLockMemoryPrivilege` alone is **not** enough |
| CUDA host allocation (`cuMemCreate` / `cudaHostAlloc`, "share ram") | driver / WDDM budget | none (driver pins in kernel) |

Basalt uses the **second** one for its RAM tier. That is the single most important fact in this document.

---

## 2. What was done, step by step

### 2.1 The symptom

At `--max-context 262144` the engine aborted during planning:

```
[server] error: inkling2: cannot pin 14515 MiB of RAM for the experts that leave the card
```

### 2.2 Measured the default lock ceiling

`VirtualLock` probe from PowerShell: 4 KB, 16 KB and 64 KB succeed; **256 KB and above fail with
`ERROR_WORKING_SET_QUOTA` (1453)**. So the account could not lock more than ~128 KB.

### 2.3 Granted the privilege

Script: `C:\LLM\Basalt\grant-lock-pages.ps1`, launched **elevated** (`Start-Process -Verb RunAs`, UAC prompt).
It rewrites local security policy with `secedit`:

```
secedit /export  /cfg C:\LLM\Basalt\secpol\secpol.inf
# insert under [Privilege Rights]:
SeLockMemoryPrivilege = *S-1-5-21-3535324772-2023699042-432813517-1001,*S-1-5-32-544
secedit /configure /db C:\LLM\Basalt\secpol\secedit.sdb /cfg ...\secpol.inf /areas USER_RIGHTS
secedit /export  /cfg ...\verify.inf      → confirms the value took
```

Resulting policy line (verified by re-export):

```
SeLockMemoryPrivilege = Phil,*S-1-5-32-544
```

Two traps worth recording:

- **`/quiet` is not a valid switch for `/export`** (only `/configure`). The first elevated run exited 1 and
  produced no file; the fix was to drop `/quiet` and capture the output.
- **LSA caches privileges per logon session.** The change does not reach the current token even when a
  process is launched *elevated* afterwards, and a fresh scheduled-task logon did not help either.
  A **reboot** was required. After it, `whoami /priv` showed:

```
SeLockMemoryPrivilege   Lock pages in memory   Disabled
```

`Disabled` is the normal state — a process enables it at runtime with `AdjustTokenPrivileges`.

### 2.4 Discovered the privilege is not the binding constraint

Even with the privilege present **and enabled**, `VirtualLock` still failed at 1453 for every size tried
(1 MB, 64 MB, 512 MB, 4 GiB). The cap is the process's **minimum working set**, not the privilege.

Proof by experiment — raise the quota first, then lock:

```csharp
SetProcessWorkingSetSizeEx(GetCurrentProcess(), 8 GiB /*min*/, 12 GiB /*max*/, 0);   // ok
VirtualLock(6 GiB region);                                                          // LOCKED OK
```

### 2.5 Tried to exploit that, then reverted

Patched `serve/engine/lifeline.py` so the engine child got a large minimum working set at spawn
(`_raise_working_set()`, `18 GiB` min / `24 GiB` max, enabling `SeIncreaseWorkingSetPrivilege` in the
spawning process). An external watchdog that raced the engine (`wsboost.ps1`) was also tried and discarded.

**This was reverted** (`serve/engine/lifeline.py.orig` holds the pristine copy) because:

- it did not fix the failure (`cannot pin` persisted),
- 18 GiB of minimum working set on a 31 GB machine is a real regression risk, and
- the actual causes turned out to be elsewhere (see §3).

---

## 3. What the real blockers were

Neither involved page locking:

1. **`expert-cache: 128`** — a fixed slot count forced 2,944 of 3,072 experts into RAM (14.18 GiB), which
   exceeded what the engine could pin. `expert-cache: auto` fixed it immediately.
2. **PR #536's host-link gate** — `if (!(link > 0 && link <= 8e9)) return nullptr;` disabled the CPU-spill
   path on this machine (link measures **15.75 GB/s**), which was worth **3.2× decode** when removed.

---

## 4. Correction: at present, nothing is page-locked by this engine

Evidence from the binary actually in use
(`C:\LLM\pr536\basalt-inkling-cpu-spill-review\bin\basalt.exe`):

| Symbol / string | Occurrences |
|---|---|
| `VirtualLock` | 0 |
| `SetProcessWorkingSetSizeEx` | 0 |
| `cuMemHostRegister` | 0 |
| `SeLockMemoryPrivilege` | 0 |
| `cuMemCreate` | 1 |

and the engine's own load log shows how the RAM tier is realised:

```
basalt load: share ram 0 15582000848        ← CUDA host VMM, not VirtualLock
```

**Therefore the `SeLockMemoryPrivilege` grant is inert.** It confers a right that this build never
exercises. It is harmless, and it would matter only for a configuration that genuinely page-locks
(e.g. a much smaller card with a larger `ram-reserve-mib`), but it did not contribute to the current
performance or to the 256k load succeeding.

Note: the log also contains a *second* family of pinned-staging failures, from 12:47 onwards today (2026-11-22),
that are **not** from the 256k experiment above. Those are 192k loads whose spill plan spilled ~29 extra experts
(2796 of 3072, share ram 14.5 GiB) because the plan saw ~128 MiB less free VRAM, pushing the pinned total
(spill 13.56 + embedding 0.86 + read staging) just past the machine's ~14.6 GiB CUDA host-allocation quota. The
fix was `kCardSpare` 1024 → 896 MiB (a `[local]` patch in `model.cpp`), which restores the proven-loadable
2767-expert plan; a fresh-process probe pins the engine's exact allocation sequence without failure, so the
quota, not the machine, was the wall. The current server loads cleanly and serves at `ctx=196608`.

---

## 5. Verifying the current state

```powershell
# the privilege (present, Disabled = normal)
whoami /priv | findstr SeLockMemory

# it still cannot lock more than ~128 KB, and that is fine
# (VirtualLock is capped by the minimum working set; see §2.4)

# what the engine actually does for the RAM tier
Get-Content "$env:USERPROFILE\.cache\basalt\55cfc8c42932d132\engine.log" |
  Select-String 'share ram|cannot pin' | Select-Object -Last 3
```

---

## 6. How to revert the policy change

Through the GUI (needs admin):

```
secpol.msc → Security Settings → Local Policies → User Rights Assignment
          → "Lock pages in memory" → remove "Phil"
```

Or through the script's own mechanism, resetting the right to unassigned and re-applying:

```
secedit /export /cfg secpol.inf
# set:  SeLockMemoryPrivilege =
secedit /configure /db secedit.sdb /cfg secpol.inf /areas USER_RIGHTS
```

Either way it applies at the **next logon**. Nothing else on the machine depends on it — Basalt does not
use it, and the scheduled task `BasaltPR` does not need it.

---

## 7. Summary

| Question | Answer |
|---|---|
| Did the privilege get granted? | Yes — `SeLockMemoryPrivilege = Phil,*S-1-5-32-544`, effective after a reboot |
| Did it enable locking? | **No** — `VirtualLock` stayed capped at ~128 KB by the minimum working set |
| What does enable large locks? | Raising the process minimum working set (`SetProcessWorkingSetSizeEx`), proven with a 6 GiB lock |
| Is that in the shipped setup? | **No** — reverted; `lifeline.py.orig` kept |
| Does the engine lock memory today? | **No** — it uses CUDA host VMM (`share ram` / `cuMemCreate`) |
| What actually fixed the load and the speed? | `expert-cache: auto`, removal of PR #536's 8 GB/s link gate, PR #525, pool pinning |
