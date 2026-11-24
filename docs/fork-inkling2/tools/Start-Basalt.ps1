#Requires -Version 5.1
<#
  Start-Basalt.ps1 - start Basalt with Inkling-2-Mini at 196,608 tokens (192k) + DSpark drafting.

  This is the launcher for the OPTIMISED build (C:\LLM\pr536\basalt-inkling-cpu-spill-review):
  PR #536 + #525, pool pinning, W1 staging reclaim and the W2 doorbell hand-off.  Measured on this
  machine: ~63-68 tok/s decode, ~54 sustained, ~1.5k tok/s prefill (stock v0.5.5 gets 11.3).

  It copies C:\LLM\Basalt\basalt_config.json into the engine tree, so THAT file is the one to edit.
  The engine is launched directly (not through the scheduled task - with this build the task context
  fails device selection) and detached, so closing this window leaves it serving.

  Usage:  Start-Basalt.cmd            (or Start-Basalt-Fast.cmd, same thing)
          Start-Profile.cmd <config>  (switch profile first)
#>
[CmdletBinding()]
param(
    [switch]$Force,        # skip the Strata prompt / free-VRAM warning
    [int]$WaitSeconds = 180
)

$ErrorActionPreference = 'Continue'
$Root     = 'C:\LLM\Basalt'
$Engine   = 'C:\LLM\rebase\b'
$ConfigSrc = Join-Path $Root  'basalt_config.json'
$ConfigDst = Join-Path $Engine 'basalt_config.json'
$KeyFile  = Join-Path $Root  'api-key.txt'

function Say($m, $c = 'Gray') { Write-Host $m -ForegroundColor $c }
function Stop-Engine {
    Get-Process basalt -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -like 'C:\LLM\*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Say '=== Basalt / Inkling-2-Mini ===' 'Cyan'
Say '    optimised build: PR #536 + #525, pinned pool, W1 staging reclaim, W2 doorbell hand-off' 'DarkGray'
Say ''

# ---- preflight ------------------------------------------------------------------------------
foreach ($p in @($Engine, $ConfigSrc, $KeyFile)) {
    if (-not (Test-Path $p)) { Say "MISSING: $p" 'Red'; Read-Host 'Press Enter to exit'; exit 1 }
}
$model = (Get-Content $ConfigSrc -Raw | ConvertFrom-Json).models[0].model
if (-not (Test-Path $model)) { Say "MISSING model file: $model" 'Red'; Read-Host 'Press Enter to exit'; exit 1 }
if (-not (Test-Path (Join-Path $Engine 'bin\basalt.exe'))) {
    Say "MISSING engine: $Engine\bin\basalt.exe  (build it with C:\LLM\rebuild.bat)" 'Red'
    Read-Host 'Press Enter to exit'; exit 1
}

# ---- free VRAM (the card needs ~10.7 GiB of its 11.9; Strata will hog it) ----------------------
$free = 0
try { $free = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1) } catch { }
Say "free VRAM: $free MiB"
if ($free -lt 11000 -and -not $Force) {
    if (Get-Process strata -ErrorAction SilentlyContinue) {
        Say 'The Strata engine is holding the GPU (~10.8 GB).' 'Yellow'
        if ((Read-Host 'Stop strata.exe? (y/n)') -match '^[Yy]') {
            Stop-Process -Name strata -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 5
            $free = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
            Say "free VRAM now: $free MiB" 'Green'
        }
    }
    if ($free -lt 11000) { Say "WARNING: only $free MiB free; the load may fail (needs ~10.7 GiB)." 'Yellow' }
}

# ---- hand-off mode (doorbell is the default; marker restores the old path) ---------------------
$mode = if (Test-Path (Join-Path $Root 'cpu-spill-hostfunc.on')) { 'hostfunc (marker file)' } else { 'doorbell' }
Say "hand-off: $mode"

# ---- restart cleanly with the canonical config ------------------------------------------------
$running = (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue) -ne $null -or
           [bool](Get-Process basalt -ErrorAction SilentlyContinue)
if ($running -and -not $Force) {
    Say 'A Basalt server is already running on 8080; it will be restarted.' 'Yellow'
}
Stop-Engine
Start-Sleep -Seconds 6
Copy-Item $ConfigSrc $ConfigDst -Force
$ctx = (Get-Content $ConfigDst -Raw | ConvertFrom-Json).models[0]."max-context"
$spec = (Get-Content $ConfigDst -Raw | ConvertFrom-Json).models[0].spec
Say "config: ctx=$ctx  spec=$spec  (from $ConfigSrc)"

Remove-Item (Join-Path $Engine 'server.out'), (Join-Path $Engine 'server.err') -ErrorAction SilentlyContinue

# ---- launch: direct, in this session, detached (the scheduled task breaks device select) ------
$log = Join-Path $Engine 'server.out'
# [prefill] prompt chunks of 8,192 rows: 2 staging passes for a ~15k prompt instead of 4 (prefill 3,018 vs 1,976 t/s).
# Lower it (e.g. 4096) when decode matters more than prefill: 4096 gives decode 99 but 4 passes.
$PromptChunk = if ($env:BASALT_PROMPT_CHUNK) { $env:BASALT_PROMPT_CHUNK } else { '8192' }
Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', ((Join-Path $Engine 'run-pr.cmd') + " --prompt-chunk $PromptChunk") -WindowStyle Minimized
Say 'loading the 18.4 GiB pack (about 40 s)...' 'Yellow'

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$ok = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $out = Get-Content $log -Raw -ErrorAction SilentlyContinue
    if ($out -match 'error:') {
        Say ('FAILED: ' + (($out -split "`n" | Where-Object { $_ -match 'error:' } | Select-Object -Last 1).Trim())) 'Red'
        break
    }
    try { $h = Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 4
          if ($h.loaded) { $ok = $true; break } } catch { }
    if (-not (Get-Process basalt -ErrorAction SilentlyContinue)) { Say 'engine exited; see server.out' 'Red'; break }
}

# ---- report ------------------------------------------------------------------------------------
if ($ok) {
    $h = Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 5
    $key = ''
    if (Test-Path $KeyFile) { $key = (Get-Content $KeyFile -Raw).Trim() }
    $ip = (Get-NetIPAddress -AddressFamily IPv4 |
           Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.InterfaceAlias -notmatch 'WSL|vEthernet' } |
           Select-Object -First 1).IPAddress
    Say ''
    # ---- warm up before reporting ready --------------------------------------------------------------
    # The first requests after a load run ~30-40% slow: the allocator and the expert residency settle, and the
    # decode graphs for each window shape are captured on first use. Measured on this box (engine log):
    # 61.2 then 107.7 tok/s, 56.3 then 108.3, 62.7 then 108.7. Two short generations plus one longer one
    # cost ~10 s at startup and take that penalty off the user's first request. A property of this engine,
    # not a benchmark trick: the same effect is documented by the DeepSeek-v4 DSpark recipe (58.5 -> 83.3).
    if ($key) {
        $warmup = @(
            @{ role = 'user'; content = 'Warm up: count from 1 to 60 in words, one line each.' },
            @{ role = 'user'; content = 'Warm up: write 300 words of continuous prose about the sea.' },
            @{ role = 'user'; content = ('Warm up cache pass: ' + ('lorem ipsum dolor sit amet consectetur adipiscing elit ' * 160) + ' Now reply OK.') }
        )
        foreach ($m in $warmup) {
            try {
                $body = @{ model = $h.model; messages = @($m); max_tokens = 320; temperature = 0 } | ConvertTo-Json -Depth 6
                $null = Invoke-RestMethod 'http://127.0.0.1:8080/v1/chat/completions' -Method POST -Body $body `
                                           -ContentType 'application/json' -Headers @{ Authorization = "Bearer $key" } -TimeoutSec 300
            } catch { Say '  (warm-up request failed; continuing)' 'Yellow' }
        }
        Say '  warmed up' 'DarkGray'
    }
    Say '================ READY ================' 'Green'
    Say ("  model    : {0}" -f $h.model) 'Cyan'
    Say ("  context  : {0:N0} tokens" -f [long]$h.max_context) 'Cyan'
    Say ''
    Say ("  local    : http://127.0.0.1:8080/v1") 'Cyan'
    if ($ip) { Say ("  network  : http://{0}:8080/v1" -f $ip) 'Cyan' }
    Say ("  model id : {0}" -f $h.model) 'Cyan'
    Say ("  auth     : Authorization: Bearer <key from {0}>" -f $KeyFile) 'Cyan'
    Say ("  chat UI  : http://127.0.0.1:8080/") 'Cyan'
    Say ''
    Say '  stop     : Stop-Basalt.cmd' 'DarkGray'
    Say '  monitor  : C:\LLM\basalt-monitor.cmd' 'DarkGray'
    Say '=======================================' 'Green'
    if ($key) { Say ("  api key  : {0}...{1}" -f $key.Substring(0, 8), $key.Substring($key.Length - 4)) 'DarkGray' }
} else {
    Say ''
    Say 'not ready - see C:\LLM\pr536\basalt-inkling-cpu-spill-review\server.out' 'Yellow'
}
Say ''
Read-Host 'Press Enter to close this window'
