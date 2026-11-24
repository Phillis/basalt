<#
  bench.ps1 - a repeatable A/B harness for this engine, following the rules that matter from Strata's standard
  benchmark protocol (docs/BENCHMARK_PROTOCOL.md) adapted to what our engine reports.

  Why each rule is here:
  - the warm-up window is discarded: the first requests after a load run 30-40% slow (measured on this box: 61.2 then
    107.7 tok/s), so a measurement that includes them is measuring the cold path;
  - every run uses a FRESH unique prompt, and a prompt that shows any cached tokens is rejected, because cache reuse
    changes what is being timed;
  - rates come from the ENGINE's own timings (the server's `s1 done pp=... tg=...` line), never from client wall clock;
  - repetitions are reported as median / min / max / n, never a single number and never a best-of;
  - a generation that stops early (end=stop) is still a valid speed sample but is flagged, since token counts differ.

  Usage:
    .\bench.ps1                        # 3 runs: ~15.4k unique prefill + 400-token prose decode
    .\bench.ps1 -Runs 5 -DecodeTokens 300 -Label "after-change"
    .\bench.ps1 -SkipPrefill           # decode only (faster), still 3 runs
#>
[CmdletBinding()]
param(
    [int]    $Runs        = 3,
    [int]    $Words       = 4600,     # ~15.4k tokens at 3.34 tokens/word here
    [int]    $DecodeTokens = 400,
    [string] $Label        = 'current',
  [int]    $CacheFloor   = 16,        # the chat template's tokens, which the engine counts as reused on every prompt
    [string] $Base         = 'http://127.0.0.1:8080',
    [string] $Log          = 'C:\LLM\rebase\b\server.out',
    [string] $KeyFile      = 'C:\LLM\Basalt\api-key.txt',
    [switch] $SkipPrefill
)

$ErrorActionPreference = 'Stop'
$key  = (Get-Content $KeyFile -Raw).Trim()
$hdr  = @{ Authorization = "Bearer $key" }
$model = 'Inkling-2-Mini'

function Send-Chat([string]$prompt, [int]$maxTokens) {
    $b = @{ model = $model; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $maxTokens; temperature = 0 } |
         ConvertTo-Json -Depth 6
    $null = Invoke-RestMethod "$Base/v1/chat/completions" -Method POST -Body $b -ContentType 'application/json' `
                             -Headers $hdr -TimeoutSec 1800
    # the engine's own line for that request
    (Get-Content $Log | Select-String -Pattern 's1 done  pp=' | Select-Object -Last 1).Line
}

function New-Prompt([int]$n) {
    # unique every call, so nothing can come from the prompt cache
    $words = 1..$n | ForEach-Object { -join (1..6 | ForEach-Object { [char](Get-Random -Minimum 97 -Maximum 123) }) }
    $tag = -join (1..12 | ForEach-Object { [char](Get-Random -Minimum 97 -Maximum 123) })
    "Read the following list, then reply OK. (session $tag)`n`n" + ($words -join ' ')
}

function Stats($xs) {
    $s = $xs | Sort-Object
    $n = $s.Count
    $med = if ($n % 2) { $s[[int](($n - 1) / 2)] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
    [pscustomobject]@{ median = [math]::Round($med, 1); min = [math]::Round($s[0], 1); max = [math]::Round($s[-1], 1); n = $n }
}

# ---- warm-up, discarded ---------------------------------------------------------------------------------------
Write-Host "warm-up (discarded)..." -ForegroundColor DarkGray
$null = Send-Chat 'Warm up: write 200 words of continuous prose about the sea.' 320

$prefill = @(); $decode = @(); $notes = @()
foreach ($i in 1..$Runs) {
    Write-Host ("run {0}/{1}" -f $i, $Runs) -ForegroundColor Cyan
    if (-not $SkipPrefill) {
        $line = Send-Chat (New-Prompt $Words) 4
        $m = [regex]::Match($line, 'pp=(\d+)/(\d+)c [\d\.]+s (\d+)t/s')
        if (-not $m.Success) { $notes += "run ${i}: no pp in [$line]"; continue }
        if ([int]$m.Groups[2].Value -gt $CacheFloor) { $notes += "run ${i}: prefill prompt came from cache ($($m.Groups[2].Value) cached, floor $CacheFloor) - rejected"; continue }
        $prefill += [double]$m.Groups[3].Value
    }
    $line = Send-Chat 'Write 400 words of continuous prose about rivers.' $DecodeTokens
    $m = [regex]::Match($line, 'tg=(\d+) ([\d\.]+)t/s')   # an early stop prints no acc/dl
    if (-not $m.Success) { $notes += "run ${i}: no tg in [$line]"; continue }
    $decode += [double]$m.Groups[2].Value
    if ($m.Groups[1].Value -ne "$DecodeTokens") { $notes += "run ${i}: stopped at $($m.Groups[1].Value) tokens (end=stop) - counted, flag it when comparing" }
}

Write-Host ""
Write-Host "=== $Label ===" -ForegroundColor Green
if ($prefill.Count) { $ps = Stats $prefill; "  prefill  median {0,7:N1} t/s   min {1,7:N1}   max {2,7:N1}   n={3}" -f $ps.median, $ps.min, $ps.max, $ps.n }
if ($decode.Count)  { $ds = Stats $decode;  "  decode   median {0,7:N1} t/s   min {1,7:N1}   max {2,7:N1}   n={3}" -f $ds.median, $ds.min, $ds.max, $ds.n }
foreach ($n in $notes) { Write-Host "  note: $n" -ForegroundColor Yellow }
if (-not $prefill.Count -and -not $decode.Count) { Write-Host "  no valid samples" -ForegroundColor Red }
