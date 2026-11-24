@echo off
rem Start Basalt (optimised build: 192k + DSpark + doorbell hand-off, ~63-68 tok/s decode).
rem The real work is in Start-Basalt.ps1; this is the double-click entry point.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\LLM\Basalt\Start-Basalt.ps1" %*
