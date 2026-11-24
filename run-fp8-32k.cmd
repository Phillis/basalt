@echo off
cd /d C:\LLM\rebase\b
set PATH=C:\LLM\rebase\b\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;%PATH%
bin\basalt.exe serve --config test-fp8-32k.json --python C:\LLM\Basalt\.venv\Scripts\python.exe --api-key-file C:\LLM\Basalt\api-key.txt %* >> server.out 2>> server.err
