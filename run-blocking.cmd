@echo off
cd /d C:\LLM\rebase\b
set PATH=C:\LLM\rebase\b\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;%PATH%
set CUDA_LAUNCH_BLOCKING=1
bin\basalt.exe serve --config basalt_config.json --python C:\LLM\Basalt\.venv\Scripts\python.exe --api-key-file C:\LLM\Basalt\api-key.txt %* >> server.out 2>> server.err
