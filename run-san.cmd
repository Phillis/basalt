@echo off
cd /d C:\LLM\rebase\b
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\compute-sanitizer\compute-sanitizer.exe" --tool memcheck --target-processes all --launch-timeout 600 --log-file C:\LLM\rebase\b\sanitizer.log bin\basalt.exe serve --config basalt_config.json --python C:\LLM\Basalt\.venv\Scripts\python.exe --api-key-file C:\LLM\Basalt\api-key.txt >> server.out 2>> server.err
