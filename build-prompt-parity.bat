@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
cd /d C:\LLM\rebase\b
"%CMAKE%" --build build-test --target inkling2_prompt_parity -j 12
exit /b %errorlevel%
