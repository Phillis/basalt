@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
cd /d C:\LLM\rebase\b
if not exist build-test (
  "%CMAKE%" -S . -B build-test -G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=inkling2 -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=ON || exit /b 1
)
"%CMAKE%" --build build-test --target inkling2_kernels_parity -j 12
exit /b %errorlevel%
