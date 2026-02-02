@echo off
REM Fixed Qwen3-VL CPU Server (Optimized + Stable)
echo ========================================
echo Qwen3-VL CPU Server - FIXED VERSION
echo ========================================
echo.

REM FIXED: Use correct stable files (NOT Q2_K)
set MODEL_PATH=%~dp0models_latest\model.gguf
set MMPROJ_PATH=%~dp0models_latest\mmproj.gguf
set LLAMA_CPP_PATH=C:\llama-cpu\llama-server.exe

REM Check llama-server.exe exists
if not exist "%LLAMA_CPP_PATH%" (
    echo ❌ ERROR: llama-server.exe not found!
    echo Download: https://github.com/ggerganov/llama.cpp/releases/latest
    echo Extract to: C:\llama-cpu\
    pause
    exit /b 1
)

REM Check model files (CRITICAL: Use Q3_K_M + Q8_0 pair)
if not exist "%MODEL_PATH%" (
    echo ❌ ERROR: Download qwen3-vl-2b-instruct-Q3_K_M.gguf
    echo Link: https://huggingface.co/enacimie/Qwen3-VL-2B-Instruct-Q3_K_M-GGUF/resolve/main/qwen3-vl-2b-instruct-Q3_K_M.gguf
    pause
    exit /b 1
)

if not exist "%MMPROJ_PATH%" (
    echo ❌ ERROR: Download mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
    echo Link: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
    pause
    exit /b 1
)

echo ✅ Model: %MODEL_PATH%
echo ✅ mmproj: %MMPROJ_PATH%
echo ✅ Server: %LLAMA_CPP_PATH%
echo.

echo Starting server on http://localhost:8088 (CPU optimized)
echo ========================================
echo.

REM OPTIMIZED COMMAND: Maximum CPU performance for Qwen3-VL
"%LLAMA_CPP_PATH%" ^
  -m "%MODEL_PATH%" ^
  --mmproj "%MMPROJ_PATH%" ^
  --host 0.0.0.0 ^
  --port 8088 ^
  -c 8192 ^
  -t %NUMBER_OF_PROCESSORS% ^
  -tb %NUMBER_OF_PROCESSORS% ^
  -b 2048 ^
  -ub 512 ^
  -n 512 ^
  --n-gpu-layers 0 ^
  -np 1 ^
  --mlock

echo.
echo Server ready! Test at: http://localhost:8088
pause