@echo off
REM CUDA+OpenMP Hybrid Search Engine Launcher for Windows
REM This script provides an easy way to run the search engine with various configurations

setlocal enabledelayedexpansion

REM Set colors for output
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

REM Default configuration
set USE_GPU=1
set OMP_THREADS=8
set CUDA_BLOCK_SIZE=256
set PROCESSING_MODE=auto
set MEMORY_STRATEGY=unified
set BATCH_SIZE=1000
set VERBOSE=0
set BENCHMARK=0

REM Project paths
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set EXECUTABLE=%PROJECT_ROOT%\\bin\\cuda_openmp_search_engine.exe

echo.
echo ========================================
echo   CUDA+OpenMP Hybrid Search Engine
echo ========================================
echo.

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--gpu" (
    set USE_GPU=1
    shift & goto parse_args
)
if "%~1"=="--no-gpu" (
    set USE_GPU=0
    shift & goto parse_args
)
if "%~1"=="-t" (
    set OMP_THREADS=%~2
    shift & shift & goto parse_args
)
if "%~1"=="--threads" (
    set OMP_THREADS=%~2
    shift & shift & goto parse_args
)
if "%~1"=="-b" (
    set CUDA_BLOCK_SIZE=%~2
    shift & shift & goto parse_args
)
if "%~1"=="--block-size" (
    set CUDA_BLOCK_SIZE=%~2
    shift & shift & goto parse_args
)
if "%~1"=="-m" (
    set PROCESSING_MODE=%~2
    shift & shift & goto parse_args
)
if "%~1"=="--mode" (
    set PROCESSING_MODE=%~2
    shift & shift & goto parse_args
)
if "%~1"=="-v" (
    set VERBOSE=1
    shift & goto parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=1
    shift & goto parse_args
)
if "%~1"=="--benchmark" (
    set BENCHMARK=1
    shift & goto parse_args
)
if "%~1"=="-q" (
    set QUERY=%~2
    shift & shift & goto parse_args
)
if "%~1"=="--query" (
    set QUERY=%~2
    shift & shift & goto parse_args
)
if "%~1"=="--help" (
    goto show_help
)
if "%~1"=="-h" (
    goto show_help
)
REM Pass through unknown arguments
set EXTRA_ARGS=!EXTRA_ARGS! %~1
shift & goto parse_args

:end_parse

REM Check if executable exists
if not exist "%EXECUTABLE%" (
    echo %RED%[ERROR]%NC% Executable not found: %EXECUTABLE%
    echo %BLUE%[INFO]%NC% Please build the project first using: make
    echo %BLUE%[INFO]%NC% Or use the Visual Studio solution if available
    exit /b 1
)

REM Check for CUDA support
set CUDA_AVAILABLE=0
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set CUDA_AVAILABLE=1
    echo %GREEN%[SUCCESS]%NC% NVIDIA GPU detected
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do (
        echo   GPU: %%i
    )
) else (
    echo %YELLOW%[WARNING]%NC% No NVIDIA GPU detected
    if %USE_GPU%==1 (
        echo %YELLOW%[WARNING]%NC% GPU requested but not available, falling back to CPU-only
        set USE_GPU=0
        set PROCESSING_MODE=cpu
    )
)

REM Display configuration
echo.
echo Current Configuration:
echo   GPU Acceleration: !USE_GPU!
echo   OpenMP Threads: !OMP_THREADS!
echo   CUDA Block Size: !CUDA_BLOCK_SIZE!
echo   Processing Mode: !PROCESSING_MODE!
echo   Memory Strategy: !MEMORY_STRATEGY!
echo   Batch Size: !BATCH_SIZE!

REM Set environment variables
set OMP_NUM_THREADS=%OMP_THREADS%

REM Build command line
set CMD_LINE=%EXECUTABLE%

if %USE_GPU%==1 (
    set CMD_LINE=!CMD_LINE! --gpu
) else (
    set CMD_LINE=!CMD_LINE! --no-gpu
)

set CMD_LINE=!CMD_LINE! --threads %OMP_THREADS%
set CMD_LINE=!CMD_LINE! --block-size %CUDA_BLOCK_SIZE%
set CMD_LINE=!CMD_LINE! --mode %PROCESSING_MODE%
set CMD_LINE=!CMD_LINE! --memory-strategy %MEMORY_STRATEGY%
set CMD_LINE=!CMD_LINE! --batch-size %BATCH_SIZE%

if %VERBOSE%==1 (
    set CMD_LINE=!CMD_LINE! --verbose
)

if %BENCHMARK%==1 (
    set CMD_LINE=!CMD_LINE! --benchmark
)

if defined QUERY (
    set CMD_LINE=!CMD_LINE! --query "!QUERY!"
)

if defined EXTRA_ARGS (
    set CMD_LINE=!CMD_LINE! !EXTRA_ARGS!
)

REM Display command and run
echo.
echo %BLUE%[INFO]%NC% Running: !CMD_LINE!
echo.

!CMD_LINE!

set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE%==0 (
    echo %GREEN%[SUCCESS]%NC% Search engine completed successfully
) else (
    echo %RED%[ERROR]%NC% Search engine exited with code %EXIT_CODE%
)

exit /b %EXIT_CODE%

:show_help
echo Usage: %~nx0 [OPTIONS] [QUERY]
echo.
echo GPU and CPU Configuration:
echo   --gpu                 Enable GPU acceleration (default if available)
echo   --no-gpu              Disable GPU, use CPU-only mode
echo   -t, --threads NUM     Set number of OpenMP threads (default: 8)
echo   -b, --block-size NUM  Set CUDA thread block size (default: 256)
echo.
echo Processing Options:
echo   -m, --mode MODE       Processing mode: cpu^|gpu^|hybrid^|auto (default: auto)
echo   --memory-strategy STR Memory strategy: basic^|pinned^|unified (default: unified)
echo   --batch-size NUM      Batch processing size (default: 1000)
echo.
echo Search Options:
echo   -q, --query QUERY     Execute search query and exit
echo.
echo Performance and Debugging:
echo   --benchmark           Run comprehensive performance benchmarks
echo   -v, --verbose         Enable verbose output
echo.
echo Examples:
echo   %~nx0 --gpu -t 8 -q "machine learning"
echo   %~nx0 --no-gpu -t 16 --mode cpu
echo   %~nx0 --benchmark --verbose
echo   %~nx0 --gpu --mode hybrid -q "artificial intelligence"
echo.
exit /b 0
