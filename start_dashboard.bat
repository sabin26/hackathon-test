@echo off
REM Sports Analytics Dashboard Startup Script for Windows
REM This script automatically handles the virtual environment and starts the dashboard

echo 🏟️  Sports Analytics Dashboard Startup
echo ======================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if required packages are installed
if not exist ".venv\Scripts\uvicorn.exe" (
    echo 📦 Installing dashboard dependencies...
    .venv\Scripts\pip install fastapi uvicorn websockets jinja2
)

REM Default values
set MODE=demo
set CONFIG=config.yaml

REM Parse command line arguments
:parse_args
if "%1"=="--mode" (
    set MODE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--config" (
    set CONFIG=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--help" goto show_help
if "%1"=="-h" goto show_help
if "%1"=="/?" goto show_help
if not "%1"=="" (
    echo Unknown option: %1
    echo Use --help for usage information
    pause
    exit /b 1
)

echo 🚀 Starting dashboard in %MODE% mode...

if "%MODE%"=="video" (
    if not exist "%CONFIG%" (
        echo ❌ Configuration file not found: %CONFIG%
        pause
        exit /b 1
    )
    echo 📹 Processing video file from configuration
)

echo 🌐 Dashboard will be available at: http://localhost:8000
echo ⏹️  Press Ctrl+C to stop
echo.

REM Start the dashboard
.venv\Scripts\python launch_dashboard.py --mode %MODE% --config %CONFIG%
goto end

:show_help
echo Usage: %0 [--mode demo^|video] [--config config.yaml]
echo.
echo Options:
echo   --mode    Run in 'demo' mode (simulated data) or 'video' mode (process actual video)
echo   --config  Configuration file path (default: config.yaml)
echo.
echo Examples:
echo   %0                           # Run in demo mode
echo   %0 --mode video              # Process actual video
echo   %0 --mode demo               # Run with simulated data
pause
exit /b 0

:end
