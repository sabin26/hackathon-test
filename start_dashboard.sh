#!/bin/bash

# Sports Analytics Dashboard Startup Script
# This script automatically handles the virtual environment and starts the dashboard

echo "🏟️  Sports Analytics Dashboard Startup"
echo "======================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if required packages are installed
if [ ! -f ".venv/bin/uvicorn" ]; then
    echo "📦 Installing dashboard dependencies..."
    .venv/bin/pip install fastapi uvicorn websockets jinja2
fi

# Default mode
MODE="demo"
CONFIG="config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--mode demo|video] [--config config.yaml]"
            echo ""
            echo "Options:"
            echo "  --mode    Run in 'demo' mode (simulated data) or 'video' mode (process actual video)"
            echo "  --config  Configuration file path (default: config.yaml)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run in demo mode"
            echo "  $0 --mode video              # Process actual video"
            echo "  $0 --mode demo               # Run with simulated data"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting dashboard in $MODE mode..."

if [ "$MODE" = "video" ]; then
    if [ ! -f "$CONFIG" ]; then
        echo "❌ Configuration file not found: $CONFIG"
        exit 1
    fi
    echo "📹 Video file: $(grep 'path:' $CONFIG | cut -d"'" -f2)"
fi

echo "🌐 Dashboard will be available at: http://localhost:8000"
echo "⏹️  Press Ctrl+C to stop"
echo ""

# Start the dashboard
.venv/bin/python launch_dashboard.py --mode "$MODE" --config "$CONFIG"
