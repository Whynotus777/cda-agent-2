#!/bin/bash

# Launch script for React API Server
# This server connects React UI to the existing Python backend

echo "=========================================="
echo "🚀 Starting FastAPI Server for React UI"
echo "=========================================="
echo ""
echo "This server:"
echo "  - Does NOT modify existing backend"
echo "  - Runs on port 8000"
echo "  - Provides REST + WebSocket API"
echo "  - Connects to PipelineOrchestrator"
echo ""
echo "URLs:"
echo "  API:       http://localhost:8000"
echo "  Docs:      http://localhost:8000/docs"
echo "  WebSocket: ws://localhost:8000/api/pipeline/logs"
echo ""
echo "=========================================="
echo ""

# Check if running in venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected"
    if [ -d "venv" ]; then
        echo "Activating venv..."
        source venv/bin/activate
    else
        echo "❌ venv not found. Please create one:"
        echo "   python3 -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing FastAPI dependencies..."
    pip install -r react_api/requirements.txt
fi

# Launch server
echo "🚀 Starting server..."
cd ~/cda-agent-2C1
python3 -m uvicorn react_api.server:app --host 0.0.0.0 --port 8000 --reload
