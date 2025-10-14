#!/bin/bash

# CDA Agent - Fast Mode
# Uses smaller LLM for 5-10x faster responses

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: ./setup_venv.sh"
    exit 1
fi

echo "=========================================="
echo "CDA Agent - FAST MODE"
echo "Using llama3.2:3b for faster responses"
echo "=========================================="
echo ""

# Check if model is available
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "⚠️  llama3.2:3b not found. Pulling model..."
    ollama pull llama3.2:3b
fi

# Activate venv and run agent with fast config
source "$SCRIPT_DIR/venv/bin/activate"
python3 "$SCRIPT_DIR/agent.py" "$SCRIPT_DIR/configs/fast_config.yaml"
