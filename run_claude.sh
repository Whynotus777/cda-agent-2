#!/bin/bash
# Claude's CDA Agent Runner
# This script runs the Claude Code working copy of the agent

set -e

echo "========================================"
echo "CDA Agent - Claude's Working Copy"
echo "========================================"
echo ""

# Navigate to agent directory
cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "⚠️  Ollama not running. Starting Ollama..."
    nohup ollama serve >/tmp/ollama_claude.log 2>&1 &
    echo "Waiting for Ollama to start..."
    until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
        sleep 2
    done
    echo "✓ Ollama started"
fi

# Check required models
echo "Checking models..."
MODELS_OK=true

for model in "llama3.2:3b" "llama3:8b"; do
    if ! ollama list | grep -q "$model"; then
        echo "⚠️  Model $model not found"
        echo "   Run: ollama pull $model"
        MODELS_OK=false
    fi
done

if [ "$MODELS_OK" = false ]; then
    echo ""
    echo "Some models are missing. Do you want to pull them now? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        echo "Pulling models..."
        ollama pull llama3.2:3b
        ollama pull llama3:8b
        echo "✓ Models ready"
    else
        echo "⚠️  Agent will fail without required models!"
    fi
fi

echo ""
echo "Starting CDA Agent..."
echo "Config: configs/default_config.yaml"
echo "Triage: ENABLED (fast 3B → 8B → 70B routing)"
echo ""

# Run agent
python3 agent.py

echo ""
echo "Agent stopped."
