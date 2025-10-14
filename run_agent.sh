#!/bin/bash

# CDA Agent Launcher Script
# Automatically activates virtual environment and runs the agent

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: ./setup_venv.sh"
    exit 1
fi

# Activate venv and run agent
source "$SCRIPT_DIR/venv/bin/activate"
python3 "$SCRIPT_DIR/agent.py" "$@"
