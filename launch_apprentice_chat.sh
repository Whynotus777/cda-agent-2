#!/bin/bash
# Launch the Chip Design Placement Apprentice Chat Interface

cd "$(dirname "$0")"

echo "=========================================="
echo "Chip Design Placement Apprentice"
echo "=========================================="
echo ""
echo "Model: placement-apprentice-v2 (GPT-2 Medium)"
echo "Size: 1.4GB (~355M parameters)"
echo "Training: 198 examples from Gold Standard corpus"
echo ""
echo "Note: For better quality, try the Mixtral model:"
echo "      ./launch_mixtral_chat.sh"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../cda-agent/venv" ]; then
    source ../cda-agent/venv/bin/activate
else
    echo "Note: No virtual environment found, using system Python"
fi

# Check dependencies
python3 -c "import torch, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Missing dependencies. Installing..."
    pip install torch transformers
fi

# Launch chat
python3 chat_with_apprentice.py "$@"
