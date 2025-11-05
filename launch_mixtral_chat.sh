#!/bin/bash
# Launch the Chip Design Specialist Chat Interface with Mixtral Model

cd "$(dirname "$0")"

echo "=========================================="
echo "Chip Design Wisdom Specialist"
echo "=========================================="
echo ""
echo "Model: Mixtral-8x7B-Instruct + LoRA Adapter"
echo "Base Model: 47B parameters (Mixture of Experts)"
echo "Adapter: wisdom-specialist-v3"
echo "Training: ~200 examples from research and real chips"
echo ""
echo "Note: First run will download the base Mixtral model (~94GB)"
echo "      Subsequent runs will be much faster!"
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
echo "Checking dependencies..."
python3 -c "import torch, transformers, peft" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Missing dependencies. Installing..."
    pip install torch transformers peft accelerate
fi

# Check for bitsandbytes (optional, for 4-bit quantization)
python3 -c "import bitsandbytes" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Note: bitsandbytes not found (optional for 4-bit mode)"
    echo "Install with: pip install bitsandbytes"
fi

echo ""
echo "Loading model..."
echo ""

# Launch chat
# Add --4bit flag to use 4-bit quantization (saves memory)
python3 chat_with_specialist.py "$@"
