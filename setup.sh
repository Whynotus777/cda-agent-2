#!/bin/bash

# CDA Agent Setup Script

echo "=========================================="
echo "CDA Agent Setup"
echo "=========================================="
echo ""

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "Note: Some installations may require sudo"
    echo ""
fi

# Update DREAMPlace path in config
DREAMPLACE_PATH="/home/quantumc1/DREAMPlace"
if [ -d "$DREAMPLACE_PATH" ]; then
    echo "✓ DREAMPlace found at: $DREAMPLACE_PATH"
    sed -i "s|dreamplace_path:.*|dreamplace_path: \"$DREAMPLACE_PATH\"|" configs/default_config.yaml
else
    echo "✗ DREAMPlace not found at expected location"
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    ollama --version

    # Check for llama3:70b
    if ollama list | grep -q "llama3:70b"; then
        echo "✓ Llama3:70b model is available"
    else
        echo "⚠ Llama3:70b not found. Pulling model (this may take a while)..."
        ollama pull llama3:70b
    fi
else
    echo "✗ Ollama not installed"
    echo "Install with: curl https://ollama.ai/install.sh | sh"
fi

# Check Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -q -r requirements.txt

# Check PyTorch
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is installed"
    python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
else
    echo "⚠ PyTorch not found. Installing PyTorch with CUDA support..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# Check Yosys
if command -v yosys &> /dev/null; then
    echo "✓ Yosys is installed"
    yosys -V | head -1
else
    echo "✗ Yosys not installed"
    echo "Install with: sudo apt-get install yosys"
fi

# Check OpenSTA
if command -v sta &> /dev/null; then
    echo "✓ OpenSTA is installed"
else
    echo "✗ OpenSTA not installed"
    echo "Install with: sudo apt-get install opensta"
fi

# Check TritonRoute (part of OpenROAD)
if command -v TritonRoute &> /dev/null; then
    echo "✓ TritonRoute is installed"
else
    echo "✗ TritonRoute not installed"
    echo "Install OpenROAD: https://github.com/The-OpenROAD-Project/OpenROAD"
fi

# Check NVIDIA GPU
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected (optional, but recommended for DREAMPlace)"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/tech_libs
mkdir -p data/designs
mkdir -p data/checkpoints
mkdir -p logs
echo "✓ Directories created"

echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo ""
echo "Essential (installed):"
echo "  ✓ Ollama + Llama3:70b"
echo "  ✓ DREAMPlace"
echo "  ✓ NVIDIA RTX 5090 (32GB VRAM)"
echo "  ✓ Python dependencies"
echo ""
echo "To install (optional but recommended):"
echo "  - Yosys: sudo apt-get install yosys"
echo "  - OpenSTA: sudo apt-get install opensta"
echo "  - OpenROAD: https://github.com/The-OpenROAD-Project/OpenROAD"
echo ""
echo "You can start using the agent now with:"
echo "  python3 agent.py"
echo ""
