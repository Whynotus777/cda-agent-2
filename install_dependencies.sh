#!/bin/bash

echo "=========================================="
echo "CDA Agent - Installing Dependencies"
echo "=========================================="
echo ""

# Install pip3 (requires sudo)
echo "Step 1: Installing pip3..."
sudo apt-get update
sudo apt-get install -y python3-pip

# Install PyTorch with CUDA support
echo ""
echo "Step 2: Installing PyTorch with CUDA 11.8..."
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
echo ""
echo "Step 3: Installing other Python dependencies..."
pip3 install --user numpy scipy pyyaml requests python-dateutil

# Optional: Install Yosys (requires sudo)
echo ""
echo "Step 4: Installing Yosys (optional)..."
read -p "Install Yosys for synthesis? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt-get install -y yosys
fi

# Optional: Install OpenSTA (requires sudo)
echo ""
echo "Step 5: Installing OpenSTA (optional)..."
read -p "Install OpenSTA for timing analysis? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt-get install -y opensta
fi

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

echo ""
echo "Python & pip:"
python3 --version
pip3 --version

echo ""
echo "PyTorch:"
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1

echo ""
echo "NumPy:"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>&1

echo ""
echo "Ollama:"
ollama --version 2>&1

echo ""
echo "DREAMPlace:"
if [ -d "/home/quantumc1/DREAMPlace" ]; then
    echo "✓ Found at /home/quantumc1/DREAMPlace"
else
    echo "✗ Not found"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "You can now run the agent with:"
echo "  python3 agent.py"
echo ""
