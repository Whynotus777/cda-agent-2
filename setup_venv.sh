#!/bin/bash

echo "=========================================="
echo "CDA Agent - Virtual Environment Setup"
echo "=========================================="
echo ""

# Install python3-venv if needed
echo "Installing python3-venv..."
sudo apt-get install -y python3-venv python3-full

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and install dependencies
echo ""
echo "Installing Python packages in virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install numpy scipy pyyaml requests python-dateutil

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 << EOF
import torch
import numpy
import yaml

print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
print(f"✓ NumPy {numpy.__version__}")
print(f"✓ PyYAML installed")
EOF

deactivate

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the agent, activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then run:"
echo "  python3 agent.py"
echo ""
echo "When done, deactivate with:"
echo "  deactivate"
echo ""
