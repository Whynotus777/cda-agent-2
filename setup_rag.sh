#!/bin/bash
# Quick Setup Script for RAG System

set -e

echo "========================================"
echo "CDA Agent - RAG Setup"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Check venv
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv"
    exit 1
fi

# Activate venv
source venv/bin/activate

echo "Step 1: Installing RAG dependencies..."
echo "----------------------------------------"
./venv/bin/python3 -m pip install -q chromadb sentence-transformers beautifulsoup4 lxml PyPDF2

echo "✓ Dependencies installed"
echo ""

echo "Step 2: Scraping EDA documentation..."
echo "----------------------------------------"
./venv/bin/python3 data/scrapers/eda_doc_scraper.py

echo ""
echo "Step 3: Indexing into RAG system..."
echo "----------------------------------------"
./venv/bin/python3 data/scrapers/index_knowledge_base.py

echo ""
echo "========================================"
echo "✅ RAG Setup Complete!"
echo "========================================"
echo ""
echo "Your agent can now use EDA documentation to answer questions!"
echo ""
echo "Test it:"
echo "  ./run_claude.sh"
echo "  > What is synthesis in Yosys?"
echo ""
echo "The agent will retrieve and use Yosys documentation."
echo ""
echo "Optional next steps:"
echo "  1. Collect Verilog code: ./venv/bin/python3 data/scrapers/verilog_github_scraper.py"
echo "  2. Prepare training data: ./venv/bin/python3 training/data_preparation/prepare_training_data.py"
echo "  3. Fine-tune model: ./venv/bin/python3 training/finetune_8b_chipdesign.py"
echo ""
echo "See RAG_AND_TRAINING_GUIDE.md for details."
