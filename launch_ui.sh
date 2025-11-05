#!/bin/bash
# Launch Spec-to-Silicon Gradio UI

echo "================================================================================"
echo "  SPEC-TO-SILICON PLATFORM - GRADIO UI"
echo "================================================================================"
echo ""
echo "Starting Gradio UI on port 7860..."
echo ""
echo "Access at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================================================"
echo ""

cd /home/quantumc1/cda-agent-2C1
python3 ui/gradio_app.py
