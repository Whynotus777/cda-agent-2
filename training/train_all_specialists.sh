#!/bin/bash

# Train All Specialist Models
# Creates fine-tuned models for each chip design phase

echo "=========================================="
echo "Training Specialist Models"
echo "=========================================="
echo ""

# Define phases and sizes to train
PHASES=(
    "specification"
    "rtl_design"
    "synthesis"
    "placement"
    "routing"
    "timing_analysis"
    "power_analysis"
)

# Train 3B models first (fastest)
echo "Step 1: Training 3B specialists..."
for phase in "${PHASES[@]}"; do
    echo ""
    echo "Training 3B specialist for: $phase"
    python3 training/finetune_specialist.py --phase "$phase" --size 3b
done

# Train 8B models
echo ""
echo "Step 2: Training 8B specialists..."
for phase in "${PHASES[@]}"; do
    echo ""
    echo "Training 8B specialist for: $phase"
    python3 training/finetune_specialist.py --phase "$phase" --size 8b
done

# Train 70B models (takes longest)
echo ""
echo "Step 3: Training 70B specialists (this will take a while)..."
read -p "Train 70B models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for phase in "${PHASES[@]}"; do
        echo ""
        echo "Training 70B specialist for: $phase"
        python3 training/finetune_specialist.py --phase "$phase" --size 70b
    done
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Available specialist models:"
ollama list | grep -E "(3b|8b|70b)-(specification|rtl_design|synthesis|placement|routing|timing_analysis|power_analysis)"

echo ""
echo "Test a specialist with:"
echo "  ollama run llama3:8b-synthesis 'How do I optimize for area?'"
echo ""
