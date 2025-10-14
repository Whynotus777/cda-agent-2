# 🔬 AI Placement Optimizer - "Insanely Great" Demo

A focused, visual demonstration of AI-powered chip placement optimization.

## What Makes It Magical

1. **Interactive Chip Canvas** - See your design as a living, visual layout
2. **Live Optimization** - Watch the AI agent think and improve the placement in real-time
3. **Animated Story** - See exactly what changed and why it matters
4. **Natural Explanations** - Understand the agent's reasoning, not just raw metrics

## Quick Start

```bash
# Run the demo
venv/bin/streamlit run placement_demo.py

# The demo will open in your browser at http://localhost:8501
```

## The Experience

### Stage 1: Upload Your Design
- Upload a synthesized Verilog netlist (from Yosys)
- Instantly see your design visualized as an interactive chip layout
- Notice the initial clustering and congestion

### Stage 2: Watch It Optimize
- Click "Optimize Placement" to start
- Watch in real-time as the AI:
  - Explores different placement strategies
  - Spreads cells to reduce congestion
  - Improves wirelength and timing
  - Narrates its thought process

### Stage 3: The Results Story
- See before/after layouts side-by-side
- Highlighted areas show where improvements happened
- Read the narrative: "I reduced wirelength by 12% by spreading cells in the core logic area"
- Understand WHY each action was taken

## What You'll See

**Live Metrics During Optimization:**
- Wirelength (μm) - reducing in real-time
- Timing Slack (ps) - improving iteration by iteration
- Thought Process - The agent narrates its strategy

**Final Report:**
- **Visual Comparison**: Before (clustered) vs After (optimized)
- **Key Improvements**: Percentage changes with concrete numbers
- **The Story**: Natural language explanation of what happened
- **Why It Worked**: Reasoning based on design patterns

## Example Results

```
Before Optimization:
- Wirelength: 9,500 μm
- Timing Slack (WNS): -135 ps
- High congestion in center

After Optimization:
- Wirelength: 8,300 μm (-12.6%)
- Timing Slack (WNS): -45 ps (+90 ps improvement)
- Cells distributed efficiently
```

**Agent's Explanation:**
> "I ran 30 optimization iterations on your design. In the first 10 iterations,
> I focused on increasing placement density in the core logic area. This is a proven
> strategy for arithmetic-heavy designs because it keeps related cells close together,
> reducing wire delay. The most impactful change was OPTIMIZE_WIRELENGTH in iteration 15,
> which reduced wirelength by 450 μm."

## Technology

- **Frontend**: Streamlit (clean, interactive web UI)
- **Visualization**: Plotly (interactive chip canvas)
- **Backend**: Real CDA Agent components
  - SimulationEngine for synthesis
  - ActionSpace for RL optimization
  - DesignState for tracking

## Design Philosophy

This demo embodies the principle: **Make the invisible visible**.

Instead of showing logs and raw metrics, it:
1. Visualizes the chip as a living design
2. Shows the transformation happening
3. Explains WHY changes were made
4. Makes AI reasoning transparent and understandable

## Files

- `placement_demo.py` - Main demo application
- `demo/optimization_tracker.py` - Tracks episodes for storytelling
- `demo/report_generator.py` - Creates visual reports with narratives

## Next Steps

After seeing the demo, you can:
1. Try different designs
2. Export detailed reports
3. Understand placement optimization strategies
4. See how RL-based optimization works in practice

---

**The goal**: Transform a powerful backend into a magical user experience where
chip design optimization feels like collaborating with an expert designer.
