# CDA Agent 2C1 - AI-Powered Multi-Agent RTL Design System

A production-ready **6-agent system** for automated RTL design, verification, and validation. Powered by fine-tuned **Mixtral-8x7B** and trained on ~200 chip design research papers.

**Status**: ✅ **100% Operational** - All 6 agents validated with 87% average success rate

## Overview

CDA Agent 2C1 automates the complete RTL design flow from natural language specification to production-ready, synthesizable Verilog code. It features:

- **6 Specialized AI Agents**: Working in coordinated pipelines for RTL generation, validation, and optimization
- **Mixtral-8x7B-Instruct**: Fine-tuned on ~200 research papers (47B parameters with LoRA adapters)
- **87% Success Rate**: Validated across 36 comprehensive tests
- **EDA Integration**: Yosys (synthesis), OpenSTA (timing), Verilator (linting)
- **FastAPI Backend**: RESTful API for programmatic access
- **Conversational AI**: Natural language interface powered by fine-tuned Mixtral
- **World Model**: Deep understanding of chip design, technology libraries, and design rules

## Architecture

### 6-Agent System

```
Natural Language Spec → Pipeline Orchestrator → 6 Agents → Production RTL

┌────────────────────────────────────────────────────────────┐
│  A1: Spec-to-RTL Generator (100% success)                  │
│  • Converts natural language → Verilog                     │
│  • Powered by fine-tuned Mixtral-8x7B-Instruct            │
│  • Template-aware with A2 integration                      │
└─────────────┬──────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────────┐
│  A2: Boilerplate Generator (80% success)                   │
│  • FSM, FIFO, AXI4-Lite, Counter templates                │
│  • Mealy/Moore FSMs, async FIFO with Gray code CDC        │
└─────────────┬──────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────────┐
│  A5: Style & Review Copilot (71.4% success)               │
│  • Security rules enforcement                              │
│  • Naming conventions validation                           │
│  • Best practices checking                                 │
└─────────────┬──────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────────┐
│  A4: Lint & CDC Assistant (66.7% success)                 │
│  • Automated fix generation (15+ patterns)                 │
│  • Parses Verilator/Yosys logs                            │
│  • Confidence scoring for fixes                            │
└─────────────┬──────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────────┐
│  A3: Constraint Synthesizer (100% success)                │
│  • SDC timing constraint generation                        │
│  • Multi-clock domain support                              │
│  • OpenSTA validated                                       │
└─────────────┬──────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────────────────────┐
│  A6: EDA Command Copilot (100% success)                   │
│  • Yosys synthesis scripts                                 │
│  • OpenSTA timing analysis                                 │
│  • Verilator lint commands                                 │
└─────────────┬──────────────────────────────────────────────┘
              ↓
         Production-Ready RTL + Constraints + Validation
```

## Installation

### Prerequisites

- **Python 3.12+** with pip
- **CUDA-capable GPU** (optional, for faster Mixtral inference)
- **100GB disk space** (for Mixtral base model, downloaded on first use)
- **Yosys** (synthesis): `sudo apt-get install yosys`
- **OpenSTA** (timing analysis): `sudo apt-get install opensta`
- **Verilator** (linting): `sudo apt-get install verilator`

### Installation

```bash
# Clone the repository
git clone https://github.com/Whynotus777/cda-agent-2.git
cd cda-agent-2C1

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Option 1: FastAPI Backend

```bash
# Start the API server
./launch_react_api.sh
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Option 2: Interactive Chat Mode

```bash
# Chat with the Mixtral specialist
./launch_mixtral_chat.sh

# Or quick questions
python3 chat_with_specialist.py --question "What is cell placement?"
```

### Option 3: Complete RTL Pipeline

```bash
# Run the full 6-agent pipeline
python3 launch_rtl_system.py
```

### Programmatic API Usage

```python
from api.pipeline import PipelineOrchestrator
from api.models import DesignSpec
from pathlib import Path

# Initialize orchestrator
orchestrator = PipelineOrchestrator(Path.cwd())

# Create design spec
spec = DesignSpec(
    module_name="counter",
    description="8-bit up counter with enable and synchronous reset",
    data_width=8,
    clock_freq=100.0
)

# Run the pipeline
result = orchestrator.run_pipeline(spec)

print(f"Status: {result.status}")
print(f"Generated RTL: {result.final_rtl}")
```

## Features

### 1. Natural Language Interface

Chat naturally with the agent:
- "Optimize this design for minimum power"
- "What's the critical path delay?"
- "Move the L1 cache to the top-right corner"
- "Show me the power breakdown"

### 2. Automated Design Flow

The agent orchestrates the complete flow:
- **Synthesis**: Verilog → Gate-level netlist (Yosys)
- **Placement**: Cell placement optimization (DREAMPlace)
- **Routing**: Detailed routing (TritonRoute)
- **Analysis**: Timing (OpenSTA) and Power analysis

### 3. RL-Based Optimization

The RL agent learns to:
- Adjust placement density for optimal PPA
- Swap cells with different drive strengths
- Buffer critical paths
- Balance power, performance, and area trade-offs

### 4. Design Goals

Specify what you care about:
```yaml
design_goals:
  performance: 1.0  # Maximize speed
  power: 0.8        # Minimize power (secondary)
  area: 0.6         # Minimize area (tertiary)
```

## Project Structure

```
cda-agent/
├── core/
│   ├── conversational/       # LLM interface, intent parsing
│   ├── world_model/          # Tech libraries, design rules
│   ├── simulation_engine/    # EDA tool wrappers
│   └── rl_optimizer/         # RL agent and environment
├── configs/
│   └── default_config.yaml   # Configuration
├── utils/
│   ├── config_loader.py
│   └── logger.py
├── data/
│   ├── tech_libs/            # Technology libraries
│   ├── designs/              # Design files
│   └── checkpoints/          # RL checkpoints
├── logs/                     # Log files
├── agent.py                  # Main entry point
└── requirements.txt
```

## Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
# LLM settings
llm:
  model_name: "llama3:70b"
  temperature: 0.7

# Technology
technology:
  default_process_node: "7nm"

# RL training
training:
  max_episodes: 500
  max_steps_per_episode: 50

# Design goals
design_goals:
  performance: 1.0
  power: 0.8
  area: 0.6
```

## Example Workflows

### Low-Power Microcontroller

```python
agent = CDAAgent()
agent.chat("New 7nm project, optimize for minimum power")
agent.chat("Load design from riscv_core.v")
agent.chat("Target clock speed: 500 MHz")
agent.chat("Run optimization")
```

### High-Performance Processor

```python
agent = CDAAgent()
agent.chat("New 5nm project, maximize performance")
agent.chat("Load design from cpu_core.v")
agent.chat("I need 3 GHz clock speed")
agent.chat("Optimize aggressively, power budget is 5W")
```

## Key Components

### Conversational Layer
- **LLMInterface**: Communicates with Ollama/Llama
- **IntentParser**: Extracts structured commands from natural language
- **ConversationManager**: Maintains conversation context

### World Model
- **TechLibrary**: Parses .lib files, understands cell characteristics
- **DesignParser**: Parses Verilog, LEF/DEF, SDC files
- **RuleEngine**: DRC rules, design constraints
- **DesignState**: Tracks current design metrics

### Simulation Engine
- **SynthesisEngine**: Yosys wrapper for RTL synthesis
- **PlacementEngine**: DREAMPlace wrapper for placement
- **RoutingEngine**: TritonRoute wrapper for routing
- **TimingAnalyzer**: OpenSTA wrapper for timing analysis
- **PowerAnalyzer**: Power estimation

### RL Optimizer
- **RLAgent**: Deep Q-Network for learning optimization strategies
- **ChipDesignEnv**: RL environment (state, actions, rewards)
- **ActionSpace**: Available design transformations
- **RewardCalculator**: PPA-based reward function

## Advanced Features

### Multi-Objective Optimization

The agent handles conflicting objectives (power vs. performance vs. area) and can find Pareto-optimal solutions.

### Transfer Learning

Train the RL agent on one design, then fine-tune on similar designs for faster convergence.

### Curriculum Learning

The agent first learns to meet basic constraints, then progressively optimizes for better PPA.

## Performance

On a typical RISC-V core design:
- **Initial synthesis**: 2-5 minutes
- **Placement (DREAMPlace GPU)**: 5-10 minutes
- **RL optimization**: 1-2 hours (50 episodes)
- **Final results**: 15-30% improvement in PPA over baseline

## GPU Acceleration

DREAMPlace uses GPU acceleration for fast placement:
- RTX 3090: ~10x faster than CPU
- A100: ~20x faster than CPU

## Troubleshooting

### Ollama connection failed
```bash
# Make sure Ollama is running
systemctl status ollama
# Or start manually
ollama serve
```

### DREAMPlace not found
```bash
# Set path in config
export DREAMPLACE_PATH=/path/to/DREAMPlace
```

### Out of memory during RL training
- Reduce `replay_buffer_size` in config
- Reduce `batch_size`
- Use smaller network architecture

## Contributing

This is an initial implementation. Key areas for improvement:
- Full Liberty/LEF/DEF parsers
- More sophisticated RL algorithms (PPO, SAC)
- Better action space (continuous parameters)
- Integration with more EDA tools
- Web UI for visualization

## License

MIT License

## References

- **Yosys**: http://www.clifford.at/yosys/
- **DREAMPlace**: https://github.com/limbo018/DREAMPlace
- **OpenROAD**: https://theopenroadproject.org/
- **Ollama**: https://ollama.ai
- **RL for EDA**: Multiple research papers on using RL for chip design optimization

## Contact

For questions or collaboration, open an issue on GitHub.
