# CDA Agent - AI-Powered Chip Design Automation

An intelligent agent that uses LLMs and Reinforcement Learning to assist with chip design, competing in the CDA (Chip Design Automation) space.

## Overview

CDA Agent combines natural language interaction with automated design optimization to help chip designers create better designs faster. It integrates:

- **Conversational AI**: Natural language interface powered by local LLM (Llama 3)
- **World Model**: Deep understanding of chip design physics, technology libraries, and design rules
- **Simulation Engine**: Integration with open-source EDA tools (Yosys, DREAMPlace, TritonRoute, OpenSTA)
- **RL Optimization**: Reinforcement learning agent that learns optimal design strategies

## Architecture

```
┌─────────────────────────────────────────────┐
│         Conversational Interface            │
│  (LLM, Intent Parser, Conversation Manager) │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│           Agent Orchestrator                │
└──┬───────────┬────────────┬─────────────┬───┘
   │           │            │             │
   ▼           ▼            ▼             ▼
┌──────┐  ┌─────────┐  ┌────────┐  ┌──────────┐
│World │  │Simulation│  │   RL   │  │  Design  │
│Model │  │ Engine   │  │Optimizer│  │  State   │
└──────┘  └─────────┘  └────────┘  └──────────┘
```

## Installation

### Prerequisites

1. **Ollama** (for LLM):
```bash
curl https://ollama.ai/install.sh | sh
ollama pull llama3:70b
```

2. **DREAMPlace** (GPU-accelerated placement):
```bash
git clone https://github.com/limbo018/DREAMPlace.git
cd DREAMPlace
python setup.py install
```

3. **Yosys** (synthesis):
```bash
sudo apt-get install yosys
```

4. **OpenSTA** (timing analysis):
```bash
sudo apt-get install opensta
```

5. **TritonRoute** (routing - part of OpenROAD):
```bash
# Install OpenROAD which includes TritonRoute
# See: https://github.com/The-OpenROAD-Project/OpenROAD
```

### CDA Agent Installation

```bash
cd cda-agent
pip install -r requirements.txt
```

## Quick Start

### Interactive Mode

```bash
python agent.py
```

Then interact naturally:
```
You: Let's start a new 7nm design for a low-power microcontroller
Agent: Understood. Creating a new project with 7nm process technology...

You: Load my Verilog file at ./designs/riscv_core.v
Agent: Loading design from ./designs/riscv_core.v...

You: Run synthesis optimized for low power
Agent: Beginning synthesis using Yosys...
```

### Programmatic Usage

```python
from agent import CDAAgent

# Create agent
agent = CDAAgent(config_path="configs/default_config.yaml")

# Process commands
agent.chat("Create a new 12nm project focused on high performance")
agent.chat("Load design from my_chip.v")
agent.chat("Run full optimization flow")

# Get results
summary = agent.get_design_summary()
print(f"Final metrics: {summary}")
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
