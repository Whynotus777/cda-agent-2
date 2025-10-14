# Getting Started with CDA Agent

## Initial Setup Completed ✓

Your CDA Agent has been successfully created with the following structure:

```
cda-agent/
├── core/
│   ├── conversational/          # LLM interface and conversation management
│   │   ├── __init__.py
│   │   ├── llm_interface.py     # Ollama/Llama3 interface
│   │   ├── intent_parser.py     # Natural language → structured commands
│   │   └── conversation_manager.py
│   │
│   ├── world_model/             # Chip design knowledge base
│   │   ├── __init__.py
│   │   ├── tech_library.py      # Parse .lib files, cell characteristics
│   │   ├── design_parser.py     # Parse Verilog, LEF/DEF, SDC
│   │   ├── rule_engine.py       # DRC rules, design constraints
│   │   └── design_state.py      # Track design metrics and state
│   │
│   ├── simulation_engine/       # EDA tool integrations
│   │   ├── __init__.py
│   │   ├── synthesis.py         # Yosys wrapper
│   │   ├── placement.py         # DREAMPlace wrapper
│   │   ├── routing.py           # TritonRoute wrapper
│   │   ├── timing_analysis.py   # OpenSTA wrapper
│   │   └── power_analysis.py    # Power estimation
│   │
│   └── rl_optimizer/            # RL optimization core
│       ├── __init__.py
│       ├── rl_agent.py          # Deep Q-Network implementation
│       ├── environment.py       # Chip design RL environment
│       ├── actions.py           # 17 optimization actions
│       └── reward.py            # PPA-based reward function
│
├── configs/
│   └── default_config.yaml      # Configuration file
├── utils/
│   ├── config_loader.py
│   └── logger.py
├── examples/
│   └── example_usage.py
├── data/
│   ├── tech_libs/               # Technology libraries
│   ├── designs/                 # Your design files
│   └── checkpoints/             # RL training checkpoints
├── logs/                        # Log files
├── agent.py                     # Main entry point
├── requirements.txt
├── setup.sh                     # Setup helper script
└── README.md
```

## What You Have

### ✅ Already Installed
- **Ollama** (v0.12.3) with **Llama3:70b** (39GB model)
- **DREAMPlace** at `/home/quantumc1/DREAMPlace`
- **NVIDIA RTX 5090** (32GB VRAM) with driver 580.65.06
- Python 3

### 🔧 Need to Install

1. **Python Virtual Environment** (recommended for Ubuntu 24.04):
```bash
cd ~/cda-agent
./setup_venv.sh
```

This will:
- Install `python3-venv`
- Create a virtual environment in `./venv/`
- Install PyTorch with CUDA 11.8 support
- Install all Python dependencies

2. **Yosys** (synthesis):
```bash
sudo apt-get install yosys
```

3. **OpenSTA** (optional, for timing analysis):
```bash
sudo apt-get install opensta
```

4. **OpenROAD** (optional, includes TritonRoute):
   - See: https://github.com/The-OpenROAD-Project/OpenROAD

## Quick Start

### 1. Install Dependencies

Run the setup script:
```bash
cd ~/cda-agent
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
# Install pip
sudo apt-get install python3-pip

# Install Python packages
pip3 install --user -r requirements.txt

# Install EDA tools
sudo apt-get install yosys opensta
```

### 2. Start Ollama

Ensure Ollama is running:
```bash
# Check if running
ollama list

# If not running, start it
ollama serve
```

### 3. Run the Agent

**Option A: Using the launcher script (easiest)**
```bash
cd ~/cda-agent
./run_agent.sh
```

**Option B: Activate venv manually**
```bash
cd ~/cda-agent
source venv/bin/activate
python3 agent.py
```

Then chat naturally:
```
You: Create a new 7nm project for a low-power microcontroller
Agent: Understood. Creating a new project with 7nm process technology...

You: Load my Verilog design from ~/designs/my_chip.v
Agent: Loading design from ~/designs/my_chip.v...
```

### 4. Programmatic Usage

```python
from agent import CDAAgent

# Create agent
agent = CDAAgent()

# Process commands
agent.chat("Create a new 12nm design focused on high performance")
agent.chat("Load design from my_chip.v")
agent.chat("Run synthesis")
agent.chat("Optimize for maximum clock speed")

# Get results
summary = agent.get_design_summary()
print(summary)
```

## What the Agent Can Do

### 1. Natural Language Interface
- Chat naturally about chip design
- Explain design concepts and metrics
- Provide guidance on optimization strategies

### 2. Design Flow Automation
- **Synthesis**: Convert Verilog to gate-level netlist (Yosys)
- **Placement**: GPU-accelerated cell placement (DREAMPlace)
- **Routing**: Detailed routing (TritonRoute, if installed)
- **Analysis**: Timing (OpenSTA) and power analysis

### 3. RL-Based Optimization
The agent learns to optimize designs through:
- Adjusting placement density
- Swapping cell drive strengths
- Buffering critical paths
- Balancing power, performance, area (PPA)

### 4. Interactive Design
- Adjust floorplan on the fly
- Lock component positions
- Query design metrics
- Visualize critical paths

## Example Workflows

### Low-Power Microcontroller

```python
agent.chat("New 7nm project, minimize power")
agent.chat("Load riscv_core.v")
agent.chat("Target 500 MHz clock")
agent.chat("Run optimization prioritizing power")
```

### High-Performance Processor

```python
agent.chat("New 5nm project, maximize performance")
agent.chat("Load cpu_core.v")
agent.chat("I need 3 GHz clock frequency")
agent.chat("Power budget is 10W")
agent.chat("Optimize aggressively")
```

### RL Training

```python
agent.chat("Create new 7nm design")
agent.chat("Load design.v")
agent.run_rl_optimization({
    'goals': {
        'performance': 1.0,
        'power': 0.8,
        'area': 0.6
    }
})
```

## Configuration

Edit `configs/default_config.yaml`:

```yaml
# LLM settings
llm:
  model_name: "llama3:70b"
  ollama_host: "http://localhost:11434"
  temperature: 0.7

# Tool paths
tools:
  yosys_binary: "yosys"
  dreamplace_path: "/home/quantumc1/DREAMPlace"
  tritonroute_binary: "TritonRoute"
  opensta_binary: "sta"

# RL training
training:
  max_episodes: 500
  max_steps_per_episode: 50

# Design goals
design_goals:
  performance: 1.0  # Priority weights
  power: 0.8
  area: 0.6
```

## Architecture Highlights

### 1. Conversational Layer
- **LLMInterface**: Communicates with Ollama/Llama3
- **IntentParser**: Extracts structured commands from natural language
- **ConversationManager**: Maintains conversation state and context

### 2. World Model
- **TechLibrary**: Parses Liberty (.lib) files for cell timing/power
- **DesignParser**: Parses Verilog, LEF/DEF, SDC constraints
- **RuleEngine**: Design rule checking (DRC) for process nodes
- **DesignState**: Tracks current design metrics (PPA)

### 3. Simulation Engine
- **SynthesisEngine**: Yosys wrapper for RTL synthesis
- **PlacementEngine**: DREAMPlace wrapper with GPU acceleration
- **RoutingEngine**: TritonRoute wrapper for detailed routing
- **TimingAnalyzer**: OpenSTA wrapper for STA
- **PowerAnalyzer**: Custom power estimation

### 4. RL Optimizer
- **RLAgent**: Deep Q-Network (DQN) for learning
- **ChipDesignEnv**: Gym-like RL environment
- **ActionSpace**: 17 optimization actions
- **RewardCalculator**: PPA-based reward function

## GPU Acceleration

Your RTX 5090 (32GB VRAM) will dramatically accelerate:

1. **DREAMPlace**: 10-20x faster placement than CPU
2. **PyTorch RL Training**: Fast neural network training
3. **Large LLM**: Can handle Llama3:70b smoothly

Expected performance:
- Placement: 5-10 minutes (vs. 1-2 hours on CPU)
- RL episode: 10-20 minutes
- Full optimization: 1-2 hours

## Next Steps

1. **Install missing dependencies** (see above)
2. **Test the agent**:
   ```bash
   python3 agent.py
   ```
3. **Try examples**:
   ```bash
   python3 examples/example_usage.py
   ```
4. **Load your own design** and start optimizing!

## Troubleshooting

### "No module named 'torch'"
```bash
pip3 install --user torch torchvision
```

### "Ollama connection failed"
```bash
# Start Ollama
ollama serve

# In another terminal, verify
ollama list
```

### "DREAMPlace not found"
Update the path in `configs/default_config.yaml`:
```yaml
tools:
  dreamplace_path: "/home/quantumc1/DREAMPlace"
```

### "Yosys command not found"
```bash
sudo apt-get install yosys
```

## Learning Resources

- **Yosys Documentation**: http://www.clifford.at/yosys/
- **DREAMPlace Paper**: https://github.com/limbo018/DREAMPlace
- **Liberty Format**: IEEE Standard 1481
- **OpenROAD**: https://theopenroadproject.org/

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review error messages
3. Verify all dependencies are installed
4. Check configuration in `configs/default_config.yaml`

---

**You're ready to start! Your AI chip design agent is set up and waiting for your first design.**
