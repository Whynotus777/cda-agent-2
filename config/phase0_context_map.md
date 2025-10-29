# Phase 0 - Context Assimilation Complete

**Date:** 2025-10-29
**Status:** ✅ COMPLETE

## Existing Infrastructure Inventory

### 1. Core Modules (~9,587 lines)

#### Simulation Engine (`core/simulation_engine/`)
- **synthesis.py** (359 lines) - Yosys RTL synthesis wrapper
  - API: `synthesize(rtl_files, top_module, output_netlist, optimization_goal)`
  - Generates Yosys TCL scripts
  - Parses synthesis statistics (cell count, area, gates)

- **timing_analysis.py** (291 lines) - OpenSTA timing analysis
  - API: `analyze_timing(netlist_file, sdc_file, lib_files, spef_file)`
  - Generates STA TCL scripts
  - Extracts WNS, TNS, critical path metrics

- **placement.py** (423 lines) - DREAMPlace integration
  - API: `place(netlist, floorplan, optimization_params)`
  - GPU-accelerated placement
  - HPWL calculation

- **routing.py** (231 lines) - TritonRoute wrapper
  - API: `route(placed_design, constraints)`

- **power_analysis.py** (410 lines) - Power estimation

#### World Model (`core/world_model/`)
- **design_state.py** (378 lines) - **PRIMARY STATE SCHEMA**
  - Classes: `DesignState`, `TimingMetrics`, `PowerMetrics`, `AreaMetrics`, `RoutingMetrics`
  - Enums: `DesignStage` (UNINITIALIZED → SIGNOFF_READY)
  - API: `update_stage()`, `update_metrics()`, `get_metrics_summary()`

- **design_parser.py** (445 lines) - Verilog/SDC parsing
  - Extracts modules, ports, instances, nets
  - Parses SDC constraints

- **tech_library.py** (378 lines) - Liberty (.lib) parser
  - Cell timing/power/area extraction
  - Drive strength inference

- **rule_engine.py** (397 lines) - Design rules & DRC

#### RL Optimizer (`core/rl_optimizer/`)
- **actions.py** (476 lines) - **17 ACTION DEFINITIONS**
  - `ActionSpace` class maps actions → EDA operations
  - API: `execute_action(action_idx) → {'success', 'metrics_delta', 'info'}`

- **environment.py** (371 lines) - Gymnasium RL environment
  - `ChipDesignEnv` class
  - `reset()`, `step()`, `render()`

- **ppo_agent.py** (279 lines) - PPO training
- **reward.py** (361 lines) - PPA-based reward calculation

#### Conversational (`core/conversational/`)
- **intent_parser.py** (249 lines) - NL → structured intent
- **action_executor.py** (411 lines) - Intent → backend execution
- **phase_router.py** (493 lines) - Routes queries to specialist LLMs
- **llm_interface.py** (356 lines) - Ollama API wrapper
- **conversation_manager.py** (638 lines) - Multi-turn state management

#### RAG System (`core/rag/`)
- **retriever.py** (213 lines) - ChromaDB semantic search
- **document_loader.py** (330 lines) - Markdown/text indexing
- **embedder.py** (108 lines) - Sentence transformers
- **vector_store.py** (161 lines) - Vector DB operations
- **Knowledge Base:** 81 documents indexed (Yosys, OpenSTA, DREAMPlace, OpenLane)

### 2. Current Schemas (DATACLASS-BASED)

All schemas use Python `dataclasses` in `core/world_model/design_state.py`:

```python
@dataclass
class TimingMetrics:
    clock_period: Optional[float]
    worst_negative_slack: Optional[float]
    total_negative_slack: Optional[float]
    worst_path_delay: Optional[float]
    setup_violations: int
    hold_violations: int
    max_frequency: Optional[float]

@dataclass
class PowerMetrics:
    total_power: Optional[float]
    dynamic_power: Optional[float]
    static_power: Optional[float]
    ...

@dataclass
class AreaMetrics:
    total_area: Optional[float]
    cell_area: Optional[float]
    utilization: Optional[float]
    ...

@dataclass
class RoutingMetrics:
    total_wirelength: Optional[float]
    via_count: Dict[str, int]
    drc_violations: int
    ...

class DesignStage(Enum):
    UNINITIALIZED, RTL_LOADED, SYNTHESIZED, PLACED,
    CTS_DONE, ROUTED, OPTIMIZED, VERIFIED, SIGNOFF_READY
```

### 3. Data Flow Architecture

```
User Input (NL)
    ↓
intent_parser.py → Intent {action, params, confidence}
    ↓
action_executor.py → Routes to backend
    ↓
simulation_engine/* → EDA tools (Yosys/STA/DREAMPlace)
    ↓
design_state.py → Updates metrics
    ↓
rl_optimizer/reward.py → Calculates reward
    ↓
ppo_agent.py → Learns policy
```

### 4. Configuration System

**File:** `configs/default_config.yaml`

Key sections:
- `llm`: Model routing, phase specialists, triage
- `technology`: Process node, lib paths
- `tools`: Binary paths (yosys, sta, dreamplace)
- `rl_agent`: Hyperparameters (lr, gamma, epsilon)
- `training`: Episode config
- `design_goals`: PPA weights

### 5. Missing Infrastructure (For Multi-Agent RTL)

#### What We Need to Build:

1. **Agent Message Bus** - Pub/sub for agent communication
2. **Run Database** - SQLite/JSON log of all agent actions
3. **Schema Definitions for New Agents:**
   - `design_intent.json` - Input spec format
   - `rtl_artifact.json` - Generated RTL metadata
   - `constraint_set.json` - SDC/TCL constraints
   - `analysis_report.json` - Lint/CDC/STA results
   - `fix_proposal.json` - Code patches
   - `run_request.json` / `run_result.json` - Execution envelopes

4. **Agent Base Class** - Common interface for A1-A6
5. **Cross-Agent Validation Pipeline**

### 6. Integration Points for New Agents

| Agent | Integrates With | Input Schema | Output Schema |
|-------|-----------------|--------------|---------------|
| A6 (EDA Cmd) | synthesis.py, timing_analysis.py | `design_state` | `run_request` (TCL) |
| A4 (Lint) | synthesis.py logs | `analysis_report` | `fix_proposal` |
| A2 (Templates) | design_parser.py | `design_intent` | `rtl_artifact` |
| A1 (Spec→RTL) | design_parser.py | `design_intent` | `rtl_artifact` |
| A3 (Constraints) | timing_analysis.py | `design_intent` | `constraint_set` |
| A5 (Style) | rule_engine.py | `rtl_artifact` | `analysis_report` |

### 7. Existing APIs to Leverage

#### Synthesis
```python
engine = SynthesisEngine(tech_lib)
result = engine.synthesize(
    rtl_files=['design.v'],
    top_module='top',
    output_netlist='out.v',
    optimization_goal='balanced'  # speed|area|power|balanced
)
# Returns: {'cell_count', 'gate_count', 'area', 'success'}
```

#### Timing Analysis
```python
analyzer = TimingAnalyzer()
result = analyzer.analyze_timing(
    netlist_file='netlist.v',
    sdc_file='constraints.sdc',
    lib_files=['tech.lib']
)
# Returns: {'wns', 'tns', 'critical_path', 'violations'}
```

#### Design State
```python
state = DesignState(project_name='my_chip')
state.update_stage(DesignStage.SYNTHESIZED)
state.update_metrics(timing={'wns': -0.5}, power={'total': 100})
summary = state.get_metrics_summary()
```

#### RL Actions
```python
action_space = ActionSpace(sim_engine, design_state, world_model)
result = action_space.execute_action(Action.OPTIMIZE_WIRELENGTH)
# Returns: {'success', 'metrics_delta', 'info'}
```

### 8. Directory Structure for New Agents

```
cda-agent-2C1/
├── core/
│   ├── rtl_agents/          # NEW - Multi-agent RTL layer
│   │   ├── __init__.py
│   │   ├── base_agent.py    # Abstract base class
│   │   ├── a1_spec_to_rtl.py
│   │   ├── a2_boilerplate_gen.py
│   │   ├── a3_constraint_synth.py
│   │   ├── a4_lint_cdc.py
│   │   ├── a5_style_review.py
│   │   ├── a6_eda_command.py
│   │   └── message_bus.py
│   ├── schemas/             # NEW - JSON schemas
│   │   ├── design_intent.json
│   │   ├── rtl_artifact.json
│   │   ├── constraint_set.json
│   │   ├── analysis_report.json
│   │   └── fix_proposal.json
│   └── ...existing modules...
├── data/
│   └── run_db/              # NEW - Execution logs
│       └── runs.db (SQLite)
└── config/
    ├── agent_directive.md   # Master mission profile
    └── phase0_context_map.md # This document
```

### 9. Validation Strategy

Each new agent must:
1. Accept standardized JSON input (via schemas)
2. Produce structured JSON output with confidence scores
3. Log all I/O to Run DB
4. Validate outputs with existing tools (Yosys/STA)
5. Calculate success metrics
6. Update RL reward signals

### 10. Next Steps (Phase 1)

Build **A6 - EDA Command Copilot** first because:
- Simplest agent (script generation)
- Can leverage existing RAG docs (Yosys/STA syntax)
- Validates integration with simulation_engine
- Provides immediate value (auto-generate TCL)

**Target:** 90% first-run script validity

---

## Phase 0 Status: ✅ COMPLETE

All infrastructure indexed. APIs documented. Schema requirements defined.
Ready to proceed to Phase 1: A6 EDA Command Copilot.
