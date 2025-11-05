# Spec-to-Silicon Platform UI

## Overview

IDE-style web interface for the 6-agent RTL generation pipeline. Built with Gradio for rapid prototyping and deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GRADIO WEB UI                           │
│  (Spec Input | Code Editor | Reports | Dashboard)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                  PIPELINE ORCHESTRATOR                       │
│      (api/pipeline.py - wraps 6-agent workflow)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                    6-AGENT PIPELINE                          │
│  A1 (RTL Gen) → A5 (Style) → A4 (Lint) →                   │
│  A3 (Constraints) → A6 (Synth Script) → Yosys               │
└─────────────────────────────────────────────────────────────┘
```

## Files Created

### Core API Layer

```
api/
├── __init__.py              - Package initialization
├── models.py                - Pydantic data models
│   ├── DesignSpec          - Input specification
│   ├── AgentResult         - Per-agent results
│   ├── PipelineResult      - Complete pipeline output
│   ├── PipelineProgress    - Real-time progress tracking
│   ├── CodeMetrics         - RTL quality metrics
│   └── SynthesisMetrics    - Yosys synthesis data
│
└── pipeline.py              - Pipeline orchestrator
    ├── PipelineOrchestrator  - Main workflow manager
    ├── execute_pipeline()    - Run 6-agent flow
    ├── _run_a1() .. _run_yosys()  - Individual agent runners
    └── _compute_metrics()    - Code quality analysis
```

### UI Layer

```
ui/
└── gradio_app.py            - Gradio web interface
    ├── Spec input panel     - Natural language + JSON params
    ├── Pipeline controls    - Enable/disable agents
    ├── Tabbed results       - RTL, reports, synthesis
    └── Dashboard            - Metrics & status
```

### Launcher

```
launch_ui.sh                 - Start Gradio on port 7860
```

## Features Implemented

### 1. Specification Input
- Module name
- Natural language description
- Data width & clock frequency
- JSON parameters (FIFO depth, etc.)
- Pre-loaded examples (SPI Master, UART)

### 2. Pipeline Configuration
- Toggle each agent on/off
- Run full or partial pipeline
- Background execution with progress tracking

### 3. Results Display
**Tabbed Interface:**
- Generated RTL (syntax-highlighted Verilog)
- Style Review Report (markdown)
- Lint & CDC Results
- SDC Constraints
- Yosys Synthesis Script
- Synthesis Report

**Dashboard Metrics:**
- Pipeline status & duration
- Lines of code generated
- Port count
- Synthesis success/fail
- Per-agent execution times

### 4. Data Models

**DesignSpec:**
```python
{
  "module_name": "SPI_MASTER_001",
  "description": "SPI Master with configurable...",
  "data_width": 8,
  "clock_freq": 100.0,
  "parameters": {"fifo_depth": 8}
}
```

**PipelineResult:**
```python
{
  "run_id": "run_20251030_123456_abc123",
  "status": "success",
  "duration_seconds": 45.2,
  "a1_rtl_generation": AgentResult(...),
  "a5_style_review": AgentResult(...),
  ...
  "rtl_file": "/path/to/generated.v",
  "synthesis_success": True
}
```

## Integration Points

### Agent Method Signatures (NEEDS FIXING)

The `api/pipeline.py` currently has **incorrect imports**. Here's what needs to be fixed:

**Current (broken):**
```python
from core.rtl_agents import (
    A1_DesignSpecPlanner,      # ❌ Does not exist
    A2_ModuleComposer,         # ❌ Does not exist
    ...
)
```

**Should be:**
```python
from core.rtl_agents import (
    A1_SpecToRTLGenerator,     # ✓ Correct
    A2_BoilerplateGenerator,   # ✓ Correct
    A3_ConstraintSynthesizer,  # ✓ Correct
    A4_LintCDCAssistant,       # ✓ Correct
    A5_StyleReviewCopilot,     # ✓ Correct
    A6_EDACommandCopilot       # ✓ Correct
)
```

### Required Agent API

Each agent needs to expose these methods (or similar):

```python
# A1 - RTL Generation
A1_SpecToRTLGenerator().generate(design_spec: Dict) -> str

# A5 - Style Review
A5_StyleReviewCopilot().review_style(rtl_file: Path) -> Dict

# A4 - Lint & CDC
A4_LintCDCAssistant().check_and_fix(rtl_file: Path) -> Dict

# A3 - Constraints
A3_ConstraintSynthesizer().generate_constraints(rtl_file: Path) -> Dict

# A6 - Synthesis Script
A6_EDACommandCopilot().generate_synthesis_script(rtl_file: Path) -> Dict
```

## Installation

```bash
# Install dependencies
pip3 install gradio fastapi uvicorn --break-system-packages

# Or use system packages
sudo apt install python3-gradio python3-fastapi python3-uvicorn
```

## Usage

```bash
# Launch UI
./launch_ui.sh

# Or directly
python3 ui/gradio_app.py
```

**Access at:** http://localhost:7860

## Current Status

✅ **Completed:**
- API data models (Pydantic)
- Pipeline orchestrator structure
- Gradio UI with IDE-style layout
- Tabbed results display
- Dashboard metrics
- Example specifications

⚠️ **Needs Fixing:**
- Agent class name mismatches (imports)
- Agent method signatures
- EDASimulator integration
- Error handling

## Next Steps

### 1. Fix Agent Integration (Priority)

Update `api/pipeline.py` lines 19-28:
```python
# Replace incorrect imports with actual agent classes
from core.rtl_agents import (
    A1_SpecToRTLGenerator,
    A2_BoilerplateGenerator,
    A3_ConstraintSynthesizer,
    A4_LintCDCAssistant,
    A5_StyleReviewCopilot,
    A6_EDACommandCopilot
)
```

Update method calls in `_run_a1()`, `_run_a5()`, etc. to match actual agent APIs.

### 2. Test End-to-End

```bash
# Start UI
python3 ui/gradio_app.py

# Load SPI Master example
# Click "Generate Silicon"
# Verify all 6 agents execute
# Check synthesis report
```

### 3. Add Real-Time Progress

Currently uses callback hooks. Enhance with WebSocket for live updates:
- Loading spinner per agent
- Progress bar (1/6, 2/6, ...)
- Live log streaming

### 4. Dashboard Enhancements

Add visualizations:
- Timing histogram (Plotly)
- Resource utilization (cell count, wires)
- Historical runs comparison
- Quality score trends

### 5. Production Upgrades

- FastAPI REST API (parallel to Gradio)
- React frontend (replace Gradio)
- Database (SQLite → PostgreSQL)
- Authentication (SSO)
- Git integration (commit generated RTL)

## Example Workflow

```
1. User enters: "SPI Master, 8-bit, 100MHz"
2. UI calls: PipelineOrchestrator().execute_pipeline(spec)
3. Orchestrator runs:
   - A1: Generate RTL → spi_master.v
   - A5: Style review → style_report.md
   - A4: Lint/CDC → lint_report.json
   - A3: Constraints → spi_master.sdc
   - A6: Synthesis script → synth_script.ys
   - Yosys: Synthesize → synth_report.txt
4. UI displays all results in tabs
5. Dashboard shows: ✅ 318 lines, 47 ports, synthesis PASS
```

## File Locations

```
data/runs/<run_id>/
├── spec.json              - Original design spec
├── spi_master.v           - Generated RTL
├── style_review.md        - A5 output
├── lint_report.json       - A4 output
├── spi_master.sdc         - A3 output
├── synth_script.ys        - A6 output
├── synthesis_report.txt   - Yosys output
└── result.json            - Complete pipeline result
```

## Dependencies

- **gradio**: Web UI framework
- **fastapi**: REST API (future)
- **pydantic**: Data validation
- **uvicorn**: ASGI server (future)

Already installed:
- **transformers, torch**: For A1 LLM generation
- **yosys**: Synthesis backend

## Known Issues

1. **Import Error**: Agent class names don't match
   - **Fix**: Update `api/pipeline.py` imports

2. **Method Signatures**: Agent APIs may differ
   - **Fix**: Check actual agent methods and adjust wrappers

3. **Pydantic Warning**: `schema_extra` → `json_schema_extra`
   - **Fix**: Update `api/models.py` line 25

4. **No Database**: Results only saved to filesystem
   - **Future**: Add SQLite for run history

## Contact & Support

- **Issues**: See agent method signatures in `core/rtl_agents/`
- **Docs**: This README
- **Examples**: `ui/gradio_app.py` has SPI/UART presets

---

**Built for chip designers** | IDE-style interface | 6-agent AI pipeline
