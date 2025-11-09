# Pipeline Feedback Loops (A1 ↔ A7 ↔ Simulation)

This document captures the structural upgrades made to the orchestrator so the rest of
the team can plug additional agents/automation into the new flow without reverse
engineering the code changes.

## High-Level Flow

```
Spec → A1 (RTL) → A5 (Style) → A4 (Lint) → A7 (Testbench) → iverilog/vvp Simulation
       ↑                                                     ↓
       └── DecisionManager ←───────────── Simulation verdict ─┘
```

- **A1/A5/A4** still run in order, but their outputs now feed into a verification
  loop instead of jumping straight to synthesis.
- **A7** is now a first-class stage. It can use either a deterministic template or a
  Qwen-based LLM backend (controlled via `USE_A7_LLM`).
- **Simulation** is mandatory when enabled. The pipeline runs `scripts/run_simulation.py`
  (iverilog + vvp) and evaluates the `status` of the run.
- **DecisionManager** (`api/decision_manager.py`) decides what to do on failure:
  1. Retry A4 once to pick up lint-driven fixes.
  2. Regenerate RTL via A1 (once) if lint retry was insufficient.
  3. Abort if both strategies are exhausted.

## Key Files to Check

| Component | Location | Notes |
|-----------|----------|-------|
| Decision logic | `api/decision_manager.py` | Centralises retry budgets & actions |
| Orchestrator | `api/pipeline.py` | `_execute_generation_with_feedback` handles loop |
| Testbench agent | `core/verification/testbench_generator.py` | Optional LLM generation |
| Simulation wrapper | `scripts/run_simulation.py` | Shared helper used by orchestrator |
| Trace log | `data/runs/<run_id>/trace.jsonl` | Structured per-stage log for downstream tooling |

## Configuration & Environment Flags

- **Enable simulation loop**: on by default; disable via `enable_agents["simulation"]=False`.
- **A7 LLM backend**:
  - `USE_A7_LLM=1` to enable.
  - Optional overrides:
    - `A7_LLM_ADAPTER=/path/to/adapter`
    - `A7_LLM_BASE_MODEL=...` (defaults to `Qwen/Qwen2.5-Coder-7B-Instruct`)
    - `A7_LLM_MAX_TOKENS`, `A7_LLM_TEMPERATURE`, `A7_LLM_TOP_P`
- **DecisionManager budgets**: currently 1 lint retry + 1 regeneration retry.
  Tune in `DecisionManager(...)` (constructor arguments).

## Run Artefacts

Every pipeline run now emits:

- `trace.jsonl`: append-only stream of stage results (status, metrics, artefact paths).
- `simulation/`: compile & simulation logs keyed by simulation job ID.
- `verification/`: generated testbench code, prompt, and RAG context.
- `result.json`: includes new fields
  (`a7_testbench`, `simulation_log`, `functional_success`, `testbench_file`).

These artefacts are designed for:

- Offline triage (replay the exact inputs that led to a failure).
- Automated dataset construction (filter trace entries where `Simulation` stage passed).

## Extending the Loop

- Hooking new fix agents (e.g., automated RTL patching) only requires adding a new
  `RemediationAction` and handling branch in `_execute_generation_with_feedback`.
- Additional metrics/telemetry can be injected via `_log_stage_result`—all consumers
  get them automatically in `trace.jsonl`.
- To integrate richer test generation, replace the template fallback in A7 or plug
  into the prompt/context files written during each run.

For any questions, ping the owner of the pipeline or reference commit history that
introduced these changes. This document should stay in sync with future iterations.
