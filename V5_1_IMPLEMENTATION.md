# V5.1 Gold-Standard Dataset Implementation

## Overview

V5.1 implements the **A7 LLM Testbench verification** strategy to recover FSM performance while maintaining V4 gains.

## Key Innovation: A7 LLM Testbenches

**Problem Identified in V4:**
- Template testbenches passed FSMs with subtle bugs
- 222 FSM training examples in V4, but only 20% test accuracy
- Inverse relationship: More training data → Worse performance
- Root cause: "You can only teach what you can test"

**V5.1 Solution:**
- Use Claude (A7) to generate comprehensive, state-aware testbenches
- A7 understands FSM semantics and can test:
  - ALL state transitions (not just happy path)
  - Edge cases (rapid inputs, back-to-back transitions)
  - Illegal states with assertions
  - Complex state sequences

## Implementation Status

### ✅ Completed

1. **A7 Testbench Generator** (`scripts/a7_testbench_generator.py`)
   - `A7TestbenchGenerator` class with Anthropic API integration
   - `generate_fsm_testbench()` with comprehensive FSM-specific prompts
   - Requires: State coverage, transition coverage, reset testing, edge cases, assertions
   - Temperature=0 for deterministic generation
   - Extracts SystemVerilog code from LLM responses

2. **V5.1 Dataset Generation Pipeline** (`scripts/generate_v5_1_dataset.py`)
   - `V51DatasetGenerator` class orchestrating full pipeline
   - Three-phase approach:
     1. Generate 150 A7-verified FSM examples
     2. Curate ~650 simple domain examples from V4 winners
     3. Combine into ~800 example gold-standard dataset
   - Integrates A7 testbench generation + RTL generation + simulation
   - Comprehensive statistics tracking

### ⚠️ Prerequisites Required

**ANTHROPIC_API_KEY must be set:**

```bash
export ANTHROPIC_API_KEY="your-key-here"
# Or add to .env file:
echo "ANTHROPIC_API_KEY=your-key-here" >> .env
```

Without this key, A7 testbench generation will fail.

## V5.1 Strategy

### Dataset Composition

**Target: ~800 examples**

1. **150 A7-verified FSM examples** (18.75%)
   - 2-state FSMs (basic)
   - 3-state FSMs (medium complexity)
   - 4-state FSMs (medium/complex)
   - 5+ state FSMs (complex)
   - Mealy/Moore machines
   - Pattern detectors, controllers, protocols

2. **~650 Simple domain examples** (81.25%)
   - Curated from V4 winners (categories with 80%+ performance)
   - Categories: edge_detect, shift_register, counters, registers, mux, comparator, etc.
   - Excludes FSMs (covered by A7-verified examples)

### Why This Will Work

**V4 Regression Analysis:**
- FSM category: 222 examples → 20% accuracy (BAD)
- Edge detect category: 87 examples → 100% accuracy (GOOD)
- Shift register category: 124 examples → 100% accuracy (GOOD)

**Key Insight:** Problem wasn't quantity, but **quality of verification**

**V5.1 Advantages:**
1. FSMs verified by comprehensive A7 testbenches (no false positives)
2. Simple domains use proven V4 winners (verified to work)
3. Clean dataset: Every example truly functional
4. Balanced: 150 complex FSMs + 650 proven simple examples

## Execution Plan

### Phase 1: Generate V5.1 Dataset

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Generate dataset (will take several hours)
source venv/bin/activate
python3 scripts/generate_v5_1_dataset.py 2>&1 | tee generate_v5_1_dataset.log
```

**Expected:**
- ~450 FSM generation attempts to get 150 passing (33% pass rate expected)
- A7 testbench generation: ~1 minute per FSM
- RTL generation + simulation: ~10 seconds per FSM
- Total time: ~3-5 hours for complete dataset

### Phase 2: Train V5.1 Model

The training script needs to be updated to use the V5.1 dataset:

```python
# In scripts/train_qwen_coder_qlora.py, line 287:
dataset_path = project_root / 'data' / 'rtl_behavioral_v5_1.jsonl'
output_dir = project_root / 'models' / 'qwen_coder_rtl' / f"run_behavioral_v5_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

Then train:

```bash
source venv/bin/activate
python3 scripts/train_qwen_coder_qlora.py 2>&1 | tee training_behavioral_v5_1.log
```

**Expected:**
- ~720 training examples, ~80 validation (90/10 split)
- Training time: ~3-4 minutes (same as V5.0)
- Loss reduction: Should see similar 50%+ reduction

### Phase 3: Benchmark V5.1

Create `scripts/benchmark_v5_1.py` (copy from `benchmark_v5.py`):

```bash
source venv/bin/activate
python3 scripts/benchmark_v5_1.py 2>&1 | tee benchmark_v5_1_results.log
```

**Target Performance:**
- **Overall Functional: 90%+** (vs V4: 86%, V5.0: 86%)
- **FSM Functional: 75%+** (vs V4: 20%, V5.0: 40%, V3: 60%)
- Maintain V4 gains in simple domains (edge detect, shifts, counters)

## Success Criteria

### V5.1 is successful if:

1. ✅ **FSM performance ≥ 75%** (recovers and exceeds V3 baseline of 60%)
2. ✅ **Overall performance ≥ 90%** (exceeds V4/V5.0 at 86%)
3. ✅ **No regressions in simple domains** (edge detect, shifts still 100%)

### If successful:
- Deploy V5.1 as production model
- Demonstrates verification-in-the-loop with LLM testbenches works
- Proves "gold standard from scratch" approach superior to remixing

### If unsuccessful:
- Proceed to V5.2 with even more A7-verified FSM examples (300+)
- Consider A7 verification for other complex categories
- Analyze failure modes in V5.1 FSM examples

## Technical Details

### A7 Testbench Prompt Structure

```
Generate comprehensive SystemVerilog testbench for FSM:

Requirements:
1. State Coverage: Test ALL states (not just happy path)
2. Transition Coverage: Test ALL valid transitions + edge cases
3. Reset Behavior: Test reset in IDLE and during active states
4. Edge Cases: Back-to-back transitions, invalid inputs
5. Assertions: Add assertions for illegal states
6. Pass/Fail: Final line must be "TEST PASSED" or "TEST FAILED"
```

### Dataset Format

```json
{
  "instruction": "Create a 4-state FSM for...",
  "input": "",
  "output": "module fsm_4state (...); endmodule",
  "metadata": {
    "source": "v5.1_a7_verified",
    "category": "fsm_4state",
    "complexity": "medium",
    "verification": "a7_testbench",
    "timestamp": "2025-11-06T..."
  }
}
```

## Files Created

1. `scripts/a7_testbench_generator.py` - A7 LLM testbench generator class
2. `scripts/generate_v5_1_dataset.py` - V5.1 dataset generation pipeline
3. `V5_1_IMPLEMENTATION.md` - This document

## Next Steps

1. **Set ANTHROPIC_API_KEY** in environment or .env file
2. **Run dataset generation:** `python3 scripts/generate_v5_1_dataset.py`
3. **Update training script** to use v5_1 dataset
4. **Train V5.1 model**
5. **Benchmark V5.1** and compare to V4/V5.0 baselines
6. **Deploy if successful** (≥90% overall, ≥75% FSM)

---

**V5.1 Hypothesis:** LLM-generated testbenches can provide comprehensive verification that template-based approaches cannot, enabling successful training on complex FSM domains.
