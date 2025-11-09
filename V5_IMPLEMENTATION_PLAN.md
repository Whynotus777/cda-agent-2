# V5 Implementation Plan: Hybrid Dataset with A7 Verification

**Date:** 2025-11-05
**Goal:** Fix V4 FSM regression and achieve 88%+ overall functional correctness
**Key Insight:** Testbench quality must match domain complexity

---

## V5 Strategy: "Greatest Hits" Hybrid Dataset

### Target Composition

```
Total: ~900 examples (optimal for 5-minute training)

Component A: V4 Examples (Domains that Improved)
  edge_detector      42 examples (100% in V4 vs 60% in V3) ✅
  async_reset        ~40 examples (100% in V4 vs 80% in V3) ✅
  enable_control     ~40 examples (100% in V4 vs 80% in V3) ✅
  shift_register     185 examples (80% in V4 vs 60% in V3) ✅
  counter_updown     56 examples (100% in V4 vs 80% in V3) ✅
  counter_load       51 examples (100% in V4 vs 80% in V3) ✅
  ─────────────────────────────────
  Subtotal:          ~414 examples (keep V4 wins)

Component B: V3 Examples (Domains that Regressed)
  clock_divider      All V3 examples (100% in V3, 80% in V4)
  enable_gating      All V3 examples (100% in V3, 80% in V4)
  fsm_*              All V3 FSM examples (60% in V3, 20% in V4)
  ─────────────────────────────────
  Subtotal:          ~300 examples (recover V3 baseline)

Component C: New A7-Verified FSM Examples
  fsm_2state         35 examples (A7-verified)
  fsm_3state         35 examples (A7-verified)
  fsm_handshake      30 examples (A7-verified)
  fsm_sequence       30 examples (A7-verified)
  fsm_mealy          20 examples (A7-verified)
  fsm_moore          20 examples (A7-verified)
  ─────────────────────────────────
  Subtotal:          170 examples (NEW gold standard)

Component D: Edge Cases & Real-World Patterns
  Pipeline trace.jsonl successful examples: ~20
  Edge cases (1-bit counters, 0-shift, etc.): ~20
  ─────────────────────────────────
  Subtotal:          ~40 examples

TOTAL:               ~924 examples
```

---

## Implementation Phases

### Phase 1: Dataset Analysis & Extraction (30 minutes)

**Task 1.1: Identify V4 Success Examples**
```bash
python3 scripts/extract_v4_winners.py
```

Input: `data/rtl_behavioral_v4.jsonl`
Output: `data/v5_component_a.jsonl`

Logic:
- Extract categories that improved in V4 benchmark:
  - edge_detector, async_reset, enable_control
  - shift_* (all shift register variants)
  - counter_updown, counter_load
- Exclude all FSM examples from V4
- Exclude clock_divider and enable_gating (regressed)

Expected: ~414 examples

**Task 1.2: Extract V3 Baseline Examples**
```bash
python3 scripts/extract_v3_baseline.py
```

Input: `data/rtl_behavioral_v3.jsonl` (if exists) or regenerate from templates
Output: `data/v5_component_b.jsonl`

Logic:
- Extract FSM examples from V3
- Extract clock_divider examples
- Extract enable_gating examples
- These provide the "floor" - model must maintain V3 performance

Expected: ~300 examples

### Phase 2: A7 LLM Testbench Integration (1 hour)

**Task 2.1: Check A7 Agent Availability**
```bash
# Check if A7 is integrated in the pipeline
grep -r "A7\|testbench.*agent" src/agents/
```

**Task 2.2: Create A7 Testbench Generator Module**

File: `scripts/a7_testbench_generator.py`

```python
"""
A7 LLM Testbench Generator for Complex RTL Verification

For FSMs and other complex domains, generates comprehensive testbenches
that test all state transitions, edge cases, and corner cases.
"""

import anthropic
from typing import Dict, Tuple

class A7TestbenchGenerator:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_fsm_testbench(
        self,
        spec: str,
        rtl_code: str,
        category: str
    ) -> str:
        """
        Generate comprehensive FSM testbench using A7 LLM.

        Returns: SystemVerilog testbench code
        """

        prompt = f"""You are an expert RTL verification engineer. Generate a comprehensive SystemVerilog testbench for the following FSM design.

**Specification:**
{spec}

**RTL Implementation:**
{rtl_code}

**Requirements:**
1. Test ALL state transitions (not just happy path)
2. Include assertions for illegal state transitions
3. Test reset during active states
4. Test rapid input changes
5. Verify output correctness in each state
6. Include edge cases (e.g., holding inputs, back-to-back transitions)
7. Use $display for clear pass/fail indication
8. Final line MUST be either "TEST PASSED" or "TEST FAILED"

**Testbench Structure:**
```systemverilog
module testbench;
    // Signals
    reg clk, rst_n;
    reg [inputs] ...;
    wire [outputs] ...;

    // Instantiate DUT
    <module_name> dut (...);

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test cases
    initial begin
        // Reset test
        // State transition tests
        // Edge case tests
        // Assertions

        $display("TEST PASSED");
        $finish;
    end
endmodule
```

Generate the complete, executable testbench:"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        testbench_code = response.content[0].text

        # Extract SystemVerilog code block
        if "```systemverilog" in testbench_code:
            testbench_code = testbench_code.split("```systemverilog")[1].split("```")[0]
        elif "```" in testbench_code:
            testbench_code = testbench_code.split("```")[1].split("```")[0]

        return testbench_code.strip()
```

**Task 2.3: Test A7 Generator with Sample FSM**
```bash
python3 scripts/test_a7_generator.py
```

Verify:
- A7 generates valid SystemVerilog testbench
- Testbench compiles with iverilog
- Testbench tests multiple state transitions
- Clear pass/fail output

### Phase 3: Generate A7-Verified FSM Examples (2-3 hours)

**Task 3.1: Create V5 FSM Generator**

File: `scripts/generate_v5_fsm_dataset.py`

```python
"""
V5 FSM Dataset Generator with A7 LLM Testbenches

Strategy:
1. Generate FSM spec and RTL (using existing V4 generator logic)
2. Use A7 to generate comprehensive testbench
3. Run simulation with A7 testbench
4. Keep only examples that pass A7 verification
5. Target: 170 gold-standard FSM examples
"""

from a7_testbench_generator import A7TestbenchGenerator
from run_iverilog_simulation import run_simulation
import json
from pathlib import Path

def generate_v5_fsm_examples(
    target_count: int = 170,
    max_attempts: int = 400,
    output_file: str = "data/v5_component_c.jsonl"
):
    """
    Generate A7-verified FSM examples.

    Expected pass rate: ~40-50% (A7 testbenches are strict!)
    """

    a7 = A7TestbenchGenerator(api_key=os.getenv("ANTHROPIC_API_KEY"))

    fsm_categories = [
        ("fsm_2state", 35),
        ("fsm_3state", 35),
        ("fsm_handshake", 30),
        ("fsm_sequence", 30),
        ("fsm_mealy", 20),
        ("fsm_moore", 20),
    ]

    results = []

    for category, target in fsm_categories:
        print(f"\n{'='*80}")
        print(f"Generating {category} (target: {target})")
        print(f"{'='*80}\n")

        passed = 0
        attempts = 0
        max_cat_attempts = max_attempts // len(fsm_categories)

        while passed < target and attempts < max_cat_attempts:
            attempts += 1

            # Generate spec and RTL (reuse V4 logic)
            spec = generate_fsm_spec(category)
            rtl_code = generate_rtl_with_model(spec)

            # Use A7 to generate testbench
            try:
                testbench = a7.generate_fsm_testbench(spec, rtl_code, category)
            except Exception as e:
                print(f"  [{attempts}] A7 generation failed: {e}")
                continue

            # Run simulation
            sim_result = run_simulation(rtl_code, testbench)

            if sim_result["passed"]:
                passed += 1
                print(f"  [{attempts}] ✅ PASSED ({passed}/{target})")

                results.append({
                    "instruction": spec,
                    "input": "",
                    "output": rtl_code,
                    "metadata": {
                        "source": "v5_a7_verified",
                        "category": category,
                        "functional": "verified",
                        "testbench_type": "a7_llm_generated",
                        "sim_passed": True,
                        "compile_success": True,
                        "a7_model": "claude-sonnet-4",
                        "generated_at": datetime.now().isoformat()
                    }
                })
            else:
                print(f"  [{attempts}] ❌ FAILED")

        print(f"\n{category} complete: {passed}/{target} ({passed/target*100:.1f}%)")

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for example in results:
            f.write(json.dumps(example) + '\n')

    print(f"\n{'='*80}")
    print(f"V5 Component C complete: {len(results)} examples")
    print(f"{'='*80}")

    return results
```

**Task 3.2: Run FSM Generation**
```bash
source venv/bin/activate
python3 scripts/generate_v5_fsm_dataset.py 2>&1 | tee generate_v5_fsm.log
```

Expected:
- Duration: 2-3 hours (A7 calls + simulation per example)
- Pass rate: 40-50% (A7 testbenches are more rigorous)
- Output: 170 gold-standard FSM examples

### Phase 4: Combine & Validate V5 Dataset (15 minutes)

**Task 4.1: Merge All Components**

File: `scripts/create_v5_dataset.py`

```python
"""
Combine V5 dataset components into final training dataset.
"""

import json
from pathlib import Path

def create_v5_dataset():
    components = [
        "data/v5_component_a.jsonl",  # V4 winners
        "data/v5_component_b.jsonl",  # V3 baseline
        "data/v5_component_c.jsonl",  # A7-verified FSMs
        # "data/v5_component_d.jsonl",  # Edge cases (optional)
    ]

    all_examples = []

    for comp_file in components:
        if not Path(comp_file).exists():
            print(f"⚠️  Missing: {comp_file}")
            continue

        with open(comp_file) as f:
            examples = [json.loads(line) for line in f]
            all_examples.extend(examples)
            print(f"✅ {comp_file}: {len(examples)} examples")

    # Shuffle for training
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Save final dataset
    output_file = "data/rtl_behavioral_v5.jsonl"
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n{'='*80}")
    print(f"V5 Dataset Created: {output_file}")
    print(f"Total examples: {len(all_examples)}")
    print(f"{'='*80}")

    # Print category breakdown
    from collections import Counter
    categories = [ex["metadata"]["category"] for ex in all_examples]
    breakdown = Counter(categories)

    print("\nCategory Breakdown:")
    for cat, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        pct = count / len(all_examples) * 100
        print(f"  {cat:25s} {count:4d} ({pct:5.1f}%)")

    return all_examples

if __name__ == "__main__":
    create_v5_dataset()
```

**Task 4.2: Validate Dataset**
```bash
python3 scripts/create_v5_dataset.py
```

Verify:
- Total count: 900-1000 examples
- FSM representation: ~470 examples (300 V3 + 170 A7)
- No duplicates
- All examples have required metadata

### Phase 5: Train V5 Model (5 minutes)

**Task 5.1: Update Training Script**

Use existing `scripts/train_qwen_coder_qlora.py` but specify V5 dataset:

```bash
source venv/bin/activate
python3 scripts/train_qwen_coder_qlora.py \
    --dataset data/rtl_behavioral_v5.jsonl \
    --output_dir models/qwen_coder_rtl/run_behavioral_v5_$(date +%Y%m%d_%H%M%S) \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    2>&1 | tee training_behavioral_v5.log
```

Expected:
- Duration: ~5 minutes
- Loss reduction: Similar to V4 (0.54 → 0.25)
- Output: `models/qwen_coder_rtl/run_behavioral_v5_*/final_model`

### Phase 6: Benchmark V5 (2 minutes)

**Task 6.1: Run Benchmark**

```bash
# Update benchmark script to use V5 model
python3 scripts/benchmark_v4_direct.py  # Modify to point to V5

# Or create dedicated script
python3 scripts/benchmark_v5.py 2>&1 | tee benchmark_v5_results.log
```

**Success Criteria:**
- Overall functional: ≥ 88% (V3: 84%, V4: 86%)
- **FSM functional: ≥ 70%** (V3: 60%, V4: 20% ❌)
- Shift registers: ≥ 80% (maintain V4 gains)
- Edge detection: ≥ 90% (maintain V4 gains)
- No category below V3 baseline

### Phase 7: Deployment (if successful)

If V5 meets success criteria:

1. **Update pipeline configuration:**
   ```python
   # In pipeline config
   BEHAVIORAL_MODEL = "models/qwen_coder_rtl/run_behavioral_v5_*/final_model"
   ```

2. **Enable continuous improvement loop:**
   - Mine successful examples from trace.jsonl
   - Add to V6 training data
   - Retrain periodically

3. **Monitor production metrics:**
   - Track functional pass rate in production
   - Identify new weak domains
   - Plan V6 targeted improvements

---

## Risk Mitigation

**Risk 1: A7 API rate limits**
- Mitigation: Generate in batches, add retry logic
- Fallback: Use smaller target (100 FSM examples vs 170)

**Risk 2: A7 testbenches too strict (low pass rate)**
- Mitigation: If pass rate < 30%, relax some test requirements
- Alternative: Use A7 for 50% of FSMs, template for rest

**Risk 3: V5 still regresses on FSMs**
- Mitigation: Increase V3 FSM component (300 → 400 examples)
- Analysis: Review A7 testbench quality, may need prompt refinement

**Risk 4: Training time exceeds budget**
- Mitigation: Reduce total examples (900 → 750)
- Priority: Keep A7-verified FSMs, reduce other categories proportionally

---

## Timeline Summary

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Dataset Analysis & Extraction | 30 min |
| 2 | A7 Integration | 1 hour |
| 3 | Generate A7-Verified FSMs | 2-3 hours |
| 4 | Combine & Validate | 15 min |
| 5 | Train V5 | 5 min |
| 6 | Benchmark V5 | 2 min |
| 7 | Deploy (if successful) | 2 hours |
| **TOTAL** | **End-to-end** | **~4-5 hours** |

---

## Success Metrics

V5 will be considered **production-ready** if:

1. ✅ Overall functional ≥ 88%
2. ✅ FSM functional ≥ 70% (fixes V4 regression)
3. ✅ All categories ≥ V3 baseline
4. ✅ Maintains V4 gains (shift registers, edge detect)

If these metrics are achieved, **V5 becomes the production model** and establishes the continuous improvement loop for V6, V7, etc.

---

## Key Insights Applied

1. **"You can only teach what you can test"** → Use A7 for complex domains
2. **Quality > Quantity** → 170 A7-verified FSMs > 222 template-verified
3. **Don't discard what works** → Keep V3 baseline + V4 wins
4. **Hybrid approach** → Template TBs for simple domains, A7 for complex

This is the path to robust, production-ready RTL generation.
