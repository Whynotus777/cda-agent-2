# Behavioral V3 Training Plan

## 📊 V2 Results Analysis (The Smoking Gun)

### Benchmark Results
```
Average Verifier Score: 0.453
Syntax Valid: 64.0%
Synthesis Success: 48.0%
All I/O Used: 0.0% ⚠️  THE SMOKING GUN
Functional %: 0.0%
```

### Category Performance
| Category | Score | Status |
|----------|-------|--------|
| clock_divider | 0.675 | Best |
| load_enable | 0.610 | Good |
| updown_counter | 0.495 | Okay |
| **shift_register** | **0.220** | ❌ Worst |
| **edge_detect** | **0.330** | ❌ Poor |
| **fsm** | **0.365** | ❌ Poor |

### Root Cause Identified

**The Problem:** Model generates syntactically correct but functionally wrong code.

**Example of the issue:**
```systemverilog
// Spec: 8-bit counter with enable and sync reset
module counter_8bit(
  input clk,
  input rst,
  input en,       // <-- DECLARED but NEVER USED ❌
  output [7:0] q  // <-- DECLARED but NEVER ASSIGNED ❌
);
  reg [7:0] count_reg;

  always @(posedge clk) begin
    if (rst)
      count_reg <= 8'd0;
    else
      count_reg <= count_reg + 1; // <-- 'en' is ignored!
  end

  // q is never assigned!
endmodule
```

**Why this happens:**
1. Training threshold of 0.7 was **too lenient**
2. Allowed "synthesizable but wrong" examples
3. Model learned structure but not behavioral correctness
4. Ports are declared but logic doesn't use them

---

## 🎯 V3 Strategy: Quality Over Quantity

### Key Changes

#### 1. **Stricter Filtering: 0.7 → 0.95**
```python
# OLD (v2): Too lenient
if result.score >= 0.7:  # Let in 1,003 examples (42.7%)

# NEW (v3): Gold standard only
if result.score >= 0.95:  # Expect ~100-200 examples (~5-10%)
```

**Expected outcome:**
- Smaller dataset (100-200 vs 1,003 examples)
- **MUCH higher quality** - only examples where:
  - All ports are used (all_io_used = True)
  - Synthesis succeeds
  - No errors
  - Few/no warnings
  - Spec keywords correctly implemented

#### 2. **Expanded Diagnostic Corpus**

**Current weakness** (from benchmark):
- shift_register: 0.220
- edge_detect: 0.330
- fsm: 0.365

**Action:** Add 15-20 more gold-standard examples for these categories.

---

## 🔧 Implementation Steps

### Step 1: Strict Re-filtering ✓ (In Progress)
```bash
cd ~/cda-agent-2C1
source venv/bin/activate
python3 scripts/verify_training_data.py  # Now uses threshold=0.95
```

**Current status:** Running
**Output:** `data/rtl_verified_training.jsonl` (expect ~100-200 examples)

### Step 2: Expand Diagnostic Corpus (TODO)

Add to `data/training/diagnostic_corpus.jsonl`:

**FSM Examples (10 new):**
- 3-state traffic light FSM with timer
- 4-state sequence detector FSM
- Moore vs Mealy machine examples
- FSM with output logic

**Shift Register Examples (10 new):**
- Right shift register variants
- Circular shift register
- Shift register with parallel load
- Bidirectional shift register

**Edge Detection Examples (5 new):**
- Falling edge detector
- Dual-edge detector
- Edge detector with synchronizer
- Pulse stretcher/debouncer

### Step 3: Combine for V3
```bash
python3 scripts/combine_verified_datasets.py
# Output: data/rtl_behavioral_v3.jsonl
```

### Step 4: Update Training Script
```python
# scripts/train_qwen_coder_qlora.py
dataset_path = project_root / 'data' / 'rtl_behavioral_v3.jsonl'
output_dir = project_root / 'models' / 'qwen_coder_rtl' / f"run_behavioral_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### Step 5: Retrain
```bash
python3 scripts/train_qwen_coder_qlora.py  # Same hyperparams: lr=2e-4, 3 epochs
```

### Step 6: Benchmark & Compare
```bash
python3 scripts/benchmark_behavioral_model.py
```

---

## 📈 Expected Improvements (V2 → V3)

| Metric | V2 Result | V3 Target | Rationale |
|--------|-----------|-----------|-----------|
| Avg Verifier Score | 0.453 | **≥ 0.75** | Stricter training data |
| Synthesis Success | 48% | **≥ 90%** | Better syntax learning |
| **All I/O Used** | **0%** | **≥ 85%** | 🎯 **Primary goal** |
| Functional % | 0% | **≥ 70%** | Proper port usage |
| shift_register | 0.220 | **≥ 0.65** | More gold examples |
| edge_detect | 0.330 | **≥ 0.70** | More gold examples |
| fsm | 0.365 | **≥ 0.70** | More gold examples |

**Success Criteria:**
- ✅ All I/O Used ≥ 85% (critical)
- ✅ Average Score ≥ 0.75
- ✅ Weakest category ≥ 0.65

---

## 🔬 Why V3 Will Work

### The Training Loop Insight
```
V1 (Comprehensive): 2,347 examples (unfiltered)
→ Result: Syntactically correct, functionally wrong
→ Lesson: Too much noise, no quality filter

V2 (Behavioral): 1,048 examples (filtered at 0.7)
→ Result: Still functionally wrong (All I/O Used: 0%)
→ Lesson: 0.7 threshold too lenient - let in bad examples

V3 (Gold Standard): ~150-250 examples (filtered at 0.95)
→ Hypothesis: Small but perfect dataset will teach correctness
→ Focus: Every example must have ALL ports used correctly
```

### Why Small+Perfect > Large+Noisy

**Quality signals the model learns:**
1. **Port usage is mandatory**
   - If `enable` is declared → it MUST appear in logic
   - If `output` is declared → it MUST be assigned

2. **Spec keywords map to code patterns**
   - "with enable" → `if (en)`
   - "synchronous reset" → `@(posedge clk) if (rst)`
   - "asynchronous reset" → `@(posedge clk or negedge rst_n)`

3. **Complete implementations only**
   - No half-finished modules
   - No unused ports
   - No unconnected signals

---

## 📝 Files Modified

### Updated for V3:
- ✅ `scripts/verify_training_data.py` - Changed threshold to 0.95
- ✅ `scripts/combine_verified_datasets.py` - Output to rtl_behavioral_v3.jsonl
- ⏳ `scripts/train_qwen_coder_qlora.py` - Need to update dataset_path

### To Create:
- ⏳ `data/training/diagnostic_corpus_expanded.jsonl` - Add 15-20 FSM/shift_reg/edge examples
- ⏳ `data/rtl_behavioral_v3.jsonl` - Final v3 training dataset

### Generated:
- ⏳ `data/rtl_verified_training.jsonl` - Strict filtered (≥0.95)
- ⏳ `verification_strict_0.95.log` - Filtering logs

---

## 🚀 Quick Start Commands

```bash
cd ~/cda-agent-2C1
source venv/bin/activate

# 1. Wait for strict filtering to complete (running in background)
tail -f verification_strict_0.95.log

# 2. Expand diagnostic corpus (TODO - manual)
#    Add 15-20 gold examples to data/training/diagnostic_corpus.jsonl

# 3. Combine verified + expanded diagnostic
python3 scripts/combine_verified_datasets.py

# 4. Update training script dataset path to v3
#    Edit scripts/train_qwen_coder_qlora.py line 287

# 5. Retrain
python3 scripts/train_qwen_coder_qlora.py

# 6. Benchmark
python3 scripts/benchmark_behavioral_model.py
```

---

## 📊 Comparison Matrix

| Version | Examples | Threshold | Avg Score | All I/O Used | Status |
|---------|----------|-----------|-----------|--------------|--------|
| V1 (Comprehensive) | 2,347 | None | ~0.45 | 0% | ❌ Too noisy |
| V2 (Behavioral) | 1,048 | 0.7 | 0.453 | 0% | ❌ Still noisy |
| **V3 (Gold)** | **~200** | **0.95** | **Target: 0.75+** | **Target: 85%+** | 🎯 **In progress** |

---

## 🎓 Lessons Learned

### From V1 → V2:
- ✅ Tool-in-the-loop validation is essential
- ✅ Semantic grading (Yosys/Verilator) > regex validation
- ✅ Diagnostic corpus helps with specific patterns

### From V2 → V3:
- ✅ **Quality threshold matters more than dataset size**
- ✅ "All I/O Used" is the key metric for functional correctness
- ✅ 0.7 score allows "synthesizable but wrong" code
- ✅ Model needs examples where ports are ALWAYS used
- ✅ Weak categories need more gold-standard examples

---

## 🔄 Future Iterations (V4+)

If V3 succeeds (All I/O Used ≥ 85%):
1. **Add simulation-based verification** (cocotb/iverilog)
2. **Reinforcement learning from verifier feedback**
3. **Expand to more complex modules** (FIFOs, arbiters, caches)
4. **Multi-turn refinement** (generate → verify → fix → verify)

If V3 still struggles:
1. **Raise threshold even higher** (0.98 or 1.0)
2. **Hand-curate all training examples** (100% manual verification)
3. **Add more diagnostic corpus examples** (500+ gold examples)
4. **Consider smaller, specialized models** (one model per category)

---

**Last Updated:** 2025-11-05 22:43 UTC
**Status:** Strict filtering (≥0.95) in progress
**Next Action:** Expand diagnostic corpus while filtering completes
