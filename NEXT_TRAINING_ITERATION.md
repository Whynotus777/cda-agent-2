# Next Training Iteration: Behavioral Correctness

## ✅ What We Built (Step 1: Tools)

### 1. RTL Semantic Verifier (`scripts/rtl_verifier.py`)
- **Replaces** regex-based "looks good" validation
- **Uses** Yosys synthesis to check:
  - Syntax validity
  - Synthesis success
  - Unconnected nets
  - Port usage
  - Spec keyword usage (enable, reset, etc.)
- **Outputs** scored verification result (0.0-1.0)

### 2. Diagnostic Corpus Generator (`scripts/generate_diagnostic_corpus.py`)
- **Generated** 45 verified examples focusing on:
  - Synchronous/asynchronous reset
  - Enable control & clock gating
  - Load/enable combinations
  - Up/down counters
  - Clock dividers
  - Edge detection
  - Shift registers
- **Located** at: `data/training/diagnostic_corpus.jsonl`
- **Next**: Expand to 200-500 examples (add FSMs, FIFOs, more edge cases)

### 3. Model Test Script (`scripts/test_model_terminal.py`)
- Quick testing of trained model
- Tests 3 RTL generation scenarios

---

## 📋 Implementation Plan: Tool-in-the-Loop Training

###  **Phase 1: Verify Existing Training Data (NOW)**

```bash
cd ~/cda-agent-2C1
source venv/bin/activate

# Run verifier on existing training examples
python3 scripts/verify_training_data.py
```

**Create `scripts/verify_training_data.py`:**
```python
#!/usr/bin/env python3
"""Verify existing training data with semantic grader"""
from pathlib import Path
import json
from rtl_verifier import RTLVerifier

verifier = RTLVerifier()
project_root = Path(__file__).parent.parent
dataset_path = project_root / 'data' / 'rtl_comprehensive_training.jsonl'

good_examples = []
bad_examples = []

with dataset_path.open('r') as f:
    for line_num, line in enumerate(f, 1):
        ex = json.loads(line)
        result = verifier.verify(
            rtl_code=ex['output'],
            spec=ex['instruction']
        )

        if result.score >= 0.7:
            good_examples.append((ex, result))
        else:
            bad_examples.append((ex, result))

        if line_num % 100 == 0:
            print(f"Processed {line_num}, Good: {len(good_examples)}, Bad: {len(bad_examples)}")

print(f"\n✅ Good examples (score ≥ 0.7): {len(good_examples)}")
print(f"❌ Bad examples (score < 0.7): {len(bad_examples)}")

# Save filtered dataset
output_path = project_root / 'data' / 'rtl_verified_training.jsonl'
with output_path.open('w') as f:
    for ex, result in good_examples:
        f.write(json.dumps(ex) + '\n')

print(f"\n💾 Saved verified dataset: {output_path}")
```

---

### **Phase 2: Combine with Diagnostic Corpus**

```bash
# Combine verified examples + diagnostic corpus
python3 scripts/combine_verified_datasets.py
```

**Create `scripts/combine_verified_datasets.py`:**
```python
#!/usr/bin/env python3
"""Combine verified training data with diagnostic corpus"""
from pathlib import Path
import json

project_root = Path(__file__).parent.parent

# Load verified examples
verified_path = project_root / 'data' / 'rtl_verified_training.jsonl'
diagnostic_path = project_root / 'data' / 'training' / 'diagnostic_corpus.jsonl'
output_path = project_root / 'data' / 'rtl_behavioral_training.jsonl'

combined = []

# Load verified examples
with verified_path.open('r') as f:
    for line in f:
        combined.append(json.loads(line))

# Load diagnostic corpus
with diagnostic_path.open('r') as f:
    for line in f:
        combined.append(json.loads(line))

# Save combined dataset
with output_path.open('w') as f:
    for ex in combined:
        f.write(json.dumps(ex) + '\n')

print(f"✅ Combined {len(combined)} examples")
print(f"   Saved to: {output_path}")
```

---

### **Phase 3: Retrain with Behavioral Dataset**

```bash
# Update training script to use new dataset
cd ~/cda-agent-2C1
source venv/bin/activate

# Modify train_qwen_coder_qlora.py to use rtl_behavioral_training.jsonl
# Then run:
python3 scripts/train_qwen_coder_qlora.py
```

**Key change in `train_qwen_coder_qlora.py`:**
```python
# Line ~287
dataset_path = project_root / 'data' / 'rtl_behavioral_training.jsonl'
output_dir = project_root / 'models' / 'qwen_coder_rtl' / f"run_behavioral_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

---

### **Phase 4: Real-Time Validation During Inference**

**Update `core/rtl_agents/a1_llm_generator.py`:**

```python
# Add at top:
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))
from rtl_verifier import RTLVerifier

class A1_LLMGenerator(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        # ... existing code ...
        self.verifier = RTLVerifier()

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        # ... generate RTL code ...

        # NEW: Verify generated code
        verification = self.verifier.verify(
            rtl_code=rtl_code,
            module_name=module_name,
            spec=specification
        )

        # Update confidence based on verification score
        confidence = verification.score

        return self.create_output(
            success=True,
            output_data={
                'rtl_code': rtl_code,
                'ports': ports,
                'verification_score': verification.score,
                'verification_details': {
                    'syntax_valid': verification.syntax_valid,
                    'synthesis_success': verification.synthesis_success,
                    'all_io_used': verification.all_io_used,
                    'spec_keywords_used': verification.spec_keywords_used,
                    'errors': verification.errors,
                    'warnings': verification.warnings
                },
                # ... existing fields ...
            },
            confidence=confidence,  # Use verification score
            execution_time_ms=execution_time
        )
```

---

## 🎯 Expected Improvements (Next Round)

| Metric | Current | Target |
|--------|---------|--------|
| Synthesis compile rate | 100% | ≥ 98% |
| **Functional intent-match** | ~0% | **≥ 80%** |
| All inputs/outputs used | ~50% | ≥ 95% |
| **Behavioral unit tests pass** | 0% | **≥ 70%** |
| Enable/reset semantics correct | ~20% | **≥ 90%** |

---

## 🚀 Quick Start: Next Training Iteration

```bash
# 1. Install Verilator (optional, for lint checking)
sudo apt install verilator

# 2. Verify existing training data
cd ~/cda-agent-2C1
source venv/bin/activate
python3 scripts/verify_training_data.py

# 3. Combine verified + diagnostic corpus
python3 scripts/combine_verified_datasets.py

# 4. Retrain model
#    Edit scripts/train_qwen_coder_qlora.py:
#    - Change dataset_path to 'rtl_behavioral_training.jsonl'
#    - Change output_dir to 'run_behavioral_YYYYMMDD_HHMMSS'
python3 scripts/train_qwen_coder_qlora.py

# 5. Test new model
python3 scripts/test_model_terminal.py
```

---

## 📈 Future Enhancements

### Phase 5: Simulation-Based Verification (Advanced)

**Add cocotb or iverilog testbenches:**
```python
# scripts/simulation_verifier.py
class SimulationVerifier:
    def verify_behavior(self, rtl_code: str, spec: str) -> bool:
        """
        1. Parse spec for expected behavior
        2. Generate testbench
        3. Run simulation with cocotb/iverilog
        4. Check outputs match expected behavior
        """
        # Generate test vectors from spec
        # Run simulation
        # Compare results
        pass
```

### Phase 6: Reinforcement Learning from Verifier Feedback

**Add RL loop:**
```python
# During training, after each generation:
# 1. Generate RTL
# 2. Verify with tools
# 3. Use verification score as reward signal
# 4. Update model weights based on reward
```

---

## 🔧 Tools Summary

### Created Tools
- [x] `scripts/rtl_verifier.py` - Semantic verification (Yosys-based)
- [x] `scripts/generate_diagnostic_corpus.py` - Behavioral corpus generator
- [x] `scripts/test_model_terminal.py` - Model testing
- [ ] `scripts/verify_training_data.py` - Filter training data
- [ ] `scripts/combine_verified_datasets.py` - Combine datasets
- [ ] `scripts/simulation_verifier.py` - Simulation-based verification (future)

### Updated Files (Next)
- [ ] `scripts/train_qwen_coder_qlora.py` - Use behavioral dataset
- [ ] `core/rtl_agents/a1_llm_generator.py` - Add real-time verification
- [ ] `api/pipeline.py` - Return verification scores

---

## 📝 Your Next Steps

1. **Review** the verifier output: `python3 scripts/rtl_verifier.py`
2. **Expand** diagnostic corpus to 200+ examples (add FSMs, FIFOs, etc.)
3. **Implement** `verify_training_data.py` and `combine_verified_datasets.py`
4. **Retrain** with behavioral dataset
5. **Test** and compare with current model
6. **Iterate** based on results

**Goal:** Move from "syntax imitation" → "behavioral correctness" ✨
