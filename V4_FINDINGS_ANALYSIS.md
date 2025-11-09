# V4 Model: Detailed Findings Analysis

**Date:** 2025-11-05
**Experiment:** Verification-in-the-Loop Training (V3 → V4)
**Result:** Mixed success (+2% overall, critical FSM regression)

---

## Executive Summary

V4 demonstrated that **verification-in-the-loop** is a viable approach for improving RTL generation quality. However, the experiment revealed a critical insight: **testbench quality matters as much as training data quality**.

**Key Result:**
- Overall functional correctness: 84% → 86% (+2%)
- 7/10 categories improved (edge detection +40% is standout)
- **FSMs regressed catastrophically: 60% → 20% (-40%)**
- **Recommendation: Do not deploy V4 to production**

---

## Detailed Results

### Success Stories

#### 1. Edge Detection: 60% → 100% (+40%)
**Why it worked:**
- Simple, well-defined behavior (detect rising/falling edges)
- Template testbench adequately tests edge detection logic
- V4 dataset likely had many correct examples

**Lesson:** When domain behavior is simple and testable, verification-in-the-loop works excellently.

#### 2. Async Reset: 80% → 100% (+20%)
**Why it worked:**
- Reset logic is straightforward to verify
- Template testbench can reliably test reset behavior
- Common pattern in V4 dataset

#### 3. Shift Registers: 60% → 80% (+20%)
**Why it worked:**
- This was a targeted weak domain (185 examples, 21.9% of dataset)
- Parametric generation created diverse shift patterns
- Template testbench can verify shift behavior adequately

**Lesson:** Targeted weak domain strategy worked for shift registers.

### The FSM Disaster

#### FSMs: 60% → 20% (-40%)

**This is the most important finding of the V4 experiment.**

**Hypotheses for FSM Regression:**

1. **Template Testbench Inadequacy (Most Likely)**
   - Simple template testbench cannot exhaustively test FSM state transitions
   - May only test happy path (IDLE → STATE1 → STATE2 → IDLE)
   - Doesn't test:
     - Invalid state transitions
     - All possible input combinations in each state
     - Edge cases (e.g., rapid input changes)
     - Reset during active states

   **Evidence:**
   - 222 FSM examples "passed" V4 generation but V4 model performs worse
   - This suggests the examples passed a weak test, not a thorough one
   - Model learned incorrect patterns that passed template testbench

2. **Overfitting to V4 FSM Patterns**
   - V3 had 857 diverse examples from templates
   - V4 has 844 parametrically-generated examples
   - If V4 FSM examples had common structural patterns, model may have memorized them
   - Lost generalization to benchmark FSM specs

3. **Distribution Shift**
   - V4 dataset FSM specs may differ from benchmark FSM specs
   - Parametric generation may have created FSMs with different characteristics
   - Model trained on one FSM style, tested on another

4. **Training Data Contamination**
   - The 222 FSM examples that "passed" simulation may have subtle bugs
   - Template testbench passed them, but they're functionally incorrect
   - Model learned from bad examples thinking they were good

**Smoking Gun Evidence:**

```
V4 Dataset: 222 FSM examples (26.3% of dataset) - MORE than any other category
V4 Benchmark: 20% functional on FSM specs - WORST of all categories
```

This inverse relationship (most training data → worst performance) strongly suggests **testbench quality issues**.

---

## Root Cause Analysis

### The Template Testbench Problem

**Current Template Testbench Approach:**
```systemverilog
// Simplified template testbench structure
always @(posedge clk) begin
    // Test case 1: Input A
    // Check output matches expected

    // Test case 2: Input B
    // Check output matches expected
end
```

**What It Tests Well:**
- Combinational logic (muxes, comparators)
- Simple sequential patterns (counters, shift registers)
- Reset behavior
- Basic clock relationships

**What It Misses for FSMs:**
- State coverage (are all states reachable?)
- Transition coverage (are all edges tested?)
- Invalid input handling
- Complex state sequences
- Corner cases in state machines

### Why Shift Registers Succeeded but FSMs Failed

| Aspect | Shift Registers | FSMs |
|--------|----------------|------|
| **Behavior Complexity** | Linear, predictable | Non-linear, state-dependent |
| **Test Coverage** | Easy (shift N times, check) | Hard (2^N state transitions) |
| **Template Testbench** | Adequate | Inadequate |
| **V4 Result** | +20% improvement | -40% regression |

**Insight:** Template testbenches work for simple, linear behaviors but fail for complex state-dependent logic.

---

## Lessons Learned

### 1. Verification-in-the-Loop Works (When Done Right)

The success with 7/10 categories proves the concept:
- Edge detection: +40%
- Async reset: +20%
- Shift registers: +20%
- Enable control: +20%

**Takeaway:** Filtering training data by simulation pass/fail improves model quality.

### 2. Testbench Quality is Critical

The FSM disaster teaches us:
- **You can only learn what you measure**
- A weak testbench creates weak training data
- 222 "passing" FSM examples were likely false positives

**Takeaway:** Invest in testbench quality as much as training data quantity.

### 3. Domain Complexity Matters

Simple domains (edge detect, reset) → Template testbench OK
Complex domains (FSMs) → Need LLM-generated testbenches

**Takeaway:** One-size-fits-all testbenches don't work.

### 4. More Data ≠ Better Results

FSMs had the MOST training data (222 examples, 26.3%) but performed WORST (20%).

**Takeaway:** 100 high-quality examples > 222 low-quality examples

---

## Recommendations for V5

### Priority 1: Fix FSM Testbenches

**Action:** Use A7 LLM-generated testbenches for all FSM examples

**Rationale:**
- LLM can generate comprehensive state transition tests
- Can create input sequences that exercise all states
- Can add assertions for invalid transitions

**Implementation:**
```python
# In generate_v5_dataset.py
if category.startswith('fsm_'):
    # Use A7 to generate testbench
    testbench = a7_agent.generate_fsm_testbench(spec, rtl)
else:
    # Template testbench is fine for simple domains
    testbench = generate_template_testbench(spec, rtl)
```

### Priority 2: Reduce FSM Quantity, Increase Quality

**V4:** 222 FSM examples (quantity focus)
**V5:** 100 FSM examples (quality focus)

**Strategy:**
- Generate 300 FSM examples
- Use LLM testbenches for all
- Keep only top 100 that pass most thorough tests
- Manually review a sample for correctness

### Priority 3: Analyze V4 Generation Failures

The 356 failed examples from V4 generation contain valuable insights:
- Which patterns consistently fail?
- Are they compile failures or functional failures?
- Can we extract common bug patterns?

**Action:** Run failure autopsy analysis (framework already created in `analyze_v4_failures.py`)

### Priority 4: Mine Pipeline Production Data

Once pipeline is deployed, mine trace.jsonl for:
- Real-world specs that succeeded
- Human-corrected RTL examples
- Edge cases not in training data

### Priority 5: Consider Hybrid Dataset

**V5 Dataset Composition:**
- 40% from V3 (857 template-based examples that worked)
- 40% from V4 (keep the 622 non-FSM examples that improved results)
- 20% new targeted examples (FSMs with LLM testbenches, edge cases)

**Rationale:** Don't throw away what worked in V3.

---

## Deployment Recommendation

### Do NOT Deploy V4 to Production

**Reasons:**
1. Critical FSM regression (60% → 20%)
2. FSMs are common in real RTL designs
3. Risk of pipeline failures on FSM-heavy specs
4. V3 is more reliable overall (only 2% worse, but more balanced)

**Alternative:** Keep V3 in production while developing V5

---

## Experiment Value

Despite not achieving deployment-ready results, V4 was **highly valuable**:

1. **Proved verification-in-the-loop concept** (7/10 categories improved)
2. **Identified testbench quality as critical factor** (FSM disaster)
3. **Established 70.3% pass rate baseline** (V3 model generates passing RTL 70% of time)
4. **Created infrastructure** (simulation runner, dataset generator, benchmarking)
5. **Provides clear path to V5** (LLM testbenches for complex domains)

**ROI:** The 4-minute training + 2-minute benchmark gave us insights worth weeks of blind iteration.

---

## Metrics for V5 Success

V5 will be considered successful if:

1. **Overall functional ≥ 88%** (V3: 84%, V4: 86%)
2. **FSM functional ≥ 70%** (V3: 60%, V4: 20% ❌)
3. **No category regresses below V3 baseline**
4. **Shift registers maintain ≥ 80%** (keep V4 gains)

If V5 achieves these metrics, deploy to production and establish continuous improvement loop.

---

## Timeline Estimate

**V5 Development Roadmap:**

1. Integrate A7 LLM testbench generation: 1 day
2. Generate V5 dataset (1500 attempts, LLM TBs for FSMs): 2-3 hours
3. Train V5 model: 5 minutes
4. Benchmark V5: 2 minutes
5. If successful, integrate into pipeline: 2 hours

**Total: 1-2 days to V5 deployment**

---

## Conclusion

V4 represents a **successful experiment that revealed critical insights**, even though it's not production-ready. The verification-in-the-loop approach works, but requires testbench quality proportional to domain complexity.

**Next step:** V5 with LLM-generated testbenches for complex domains.

**Key insight for future work:**
> "In machine learning for RTL generation, you can only teach what you can test."

The FSM regression is not a failure—it's a discovery that testbench sophistication must match domain complexity. This insight will make V5 significantly stronger.
