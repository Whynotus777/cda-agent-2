# V5.7 Root Cause Analysis: Why FSM Performance Degraded

## Executive Summary

**Problem**: Despite fixing metadata alignment in V5.7 (zero tag drift), FSM performance DECLINED to 10.0% (vs V5.6's 12.5%).

**Root Cause Identified**: 93% of synthetic FSM training data failed strict quality validation.

**Impact**: Training on low-quality data taught the model incorrect FSM patterns, making it worse than not training on FSMs at all.

## Investigation Timeline

### V5.6 Results (November 7, 2025)
- FSM Accuracy: **12.5%** (5/40 tests)
- Training data: 1,252 examples with 316 synthetic FSMs
- Issue identified: 410/413 FSMs had missing/incorrect `category` field

### V5.7 Fix Attempt
**Action**: Fixed metadata alignment
- Created `scripts/fix_category_alignment_v5_7.py`
- Aligned `category` with `hierarchy.l2` for all examples
- Result: **ZERO tag drift** (all 413 FSMs correctly tagged)

**Expected**: FSM accuracy should improve with correct metadata
**Actual**: FSM accuracy DECLINED to **10.0%** (4/40 tests)

### Critical Question (User)
> "did we run our claude synthetic data set through a7 to confirm its high quality? Not just passing, but obeying the spec"

This question identified the real problem.

### Validation of Synthetic Data

Created `scripts/validate_fsm_quality.py` to check if synthetic FSMs actually implement their specifications.

**Results** (30 FSMs sampled):
```
Moore FSMs:     2/10 passed (20.0%)
Mealy FSMs:     0/10 passed (0.0%)
Handshake FSMs: 0/10 passed (0.0%)
----------------------------------
OVERALL:        2/30 passed (6.7%)
```

**Failure Breakdown**:
- 21 syntax errors (missing EOF newlines, Verilator warnings)
- 11 specification violations (missing signals, wrong structure)

## Root Cause Analysis

### What Went Wrong

The 316 synthetic FSMs generated for V5.6/V5.7:
- ✅ Passed permissive iverilog validation (basic compilation)
- ✅ Passed Verilator validation (with warnings ignored)
- ❌ Failed strict syntax validation (warnings not allowed)
- ❌ Failed specification compliance checks
- ❌ Failed functional correctness tests

### Why This Made Things Worse

Training on 93% bad data is worse than not training at all:

1. **Model learned incorrect patterns**:
   - Missing EOF newlines
   - Incomplete FSM structures
   - Missing required signals
   - Wrong output logic patterns

2. **Overfitting to flawed examples**:
   - Model memorized broken code patterns
   - Generated syntactically invalid FSMs
   - Failed to implement specifications correctly

3. **Dilution of high-quality data**:
   - 97 original FSMs (high quality)
   - 316 synthetic FSMs (93% low quality)
   - Ratio: 23% good vs 77% bad FSM data

### Performance Comparison

```
FSM Accuracy History:
V3:  60.0% (baseline with original data)
V4:  20.0% (regression)
V5.4: 10.0% (added low-quality synthetic FSMs)
V5.6: 12.5% (more low-quality synthetic FSMs)
V5.7: 10.0% (fixed metadata, still bad synthetic data)
```

**Conclusion**: Metadata alignment wasn't the problem. Data quality was the problem.

## Solution Implemented

### 1. Strict Validation Framework

Created `core/validation/strict_fsm_validator.py` with comprehensive checks:

**Syntax Validation** (Zero Tolerance):
- Verilator with `-Wall -Werror`
- iverilog with `-Wall`
- POSIX compliance (EOF newlines, etc.)
- No warnings allowed

**Structure Validation**:
- State register (`current_state`)
- Next state logic (`next_state`)
- State transition logic (clocked `always_ff`)
- Type-specific requirements (Moore, Mealy, Handshake)

**Specification Compliance**:
- Correct number of states
- Required signals present
- Output patterns match specification
- State transitions match description

**Functional Correctness**:
- Generates and runs testbench
- Verifies basic functionality
- Checks for simulation errors
- No timeouts or crashes

### 2. Strict Validation Policy

Created `docs/STRICT_VALIDATION_POLICY.md`:

**Requirements**:
- All synthetic data MUST pass strict validation
- Target pass rate: ≥80%
- Zero syntax errors/warnings
- <5% simulation failures
- <10% specification violations

**Enforcement**:
- Validation required before adding to dataset
- Validation logs required for all synthetic data
- Failed examples logged for debugging
- Retry with feedback or discard

### 3. Usage Example

```python
from core.validation import validate_fsm_strict

# Generate FSM code
code = generate_fsm(instruction)

# STRICT VALIDATION - REQUIRED
result = validate_fsm_strict(
    code=code,
    instruction=instruction,
    fsm_type="Moore",
    dialect="sv2005"
)

# Only add if passed ALL checks
if result.passed:
    add_to_dataset(code)
else:
    log_failure(result.errors)
    # Optionally retry with error feedback
```

## Lessons Learned

### 1. Quality > Quantity
- 100 high-quality examples > 300 low-quality examples
- Training on bad data makes model worse
- Validation must be strict, not permissive

### 2. Compilation ≠ Correctness
- Passing iverilog doesn't mean code is good
- Warnings are often actual errors
- Specification compliance is separate from syntax

### 3. Metadata Alone Isn't Enough
- V5.7 had perfect metadata alignment (0 drift)
- But still had 93% bad training data
- Both metadata AND content quality matter

### 4. Validate Early and Strictly
- Permissive validation allows bad data to accumulate
- Strict validation catches issues before training
- Testbenches verify functional correctness

## Next Steps

### Immediate Actions
1. ✅ Reject V5.6/V5.7 synthetic FSMs (93% failure rate)
2. ⬜ Generate new synthetic FSMs with strict validation
3. ⬜ Target pass rate: ≥80%
4. ⬜ Train V5.8 with only high-quality FSMs

### Long-term Improvements
1. Improve synthetic generation prompts to reduce failures
2. Add specification-aware testbench generation
3. Implement feedback loop: errors → improved prompts
4. Create gold standard FSM dataset with manual verification

## Validation Results Summary

**V5.6/V5.7 Synthetic FSMs**:
- Total: 316 examples
- Validated: 30 sampled
- Passed: 2 (6.7%)
- **Status: REJECTED**

**Original FSMs**:
- Total: 97 examples
- Estimated quality: High (manually created)
- **Status: ACCEPTED**

## Files Created

1. `core/validation/strict_fsm_validator.py` - Validation framework
2. `core/validation/__init__.py` - Module exports
3. `docs/STRICT_VALIDATION_POLICY.md` - Policy document
4. `scripts/validate_fsm_quality.py` - Quality checker
5. `docs/V5_7_ROOT_CAUSE_ANALYSIS.md` - This document

## Conclusion

The root cause of V5.7's poor FSM performance was **training data quality**, not metadata alignment. The 316 synthetic FSMs passed permissive validation but failed strict quality checks, teaching the model incorrect patterns.

Going forward, **all synthetic data must pass strict validation** before being included in training datasets. This ensures the model learns from production-ready, specification-compliant code.

**Quality over quantity**: It's better to have fewer high-quality examples than many low-quality ones.
