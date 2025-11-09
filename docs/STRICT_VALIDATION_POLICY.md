# Strict Validation Policy for Synthetic Training Data

**CRITICAL**: This document defines the policy for all synthetic data generation going forward.

## Problem

In V5.6 and V5.7, we discovered that **93% of synthetic FSMs failed strict validation**. These examples:
- ✅ Passed permissive iverilog/Verilator validation
- ❌ Failed strict syntax checks (missing newlines, warnings)
- ❌ Failed specification compliance (missing signals, wrong structure)
- ❌ Failed functional correctness tests

**Result**: Training on low-quality data made the model WORSE at FSMs (10-12.5% accuracy vs 60% baseline).

## New Policy: ZERO TOLERANCE for Low-Quality Data

### All Synthetic Data Must Pass

1. **Strict Syntax Validation**
   - Verilator with `-Wall -Werror` (no warnings allowed)
   - iverilog with `-Wall` (all warnings treated as errors)
   - Proper POSIX formatting (EOF newlines, etc.)

2. **FSM Structure Validation**
   - State register (`current_state`)
   - Next state logic (`next_state`)
   - State transitions (clocked `always_ff` or `always @(posedge)`)
   - Type-specific requirements:
     - **Moore**: Outputs depend only on state
     - **Mealy**: Outputs depend on state AND inputs
     - **Handshake**: Must have `req` and `ack` signals

3. **Specification Compliance**
   - Correct number of states (if specified)
   - Required signals present (req, ack, valid, ready, etc.)
   - Output patterns match specification
   - State transitions match description

4. **Functional Correctness**
   - Compiles with generated testbench
   - Simulates without errors
   - Testbench verifies basic functionality
   - No simulation timeout or crashes

## Implementation

### Framework: `core/validation/strict_fsm_validator.py`

```python
from core.validation import validate_fsm_strict

# Validate FSM with all checks
result = validate_fsm_strict(
    code=generated_code,
    instruction=user_instruction,
    fsm_type="Moore",  # or "Mealy", "Handshake"
    dialect="sv2005"
)

if result.passed:
    # ONLY include in training data if passed
    add_to_dataset(code)
else:
    # Log failures for debugging
    log_validation_failure(result.errors)
    # Try regenerating or discard
```

### Required for All Synthetic Generation

Every script that generates synthetic data MUST:

1. **Import the strict validator**
   ```python
   from core.validation import validate_fsm_strict
   ```

2. **Validate before adding to dataset**
   ```python
   result = validate_fsm_strict(code, instruction, fsm_type, dialect)
   if not result.passed:
       continue  # Skip this example
   ```

3. **Report validation statistics**
   ```python
   print(f"Generated: {total_attempts}")
   print(f"Passed validation: {passed}")
   print(f"Failed validation: {failed}")
   print(f"Pass rate: {passed/total_attempts*100:.1f}%")
   ```

4. **Allow retry with feedback**
   ```python
   if not result.passed:
       # Optionally: retry generation with error feedback
       # Or: discard and move on
       pass
   ```

## Target Metrics

For synthetic data to be acceptable:

- **Pass Rate**: ≥80% of generated examples must pass strict validation
- **Syntax Errors**: 0 syntax errors or warnings allowed
- **Functional Failures**: <5% simulation failures
- **Specification Violations**: <10% spec violations (these get retried)

If pass rates are below 80%, the generation prompts/process needs improvement.

## What This Means for Existing Data

### V5.6/V5.7 Synthetic FSMs (316 examples)

**Status**: REJECTED
- **Pass Rate**: 6.7% (2/30 sampled)
- **Failures**: 70% syntax, 30% specification
- **Action**: Do NOT use for training

### Future Synthetic Generation

All new synthetic data MUST use the strict validator or it will be REJECTED for training.

## Example: Strict FSM Generation Script Template

```python
#!/usr/bin/env python3
from core.validation import validate_fsm_strict
from core.rag import RAGEnhancedClient
import asyncio

async def generate_fsm_synthetic_strict(fsm_type: str, num_examples: int):
    """Generate high-quality synthetic FSMs with strict validation"""

    client = RAGEnhancedClient()

    generated = 0
    attempts = 0
    failed_reasons = []

    output_examples = []

    while generated < num_examples and attempts < num_examples * 3:
        attempts += 1

        # Generate FSM
        instruction = create_fsm_instruction(fsm_type)
        response = await client.generate_with_rag(
            user_prompt=instruction,
            query_for_rag=f"{fsm_type} FSM design patterns",
            system_prompt=SYSTEM_PROMPT
        )

        code = extract_code(response)

        # STRICT VALIDATION
        result = validate_fsm_strict(
            code=code,
            instruction=instruction,
            fsm_type=fsm_type,
            dialect="sv2005"
        )

        if result.passed:
            # SUCCESS - add to dataset
            output_examples.append({
                "instruction": instruction,
                "output": code,
                "hierarchy": {"l1": "Sequential", "l2": "FSM", "l3": fsm_type},
                "metadata": {"validated": True, "strict": True}
            })
            generated += 1
            print(f"✓ {generated}/{num_examples} generated (attempt {attempts})")
        else:
            # FAILURE - log and continue
            failed_reasons.append(result.errors)
            print(f"✗ Validation failed: {result.errors[0][:100]}")

    # Report statistics
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Attempts:     {attempts}")
    print(f"Passed:       {generated} ({generated/attempts*100:.1f}%)")
    print(f"Failed:       {attempts - generated}")
    print(f"Pass rate:    {'✓ ACCEPTABLE' if generated/attempts >= 0.8 else '✗ TOO LOW'}")

    return output_examples
```

## Enforcement

- **All pull requests** adding synthetic data must include validation logs
- **All dataset files** must have validation metadata
- **All training runs** must report % of validated vs unvalidated data

## Rationale

Training on 93% bad data is worse than not training at all. The model learns incorrect patterns and generates broken code.

**Quality over quantity**: 100 high-quality examples > 300 low-quality examples.

The strict validator ensures we only train on production-ready code that actually implements specifications correctly.
