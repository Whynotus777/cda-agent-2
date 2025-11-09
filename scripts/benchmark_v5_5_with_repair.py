#!/usr/bin/env python3
"""
V5.5 Benchmark with A7 Smart Validation + Auto-Repair Loop
==========================================================

Flow:
1. Model generates RTL
2. A7 validates with smart testbench (or template fallback)
3. If validation fails → Run fsm_autofix repair
4. Re-validate repaired RTL with A7
5. Track metrics: baseline vs repaired pass rates

Expected outcome: 25% → 40-50% FSM pass rate with repairs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from scripts.benchmark_v5_4 import *
from scripts.fsm_autofix import repair_fsm_example  # Import existing repair function

# =============================================================================
# RTL GENERATION HELPERS (extracted from benchmark_v5_4.py)
# =============================================================================

def extract_module_code(raw_output: str) -> str:
    """
    Extract Verilog module code from raw LLM output.
    Handles:
    - Conversational text ("module should have...", "module takes...")
    - Trailing explanations after endmodule
    - Better validation of actual module declarations
    """
    # Pattern matches "module <identifier> (" or "module <identifier> #" or "module <identifier>;"
    # This filters out conversational sentences like "module should" or "module takes"
    module_pattern = r'module\s+\w+\s*[(\[#;]'
    module_match = re.search(module_pattern, raw_output, re.MULTILINE)

    if not module_match:
        # Fallback: try finding any "module identifier" pattern
        fallback_match = re.search(r'module\s+\w+', raw_output, re.IGNORECASE)
        if not fallback_match:
            return raw_output  # Return as-is if no module found
        start = fallback_match.start()
    else:
        start = module_match.start()

    # Find endmodule and strip everything after
    endmodule_pattern = r'endmodule\s*;?\s*'
    remaining_text = raw_output[start:]
    endmodule_match = re.search(endmodule_pattern, remaining_text, re.MULTILINE)

    if endmodule_match:
        # Extract from module to end of endmodule (inclusive)
        end = start + endmodule_match.end()
        extracted = raw_output[start:end].strip()

        # Additional cleanup: remove any trailing non-code text
        # If there's significant text after endmodule, it's likely an explanation
        remaining = raw_output[end:].strip()
        if remaining and len(remaining) > 50:
            # Long trailing text detected - we've successfully isolated the module
            pass

        return extracted

    # No endmodule found - return from module to end
    return raw_output[start:].strip()


def generate_rtl(model, tokenizer, spec_text: str):
    """Generate RTL from specification using the model"""
    prompt = f"""Generate SystemVerilog (IEEE 1800-2012) RTL for the following specification.

REQUIRED SYNTAX:
- Use 'typedef enum' for state types
- Use 'always_ff @(posedge clk or negedge rst_n)' for sequential logic
- Use 'always_comb' for combinational logic
- Use 'logic' instead of 'reg'/'wire' where appropriate

Specification:
{spec_text}

Provide only the SystemVerilog module code."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rtl_code_raw = generated[len(prompt):].strip()

    # CRITICAL FIX: Extract module code, strip explanatory text
    rtl_code = extract_module_code(rtl_code_raw)

    return rtl_code, {"prompt": prompt, "raw_output": rtl_code_raw}


def extract_module_name(rtl_code: str) -> str:
    """Extract module name from RTL code"""
    module_match = re.search(r'module\s+(\w+)', rtl_code, re.IGNORECASE)
    if module_match:
        return module_match.group(1)
    return "unknown_module"

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_for_generation(model_path: str):
    """Load Qwen model with PEFT adapters for RTL generation"""
    print("Loading model...")

    # Load base model with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ),
        device_map="auto"
    )

    # Resolve adapter path
    adapter_path = Path(model_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    # Load PEFT adapters
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded successfully")
    return model, tokenizer


# =============================================================================
# REPAIR LOOP INTEGRATION
# =============================================================================

def repair_rtl_code(code: str, spec: str, errors: str = "") -> str:
    """
    Wrapper for repair_fsm_example that adapts interface for benchmark use.

    Args:
        code: RTL code to repair
        spec: Specification text
        errors: Error messages (not used by deterministic repair)

    Returns:
        Repaired RTL code (or original if no repairs possible)
    """
    # Create minimal example dict for repair_fsm_example
    example = {
        'instruction': spec,
        'output': code
    }

    repair_log = []
    repaired_example, was_repaired = repair_fsm_example(example, repair_log)

    if was_repaired:
        return repaired_example['output']
    else:
        return code


def run_test_with_repair(spec_entry, model, tokenizer, run_number, max_repair_attempts=2):
    """
    Run test with automatic repair loop.

    Args:
        spec_entry: Test specification
        model, tokenizer: Model for RTL generation
        run_number: Current run number
        max_repair_attempts: Max number of repair cycles (default: 2)

    Returns:
        dict: Result with repair metadata
    """
    spec_text = spec_entry["spec"]
    test_name = spec_entry["name"]
    category = spec_entry.get("category", "unknown")

    # STEP 1: Model generates RTL
    rtl_code, gen_metadata = generate_rtl(model, tokenizer, spec_text)
    module_name = extract_module_name(rtl_code)

    # STEP 2: Generate smart testbench with A7
    testbench, testbench_type = generate_smart_testbench(rtl_code, module_name, spec_text, use_a7=True)

    # STEP 3: First validation attempt (baseline)
    result_original = run_test(test_name, category, rtl_code, testbench, module_name, spec_entry, run_number)
    result_original["testbench_type"] = testbench_type
    result_original["repair_attempt"] = 0
    result_original["repaired"] = False

    if result_original["passed"]:
        # Original passed - no repair needed
        return result_original

    # STEP 4: Repair loop (if original failed)
    print(f"  ⚙️  Original failed, attempting auto-repair...")

    repaired_rtl = rtl_code
    for attempt in range(1, max_repair_attempts + 1):
        try:
            # Run deterministic repairs
            repaired_rtl = repair_rtl_code(
                code=repaired_rtl,
                spec=spec_text,
                errors=result_original.get("error", "")
            )

            if repaired_rtl == rtl_code:
                # No changes made - repair couldn't fix it
                print(f"  ⚠️  Repair attempt {attempt}: No changes possible")
                break

            # Re-validate repaired RTL
            print(f"  ♻️  Repair attempt {attempt}: Testing repaired RTL...")
            result_repaired = run_test(
                test_name, category, repaired_rtl, testbench,
                module_name, spec_entry, run_number
            )
            result_repaired["testbench_type"] = testbench_type
            result_repaired["repair_attempt"] = attempt
            result_repaired["repaired"] = True

            if result_repaired["passed"]:
                print(f"  ✓ REPAIRED! Pass after {attempt} repair(s)")
                return result_repaired

            # Update for next attempt
            rtl_code = repaired_rtl

        except Exception as e:
            print(f"  ⚠️  Repair attempt {attempt} failed: {e}")
            break

    # All repair attempts exhausted
    result_final = result_original.copy()
    result_final["repair_attempts"] = attempt
    result_final["repair_failed"] = True
    print(f"  ✗ FAILED after {attempt} repair attempt(s)")

    return result_final


# =============================================================================
# BENCHMARK WITH REPAIR STATS
# =============================================================================

def benchmark_v5_5_with_repair(model_path, num_runs):
    """Run benchmark with repair loop and track before/after metrics"""

    print("=" * 80)
    print("  V5.5 BENCHMARK: A7 SMART VALIDATION + AUTO-REPAIR")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Test Suite: {len(TEST_SPECS)} specs × {num_runs} runs = {len(TEST_SPECS) * num_runs} total tests")
    print(f"Max repair attempts: 2 per failure")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_for_generation(model_path)
    print("✓ Model loaded\n")

    # Initialize A7
    print("=" * 80)
    print("  INITIALIZING A7 LLM TESTBENCH GENERATOR")
    print("=" * 80)
    init_a7_generator(PROJECT_ROOT)
    print()

    # Run tests
    all_results = []
    for run in range(num_runs):
        print(f"=" * 80)
        print(f"  RUN {run+1}/{num_runs}")
        print(f"=" * 80)
        print()

        for spec_entry in TEST_SPECS:
            print(f"Testing: {spec_entry['name']} ({spec_entry.get('category', 'unknown')})")

            result = run_test_with_repair(
                spec_entry=spec_entry,
                model=model,
                tokenizer=tokenizer,
                run_number=run+1
            )
            all_results.append(result)

    # Analyze results
    print("\n" + "=" * 80)
    print("  V5.5 BENCHMARK RESULTS WITH REPAIR")
    print("=" * 80)
    print()

    baseline_results = [r for r in all_results if r.get("repair_attempt", 0) == 0]
    repaired_results = [r for r in all_results if r.get("repaired", False) and r["passed"]]

    baseline_passed = sum(1 for r in baseline_results if r["passed"])
    baseline_total = len(baseline_results)

    total_passed = sum(1 for r in all_results if r["passed"])
    total_tests = len(all_results)

    repairs_successful = len(repaired_results)

    print("Baseline Performance (No Repairs):")
    print(f"  Pass Rate: {baseline_passed}/{baseline_total} ({100*baseline_passed/baseline_total:.1f}%)")
    print()

    print("With Auto-Repair:")
    print(f"  Pass Rate: {total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)")
    print(f"  Successful Repairs: {repairs_successful}")
    print(f"  Improvement: +{100*(total_passed-baseline_passed)/baseline_total:.1f}%")
    print()

    # FSM-specific stats
    fsm_baseline = [r for r in baseline_results if "fsm_" in r.get("category", "")]
    fsm_total = [r for r in all_results if "fsm_" in r.get("category", "")]

    if fsm_baseline:
        fsm_baseline_passed = sum(1 for r in fsm_baseline if r["passed"])
        fsm_total_passed = sum(1 for r in fsm_total if r["passed"])

        print("FSM Performance:")
        print(f"  Baseline: {fsm_baseline_passed}/{len(fsm_baseline)} ({100*fsm_baseline_passed/len(fsm_baseline):.1f}%)")
        print(f"  With Repair: {fsm_total_passed}/{len(fsm_total)} ({100*fsm_total_passed/len(fsm_total):.1f}%)")
        print(f"  FSM Improvement: +{100*(fsm_total_passed-fsm_baseline_passed)/len(fsm_baseline):.1f}%")

    # Testbench stats
    print("\nTestbench Type Distribution:")
    tb_types = {}
    for r in all_results:
        tb_type = r.get("testbench_type", "UNKNOWN")
        tb_types[tb_type] = tb_types.get(tb_type, 0) + 1

    for tb_type, count in sorted(tb_types.items()):
        print(f"  {tb_type}: {count} ({100*count/len(all_results):.1f}%)")

    return all_results



def find_latest_model():
    """Find the latest trained model"""
    from pathlib import Path
    models_dir = Path("models/qwen_coder_rtl")
    if not models_dir.exists():
        return "models/qwen_coder_rtl/latest"
    
    # Return the latest symlink or default
    latest = models_dir / "latest"
    if latest.exists():
        return str(latest)
    
    return "models/qwen_coder_rtl/latest"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V5.5 Benchmark with A7 + Auto-Repair")
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per test")
    args = parser.parse_args()

    model_path = args.model if args.model else find_latest_model()

    results = benchmark_v5_5_with_repair(model_path, args.runs)

    # Save results
    import json
    output_path = "benchmark_v5_5_with_repair_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
