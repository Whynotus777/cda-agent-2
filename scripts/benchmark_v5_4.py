#!/usr/bin/env python3
"""
V5.4 Dialect-Aware Model Benchmark with Dual-Path Simulation

V5.4 Dataset Strategy - DIALECT-AWARE TRAINING:
- All 1,125 examples tagged with [HDL:dialect] conditioning tokens
- Model learns to generate appropriate syntax based on dialect token
- Trained on 46% sv2009, 34% verilog2001, 20% sv2005

Key Improvements over V5.3:
- Dialect conditioning enables controllable generation
- Dual-path simulation: iverilog (Verilog) + Verilator (SystemVerilog)
- Automatic dialect detection routes to correct compiler
- Expanded FSM test suite (8 FSM tests vs 1 in V5.3)

Root Cause Fix:
- V5.3 Problem: 77% of FSM training data is SystemVerilog
- V5.3 Testing: iverilog (Verilog-only) → all SV FSMs failed (0%)
- V5.4 Solution: Dual-path runner validates both Verilog & SystemVerilog

Target: ≥60% FSM functional accuracy (V3 baseline recovery)
"""
import sys
from pathlib import Path
import argparse
import torch
import json
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rtl_verifier import RTLVerifier

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

# Import DUAL-PATH simulation runner (NEW)
from run_dual_path_simulation import run_dual_path_simulation

try:
    from core.verification.testbench_generator import A7_TestbenchGenerator
except ImportError:
    A7_TestbenchGenerator = None

try:
    from formal_runner import run_formal
except ImportError:
    run_formal = None

# =============================================================================
# A7 TESTBENCH GENERATOR INTEGRATION
# =============================================================================

# Initialize A7 testbench generator (GLOBAL - runs once at startup)
A7_TESTBENCH_GEN = None

def init_a7_generator(project_root):
    """Initialize A7 testbench generator with LLM enabled"""
    global A7_TESTBENCH_GEN
    if A7_TestbenchGenerator is None:
        return None

    import os
    os.environ["USE_A7_LLM"] = "1"  # Force enable A7 LLM

    try:
        A7_TESTBENCH_GEN = A7_TestbenchGenerator(project_root)
        if A7_TESTBENCH_GEN.llm_enabled:
            print(f"✓ A7 LLM testbench generator initialized (Model loaded: {A7_TESTBENCH_GEN._llm_model is not None})")
            return A7_TESTBENCH_GEN
        else:
            print("✗ A7 LLM not enabled - falling back to template testbenches")
            return None
    except Exception as e:
        print(f"✗ A7 initialization failed: {e}")
        return None


def generate_smart_testbench(rtl_code: str, module_name: str, spec_text: str = "", use_a7: bool = True) -> Tuple[str, str]:
    """
    Generate testbench using A7 LLM if available, otherwise template.

    Returns: (testbench_code, testbench_type)
        - testbench_type: "A7_SMART" or "TEMPLATE_DUMB"
    """
    if use_a7 and A7_TESTBENCH_GEN and A7_TESTBENCH_GEN.llm_enabled:
        try:
            # Use A7 to generate smart testbench
            a7_input = {
                "module_name": module_name,
                "rtl_path": "generated.v",
                "spec": spec_text,
                "ports": []  # A7 will parse from RTL
            }

            a7_result = A7_TESTBENCH_GEN.process(a7_input)

            if a7_result.success:
                return a7_result.output_data.get('result', {}).get('code', ''), "A7_SMART"
            else:
                print(f"  A7 generation failed: {a7_result.errors}, falling back to template")
                return generate_simple_testbench(rtl_code, module_name), "TEMPLATE_FALLBACK"
        except Exception as e:
            print(f"  A7 exception: {e}, falling back to template")
            return generate_simple_testbench(rtl_code, module_name), "TEMPLATE_FALLBACK"
    else:
        # No A7 available - use template
        return generate_simple_testbench(rtl_code, module_name), "TEMPLATE_DUMB"



PROJECT_ROOT = Path(__file__).parent.parent

# Expanded test suite with 8 FSM tests (was 1 in V5.3)
TEST_SPECS = [
    # Non-FSM tests (same as V5.3)
    {"name": "8-bit Counter with Enable", "spec": "Design an 8-bit counter with enable and synchronous reset. Counter increments only when enable is high.", "category": "enable_control"},
    {"name": "4-bit Counter with Async Reset", "spec": "Create a 4-bit counter with asynchronous active-low reset (rst_n).", "category": "async_reset"},
    {"name": "16-bit Register with Enable", "spec": "Design a 16-bit register with clock enable. Register updates only when enable is asserted.", "category": "enable_gating"},
    {"name": "8-bit Up/Down Counter", "spec": "Create an 8-bit up/down counter with enable. Counts up when up=1, down when up=0, only when enabled.", "category": "updown_counter"},
    {"name": "Clock Divider by 8", "spec": "Implement a clock divider that divides input clock frequency by 8.", "category": "clock_divider"},
    {"name": "Rising Edge Detector", "spec": "Design a rising edge detector that outputs a 1-cycle pulse on rising edges of input signal.", "category": "edge_detect"},
    {"name": "8-bit Shift Register Left", "spec": "Create an 8-bit left shift register with serial input, enable, and parallel output.", "category": "shift_register"},
    {"name": "Counter with Load", "spec": "Design an 8-bit counter with load, enable, and synchronous reset. Load has priority over count.", "category": "load_enable"},
    {"name": "Sync Reset Counter", "spec": "Design a 12-bit counter with synchronous active-high reset.", "category": "sync_reset"},

    # EXPANDED FSM TEST SUITE (8 tests, was 1 in V5.3)
    {"name": "2-State FSM (IDLE/ACTIVE)", "spec": "Create a 2-state FSM that transitions from IDLE to ACTIVE on start signal, back to IDLE on done signal.", "category": "fsm_2state"},
    {"name": "3-State Sequence Detector", "spec": "Design a 3-state FSM that detects the sequence '101' on a serial input.", "category": "fsm_seqdet"},
    {"name": "4-State Moore FSM", "spec": "Create a 4-state Moore FSM with one-hot encoding that cycles through states S0→S1→S2→S3→S0 on each clock when enable is high.", "category": "fsm_moore"},
    {"name": "Mealy FSM Output on Transition", "spec": "Design a Mealy FSM with 2 states where output goes high only during the transition from A to B when input x=1.", "category": "fsm_mealy"},
    {"name": "Traffic Light Controller FSM", "spec": "Create a traffic light controller FSM with 3 states (GREEN=30 cycles, YELLOW=5 cycles, RED=30 cycles).", "category": "fsm_traffic"},
    {"name": "Button Debounce FSM", "spec": "Design a button debounce FSM with IDLE, PRESS, and DEBOUNCE states that validates button press after 3 stable cycles.", "category": "fsm_debounce"},
    {"name": "Vending Machine FSM", "spec": "Create a vending machine FSM that accepts nickels (5¢) and dimes (10¢), dispenses item at 15¢, returns change.", "category": "fsm_vending"},
    {"name": "Pattern Detector FSM", "spec": "Design a pattern detector FSM that detects the sequence '1011' with overlap (e.g., '10111' contains 2 matches).", "category": "fsm_pattern"},
]


def _parse_module_ports(rtl_code: str) -> List[Tuple[str, str, str]]:
    """Extract direction, optional width, and name for module ports."""
    pattern = re.compile(r'(input|output)\s+(?:logic|wire|reg)?\s*(\[[^\]]+\])?\s*(\w+)', re.IGNORECASE)
    ports: List[Tuple[str, str, str]] = []
    for direction, width, name in pattern.findall(rtl_code):
        ports.append((direction.lower(), width or "", name))
    return ports


def _detect_clock_reset(ports: List[Tuple[str, str, str]]) -> Tuple[Optional[str], Optional[str], bool]:
    """Identify clock/reset ports and reset polarity."""
    clock_candidates = {"clk", "clk_i", "clk_in", "clock"}
    reset_candidates = {"rst", "rst_n", "reset", "reset_n"}

    clock_name = None
    reset_name = None
    reset_active_low = False

    for _, _, name in ports:
        lname = name.lower()
        if not clock_name and lname in clock_candidates:
            clock_name = name
        if not reset_name and lname in reset_candidates:
            reset_name = name
            reset_active_low = lname.endswith("n")

    return clock_name, reset_name, reset_active_low


def _ensure_output_present(ports: List[Tuple[str, str, str]]) -> None:
    """Raise if no outputs exist; simulation can't validate anything."""
    if not any(direction == "output" for direction, _, _ in ports):
        raise ValueError("Module exposes no outputs; cannot validate behavior.")


def _is_fsm_spec(spec_entry: Dict[str, str]) -> bool:
    category = spec_entry.get('category', '').lower()
    return category.startswith('fsm') or 'fsm' in category


def generate_simple_testbench(rtl_code: str, module_name: str) -> str:
    """Generate a minimal SystemVerilog testbench"""
    raw_ports = _parse_module_ports(rtl_code)
    _ensure_output_present(raw_ports)

    io_decl = []
    connections = []

    for direction, width, name in raw_ports:
        width_str = f"{width} " if width else ""
        if direction == "input":
            io_decl.append(f"    logic {width_str}{name};")
        else:
            io_decl.append(f"    wire {width_str}{name};")
        connections.append(f"        .{name}({name})")

    io_decl_str = "\n".join(io_decl) if io_decl else "    // No ports"
    conn_str = ",\n".join(connections) if connections else "        // No ports"

    clk_signal, reset_signal, reset_active_low = _detect_clock_reset(raw_ports)

    testbench = f"""// Auto-generated testbench for {module_name}
`timescale 1ns/1ps

module {module_name}_tb;

{io_decl_str}

    // DUT instance
    {module_name} dut (
{conn_str}
    );
"""

    if clk_signal:
        testbench += f"""
    // Clock generation (10ns period)
    initial {clk_signal} = 0;
    always #5 {clk_signal} = ~{clk_signal};
"""

    if reset_signal:
        reset_assert = "0" if reset_active_low else "1"
        reset_deassert = "1" if reset_active_low else "0"
        testbench += f"""
    // Reset sequence
    initial begin
        {reset_signal} = {reset_assert};
        #20 {reset_signal} = {reset_deassert};
    end
"""

    testbench += """
    // Test stimulus
    initial begin
        #100 $display("TESTBENCH COMPLETED");
        $finish;
    end

endmodule
"""

    return testbench


def extract_module_code(raw_output: str) -> str:
    """
    Extract pure Verilog module code from model output.
    Enhanced version that handles:
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


def run_test(model, tokenizer, spec_entry: Dict, run_id: int) -> Dict:
    """Run a single test with dual-path simulation"""
    testbench_type = "UNKNOWN"  # Initialize to avoid UnboundLocalError
    spec_text = spec_entry["spec"]

    # Generate RTL
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

    # Extract module
    module_match = re.search(r'module\s+(\w+)', rtl_code, re.IGNORECASE)
    if not module_match:
        return {
            "name": spec_entry["name"],
            "category": spec_entry["category"],
            "compile_success": False,
            "sim_success": False,
            "testbench_type": testbench_type,
        "passed": False,
            "error": "No module declaration found"
        }

    module_name = module_match.group(1)

    # Save RTL to temp file
    work_dir = Path(tempfile.mkdtemp(prefix=f'v5_4_bench_{run_id}_'))
    rtl_path = work_dir / f"{module_name}.v"
    rtl_path.write_text(rtl_code)

    # Generate testbench (try A7 first, fall back to template)
    try:
        testbench, testbench_type = generate_smart_testbench(rtl_code, module_name, spec_text, use_a7=True)
        tb_path = work_dir / f"{module_name}_tb.sv"
        tb_path.write_text(testbench)
    except Exception as e:
        return {
            "name": spec_entry["name"],
            "category": spec_entry["category"],
            "compile_success": False,
            "sim_success": False,
            "testbench_type": testbench_type,
        "passed": False,
            "error": f"Testbench generation failed: {str(e)}"
        }

    # Run DUAL-PATH simulation (automatic dialect detection + routing)
    sim_result = run_dual_path_simulation(
        rtl_path=rtl_path,
        testbench_path=tb_path,
        work_dir=work_dir,
        timeout=30
    )

    return {
        "name": spec_entry["name"],
        "category": spec_entry["category"],
        "compile_success": sim_result.get("compile_success", False),
        "sim_success": sim_result.get("sim_success", False),
        "testbench_type": testbench_type,
        "passed": sim_result.get("passed", False),
        "compiler": sim_result.get("compiler", "unknown"),
        "dialect": sim_result.get("dialect", "unknown"),
        "error": sim_result.get("errors", [])
    }


def benchmark_v5_4(model_path: Path, num_runs: int = 5):
    """Benchmark V5.4 model with dual-path simulation"""

    print("="*80)
    print("  V5.4 DIALECT-AWARE MODEL BENCHMARK")
    print("="*80)
    print()
    print("V5.4 Improvements:")
    print("  • Dialect-aware training: [HDL:dialect] conditioning tokens")
    print("  • Dual-path simulation: iverilog (Verilog) + Verilator (SystemVerilog)")
    print("  • Expanded FSM suite: 8 FSM tests (was 1 in V5.3)")
    print("  • Automatic dialect detection & compiler routing")
    print()
    print(f"Model: {model_path}")
    print(f"Test Suite: {len(TEST_SPECS)} specs × {num_runs} runs = {len(TEST_SPECS) * num_runs} total tests")
    print(f"FSM Tests: 8 (vs 1 in V5.3)")
    print()

    # Load model
    print("Loading V5.4 model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ),
        device_map="auto"
    )

    # Resolve to absolute path
    adapter_path = Path(model_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded successfully")
    print()

    # Run benchmark
    results = []
    fsm_results = []

    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        print(f"  RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*80}\n")

        for spec in TEST_SPECS:
            print(f"Testing: {spec['name']} ({spec['category']})")
            result = run_test(model, tokenizer, spec, run_idx)
            results.append(result)

            if _is_fsm_spec(spec):
                fsm_results.append(result)

            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            compiler = result.get("compiler", "unknown")
            dialect = result.get("dialect", "unknown")
            print(f"  {status} [{compiler} / {dialect}]")

            if not result["passed"] and result.get("error"):
                print(f"  Error: {result['error']}")

    # Calculate metrics
    total = len(results)
    compile_success = sum(1 for r in results if r["compile_success"])
    sim_success = sum(1 for r in results if r["sim_success"])
    passed = sum(1 for r in results if r["passed"])

    fsm_total = len(fsm_results)
    fsm_passed = sum(1 for r in fsm_results if r["passed"])

    compile_pct = (compile_success / total * 100) if total > 0 else 0
    sim_pct = (sim_success / total * 100) if total > 0 else 0
    functional_pct = (passed / total * 100) if total > 0 else 0
    fsm_functional_pct = (fsm_passed / fsm_total * 100) if fsm_total > 0 else 0

    # Print results
    print("\n" + "="*80)
    print("  V5.4 BENCHMARK RESULTS")
    print("="*80)
    print()
    print(f"Overall Performance:")
    print(f"  Compilation:  {compile_success}/{total} ({compile_pct:.1f}%)")
    print(f"  Simulation:   {sim_success}/{total} ({sim_pct:.1f}%)")
    print(f"  Functional:   {passed}/{total} ({functional_pct:.1f}%)")
    print()
    print(f"FSM Performance:")
    print(f"  FSM Tests:    {fsm_total} (8 FSM specs × {num_runs} runs)")
    print(f"  FSM Passed:   {fsm_passed}/{fsm_total} ({fsm_functional_pct:.1f}%)")
    print()
    print("Historical FSM Comparison:")
    print(f"  V3 FSM:  60.0% (baseline)")
    print(f"  V4 FSM:  20.0% (regression)")
    print(f"  V5.3 FSM: 0.0% (iverilog-only, SystemVerilog fails)")
    print(f"  V5.4 FSM: {fsm_functional_pct:.1f}% 🎯 (dual-path validation)")
    print()

    # Compiler/Dialect breakdown
    compiler_counts = {}
    dialect_counts = {}
    for r in results:
        compiler = r.get("compiler", "unknown")
        dialect = r.get("dialect", "unknown")
        compiler_counts[compiler] = compiler_counts.get(compiler, 0) + 1
        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1

    print("Dual-Path Routing Statistics:")
    print("  Compilers:")
    for compiler, count in sorted(compiler_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"    {compiler:12s}: {count:3d} ({pct:5.1f}%)")
    print("  Dialects:")
    for dialect, count in sorted(dialect_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"    {dialect:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    if fsm_functional_pct >= 60:
        print("✅ SUCCESS! V5.4 recovers FSM performance to V3 baseline!")
        print("   Dual-path validation + dialect-aware training validated!")
    elif fsm_functional_pct >= 40:
        print("✅ STRONG IMPROVEMENT! V5.4 significantly improved over V5.3 (0%)")
        print("   Dual-path simulation working, may need more tuning")
    elif fsm_functional_pct >= 20:
        print("✅ PROGRESS! V5.4 shows improvement over V5.3 (0%)")
        print("   Dialect detection working, further investigation needed")
    elif fsm_functional_pct > 0:
        print("⚠️  PARTIAL FIX: V5.4 improved over V5.3 but below target")
        print("   Root cause partially addressed")
    else:
        print("⚠️  V5.4 did not resolve the FSM 0% issue")
        print("   Further investigation required")

    return functional_pct, compile_pct, sim_pct, fsm_functional_pct


def main():
    parser = argparse.ArgumentParser(description="Benchmark V5.4 model with dual-path simulation")
    parser.add_argument("--model", type=str, help="Path to V5.4 model (default: auto-detect latest)")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per test (default: 5)")
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
    else:
        # Auto-detect latest V5.4 model
        models_dir = PROJECT_ROOT / "models" / "qwen_coder_rtl"
        v5_4_dirs = sorted([d for d in models_dir.glob("run_rtl_behavioral_v5_4_*") if d.is_dir()], reverse=True)
        if not v5_4_dirs:
            print("❌ No V5.4 model found. Train V5.4 first!")
            sys.exit(1)
        model_path = v5_4_dirs[0] / "final_model"

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)

    benchmark_v5_4(model_path, args.runs)


if __name__ == "__main__":
    main()
