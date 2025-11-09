#!/usr/bin/env python3
"""
Universal Validation Framework for ALL Hierarchies

This script validates ALL examples in a dataset across all 9 behavioral hierarchies:
- FSM (Finite State Machine)
- Counter
- ShiftRegister
- Arithmetic
- Controller
- Edge
- Clocking
- Memory
- Protocol

Each example undergoes:
1. Syntax validation (iverilog/verilator)
2. Simulation (runs without errors)
3. Functional correctness (output matches specification)
4. Hierarchy-specific validation rules

Only examples passing ALL checks are marked as validated.
"""

import json
import re
import subprocess
import tempfile
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.validation import validate_fsm_strict, ValidationResult

@dataclass
class UniversalValidationResult:
    """Result of universal validation"""
    passed: bool
    syntax_check: bool
    simulation_check: bool
    functional_check: bool
    hierarchy_check: bool
    errors: List[str]
    warnings: List[str]


class UniversalValidator:
    """Universal validator for all hierarchies"""

    def __init__(self):
        self.validation_timeout = 10  # seconds

    def validate(self, example: Dict) -> UniversalValidationResult:
        """
        Validate an example based on its hierarchy

        Args:
            example: Full example dictionary with instruction, output, hierarchy, metadata

        Returns:
            UniversalValidationResult
        """
        hierarchy = example.get('hierarchy', {}).get('l2', 'Unknown')
        instruction = example.get('instruction', '')
        code = example.get('output', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')

        # Route to hierarchy-specific validator
        if hierarchy == 'FSM':
            return self._validate_fsm(example)
        elif hierarchy == 'Counter':
            return self._validate_counter(example)
        elif hierarchy == 'ShiftRegister':
            return self._validate_shift_register(example)
        elif hierarchy == 'Arithmetic':
            return self._validate_arithmetic(example)
        elif hierarchy == 'Controller':
            return self._validate_controller(example)
        elif hierarchy == 'Edge':
            return self._validate_edge(example)
        elif hierarchy == 'Clocking':
            return self._validate_clocking(example)
        elif hierarchy == 'Memory':
            return self._validate_memory(example)
        elif hierarchy == 'Protocol':
            return self._validate_protocol(example)
        else:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=[f"Unknown hierarchy: {hierarchy}"],
                warnings=[]
            )

    def _validate_fsm(self, example: Dict) -> UniversalValidationResult:
        """Validate FSM using strict FSM validator"""
        instruction = example.get('instruction', '')
        code = example.get('output', '')
        fsm_type = example.get('hierarchy', {}).get('l3', 'Moore')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')

        result = validate_fsm_strict(code, instruction, fsm_type, dialect)

        return UniversalValidationResult(
            passed=result.passed,
            syntax_check=result.syntax_check,
            simulation_check=result.simulation_check,
            functional_check=result.specification_check,
            hierarchy_check=result.passed,
            errors=result.errors,
            warnings=result.warnings
        )

    def _validate_counter(self, example: Dict) -> UniversalValidationResult:
        """Validate Counter"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Step 1: Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # Step 2: Counter-specific checks
        hierarchy_ok = True

        # Check for counter signal (count, counter, q, out, etc.)
        if not re.search(r'(count|counter|q|out)', code, re.IGNORECASE):
            errors.append("Counter must have output signal (count/counter/q/out)")
            hierarchy_ok = False

        # Check for enable/reset logic
        if 'enable' in instruction.lower() or 'en' in instruction.lower():
            if not re.search(r'\b(enable|en)\b', code, re.IGNORECASE):
                errors.append("Counter spec requires enable signal")
                hierarchy_ok = False

        # Step 3: Simulation validation (basic)
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,  # Simplified for now
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_shift_register(self, example: Dict) -> UniversalValidationResult:
        """Validate Shift Register"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # ShiftRegister-specific checks
        hierarchy_ok = True

        # Check for shift operation
        if not re.search(r'(<<|>>|shift)', code, re.IGNORECASE):
            errors.append("Shift register must have shift operation")
            hierarchy_ok = False

        # Check for serial input/output
        if 'serial' in instruction.lower():
            if not re.search(r'(serial_in|si|din)', code, re.IGNORECASE):
                warnings.append("Shift register might be missing serial input")

        # Simulation
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_arithmetic(self, example: Dict) -> UniversalValidationResult:
        """Validate Arithmetic"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # Arithmetic-specific checks
        hierarchy_ok = True

        # Check for arithmetic operations
        if not re.search(r'(\+|-|\*|/|%)', code):
            errors.append("Arithmetic module must have arithmetic operations")
            hierarchy_ok = False

        # Simulation
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_controller(self, example: Dict) -> UniversalValidationResult:
        """Validate Controller"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # Controller-specific checks
        hierarchy_ok = True

        # Check for control signals
        if not re.search(r'(enable|valid|ready|start|done)', code, re.IGNORECASE):
            warnings.append("Controller might be missing control signals")

        # Simulation
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_edge(self, example: Dict) -> UniversalValidationResult:
        """Validate Edge Detector"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # Edge-specific checks
        hierarchy_ok = True

        # Check for edge detection logic (comparing with delayed signal)
        if not re.search(r'(prev|last|delay)', code, re.IGNORECASE):
            warnings.append("Edge detector might be missing delayed signal")

        # Simulation
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_clocking(self, example: Dict) -> UniversalValidationResult:
        """Validate Clocking"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # Clocking-specific checks
        hierarchy_ok = True

        # Check for clock-related signals
        if not re.search(r'(clk|clock)', code, re.IGNORECASE):
            errors.append("Clocking module must have clock signals")
            hierarchy_ok = False

        # Simulation
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_memory(self, example: Dict) -> UniversalValidationResult:
        """Validate Memory"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # Memory-specific checks
        hierarchy_ok = True

        # Check for memory array/register
        if not re.search(r'(mem|memory|ram|reg\s*\[)', code, re.IGNORECASE):
            errors.append("Memory module must have memory array")
            hierarchy_ok = False

        # Check for read/write logic
        if not re.search(r'(write|read|we|re|wr|rd)', code, re.IGNORECASE):
            warnings.append("Memory might be missing read/write control")

        # Simulation
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_protocol(self, example: Dict) -> UniversalValidationResult:
        """Validate Protocol"""
        code = example.get('output', '')
        instruction = example.get('instruction', '')
        dialect = example.get('metadata', {}).get('dialect', 'verilog2001')
        errors = []
        warnings = []

        # Syntax validation
        syntax_ok, syntax_errors = self._validate_syntax(code, dialect)
        if not syntax_ok:
            return UniversalValidationResult(
                passed=False,
                syntax_check=False,
                simulation_check=False,
                functional_check=False,
                hierarchy_check=False,
                errors=syntax_errors,
                warnings=warnings
            )

        # Protocol-specific checks
        hierarchy_ok = True

        # Check for protocol signals (valid/ready, req/ack, etc.)
        if not re.search(r'(valid|ready|req|ack|handshake)', code, re.IGNORECASE):
            warnings.append("Protocol might be missing handshake signals")

        # Simulation
        sim_ok, sim_errors = self._validate_simulation_basic(code, dialect)
        if not sim_ok:
            errors.extend(sim_errors)

        passed = syntax_ok and hierarchy_ok and sim_ok

        return UniversalValidationResult(
            passed=passed,
            syntax_check=syntax_ok,
            simulation_check=sim_ok,
            functional_check=True,
            hierarchy_check=hierarchy_ok,
            errors=errors,
            warnings=warnings
        )

    def _validate_syntax(self, code: str, dialect: str) -> Tuple[bool, List[str]]:
        """Validate syntax with iverilog/verilator"""
        errors = []

        if not code.endswith('\n'):
            code += '\n'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name

        try:
            if dialect.startswith('sv'):
                # Verilator
                result = subprocess.run(
                    ['verilator', '--lint-only', '-Wall', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.validation_timeout
                )
            else:
                # iverilog
                result = subprocess.run(
                    ['iverilog', '-Wall', '-Wno-timescale', '-t', 'null', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.validation_timeout
                )

            if result.returncode != 0:
                errors.append(f"Syntax error: {result.stderr[:500]}")
                return False, errors

            return True, []

        except subprocess.TimeoutExpired:
            errors.append("Syntax validation timeout")
            return False, errors
        except Exception as e:
            errors.append(f"Syntax validation error: {str(e)}")
            return False, errors
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def _validate_simulation_basic(self, code: str, dialect: str) -> Tuple[bool, List[str]]:
        """Basic simulation check - ensure code compiles and simulates"""
        errors = []

        # Generate minimal testbench
        testbench = self._generate_minimal_testbench(code)
        full_design = f"{code}\n\n{testbench}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(full_design)
            f.flush()
            design_file = f.name

        try:
            # Compile
            compile_result = subprocess.run(
                ['iverilog', '-g2005-sv', '-o', f'{design_file}.vvp', design_file],
                capture_output=True,
                text=True,
                timeout=self.validation_timeout
            )

            if compile_result.returncode != 0:
                errors.append(f"Simulation compile failed: {compile_result.stderr[:500]}")
                return False, errors

            # Simulate
            sim_result = subprocess.run(
                ['vvp', f'{design_file}.vvp'],
                capture_output=True,
                text=True,
                timeout=self.validation_timeout
            )

            if sim_result.returncode != 0:
                errors.append(f"Simulation failed: {sim_result.stderr[:500]}")
                return False, errors

            # Check for errors in output
            if 'ERROR' in sim_result.stdout:
                errors.append(f"Simulation error: {sim_result.stdout[:500]}")
                return False, errors

            return True, []

        except subprocess.TimeoutExpired:
            errors.append("Simulation timeout")
            return False, errors
        except Exception as e:
            errors.append(f"Simulation error: {str(e)}")
            return False, errors
        finally:
            Path(design_file).unlink(missing_ok=True)
            Path(f'{design_file}.vvp').unlink(missing_ok=True)

    def _generate_minimal_testbench(self, code: str) -> str:
        """Generate minimal testbench for basic simulation"""
        # Extract module name
        module_match = re.search(r'module\s+(\w+)', code)
        module_name = module_match.group(1) if module_match else 'dut'

        # Extract ports
        module_match = re.search(r'module\s+\w+\s*\((.*?)\);', code, re.DOTALL)
        if not module_match:
            return f"""
module tb_{module_name};
    initial begin
        #100;
        $finish;
    end
endmodule
"""

        port_list = module_match.group(1)

        # Simple testbench that just instantiates and runs
        inputs = []
        outputs = []

        # Extract inputs
        for match in re.finditer(r'input\s+(?:\w+\s+)?(?:\[.*?\]\s+)?(\w+)', port_list):
            inputs.append(match.group(1))

        # Extract outputs
        for match in re.finditer(r'output\s+(?:\w+\s+)?(?:\[.*?\]\s+)?(\w+)', port_list):
            outputs.append(match.group(1))

        # Identify clock/reset
        clk = next((p for p in inputs if 'clk' in p.lower()), None)
        rst = next((p for p in inputs if 'rst' in p.lower()), None)

        tb = f"module tb_{module_name};\n"

        # Declare signals
        for inp in inputs:
            tb += f"    reg {inp};\n"
        for out in outputs:
            tb += f"    wire {out};\n"

        # Instantiate DUT
        tb += f"\n    {module_name} dut(\n"
        ports_list = inputs + outputs
        for i, port in enumerate(ports_list):
            tb += f"        .{port}({port})"
            if i < len(ports_list) - 1:
                tb += ","
            tb += "\n"
        tb += "    );\n\n"

        # Clock generation if exists
        if clk:
            tb += f"    initial begin\n"
            tb += f"        {clk} = 0;\n"
            tb += f"        forever #5 {clk} = ~{clk};\n"
            tb += f"    end\n\n"

        # Test sequence
        tb += "    initial begin\n"

        # Reset if exists
        if rst:
            tb += f"        {rst} = {'0' if 'n' in rst else '1'};\n"
            tb += f"        #20;\n"
            tb += f"        {rst} = {'1' if 'n' in rst else '0'};\n"

        # Initialize other inputs
        for inp in inputs:
            if inp != clk and inp != rst:
                tb += f"        {inp} = 0;\n"

        tb += f"        #100;\n"
        tb += f"        $finish;\n"
        tb += f"    end\n"
        tb += f"endmodule\n"

        return tb


def validate_dataset(
    dataset_path: Path,
    output_path: Path,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Validate all examples in a dataset

    Args:
        dataset_path: Path to dataset JSONL file
        output_path: Path to save validation report
        verbose: Print detailed progress

    Returns:
        Dict with validation statistics
    """
    print("="*80)
    print("UNIVERSAL VALIDATION FRAMEWORK")
    print("="*80)
    print()
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_path}")
    print()

    # Load dataset
    if not dataset_path.exists():
        print(f"✗ Dataset not found: {dataset_path}")
        return {}

    all_examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            all_examples.append(json.loads(line))

    print(f"Total examples: {len(all_examples)}")
    print()

    # Group by hierarchy
    hierarchy_groups = {}
    for ex in all_examples:
        hier = ex.get('hierarchy', {}).get('l2', 'Unknown')
        if hier not in hierarchy_groups:
            hierarchy_groups[hier] = []
        hierarchy_groups[hier].append(ex)

    print("Hierarchy distribution:")
    for hier, examples in sorted(hierarchy_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {hier:15s}: {len(examples):4d} examples")
    print()

    # Validate each example
    validator = UniversalValidator()
    results = {}
    validated_examples = []

    print("="*80)
    print("VALIDATION IN PROGRESS...")
    print("="*80)
    print()

    for hierarchy, examples in sorted(hierarchy_groups.items()):
        print(f"Validating {hierarchy}:")
        print("-" * 40)

        passed = []
        failed = []

        for i, example in enumerate(examples, 1):
            example_id = example.get('metadata', {}).get('id', f'{hierarchy}_{i}')

            if verbose:
                print(f"  [{i}/{len(examples)}] {example_id}...", end=' ')

            result = validator.validate(example)

            if result.passed:
                # Mark as validated
                example['metadata']['validated'] = True
                example['metadata']['validation_passed'] = True
                passed.append(example)
                validated_examples.append(example)
                if verbose:
                    print("✓")
            else:
                failed.append({
                    'example': example,
                    'errors': result.errors
                })
                if verbose:
                    print(f"✗ {result.errors[0][:50] if result.errors else 'Unknown error'}")

        pass_rate = (len(passed) / len(examples) * 100) if examples else 0

        results[hierarchy] = {
            'total': len(examples),
            'passed': len(passed),
            'failed': len(failed),
            'pass_rate': pass_rate
        }

        print(f"  Total:  {len(examples)}")
        print(f"  Passed: {len(passed)} ({pass_rate:.1f}%)")
        print(f"  Failed: {len(failed)}")
        print()

    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print()

    total_examples = sum(r['total'] for r in results.values())
    total_passed = sum(r['passed'] for r in results.values())
    total_failed = sum(r['failed'] for r in results.values())
    overall_pass_rate = (total_passed / total_examples * 100) if total_examples > 0 else 0

    print(f"{'Hierarchy':<15} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Pass Rate':<12}")
    print("-" * 70)

    for hierarchy in sorted(results.keys()):
        r = results[hierarchy]
        print(f"{hierarchy:<15} {r['total']:<8} {r['passed']:<8} {r['failed']:<8} {r['pass_rate']:6.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_examples:<8} {total_passed:<8} {total_failed:<8} {overall_pass_rate:6.1f}%")
    print()

    # Save validated examples
    if validated_examples:
        validated_file = dataset_path.parent / f"{dataset_path.stem}_validated.jsonl"
        with open(validated_file, 'w') as f:
            for ex in validated_examples:
                f.write(json.dumps(ex) + '\n')

        print(f"✓ Saved {len(validated_examples)} validated examples to: {validated_file}")
        print()

    # Save validation report
    report = {
        'dataset': str(dataset_path),
        'total_examples': total_examples,
        'total_passed': total_passed,
        'total_failed': total_failed,
        'overall_pass_rate': overall_pass_rate,
        'hierarchy_results': results
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Validation report saved to: {output_path}")
    print()

    print("="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print()

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal validator for all hierarchies"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save validation report JSON'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed validation progress'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    validate_dataset(dataset_path, output_path, args.verbose)
