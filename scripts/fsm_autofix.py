#!/usr/bin/env python3
"""
FSM Auto-Repair Script
Deterministically repairs FSM training data with mechanical issues.
Focus: Adding missing ports mentioned in spec but absent from module.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import subprocess
import tempfile

def extract_signals_from_spec(spec: str) -> Dict[str, Set[str]]:
    """
    Extract signal names from specification text.
    Returns dict with 'inputs', 'outputs', 'inouts' keys.
    """
    signals = {
        'inputs': set(),
        'outputs': set(),
        'inouts': set()
    }

    # Common signal patterns in specs
    # Pattern 1: "input signal_name" or "output signal_name"
    input_patterns = [
        r'input\s+(?:wire\s+|logic\s+|reg\s+)?(\w+)',
        r'(\w+)\s+input',
        r'takes?\s+(\w+)\s+as\s+(?:an\s+)?input',
    ]

    output_patterns = [
        r'output\s+(?:wire\s+|logic\s+|reg\s+)?(\w+)',
        r'(\w+)\s+output',
        r'produces?\s+(\w+)\s+as\s+(?:an\s+)?output',
        r'outputs?\s+(\w+)',
    ]

    for pattern in input_patterns:
        matches = re.findall(pattern, spec, re.IGNORECASE)
        signals['inputs'].update(matches)

    for pattern in output_patterns:
        matches = re.findall(pattern, spec, re.IGNORECASE)
        signals['outputs'].update(matches)

    # Common FSM signals - infer direction from name
    common_inputs = ['clk', 'clock', 'rst', 'rst_n', 'reset', 'reset_n',
                     'start', 'enable', 'en', 'data_in', 'din', 'x', 'req', 'valid']
    common_outputs = ['done', 'ready', 'q', 'out', 'data_out', 'dout', 'y',
                      'ack', 'match', 'active', 'state_out', 'count', 'count_out']

    # Check for these signals mentioned anywhere in spec
    spec_lower = spec.lower()
    for sig in common_inputs:
        if sig in spec_lower or sig.replace('_', '') in spec_lower:
            signals['inputs'].add(sig)

    for sig in common_outputs:
        if sig in spec_lower or sig.replace('_', '') in spec_lower:
            signals['outputs'].add(sig)

    # Remove noise words
    noise = {'input', 'output', 'wire', 'reg', 'logic', 'the', 'a', 'an', 'is', 'are', 'be', 'to'}
    signals['inputs'] = {s for s in signals['inputs'] if s.lower() not in noise}
    signals['outputs'] = {s for s in signals['outputs'] if s.lower() not in noise}

    return signals


def extract_existing_ports(code: str) -> Dict[str, List[str]]:
    """
    Extract existing port declarations from module.
    Returns dict with 'inputs', 'outputs', 'inouts' keys.
    """
    ports = {
        'inputs': [],
        'outputs': [],
        'inouts': []
    }

    # Find module declaration
    module_match = re.search(r'module\s+\w+\s*[#(](.*?)\);', code, re.DOTALL)
    if not module_match:
        return ports

    port_section = module_match.group(1)

    # Extract input ports
    input_pattern = r'input\s+(?:wire\s+|logic\s+|reg\s+)?(?:\[[^\]]+\]\s+)?(\w+)'
    ports['inputs'] = re.findall(input_pattern, port_section, re.MULTILINE)

    # Extract output ports
    output_pattern = r'output\s+(?:wire\s+|logic\s+|reg\s+)?(?:\[[^\]]+\]\s+)?(\w+)'
    ports['outputs'] = re.findall(output_pattern, port_section, re.MULTILINE)

    # Extract inout ports
    inout_pattern = r'inout\s+(?:wire\s+|logic\s+|reg\s+)?(?:\[[^\]]+\]\s+)?(\w+)'
    ports['inouts'] = re.findall(inout_pattern, port_section, re.MULTILINE)

    return ports


def normalize_signal_name(signal: str, existing_ports: Dict[str, List[str]]) -> str:
    """
    Normalize signal name to match existing convention in module.
    E.g., if spec says 'reset' but module uses 'rst_n', return 'rst_n'
    """
    all_ports = existing_ports['inputs'] + existing_ports['outputs'] + existing_ports['inouts']

    # Check for exact match
    if signal in all_ports:
        return signal

    # Common aliases
    aliases = {
        'reset': ['rst', 'rst_n', 'reset_n'],
        'rst': ['reset', 'rst_n', 'reset_n'],
        'rst_n': ['reset', 'rst', 'reset_n'],
        'clock': ['clk'],
        'clk': ['clock'],
        'enable': ['en'],
        'en': ['enable'],
        'data_in': ['din', 'd', 'data'],
        'data_out': ['dout', 'q', 'out'],
    }

    signal_lower = signal.lower()
    if signal_lower in aliases:
        for alias in aliases[signal_lower]:
            if alias in all_ports:
                return alias

    # Check reverse direction
    for key, alias_list in aliases.items():
        if signal_lower in alias_list:
            if key in all_ports:
                return key
            for alias in alias_list:
                if alias in all_ports:
                    return alias

    return signal


def detect_dialect_from_code(code: str) -> str:
    """Detect HDL dialect from code."""
    if 'always_ff' in code or 'always_comb' in code:
        return 'sv2009'
    if 'typedef enum' in code or 'typedef struct' in code:
        return 'sv2012'
    if re.search(r'\blogic\b', code):
        return 'sv2005'
    return 'verilog2001'


def insert_missing_port(code: str, port_name: str, direction: str, dialect: str) -> str:
    """
    Insert a missing port into the module declaration.
    """
    # Find module declaration
    module_match = re.search(r'(module\s+\w+\s*(?:#[^(]*)?\()(.*?)(\);)', code, re.DOTALL)
    if not module_match:
        return code

    module_pre = module_match.group(1)
    port_section = module_match.group(2)
    module_post = module_match.group(3)

    # Determine wire type based on dialect
    if dialect.startswith('sv'):
        wire_type = 'logic'
    else:
        if direction == 'output':
            wire_type = 'reg'
        else:
            wire_type = 'wire'

    # Construct new port declaration
    new_port = f"    {direction} {wire_type} {port_name}"

    # Add to end of port list
    port_section_stripped = port_section.rstrip()
    if port_section_stripped and not port_section_stripped.endswith(','):
        port_section_stripped += ','

    port_section_new = port_section_stripped + f"\n{new_port}\n"

    # Reconstruct module
    new_code = module_pre + port_section_new + module_post

    # Insert rest of code
    rest_of_code = code[module_match.end():]
    new_code += rest_of_code

    return new_code


def repair_fsm_example(example: Dict, repair_log: List[str]) -> Tuple[Dict, bool]:
    """
    Attempt to repair a single FSM example.
    Returns (repaired_example, was_repaired)
    """
    spec = example.get('instruction', '')
    code = example.get('output', '')

    # Strip HDL conditioning token from spec for analysis
    spec_clean = re.sub(r'\[HDL:\w+\]\s*', '', spec)

    # Detect dialect
    dialect = detect_dialect_from_code(code)

    # Extract signals from spec
    spec_signals = extract_signals_from_spec(spec_clean)

    # Extract existing ports
    existing_ports = extract_existing_ports(code)

    # Find missing ports
    missing_inputs = []
    missing_outputs = []

    for sig in spec_signals['inputs']:
        normalized = normalize_signal_name(sig, existing_ports)
        if normalized not in existing_ports['inputs']:
            # Check if it might already be there under a different name
            if not any(s.lower() == sig.lower() or s.lower() == normalized.lower()
                      for s in existing_ports['inputs']):
                missing_inputs.append(normalized)

    for sig in spec_signals['outputs']:
        normalized = normalize_signal_name(sig, existing_ports)
        if normalized not in existing_ports['outputs']:
            if not any(s.lower() == sig.lower() or s.lower() == normalized.lower()
                      for s in existing_ports['outputs']):
                missing_outputs.append(normalized)

    # If no missing ports, mark as clean
    if not missing_inputs and not missing_outputs:
        return example, False

    # Attempt repair
    repaired_code = code
    repairs_made = []

    for port in missing_inputs:
        repaired_code = insert_missing_port(repaired_code, port, 'input', dialect)
        repairs_made.append(f"Added input: {port}")

    for port in missing_outputs:
        repaired_code = insert_missing_port(repaired_code, port, 'output', dialect)
        repairs_made.append(f"Added output: {port}")

    # Create repaired example
    repaired_example = example.copy()
    repaired_example['output'] = repaired_code

    # Add repair metadata
    if 'metadata' not in repaired_example:
        repaired_example['metadata'] = {}

    repaired_example['metadata']['autofix'] = {
        'repaired': True,
        'repairs': repairs_made,
        'original_hash': hash(code) % (10 ** 8)
    }

    repair_log.extend(repairs_made)

    return repaired_example, True


def lint_with_compiler(code: str, dialect: str, work_dir: Path) -> Tuple[bool, str]:
    """
    Lint code with appropriate compiler (iverilog or verilator).
    Returns (success, error_message)
    """
    # Write code to temp file
    code_file = work_dir / "test.v"
    code_file.write_text(code)

    try:
        if dialect == 'verilog2001':
            # Use iverilog
            result = subprocess.run(
                ['iverilog', '-tnull', '-Wall', str(code_file)],
                capture_output=True,
                text=True,
                timeout=5
            )
        else:
            # Use verilator
            result = subprocess.run(
                ['verilator', '--lint-only', '-Wall', str(code_file)],
                capture_output=True,
                text=True,
                timeout=5
            )

        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except FileNotFoundError:
        return False, "Compiler not found"
    except Exception as e:
        return False, str(e)


def main():
    print("="*80)
    print("FSM AUTO-REPAIR PIPELINE")
    print("="*80)
    print()

    # Load dataset
    dataset_path = Path("/home/quantumc1/cda-agent-2C1/data/rtl_behavioral_v5_4.jsonl")
    print(f"Loading dataset from {dataset_path}...")

    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Total examples: {len(examples)}")

    # Filter for FSM examples
    fsm_examples = [ex for ex in examples if ex.get('hierarchy', {}).get('l2') == 'FSM']
    print(f"FSM examples: {len(fsm_examples)}")
    print()

    # Repair FSMs
    print("Starting repair process...")
    print()

    repaired_examples = []
    clean_examples = []
    failed_repairs = []

    repair_stats = {
        'total': len(fsm_examples),
        'clean': 0,
        'repaired': 0,
        'failed_lint': 0,
        'failed_repair': 0
    }

    work_dir = Path(tempfile.mkdtemp())

    for idx, example in enumerate(fsm_examples):
        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{len(fsm_examples)}...")

        repair_log = []
        repaired_example, was_repaired = repair_fsm_example(example, repair_log)

        if not was_repaired:
            # Already clean
            clean_examples.append(example)
            repair_stats['clean'] += 1
        else:
            # Attempt was made - now lint
            dialect = detect_dialect_from_code(repaired_example['output'])
            lint_ok, lint_error = lint_with_compiler(repaired_example['output'], dialect, work_dir)

            if lint_ok:
                repaired_examples.append(repaired_example)
                repair_stats['repaired'] += 1
                print(f"  [REPAIRED {idx}] {', '.join(repair_log)}")
            else:
                failed_repairs.append({
                    'example': example,
                    'repair_log': repair_log,
                    'lint_error': lint_error
                })
                repair_stats['failed_lint'] += 1

    print()
    print("="*80)
    print("REPAIR SUMMARY")
    print("="*80)
    print(f"  Total FSM examples: {repair_stats['total']}")
    print(f"  Already clean: {repair_stats['clean']} ({repair_stats['clean']/repair_stats['total']*100:.1f}%)")
    print(f"  Successfully repaired: {repair_stats['repaired']} ({repair_stats['repaired']/repair_stats['total']*100:.1f}%)")
    print(f"  Failed lint after repair: {repair_stats['failed_lint']} ({repair_stats['failed_lint']/repair_stats['total']*100:.1f}%)")
    print()
    print(f"  Total usable FSMs: {repair_stats['clean'] + repair_stats['repaired']} ({(repair_stats['clean'] + repair_stats['repaired'])/repair_stats['total']*100:.1f}%)")
    print()

    # Save results
    clean_fsm_path = Path("/home/quantumc1/cda-agent-2C1/data/fsm_clean_v5_4.jsonl")
    repaired_fsm_path = Path("/home/quantumc1/cda-agent-2C1/data/fsm_repaired_v5_4.jsonl")
    failed_fsm_path = Path("/home/quantumc1/cda-agent-2C1/data/fsm_failed_repair_v5_4.jsonl")

    with open(clean_fsm_path, 'w') as f:
        for ex in clean_examples:
            f.write(json.dumps(ex) + '\n')

    with open(repaired_fsm_path, 'w') as f:
        for ex in repaired_examples:
            f.write(json.dumps(ex) + '\n')

    with open(failed_fsm_path, 'w') as f:
        for item in failed_repairs:
            f.write(json.dumps(item) + '\n')

    print(f"Clean FSMs saved to: {clean_fsm_path}")
    print(f"Repaired FSMs saved to: {repaired_fsm_path}")
    print(f"Failed repairs saved to: {failed_fsm_path}")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(work_dir)

    print("="*80)
    print("NEXT STEPS:")
    print("  1. Review failed repairs to understand irreparable patterns")
    print(f"  2. Generate {repair_stats['failed_lint'] + (repair_stats['total'] - repair_stats['clean'] - repair_stats['repaired'])} synthetic FSMs to fill gaps")
    print("  3. Merge clean + repaired + synthetic into V5.5 dataset")
    print("="*80)


if __name__ == "__main__":
    main()
