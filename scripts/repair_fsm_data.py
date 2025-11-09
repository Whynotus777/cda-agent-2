#!/usr/bin/env python3
"""
FSM Data Repair Pipeline

Strategy: Clean first, synthesize second.
1. Identify FSMs with missing/broken interfaces (missing output ports)
2. Attempt to repair by analyzing module body and inferring outputs
3. Validate repairs with iverilog compilation
4. Create cleaned, verified FSM dataset

This restores the structural integrity before any synthesis.
"""

import json
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

def analyze_module_interface(rtl: str) -> dict:
    """Analyze module interface to detect issues"""
    issues = []

    # Extract module declaration
    module_match = re.search(r'module\s+(\w+)\s*\(([^;]+)\);', rtl, re.DOTALL)
    if not module_match:
        return {'valid': False, 'issues': ['No module declaration found']}

    module_name = module_match.group(1)
    port_list = module_match.group(2)

    # Check for outputs
    has_output = 'output' in port_list

    # Check for inputs
    has_input = 'input' in port_list

    if not has_output:
        issues.append('missing_output_port')

    if not has_input:
        issues.append('missing_input_port')

    return {
        'valid': len(issues) == 0,
        'module_name': module_name,
        'port_list': port_list,
        'has_output': has_output,
        'has_input': has_input,
        'issues': issues
    }

def infer_missing_outputs(rtl: str, module_info: dict) -> list:
    """Infer what output ports should exist by analyzing module body"""
    outputs = []

    # Look for assign statements (implies output)
    for match in re.finditer(r'assign\s+(\w+)\s*=', rtl):
        signal = match.group(1)
        outputs.append(signal)

    # Look for output registers in always blocks
    # Pattern: Check for signals that are only assigned to (never read from first)
    # This is heuristic-based

    # Look for common FSM output patterns
    if 'state' in rtl.lower():
        # Check for active/busy/valid type signals
        for pattern in [r'\b(active|busy|valid|ready|done)\b', r'\b(\w*_out)\b']:
            for match in re.finditer(pattern, rtl):
                signal = match.group(1)
                if signal not in outputs:
                    outputs.append(signal)

    return list(set(outputs))  # Remove duplicates

def repair_missing_output(rtl: str, module_info: dict) -> Optional[str]:
    """Attempt to repair module by adding missing output port"""

    # Infer missing outputs
    inferred_outputs = infer_missing_outputs(rtl, module_info)

    if not inferred_outputs:
        return None  # Can't infer outputs, repair failed

    # Build new port list with outputs
    module_name = module_info['module_name']
    old_port_list = module_info['port_list']

    # Add output declarations (assume logic type, 1-bit)
    output_decls = []
    for out in inferred_outputs:
        # Try to infer width from module body
        width_match = re.search(rf'{out}\s*\[(\d+):0\]', rtl)
        if width_match:
            width = int(width_match.group(1)) + 1
            output_decls.append(f'output logic [{width-1}:0] {out}')
        else:
            output_decls.append(f'output logic {out}')

    # Reconstruct module declaration
    new_port_list = old_port_list.rstrip()
    if not new_port_list.endswith(','):
        new_port_list += ','
    new_port_list += '\n    ' + ',\n    '.join(output_decls)

    # Replace module declaration
    new_rtl = re.sub(
        r'module\s+\w+\s*\([^;]+\);',
        f'module {module_name}({new_port_list});',
        rtl,
        count=1,
        flags=re.DOTALL
    )

    return new_rtl

def validate_rtl_syntax(rtl: str) -> Tuple[bool, str]:
    """Validate RTL syntax with iverilog"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
        f.write(rtl)
        rtl_file = f.name

    try:
        result = subprocess.run(
            ['iverilog', '-g2012', '-o', '/dev/null', rtl_file],
            capture_output=True,
            text=True,
            timeout=5
        )

        success = result.returncode == 0
        log = result.stderr if result.stderr else result.stdout

        return success, log
    except Exception as e:
        return False, str(e)
    finally:
        Path(rtl_file).unlink(missing_ok=True)

def repair_fsm_dataset(input_path: Path, output_path: Path):
    """Main repair pipeline"""

    print("=" * 80)
    print("  FSM DATA REPAIR PIPELINE")
    print("=" * 80)
    print()
    print("Strategy: Clean first, synthesize second")
    print("  1. Identify broken FSMs (missing outputs)")
    print("  2. Attempt structural repair")
    print("  3. Validate with iverilog")
    print("  4. Output cleaned dataset")
    print()

    stats = {
        'total': 0,
        'fsm_count': 0,
        'broken_fsms': 0,
        'repair_attempted': 0,
        'repair_success': 0,
        'repair_failed': 0,
        'already_clean': 0
    }

    repaired_examples = []
    broken_examples = []

    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)
            stats['total'] += 1

            # Only process FSMs
            if ex.get('hierarchy', {}).get('l2') != 'FSM':
                repaired_examples.append(ex)
                continue

            stats['fsm_count'] += 1
            rtl = ex.get('output', '')

            # Analyze interface
            module_info = analyze_module_interface(rtl)

            if module_info['valid']:
                stats['already_clean'] += 1
                repaired_examples.append(ex)
                continue

            # Broken FSM detected
            stats['broken_fsms'] += 1

            if 'missing_output_port' in module_info.get('issues', []):
                print(f"🔧 Repairing FSM: {ex.get('instruction', '')[:60]}...")

                # Attempt repair
                stats['repair_attempted'] += 1
                repaired_rtl = repair_missing_output(rtl, module_info)

                if repaired_rtl:
                    # Validate repair
                    valid, log = validate_rtl_syntax(repaired_rtl)

                    if valid:
                        stats['repair_success'] += 1
                        ex['output'] = repaired_rtl
                        ex['metadata']['repaired'] = True
                        ex['metadata']['repair_log'] = 'Added missing output port'
                        repaired_examples.append(ex)
                        print(f"   ✅ Repair successful")
                    else:
                        stats['repair_failed'] += 1
                        broken_examples.append({
                            'example': ex,
                            'repair_log': log
                        })
                        print(f"   ❌ Repair failed: {log[:100]}")
                else:
                    stats['repair_failed'] += 1
                    print(f"   ❌ Could not infer outputs")
            else:
                # Other issues, skip
                stats['repair_failed'] += 1

    # Save cleaned dataset
    print()
    print(f"💾 Saving cleaned dataset: {output_path}")
    with open(output_path, 'w') as f:
        for ex in repaired_examples:
            f.write(json.dumps(ex) + '\n')

    # Save broken examples for review
    broken_path = output_path.parent / 'fsm_repair_failed.json'
    print(f"📋 Saving failed repairs: {broken_path}")
    with open(broken_path, 'w') as f:
        json.dump(broken_examples, f, indent=2)

    # Print summary
    print()
    print("=" * 80)
    print("  REPAIR SUMMARY")
    print("=" * 80)
    print()
    print(f"Total examples: {stats['total']}")
    print(f"Total FSMs: {stats['fsm_count']}")
    print(f"  Already clean: {stats['already_clean']} ({stats['already_clean']/stats['fsm_count']*100:.1f}%)")
    print(f"  Broken (missing outputs): {stats['broken_fsms']} ({stats['broken_fsms']/stats['fsm_count']*100:.1f}%)")
    print()
    print(f"Repair attempts: {stats['repair_attempted']}")
    print(f"  ✅ Successful: {stats['repair_success']} ({stats['repair_success']/stats['repair_attempted']*100:.1f}%)")
    print(f"  ❌ Failed: {stats['repair_failed']} ({stats['repair_failed']/stats['repair_attempted']*100:.1f}%)")
    print()
    print(f"📊 Final cleaned dataset:")
    print(f"   Total examples: {len(repaired_examples)}")
    print(f"   FSMs: {stats['already_clean'] + stats['repair_success']}")
    print(f"   Removed: {stats['repair_failed']}")
    print()

    return stats

def main():
    project_root = Path(__file__).parent.parent

    input_path = project_root / 'data' / 'rtl_behavioral_v5_2_classified.jsonl'
    output_path = project_root / 'data' / 'rtl_behavioral_v5_2_repaired.jsonl'

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    stats = repair_fsm_dataset(input_path, output_path)

    print("✅ Data repair complete!")
    print()
    print("Next steps:")
    print("  1. Review repaired dataset: data/rtl_behavioral_v5_2_repaired.jsonl")
    print("  2. Add 60 FSM gold examples from data/fsm_gold_consolidated.jsonl")
    print("  3. Train V5.3 on cleaned + gold FSM data")
    print("  4. Benchmark to verify FSM recovery")
    print("  5. THEN consider synthesis for expansion")
    print()

if __name__ == "__main__":
    main()
