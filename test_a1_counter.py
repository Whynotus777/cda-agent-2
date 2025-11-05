#!/usr/bin/env python3
"""
Diagnostic Test: Can A1 Generate a Simple 4-bit Counter?

This tests the core reasoning capability of the A1 agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A1_SpecToRTLGenerator


def main():
    """Test A1 with simple counter specification"""

    print("\n" + "="*80)
    print("  A1 DIAGNOSTIC: Simple 4-bit Counter Test")
    print("="*80 + "\n")

    # Simple counter specification
    counter_spec = {
        "module_name": "counter_4bit",
        "specification": """A simple 4-bit counter that increments on each clock edge
        when enabled. Includes active-low asynchronous reset.""",
        "intent_type": "counter",
        "parameters": {
            "width": 4
        },
        "constraints": {
            "clock_period_ns": 10.0,
            "target_frequency_mhz": 100.0
        }
    }

    print("📝 Test Specification:")
    print(f"   Module: {counter_spec['module_name']}")
    print(f"   Intent: {counter_spec['intent_type']}")
    print(f"   Width: {counter_spec['parameters']['width']}-bit")
    print(f"   Description: {counter_spec['specification'][:80]}...\n")

    # Initialize A1
    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    # Generate RTL
    print("⚙️  Calling A1.process()...\n")
    result = agent.process(counter_spec)

    print("="*80)
    print("  A1 GENERATION RESULTS")
    print("="*80 + "\n")

    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Execution Time: {result.execution_time_ms:.2f}ms")

    if result.success:
        rtl_code = result.output_data['rtl_code']
        ports = result.output_data.get('ports', [])
        generation_method = result.output_data.get('generation_method', 'unknown')
        validation = result.output_data.get('validation', {})

        print(f"\n📊 Generation Metrics:")
        print(f"   Method: {generation_method}")
        print(f"   Lines: {len(rtl_code.split(chr(10)))}")
        print(f"   Ports: {len(ports)}")
        print(f"   Syntax Valid: {validation.get('syntax_valid', False)}")

        print(f"\n📌 Extracted Ports:")
        if ports:
            for port in ports:
                print(f"   {port['direction']:6} {port['name']:15} [{port['width']} bits]")
        else:
            print("   ⚠️  No ports extracted!")

        print(f"\n📄 Generated RTL Code:")
        print("="*80)
        print(rtl_code)
        print("="*80)

        # Write to file
        output_file = Path('/tmp/counter_4bit_diagnostic.v')
        output_file.write_text(rtl_code)
        print(f"\n✅ RTL written to: {output_file}")

        # Analysis
        print(f"\n🔍 DIAGNOSTIC ANALYSIS:")
        print("="*80)

        has_module = 'module' in rtl_code
        has_ports = 'input' in rtl_code or 'output' in rtl_code
        has_logic = 'always' in rtl_code or 'assign' in rtl_code
        has_clk = 'clk' in rtl_code.lower()
        has_reset = 'rst' in rtl_code.lower()
        line_count = len(rtl_code.split('\n'))

        print(f"   Has module declaration: {has_module}")
        print(f"   Has ports: {has_ports}")
        print(f"   Has logic (always/assign): {has_logic}")
        print(f"   Has clock reference: {has_clk}")
        print(f"   Has reset reference: {has_reset}")
        print(f"   Line count: {line_count}")

        # Verdict
        print(f"\n📋 VERDICT:")
        if line_count <= 5:
            print("   ❌ FAILURE: Generated stub/empty module (≤5 lines)")
            return False
        elif not has_ports:
            print("   ❌ FAILURE: No ports declared")
            return False
        elif not has_logic:
            print("   ❌ FAILURE: No behavioral logic")
            return False
        elif not (has_clk and has_reset):
            print("   ⚠️  WARNING: Missing clock or reset")
            return False
        else:
            print("   ✅ SUCCESS: Generated functional RTL")
            return True

    else:
        print(f"\n❌ A1 Generation Failed!")
        print(f"   Errors: {result.errors}")
        print(f"   Warnings: {result.warnings}")
        return False


if __name__ == '__main__':
    success = main()

    print("\n" + "="*80)
    if success:
        print("  ✅ A1 CAN generate logic for simple specifications")
    else:
        print("  ❌ A1 CANNOT generate logic - Core failure detected")
    print("="*80 + "\n")

    sys.exit(0 if success else 1)
