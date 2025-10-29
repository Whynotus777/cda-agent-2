#!/usr/bin/env python3
"""
Test script for A6 - EDA Command Copilot

Tests script generation and validation capabilities.
"""

import sys
import json
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A6_EDACommandCopilot


def test_yosys_script_generation():
    """Test Yosys synthesis script generation"""
    print("\n" + "="*70)
    print("TEST 1: Yosys Synthesis Script Generation")
    print("="*70)

    config = {
        'yosys_binary': 'yosys',
        'opensta_binary': 'sta'
    }

    agent = A6_EDACommandCopilot(config=config)

    # Create test input
    input_data = {
        'tool': 'yosys',
        'command_type': 'synthesis',
        'input_files': ['counter.v', 'alu.v'],
        'output_files': ['output.v'],
        'parameters': {
            'top_module': 'counter',
            'optimization_goal': 'speed',
            'tech_library': '/path/to/tech.lib'
        },
        'dry_run': False
    }

    # Process
    result = agent.process(input_data)

    # Display results
    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Execution Time: {result.execution_time_ms:.2f}ms")
    print(f"\n--- Generated Script ---")
    print(result.output_data.get('script_content', 'N/A'))
    print(f"\n--- Errors ({len(result.errors)}) ---")
    for error in result.errors:
        print(f"  ❌ {error}")
    print(f"\n--- Warnings ({len(result.warnings)}) ---")
    for warning in result.warnings:
        print(f"  ⚠️  {warning}")

    return result.success


def test_opensta_script_generation():
    """Test OpenSTA timing analysis script generation"""
    print("\n" + "="*70)
    print("TEST 2: OpenSTA Timing Analysis Script Generation")
    print("="*70)

    agent = A6_EDACommandCopilot()

    input_data = {
        'tool': 'opensta',
        'command_type': 'timing_analysis',
        'parameters': {
            'netlist_file': 'design.v',
            'sdc_file': 'constraints.sdc',
            'lib_files': ['tech_fast.lib', 'tech_slow.lib'],
            'spef_file': 'parasitics.spef',
            'top_module': 'top'
        },
        'dry_run': False
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Execution Time: {result.execution_time_ms:.2f}ms")
    print(f"\n--- Generated Script ---")
    print(result.output_data.get('script_content', 'N/A'))
    print(f"\n--- Errors ({len(result.errors)}) ---")
    for error in result.errors:
        print(f"  ❌ {error}")
    print(f"\n--- Warnings ({len(result.warnings)}) ---")
    for warning in result.warnings:
        print(f"  ⚠️  {warning}")

    return result.success


def test_verilator_lint_generation():
    """Test Verilator lint command generation"""
    print("\n" + "="*70)
    print("TEST 3: Verilator Lint Command Generation")
    print("="*70)

    agent = A6_EDACommandCopilot()

    input_data = {
        'tool': 'verilator',
        'command_type': 'lint',
        'input_files': ['design.v', 'sub_module.v'],
        'parameters': {
            'top_module': 'my_design'
        }
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Execution Time: {result.execution_time_ms:.2f}ms")
    print(f"\n--- Generated Script ---")
    print(result.output_data.get('script_content', 'N/A'))
    print(f"\n--- Errors ({len(result.errors)}) ---")
    for error in result.errors:
        print(f"  ❌ {error}")

    return result.success


def test_invalid_input():
    """Test error handling with invalid input"""
    print("\n" + "="*70)
    print("TEST 4: Invalid Input Handling")
    print("="*70)

    agent = A6_EDACommandCopilot()

    # Missing required fields
    input_data = {
        'input_files': ['test.v']
        # Missing 'tool' and 'command_type'
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success} (should be False)")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"\n--- Errors ({len(result.errors)}) ---")
    for error in result.errors:
        print(f"  ❌ {error}")

    return not result.success  # Test passes if result fails


def test_schema_access():
    """Test schema retrieval"""
    print("\n" + "="*70)
    print("TEST 5: Schema Access")
    print("="*70)

    agent = A6_EDACommandCopilot()
    schema = agent.get_schema()

    print(f"\n✓ Schema loaded: {bool(schema)}")
    if schema:
        print(f"✓ Schema title: {schema.get('title', 'N/A')}")
        print(f"✓ Required fields: {schema.get('required', [])}")

    return bool(schema)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("A6 EDA COMMAND COPILOT - TEST SUITE")
    print("="*70)

    tests = [
        ("Yosys Script Generation", test_yosys_script_generation),
        ("OpenSTA Script Generation", test_opensta_script_generation),
        ("Verilator Lint Generation", test_verilator_lint_generation),
        ("Invalid Input Handling", test_invalid_input),
        ("Schema Access", test_schema_access)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    # Phase 1 target: ≥90% validity
    if success_rate >= 90:
        print(f"\n🎉 Phase 1 Target Achieved: {success_rate:.1f}% ≥ 90%")
    else:
        print(f"\n⚠️  Phase 1 Target Not Met: {success_rate:.1f}% < 90%")

    return passed_count == total_count


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
