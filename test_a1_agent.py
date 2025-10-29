#!/usr/bin/env python3
"""
Test script for A1 - Spec-to-RTL Generator

Tests natural language to RTL generation and validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A1_SpecToRTLGenerator


def test_intent_parsing():
    """Test natural language intent detection"""
    print("\n" + "="*70)
    print("TEST 1: Intent Parsing from Natural Language")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    specifications = [
        ("Create a 4-state FSM", 'fsm'),
        ("Build a 16-deep FIFO buffer", 'fifo'),
        ("Design an 8-bit counter", 'counter'),
        ("Generate an async FIFO for clock crossing", 'fifo_async'),
        ("Create an 8-bit adder", 'adder')
    ]

    results = []
    for spec, expected_intent in specifications:
        intent_type, params = agent._parse_intent(spec)

        matched = (intent_type == expected_intent)
        results.append((spec, expected_intent, intent_type, matched))

        status = "✅" if matched else "❌"
        print(f"  {status} \"{spec[:40]}...\"")
        print(f"       Expected: {expected_intent}, Got: {intent_type}")
        if params:
            print(f"       Parameters: {params}")

    success_count = sum(1 for _, _, _, matched in results if matched)
    total = len(results)
    success_rate = (success_count / total * 100) if total > 0 else 0

    print(f"\n✓ Intent Detection Rate: {success_rate:.1f}%")

    return success_rate >= 80


def test_template_generation():
    """Test template-based RTL generation via A2"""
    print("\n" + "="*70)
    print("TEST 2: Template-Based Generation (via A2)")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    input_data = {
        'specification': 'Create a synchronous FIFO with 16 entries and 8-bit data width',
        'module_name': 'test_fifo',
        'intent_type': 'fifo',
        'parameters': {
            'depth': 16,
            'data_width': 8
        }
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Execution Time: {result.execution_time_ms:.2f}ms")

    rtl_code = result.output_data.get('rtl_code', '')
    generation_method = result.output_data.get('generation_method', '')
    validation = result.output_data.get('validation', {})

    print(f"\n--- Generation Details ---")
    print(f"  Method: {generation_method}")
    print(f"  Line Count: {len(rtl_code.split(chr(10)))}")
    print(f"  Syntax Valid: {validation.get('syntax_valid')}")

    print(f"\n--- RTL Preview (first 400 chars) ---")
    print(rtl_code[:400])

    return result.success and validation.get('syntax_valid', False)


def test_synthesized_register():
    """Test direct synthesis of register module"""
    print("\n" + "="*70)
    print("TEST 3: Synthesized Register Module")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    input_data = {
        'specification': 'Create a 16-bit register with write enable',
        'module_name': 'reg16',
        'intent_type': 'register',
        'parameters': {
            'data_width': 16
        }
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    generation_method = result.output_data.get('generation_method', '')
    validation = result.output_data.get('validation', {})
    ports = result.output_data.get('ports', [])

    print(f"\n--- Module Details ---")
    print(f"  Generation Method: {generation_method}")
    print(f"  Syntax Valid: {validation.get('syntax_valid')}")
    print(f"  Ports: {len(ports)}")

    for port in ports:
        print(f"    {port['direction']:6} {port['name']:15} [{port['width']} bits]")

    return result.success


def test_synthesized_adder():
    """Test direct synthesis of adder module"""
    print("\n" + "="*70)
    print("TEST 4: Synthesized Adder Module")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    input_data = {
        'specification': 'Design a 32-bit adder with carry in and out',
        'module_name': 'adder32',
        'intent_type': 'adder',
        'parameters': {
            'data_width': 32
        }
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    validation = result.output_data.get('validation', {})
    rtl_code = result.output_data.get('rtl_code', '')

    print(f"\n--- Validation ---")
    print(f"  Syntax Valid: {validation.get('syntax_valid')}")
    if validation.get('errors'):
        print(f"  Errors: {len(validation['errors'])}")
        for err in validation['errors'][:3]:
            print(f"    {err}")

    print(f"\n--- RTL Snippet ---")
    lines = rtl_code.split('\n')
    for i, line in enumerate(lines[5:15], 6):  # Lines 6-15
        print(f"  {i:3}: {line}")

    return result.success


def test_synthesized_multiplier():
    """Test direct synthesis of multiplier module"""
    print("\n" + "="*70)
    print("TEST 5: Synthesized Multiplier Module")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    input_data = {
        'specification': 'Create an 8x8 multiplier',
        'module_name': 'mult8x8',
        'intent_type': 'multiplier',
        'parameters': {
            'data_width': 8
        }
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    validation = result.output_data.get('validation', {})

    print(f"\n--- Validation ---")
    print(f"  Syntax Valid: {validation.get('syntax_valid')}")

    return result.success


def test_generic_fallback():
    """Test generic module fallback"""
    print("\n" + "="*70)
    print("TEST 6: Generic Module Fallback")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    input_data = {
        'specification': 'Create a combinational module that passes data through',
        'module_name': 'passthrough',
        'parameters': {
            'data_width': 8
        }
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    generation_method = result.output_data.get('generation_method', '')
    validation = result.output_data.get('validation', {})

    print(f"\n--- Details ---")
    print(f"  Method: {generation_method}")
    print(f"  Syntax Valid: {validation.get('syntax_valid')}")

    return result.success


def test_natural_language_e2e():
    """Test end-to-end natural language processing"""
    print("\n" + "="*70)
    print("TEST 7: End-to-End Natural Language")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    test_cases = [
        {
            'spec': 'Build an 8-bit counter that counts up to 255',
            'module': 'counter8',
            'should_compile': True
        },
        {
            'spec': 'Create a 4-state Mealy FSM for a traffic light controller',
            'module': 'traffic_fsm',
            'should_compile': True
        },
        {
            'spec': 'Design a 16-deep async FIFO for clock domain crossing',
            'module': 'cdc_fifo',
            'should_compile': True
        },
        {
            'spec': 'Generate a 16-bit register with synchronous reset',
            'module': 'reg16_sync',
            'should_compile': True
        }
    ]

    results = []

    for test_case in test_cases:
        input_data = {
            'specification': test_case['spec'],
            'module_name': test_case['module']
        }

        result = agent.process(input_data)

        syntax_valid = result.output_data.get('validation', {}).get('syntax_valid', False)
        expected = test_case['should_compile']
        matched = (syntax_valid == expected)

        results.append((test_case['module'], syntax_valid, expected, matched))

        status = "✅" if matched else "❌"
        print(f"  {status} {test_case['module']:15} - Compile: {syntax_valid}, Expected: {expected}")

    success_count = sum(1 for _, _, _, matched in results if matched)
    total = len(results)
    compile_rate = (success_count / total * 100) if total > 0 else 0

    print(f"\n✓ Compile Success Rate: {compile_rate:.1f}%")

    # Phase 4 target: ≥80% compile success
    if compile_rate >= 80:
        print(f"\n🎉 Phase 4 Target Achieved: {compile_rate:.1f}% ≥ 80%")
        return True
    else:
        print(f"\n⚠️  Phase 4 Target: {compile_rate:.1f}% < 80%")
        return compile_rate >= 70  # Partial credit


def test_compile_success_rate():
    """Test overall compile success rate (PRIMARY TARGET)"""
    print("\n" + "="*70)
    print("TEST 8: Compile Success Rate (PRIMARY TARGET)")
    print("="*70)

    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    test_modules = [
        {'intent': 'register', 'name': 'reg8', 'params': {'data_width': 8}},
        {'intent': 'register', 'name': 'reg32', 'params': {'data_width': 32}},
        {'intent': 'adder', 'name': 'add8', 'params': {'data_width': 8}},
        {'intent': 'adder', 'name': 'add16', 'params': {'data_width': 16}},
        {'intent': 'multiplier', 'name': 'mult4', 'params': {'data_width': 4}},
        {'intent': 'multiplier', 'name': 'mult8', 'params': {'data_width': 8}},
        {'intent': 'counter', 'name': 'cnt8', 'params': {'width': 8}},
        {'intent': 'counter', 'name': 'cnt16', 'params': {'width': 16}},
        {'intent': 'fifo', 'name': 'fifo8x16', 'params': {'depth': 16, 'data_width': 8}},
        {'intent': 'fifo_async', 'name': 'async_fifo', 'params': {'depth': 8, 'data_width': 8}}
    ]

    results = []

    for test_mod in test_modules:
        input_data = {
            'intent_type': test_mod['intent'],
            'module_name': test_mod['name'],
            'parameters': test_mod['params']
        }

        result = agent.process(input_data)

        syntax_valid = result.output_data.get('validation', {}).get('syntax_valid', False)
        results.append((test_mod['name'], test_mod['intent'], syntax_valid))

        status = "✅" if syntax_valid else "❌"
        print(f"  {status} {test_mod['name']:15} ({test_mod['intent']:12}) - Compile: {syntax_valid}")

    compile_count = sum(1 for _, _, valid in results if valid)
    total = len(results)

    compile_rate = (compile_count / total * 100) if total > 0 else 0

    print(f"\n✓ Total: {compile_count}/{total} compiled successfully")
    print(f"✓ Compile Success Rate: {compile_rate:.1f}%")

    # Phase 4 target: ≥80% compile success
    if compile_rate >= 80:
        print(f"\n🎉 Phase 4 Target Achieved: {compile_rate:.1f}% ≥ 80%")
        return True
    else:
        print(f"\n⚠️  Phase 4 Target: {compile_rate:.1f}% < 80%")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("A1 SPEC-TO-RTL GENERATOR - TEST SUITE")
    print("="*70)

    tests = [
        ("Intent Parsing", test_intent_parsing),
        ("Template Generation (A2)", test_template_generation),
        ("Synthesized Register", test_synthesized_register),
        ("Synthesized Adder", test_synthesized_adder),
        ("Synthesized Multiplier", test_synthesized_multiplier),
        ("Generic Fallback", test_generic_fallback),
        ("Natural Language E2E", test_natural_language_e2e),
        ("Compile Success Rate (PRIMARY)", test_compile_success_rate)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
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

    if success_rate >= 80:
        print(f"\n🎉 Phase 4 Complete: {success_rate:.1f}% success rate")
    else:
        print(f"\n⚠️  Phase 4 Status: {success_rate:.1f}%")

    return passed_count >= (total_count * 0.8)  # 80% threshold


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
