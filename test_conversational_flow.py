#!/usr/bin/env python3
"""
Test Conversational Flow

Tests the conversational layer without requiring Ollama LLM.
Uses direct intent creation to test action execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
from core.conversational import ActionExecutor
from core.conversational.intent_parser import ParsedIntent, ActionType, DesignGoal

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_conversational_flow():
    """Test the conversational flow with synthetic intents"""
    print("="*70)
    print("CONVERSATIONAL FLOW TEST (WITHOUT LLM)")
    print("="*70)
    print()

    # Initialize action executor
    action_executor = ActionExecutor()
    print("✓ ActionExecutor initialized")
    print()

    test_cases = []

    # Test 1: Query (RAG)
    print("="*70)
    print("TEST 1: RAG Query")
    print("="*70)
    query_intent = ParsedIntent(
        action=ActionType.QUERY,
        parameters={},
        goals=[DesignGoal.BALANCED],
        confidence=0.95,
        raw_text="What is Yosys?"
    )

    print(f"Intent: {query_intent.action.value}")
    print(f"Query: {query_intent.raw_text}")

    result = action_executor.execute(query_intent)

    if result['success']:
        print(f"✓ Success: {result['message']}")
        if result['data'] and result['data'].get('results'):
            print(f"✓ Retrieved {len(result['data']['results'])} documents")
            for i, doc in enumerate(result['data']['results'][:2], 1):
                source = doc['metadata'].get('source', 'Unknown')
                preview = doc['document'][:100] + "..."
                print(f"\n  Result {i}:")
                print(f"    Source: {source}")
                print(f"    Preview: {preview}")
    else:
        print(f"✗ Failed: {result['message']}")

    test_cases.append(('RAG Query', result['success']))
    print()

    # Test 2: Create Project
    print("="*70)
    print("TEST 2: Create New Design Project")
    print("="*70)
    create_intent = ParsedIntent(
        action=ActionType.CREATE_PROJECT,
        parameters={
            'process_node': '7nm',
            'project_name': 'test_chip',
            'design_type': 'microcontroller'
        },
        goals=[DesignGoal.MINIMIZE_POWER],
        confidence=0.90,
        raw_text="Start a new 7nm design for a low-power microcontroller"
    )

    print(f"Intent: {create_intent.action.value}")
    print(f"Parameters: {create_intent.parameters}")

    result = action_executor.execute(create_intent)

    if result['success']:
        print(f"✓ Success: {result['message']}")
        print(f"✓ Project: {result['data']['project_name']}")
        print(f"✓ Process: {result['data']['process_node']}")
        print(f"✓ Goals: {result['data']['goals']}")
    else:
        print(f"✗ Failed: {result['message']}")

    test_cases.append(('Create Project', result['success']))
    print()

    # Test 3: Load Design
    print("="*70)
    print("TEST 3: Load Design File")
    print("="*70)
    load_intent = ParsedIntent(
        action=ActionType.LOAD_DESIGN,
        parameters={
            'file_path': '/tmp/test_counter.v',
            'top_module': 'simple_counter'
        },
        goals=[DesignGoal.BALANCED],
        confidence=0.95,
        raw_text="Load the design from /tmp/test_counter.v"
    )

    print(f"Intent: {load_intent.action.value}")
    print(f"File: {load_intent.parameters['file_path']}")

    result = action_executor.execute(load_intent)

    if result['success']:
        print(f"✓ Success: {result['message']}")
        print(f"✓ Top module: {result['data']['top_module']}")
    else:
        print(f"✗ Failed: {result['message']}")

    test_cases.append(('Load Design', result['success']))
    print()

    # Test 4: Synthesize
    print("="*70)
    print("TEST 4: Run Synthesis")
    print("="*70)
    synth_intent = ParsedIntent(
        action=ActionType.SYNTHESIZE,
        parameters={'goal': 'balanced'},
        goals=[DesignGoal.BALANCED],
        confidence=0.95,
        raw_text="Run synthesis on the design"
    )

    print(f"Intent: {synth_intent.action.value}")

    result = action_executor.execute(synth_intent)

    if result['success']:
        print(f"✓ Success: {result['message']}")
        if result['data']:
            print(f"✓ Cells: {result['data'].get('cell_count', 'N/A')}")
            print(f"✓ Flip-flops: {result['data'].get('flip_flops', 'N/A')}")
    else:
        print(f"✗ Failed: {result['message']}")

    test_cases.append(('Synthesis', result['success']))
    print()

    # Test 5: Get Status
    print("="*70)
    print("TEST 5: System Status")
    print("="*70)
    status = action_executor.get_status()

    if status['active']:
        print(f"✓ Active Project: {status['project_name']}")
        print(f"✓ Design Stage: {status['stage']}")
        print(f"✓ Process Node: {status['process_node']}")
        print(f"✓ RTL Files: {status['rtl_files']}")
        print(f"✓ Netlist: {status['netlist_file']}")
    else:
        print("✗ No active project")

    test_cases.append(('Get Status', status['active']))
    print()

    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print()

    passed = sum(1 for _, success in test_cases if success)
    total = len(test_cases)

    print(f"Tests Passed: {passed}/{total}")
    print()

    for name, success in test_cases:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print()

    if passed == total:
        print("="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print()
        print("Key Achievements:")
        print("✓ RAG system can retrieve documentation")
        print("✓ Can create new design projects")
        print("✓ Can load Verilog designs")
        print("✓ Can run synthesis with Yosys")
        print("✓ Design state is tracked correctly")
        print()
        print("The conversational layer is fully connected to the backend!")
        print()
    else:
        print(f"\n{total - passed} test(s) failed. See details above.")

    return passed == total


if __name__ == "__main__":
    success = test_conversational_flow()
    sys.exit(0 if success else 1)
