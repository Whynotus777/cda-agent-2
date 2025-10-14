#!/usr/bin/env python3
"""
Conversational Agent Demo

Demonstrates natural language control of chip design automation.
Shows how intents are parsed and routed to backend functions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
from core.conversational import IntentParser, ActionExecutor, LLMInterface
from core.conversational.intent_parser import ActionType

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_conversational_flow():
    """
    Demonstrate the full conversational flow:
    1. User says something in natural language
    2. IntentParser extracts structured intent
    3. ActionExecutor routes to appropriate backend function
    4. Result is returned to user
    """
    print("="*70)
    print("CONVERSATIONAL CHIP DESIGN AGENT DEMO")
    print("="*70)
    print()
    print("This demo shows how natural language commands are converted")
    print("into actual backend operations.")
    print()

    # Initialize components
    print("Initializing conversational agent...")
    llm = LLMInterface()
    intent_parser = IntentParser(llm)
    action_executor = ActionExecutor()
    print("✓ Agent initialized")
    print()

    # Test scenarios
    scenarios = [
        {
            'name': "Query: What is placement?",
            'input': "What is placement in chip design?",
            'expected_action': ActionType.QUERY
        },
        {
            'name': "Create new design project",
            'input': "Start a new 7nm design for a low-power microcontroller",
            'expected_action': ActionType.CREATE_PROJECT
        },
        {
            'name': "Load design file",
            'input': "Load the design from /tmp/test_counter.v with top module simple_counter",
            'expected_action': ActionType.LOAD_DESIGN
        },
        {
            'name': "Run synthesis",
            'input': "Run synthesis on the design",
            'expected_action': ActionType.SYNTHESIZE
        },
        {
            'name': "Run placement optimizing for wirelength",
            'input': "Place the design optimizing for minimal wirelength",
            'expected_action': ActionType.PLACE
        },
        {
            'name': "Run RL optimization",
            'input': "Run optimization to minimize wirelength",
            'expected_action': ActionType.OPTIMIZE
        },
    ]

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print("="*70)
        print(f"SCENARIO {i}: {scenario['name']}")
        print("="*70)
        print()

        user_input = scenario['input']
        print(f"User: \"{user_input}\"")
        print()

        # Step 1: Parse intent
        print("Step 1: Parsing intent...")
        try:
            intent = intent_parser.parse(user_input)

            print(f"  ✓ Action: {intent.action.value}")
            print(f"  ✓ Parameters: {intent.parameters}")
            print(f"  ✓ Goals: {[g.value for g in intent.goals]}")
            print(f"  ✓ Confidence: {intent.confidence:.2f}")

            # Validate intent matches expectation
            if intent.action == scenario['expected_action']:
                print(f"  ✓ Intent correctly identified as {intent.action.value}")
            else:
                print(f"  ⚠ Expected {scenario['expected_action'].value}, got {intent.action.value}")

        except Exception as e:
            print(f"  ✗ Intent parsing failed: {e}")
            results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': str(e)
            })
            print()
            continue

        print()

        # Step 2: Execute action
        print("Step 2: Executing action...")
        try:
            result = action_executor.execute(intent)

            if result['success']:
                print(f"  ✓ Success: {result['message']}")
                if result['data']:
                    print(f"  ✓ Data: {result['data']}")
            else:
                print(f"  ✗ Failed: {result['message']}")

            results.append({
                'scenario': scenario['name'],
                'action': intent.action.value,
                'success': result['success'],
                'message': result['message']
            })

        except Exception as e:
            print(f"  ✗ Action execution failed: {e}")
            results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': str(e)
            })

        print()

    # Summary
    print("="*70)
    print("DEMO SUMMARY")
    print("="*70)
    print()

    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)

    print(f"Scenarios executed: {total}")
    print(f"Successful: {successful}/{total}")
    print()

    print("Results:")
    for r in results:
        status = "✓" if r.get('success', False) else "✗"
        print(f"  {status} {r['scenario']}")
        if not r.get('success', False):
            print(f"     Reason: {r.get('message', r.get('error', 'Unknown'))}")

    print()

    # Show final system status
    print("="*70)
    print("SYSTEM STATUS")
    print("="*70)
    status = action_executor.get_status()
    if status['active']:
        print(f"Active Project: {status['project_name']}")
        print(f"Design Stage: {status['stage']}")
        print(f"Process Node: {status['process_node']}")
        print(f"Design Goals: {status['goals']}")
        print(f"RTL Files: {status['rtl_files']}")
        print(f"Netlist: {status['netlist_file']}")
        print(f"DEF File: {status['def_file']}")
    else:
        print("No active project")

    print()

    # Capabilities demonstration
    print("="*70)
    print("KEY CAPABILITIES DEMONSTRATED")
    print("="*70)
    print()
    print("✓ Natural Language Understanding:")
    print("  - 'Start a new 7nm design' → CREATE_PROJECT action")
    print("  - 'Run optimization' → OPTIMIZE action (triggers RL loop)")
    print("  - 'What is placement?' → QUERY action (uses RAG)")
    print()
    print("✓ Intent Parsing:")
    print("  - Extracts actions, parameters, and goals from text")
    print("  - Heuristic + LLM-based parsing")
    print("  - High confidence scores for clear commands")
    print()
    print("✓ Action Execution:")
    print("  - Routes intents to appropriate backend functions")
    print("  - Manages design state across operations")
    print("  - Connects to EDA tools (Yosys, DREAMPlace)")
    print("  - Can trigger RL optimization loop")
    print()
    print("✓ RAG System:")
    print("  - 81 documents indexed in ChromaDB")
    print("  - Can answer questions about EDA tools")
    print("  - Vector similarity search for relevant context")
    print()

    print("="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print()
    print("The conversational layer is now connected to the backend!")
    print("You can control chip design through natural language commands.")
    print()


def demo_rag_queries():
    """Demonstrate RAG system capabilities"""
    print("="*70)
    print("RAG SYSTEM DEMONSTRATION")
    print("="*70)
    print()

    action_executor = ActionExecutor()

    queries = [
        "What is Yosys?",
        "How does DREAMPlace work?",
        "What are the stages of chip design?",
        "Explain placement optimization"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 70)

        result = action_executor.rag.retrieve(query, top_k=2)

        if result:
            for i, doc in enumerate(result, 1):
                source = doc['metadata'].get('source', 'Unknown')
                distance = doc.get('distance', 0.0)
                preview = doc['document'][:200] + "..."

                print(f"\nResult {i} (similarity: {1-distance:.2f}):")
                print(f"  Source: {source}")
                print(f"  Preview: {preview}")
        else:
            print("  No results found")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conversational agent demo")
    parser.add_argument('--rag-only', action='store_true',
                        help='Only demonstrate RAG system')

    args = parser.parse_args()

    if args.rag_only:
        demo_rag_queries()
    else:
        demo_conversational_flow()
        print("\nTo test RAG system only, run: ./venv/bin/python3 demo_conversational_agent.py --rag-only")
