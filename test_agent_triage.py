#!/usr/bin/env python3
"""
Quick Integration Test - Agent with Triage Router

Tests the full agent flow with triage enabled to ensure:
1. Fast initial responses from 3B
2. Proper escalation for complex queries
3. Intent parsing and action determination still work
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.conversational import LLMInterface, ConversationManager, IntentParser
from utils import load_config, setup_logging
import time


def test_agent_with_triage():
    """Test agent with triage routing enabled"""

    print("=" * 70)
    print("AGENT TRIAGE INTEGRATION TEST")
    print("=" * 70)
    print()

    # Setup
    config = load_config()
    setup_logging(config.get('logging', {}))

    # Ensure triage is enabled
    config['llm']['triage']['enable'] = True

    # Initialize components
    llm = LLMInterface(
        model_name="llama3:8b",
        ollama_host="http://localhost:11434"
    )

    parser = IntentParser(llm)
    manager = ConversationManager(llm, parser)

    print(f"✓ Triage enabled: {manager.triage_enabled}")
    print(f"✓ Components initialized")
    print()

    # Test queries with increasing complexity
    test_queries = [
        {
            'query': "What is synthesis?",
            'expected': 'SIMPLE - should get fast response from 3B',
        },
        {
            'query': "Create a new SoC project for 7nm targeting low power",
            'expected': 'MODERATE - 3B responds, may escalate to 8B',
        },
        {
            'query': "How should I optimize placement for both power and timing?",
            'expected': 'MODERATE/COMPLEX - likely escalates to 8B',
        },
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}")
        print(f"{'─' * 70}")
        print(f"\nQuery: {test['query']}")
        print(f"Expected: {test['expected']}")
        print()

        # Time the response
        start_time = time.time()

        try:
            result = manager.process_message(test['query'])

            elapsed = time.time() - start_time

            print(f"⏱  Response time: {elapsed:.2f}s")
            print()
            print("┌─ RESPONSE")
            print(f"│  {result['response'][:300]}{'...' if len(result['response']) > 300 else ''}")
            print("└────────────────────────────────────────────────")
            print()
            print(f"Intent: {result['intent'].action.value if result.get('intent') else 'N/A'}")
            print(f"Actions: {len(result.get('actions', []))} actions determined")

            # Show triage info if available
            if 'triage_info' in result:
                print(f"Complexity: {result['triage_info']['complexity']}")
                print(f"Escalated: {result['triage_info']['escalated']}")

            print()

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Show conversation stats
    print("\n" + "=" * 70)
    print("CONVERSATION STATISTICS")
    print("=" * 70)
    print(f"\nTotal turns: {len(manager.context.conversation_history) // 2}")
    print(f"Current stage: {manager.context.current_stage}")
    print(f"Project: {manager.context.project_name or 'None'}")
    print(f"Process node: {manager.context.process_node or 'None'}")

    # Show triage router stats
    print("\n" + "=" * 70)
    print("TRIAGE ROUTER STATISTICS")
    print("=" * 70)

    try:
        stats = manager.triage_router.get_routing_stats()
        print(f"\nTotal turns: {stats['total_turns']}")
        print(f"Escalations: {stats['escalations']}")
        print(f"Escalation rate: {stats['escalation_rate']:.1%}")
        print(f"Average complexity: {stats['avg_complexity']:.2f}")
    except Exception as e:
        print(f"Could not get triage stats: {e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("\n🚀 Testing Agent with Triage Integration\n")

    try:
        test_agent_with_triage()
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
