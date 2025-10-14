#!/usr/bin/env python3
"""
Test Triage Router

Demonstrates how the streaming triage system works with different query types.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversational import LLMInterface, TriageRouter
from utils import load_config, setup_logging


def test_triage_system():
    """Test the triage router with various query types"""

    # Setup
    config = load_config()
    setup_logging(config.get('logging', {}))

    llm = LLMInterface(
        model_name="llama3:8b",
        ollama_host="http://localhost:11434"
    )

    triage = TriageRouter(llm)

    print("=" * 70)
    print("TRIAGE ROUTER TEST")
    print("=" * 70)
    print()

    # Test queries with increasing complexity
    test_queries = [
        {
            'query': "What is synthesis?",
            'expected': 'SIMPLE',
            'description': 'Simple definition'
        },
        {
            'query': "How do I optimize my design for minimum area using Yosys with a 7nm technology library?",
            'expected': 'MODERATE',
            'description': 'Technical how-to'
        },
        {
            'query': "I need to choose between TSMC 7nm and Samsung 5nm for my robotics SoC with custom NPU. Evaluate power, performance, cost, and integration complexity.",
            'expected': 'COMPLEX',
            'description': 'Architecture decision'
        },
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}: {test['description']}")
        print(f"{'─' * 70}")
        print(f"\nQuery: {test['query']}")
        print(f"Expected: {test['expected']}")
        print()

        # Route the query
        result = triage.route_streaming(test['query'])

        # Show results
        print("┌─ IMMEDIATE RESPONSE (3B, ~1-2 sec)")
        print(f"│  {result['immediate_response']}")
        print("└────────────────────────────────────────────────")

        print()
        print(f"Complexity: {result['escalation_level'].name}")
        print(f"Escalation: {result['needs_escalation']}")
        print(f"Reasoning: {result['routing_reasoning']}")
        print(f"Conv Depth: {result['conversation_depth']}")

        if result['refined_response']:
            print()
            print("┌─ REFINED RESPONSE (8B/70B, ~3-20 sec)")
            print(f"│  {result['refined_response'][:200]}...")
            print("└────────────────────────────────────────────────")

        print()

    # Show routing statistics
    print("\n" + "=" * 70)
    print("ROUTING STATISTICS")
    print("=" * 70)

    stats = triage.get_routing_stats()
    print(f"\nTotal turns: {stats['total_turns']}")
    print(f"Escalations: {stats['escalations']}")
    print(f"Escalation rate: {stats['escalation_rate']:.1%}")
    print(f"Average complexity: {stats['avg_complexity']:.2f}")
    print(f"\nConversation history:")
    for turn in stats['conversation_history']:
        print(f"  Turn {turn['turn']}: {turn['complexity']} "
              f"({'escalated' if turn['escalated'] else 'handled by 3B'})")

    # Test proactive escalation
    print("\n" + "=" * 70)
    print("PROACTIVE ESCALATION TEST")
    print("=" * 70)

    # Simulate 8 more turns of moderate complexity
    print("\nSimulating sustained technical conversation...")
    for i in range(8):
        result = triage.route_streaming(
            f"Technical question {i+4} about timing optimization",
            context={'simulated': True}
        )
        print(f"Turn {triage.conversation_depth}: {result['escalation_level'].name}")

    should_escalate = triage.should_proactively_escalate()
    print(f"\nShould proactively escalate? {should_escalate}")
    print(f"Reason: Conversation depth = {triage.conversation_depth}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("\n🚀 Testing Triage Router System\n")

    try:
        test_triage_system()
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
