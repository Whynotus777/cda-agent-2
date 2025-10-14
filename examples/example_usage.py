#!/usr/bin/env python3
"""
Example usage of the CDA Agent

This demonstrates the main features and workflows.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import CDAAgent


def example_basic_interaction():
    """Basic conversational interaction"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Interaction")
    print("=" * 60 + "\n")

    agent = CDAAgent()

    # Simple conversation
    responses = [
        agent.chat("Hello, what can you help me with?"),
        agent.chat("I want to create a new chip design for a 7nm process"),
        agent.chat("The design should prioritize low power consumption"),
    ]

    for i, response in enumerate(responses, 1):
        print(f"\nResponse {i}:")
        print(response)


def example_design_flow():
    """Complete design flow example"""
    print("\n" + "=" * 60)
    print("Example 2: Complete Design Flow")
    print("=" * 60 + "\n")

    agent = CDAAgent()

    # Step-by-step design flow
    steps = [
        "Create a new project for a 12nm RISC-V core, targeting high performance",
        "I have a Verilog file at ./designs/example_core.v - can you load it?",
        "Run synthesis with high-speed optimization",
        "Perform placement optimized for minimal wirelength",
        "Run timing analysis and tell me the critical path",
        "What's the estimated power consumption?",
    ]

    for step in steps:
        print(f"\n{'User:':<10} {step}")
        response = agent.chat(step)
        print(f"{'Agent:':<10} {response}")
        print("-" * 60)


def example_rl_optimization():
    """RL-based optimization example"""
    print("\n" + "=" * 60)
    print("Example 3: RL Optimization")
    print("=" * 60 + "\n")

    agent = CDAAgent()

    # Setup design
    agent.chat("Create a new 7nm project")
    agent.chat("Load design from ./designs/example.v")
    agent.chat("Set design goals: performance=1.0, power=0.8, area=0.6")

    # Run RL optimization
    print("\nStarting RL-based optimization...")
    print("This will run the RL agent to iteratively improve the design.")
    print("(In a real scenario, this would take 1-2 hours)")

    # In practice:
    # agent.run_rl_optimization({'goals': {'performance': 1.0, 'power': 0.8, 'area': 0.6}})

    print("\nRL optimization would now:")
    print("1. Try different placement densities")
    print("2. Swap cells with different drive strengths")
    print("3. Buffer critical paths")
    print("4. Learn which actions improve PPA metrics")
    print("5. Converge to an optimal design")


def example_design_queries():
    """Example design queries and analysis"""
    print("\n" + "=" * 60)
    print("Example 4: Design Queries")
    print("=" * 60 + "\n")

    agent = CDAAgent()

    queries = [
        "What's the worst negative slack in my design?",
        "Show me the top 5 power-consuming cells",
        "What's the current area utilization?",
        "Are there any DRC violations?",
        "Which paths are critical for timing?",
        "Can we meet a 2 GHz clock target?",
    ]

    for query in queries:
        print(f"\n{'User:':<10} {query}")
        response = agent.chat(query)
        print(f"{'Agent:':<10} {response}")


def example_interactive_adjustments():
    """Interactive design adjustments"""
    print("\n" + "=" * 60)
    print("Example 5: Interactive Adjustments")
    print("=" * 60 + "\n")

    agent = CDAAgent()

    adjustments = [
        "Move the L1 cache macro to the top-right corner",
        "Increase the placement density by 10%",
        "Lock the position of the memory controller",
        "Buffer all paths with slack < 100ps",
        "Replace non-critical AND gates with lower power versions",
    ]

    for adjustment in adjustments:
        print(f"\n{'User:':<10} {adjustment}")
        response = agent.chat(adjustment)
        print(f"{'Agent:':<10} {response}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print(" " * 20 + "CDA Agent - Example Usage")
    print("=" * 80)

    examples = [
        ("Basic Interaction", example_basic_interaction),
        ("Design Flow", example_design_flow),
        ("RL Optimization", example_rl_optimization),
        ("Design Queries", example_design_queries),
        ("Interactive Adjustments", example_interactive_adjustments),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nPress Enter to run all examples, or enter a number to run one:")
    choice = input("> ").strip()

    if choice == "":
        # Run all
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\nExample '{name}' encountered an error: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        # Run specific example
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"\nExample '{name}' encountered an error: {e}")
    else:
        print("Invalid choice")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # For demonstration, just show what the examples would do
    # without actually running the agent (which requires full setup)
    print("\n" + "=" * 80)
    print(" " * 20 + "CDA Agent - Example Scenarios")
    print("=" * 80 + "\n")

    print("This file demonstrates various usage patterns for the CDA Agent.")
    print("\nTo run these examples with a real agent, ensure:")
    print("  1. Ollama is running with llama3:70b")
    print("  2. All dependencies are installed (run setup.sh)")
    print("  3. You have design files to work with")
    print("\nThen run: python3 examples/example_usage.py")
    print("\n" + "=" * 80 + "\n")
