#!/usr/bin/env python3
"""Detailed A2 FIFO diagnostic"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A2_BoilerplateGenerator


def main():
    """Test A2 FIFO generation with detailed validation info"""

    agent = A2_BoilerplateGenerator({'yosys_binary': 'yosys'})

    result = agent.process({
        'intent_type': 'fifo_sync',
        'module_name': 'tx_fifo',
        'parameters': {'depth': 8, 'data_width': 32}
    })

    print(f"Success: {result.success}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")

    if 'validation' in result.output_data:
        val = result.output_data['validation']
        print(f"\nValidation Details:")
        print(f"  syntax_valid: {val.get('syntax_valid', 'N/A')}")
        print(f"  lint_clean: {val.get('lint_clean', 'N/A')}")
        print(f"  errors: {val.get('errors', [])}")
        print(f"  warnings: {val.get('warnings', [])[:3]}")  # First 3

    if 'rtl_code' in result.output_data:
        print(f"\nRTL Generated: YES ({len(result.output_data['rtl_code'])} chars)")
        print(f"Ports: {len(result.output_data.get('ports', []))}")


if __name__ == '__main__':
    main()
