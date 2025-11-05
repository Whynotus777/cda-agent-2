#!/usr/bin/env python3
"""Quick diagnostic: Test A2 FIFO generation directly"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A2_BoilerplateGenerator


def main():
    """Test A2 FIFO generation"""

    print("\n" + "="*80)
    print("  A2 FIFO DIAGNOSTIC")
    print("="*80 + "\n")

    agent = A2_BoilerplateGenerator({'yosys_binary': 'yosys'})

    # Test FIFO sync generation (same params as SPI Master Planner)
    fifo_spec = {
        'intent_type': 'fifo_sync',
        'module_name': 'tx_fifo',
        'parameters': {
            'depth': 8,
            'data_width': 32
        }
    }

    print("📝 Testing FIFO Sync Generation:")
    print(f"   Module: {fifo_spec['module_name']}")
    print(f"   Type: {fifo_spec['intent_type']}")
    print(f"   Depth: {fifo_spec['parameters']['depth']}")
    print(f"   Width: {fifo_spec['parameters']['data_width']}\n")

    result = agent.process(fifo_spec)

    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")

    if result.success:
        rtl_code = result.output_data['rtl_code']
        print(f"\n✅ Generated {len(rtl_code.split(chr(10)))} lines")
        print(f"\nFirst 30 lines:")
        print("="*80)
        for i, line in enumerate(rtl_code.split('\n')[:30], 1):
            print(f"  {i:3}: {line}")
    else:
        print(f"\n❌ FIFO generation failed!")
        print(f"Full output_data keys: {list(result.output_data.keys())}")


if __name__ == '__main__':
    main()
