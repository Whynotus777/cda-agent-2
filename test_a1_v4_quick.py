#!/usr/bin/env python3
"""
Quick A1 V4 Test - Uses LLM Generator Wrapper
Fast test without reloading the model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A1_LLMGenerator


def main():
    """Quick test of A1 V4 with 4096 tokens"""

    print("\n" + "="*80)
    print("  A1 V4 QUICK TEST (4096 tokens)")
    print("="*80 + "\n")

    # SPI Master spec
    spec = {
        "module_name": "SPI_MASTER_QUICK",
        "specification": """SPI Master controller with configurable clock polarity, phase,
        and data width (8/16/32-bit). Supports full-duplex operation, programmable clock
        divider (divide by 2 to 256), and FIFO buffers (8-deep TX/RX). Includes busy
        status flag and interrupt generation on transfer complete.""",
        "parameters": {
            "data_width": 32,
            "fifo_depth": 8,
            "max_clock_div": 256
        }
    }

    print("📝 Test Specification:")
    print(f"   Module: {spec['module_name']}")
    print(f"   Data Width: {spec['parameters']['data_width']}-bit")
    print(f"   FIFO Depth: {spec['parameters']['fifo_depth']}\n")

    # Initialize A1 LLM Generator
    print("🔧 Initializing A1 LLM Generator (this will take ~15 min)...")

    config = {
        'model_path': 'models/mixtral_rtl/run_pure_20251030_121523/final_model',
        'max_new_tokens': 4096,  # 🔥 Increased from 2048!
        'temperature': 0.7,
        'top_p': 0.95
    }

    generator = A1_LLMGenerator(config)

    # Generate RTL
    print("\n⚙️  Generating RTL with A1 V4 (4096 tokens)...")
    result = generator.process(spec)

    print("\n" + "="*80)
    print("  GENERATION RESULTS")
    print("="*80 + "\n")

    if result.success:
        rtl_code = result.output_data['rtl_code']
        lines = rtl_code.split('\n')
        line_count = len(lines)

        print(f"📊 Basic Metrics:")
        print(f"   Lines: {line_count}")
        print(f"   Characters: {len(rtl_code)}")
        print(f"   Generation Time: {result.output_data.get('generation_time_s', 0):.2f}s")
        print(f"   Tokens Generated: {result.output_data.get('tokens_generated', 0)}")

        # Check for completion
        has_endmodule = 'endmodule' in rtl_code.lower()
        validation = result.output_data.get('validation', {})

        print(f"\n🔍 Validation:")
        print(f"   Has endmodule: {has_endmodule} {'✅' if has_endmodule else '❌'}")
        print(f"   Yosys Valid: {validation.get('syntax_valid', False)}")

        if not has_endmodule:
            print(f"   ⚠️  Code was truncated!")
        else:
            print(f"   ✅ Code is complete!")

        if validation.get('errors'):
            print(f"\n⚠️  Syntax Errors: {len(validation['errors'])}")
            for error in validation['errors'][:5]:
                print(f"      - {error}")

        # Display code
        print(f"\n📄 Generated RTL Code:")
        print("="*80)
        for i, line in enumerate(lines[:50], 1):
            print(f"  {i:3}: {line}")
        if len(lines) > 50:
            print(f"\n  ... and {len(lines) - 50} more lines")

            # Show last 10 lines
            print(f"\n  Last 10 lines:")
            for i, line in enumerate(lines[-10:], len(lines) - 9):
                print(f"  {i:3}: {line}")
        print("="*80)

        # Write to file
        output_file = Path(f'/tmp/{spec["module_name"]}.v')
        output_file.write_text(rtl_code)
        print(f"\n✅ RTL written to: {output_file}")

        # Verdict
        print(f"\n📋 VERDICT:")
        if has_endmodule and line_count >= 150:
            print(f"   ✅ SUCCESS: Complete RTL generated ({line_count} lines)")
            print(f"   🎉 4096 token limit resolved truncation!")
            return True
        elif has_endmodule:
            print(f"   ✅ COMPLETE: RTL has endmodule ({line_count} lines)")
            print(f"   💡 May need more complex logic")
            return True
        else:
            print(f"   ❌ TRUNCATED: Missing endmodule at {line_count} lines")
            print(f"   💡 May need even higher token limit")
            return False

    else:
        print(f"❌ Generation Failed!")
        print(f"   Errors: {result.errors}")
        return False


if __name__ == '__main__':
    success = main()

    print("\n" + "="*80)
    if success:
        print("  ✅ A1 V4 QUICK TEST PASSED")
        print("  🎉 Token limit increase resolved truncation")
    else:
        print("  ⚠️  A1 V4 STILL TRUNCATING")
        print("  💡 Need further investigation")
    print("="*80 + "\n")

    sys.exit(0 if success else 1)
