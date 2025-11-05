#!/usr/bin/env python3
"""
A1 V2 Test - SPI Master with Planner & Composer
Test the upgraded A1 with hierarchical design generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A1_SpecToRTLGenerator


def main():
    """Test A1 V2 with SPI Master - Planner & Composer Architecture"""

    print("\n" + "="*80)
    print("  A1 V2 TEST: SPI Master (Planner & Composer)")
    print("="*80 + "\n")

    # SPI Master spec - Same one that FAILED with original A1 (generated 3 lines)
    spi_spec = {
        "module_name": "SPI_MASTER_V2",
        "specification": """SPI Master controller with configurable clock polarity, phase,
        and data width (8/16/32-bit). Supports full-duplex operation, programmable clock
        divider (divide by 2 to 256), and FIFO buffers (8-deep TX/RX). Includes busy
        status flag and interrupt generation on transfer complete.""",
        "intent_type": "spi_master",  # Explicitly set intent for SPI Master
        "parameters": {
            "data_width": 32,
            "fifo_depth": 8,
            "max_clock_div": 256
        },
        "constraints": {
            "clock_period_ns": 10.0,
            "target_frequency_mhz": 100.0
        }
    }

    print("📝 Test Specification:")
    print(f"   Module: {spi_spec['module_name']}")
    print(f"   Intent: {spi_spec['intent_type']}")
    print(f"   Data Width: {spi_spec['parameters']['data_width']}-bit")
    print(f"   FIFO Depth: {spi_spec['parameters']['fifo_depth']}")
    print(f"   Spec: {spi_spec['specification'][:120]}...\n")

    # Initialize A1 V2
    print("🔧 Initializing A1 V2 (Planner & Composer)...")
    agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})

    # Generate RTL with V2 architecture
    print("⚙️  Calling A1 V2.process()...\n")
    result = agent.process(spi_spec)

    print("="*80)
    print("  A1 V2 GENERATION RESULTS")
    print("="*80 + "\n")

    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Execution Time: {result.execution_time_ms:.2f}ms")

    if result.success:
        rtl_code = result.output_data['rtl_code']
        ports = result.output_data.get('ports', [])
        generation_method = result.output_data.get('generation_method', 'unknown')
        intent_detected = result.output_data.get('intent_type', 'unknown')
        validation = result.output_data.get('validation', {})
        design_plan = result.output_data.get('design_plan', None)

        print(f"\n📊 Generation Metrics:")
        print(f"   Detected Intent: {intent_detected}")
        print(f"   Method: {generation_method}")
        print(f"   Lines: {len(rtl_code.split(chr(10)))}")
        print(f"   Ports: {len(ports)}")
        print(f"   Syntax Valid: {validation.get('syntax_valid', False)}")

        # Display DesignPlan if available
        if design_plan:
            print(f"\n📋 DesignPlan Generated:")
            print(f"   Module: {design_plan.get('module_name', 'N/A')}")
            print(f"   Description: {design_plan.get('description', 'N/A')}")
            submodules = design_plan.get('submodules', [])
            print(f"   Submodules: {len(submodules)}")
            if submodules:
                for i, submod in enumerate(submodules, 1):
                    print(f"      {i}. {submod['type']:20} ({submod['instance_name']}) - {submod['role']}")

        print(f"\n📌 Extracted Ports:")
        if ports:
            for port in ports[:15]:
                print(f"   {port['direction']:6} {port['name']:25} [{port['width']} bits]")
            if len(ports) > 15:
                print(f"   ... and {len(ports) - 15} more")
        else:
            print("   ⚠️  No ports extracted!")

        print(f"\n📄 Generated RTL Code (first 80 lines):")
        print("="*80)
        lines = rtl_code.split('\n')
        for i, line in enumerate(lines[:80], 1):
            print(f"  {i:3}: {line}")
        if len(lines) > 80:
            print(f"\n  ... and {len(lines) - 80} more lines")
        print("="*80)

        # Write to file
        output_file = Path('/tmp/SPI_MASTER_V2.v')
        output_file.write_text(rtl_code)
        print(f"\n✅ RTL written to: {output_file}")

        # Analysis
        print(f"\n🔍 DIAGNOSTIC ANALYSIS:")
        print("="*80)

        has_spi_keywords = any(kw in rtl_code.lower() for kw in ['spi', 'mosi', 'miso', 'sclk', 'cs'])
        has_fifo = 'fifo' in rtl_code.lower()
        has_state_machine = any(kw in rtl_code.lower() for kw in ['state', 'fsm'])
        has_shift_register = 'shift' in rtl_code.lower()
        has_clock_divider = any(kw in rtl_code.lower() for kw in ['divider', 'divisor', 'clk_out'])
        line_count = len(lines)

        print(f"   Line count: {line_count}")
        print(f"   Has SPI keywords (mosi/miso/sclk): {has_spi_keywords}")
        print(f"   Has FIFO logic: {has_fifo}")
        print(f"   Has state machine: {has_state_machine}")
        print(f"   Has shift register: {has_shift_register}")
        print(f"   Has clock divider: {has_clock_divider}")
        print(f"   Port count: {len(ports)}")
        print(f"   Submodule count: {len(submodules) if design_plan else 0}")

        # Comparison to original A1
        print(f"\n📊 COMPARISON TO ORIGINAL A1:")
        print("="*80)
        print(f"   Original A1 (FAILED):")
        print(f"      Lines: 3")
        print(f"      Method: template_register_file")
        print(f"      Ports: 0")
        print(f"      Quality: Empty stub")
        print(f"\n   A1 V2 (PLANNER & COMPOSER):")
        print(f"      Lines: {line_count}")
        print(f"      Method: {generation_method}")
        print(f"      Ports: {len(ports)}")
        print(f"      Submodules: {len(submodules) if design_plan else 0}")

        # Verdict
        print(f"\n📋 VERDICT:")
        if line_count <= 10:
            print("   ❌ FAILURE: Still generating stub/empty module (≤10 lines)")
            print("   ⚠️  A1 V2 did not improve generation")
            return False
        elif line_count < 50:
            print("   ⚠️  PARTIAL: Generated more than stub but still minimal (10-50 lines)")
            if has_spi_keywords or has_fifo or has_shift_register:
                print("   ✅ Contains domain-specific logic")
                print("   💡 May need more comprehensive composition logic")
                return True
            else:
                print("   ⚠️  Generic code, may not be SPI-specific")
                return False
        else:
            print("   ✅ SUCCESS: Generated substantial code (≥50 lines)")
            improvement = line_count / 3  # Original was 3 lines
            print(f"   📈 Improvement: {improvement:.1f}x more lines than original A1")

            if design_plan and len(submodules) >= 3:
                print(f"   ✅ Hierarchical design with {len(submodules)} submodules")

            if has_spi_keywords or has_fifo or has_shift_register:
                print("   ✅ Contains domain-specific logic")

            if generation_method == 'composed_hierarchical':
                print("   ✅ Used Planner & Composer architecture")

            return True

    else:
        print(f"\n❌ A1 V2 Generation Failed!")
        print(f"   Errors: {result.errors}")
        print(f"   Warnings: {result.warnings}")
        return False


if __name__ == '__main__':
    success = main()

    print("\n" + "="*80)
    if success:
        print("  ✅ A1 V2 UPGRADE SUCCESSFUL")
        print("  🎉 Planner & Composer architecture solved the empty module problem")
        print("  📋 Ready to proceed with Track B: LLM Fine-tuning")
    else:
        print("  ❌ A1 V2 UPGRADE INCOMPLETE")
        print("  💡 Additional refinement needed for Planner & Composer")
    print("="*80 + "\n")

    sys.exit(0 if success else 1)
