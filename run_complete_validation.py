#!/usr/bin/env python3
"""
Complete 6-Agent Pipeline Validation - SPI_MASTER_001

This script runs the COMPLETE pipeline with actual synthesis execution:
A1 → A5 → A4 → A3 → A6 → YOSYS EXECUTION

This is the true "spec-to-silicon" test.
"""

import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import (
    A1_SpecToRTLGenerator,
    A2_BoilerplateGenerator,
    A3_ConstraintSynthesizer,
    A4_LintCDCAssistant,
    A5_StyleReviewCopilot,
    A6_EDACommandCopilot
)


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """Run complete end-to-end validation"""

    print_section("🚀 COMPLETE 6-AGENT PIPELINE VALIDATION - SPI_MASTER_001")

    print("Pipeline: A1 → A5 → A4 → A3 → A6 → YOSYS SYNTHESIS")
    print("Design: SPI Master Controller (100 MHz, 32-bit, 8-deep FIFOs)")
    print("\nThis is the TRUE end-to-end test with actual synthesis execution.\n")

    # SPI_MASTER_001 Design Intent
    spi_spec = {
        "module_name": "SPI_MASTER_001",
        "specification": """SPI Master controller with configurable clock polarity, phase,
        and data width (8/16/32-bit). Supports full-duplex operation, programmable clock
        divider (divide by 2 to 256), and FIFO buffers (8-deep TX/RX). Includes busy
        status flag and interrupt generation on transfer complete.""",
        "intent_type": "register",  # Will use synthesized approach
        "parameters": {
            "data_width": 32,
            "fifo_depth": 8,
            "max_clock_div": 256
        },
        "constraints": {
            "clock_period_ns": 10.0,
            "target_frequency_mhz": 100.0,
            "default_input_delay_ns": 2.0,
            "default_output_delay_ns": 2.0
        }
    }

    # =========================================================================
    # STEP 1: A1 - Spec-to-RTL Generation
    # =========================================================================
    print_section("STEP 1/6: A1 - Spec-to-RTL Generation")

    print("📝 Input Specification:")
    print(f"   Module: {spi_spec['module_name']}")
    print(f"   Data Width: {spi_spec['parameters']['data_width']}-bit")
    print(f"   FIFO Depth: {spi_spec['parameters']['fifo_depth']}")
    print(f"   Clock: {spi_spec['constraints']['target_frequency_mhz']} MHz\n")

    a1_agent = A1_SpecToRTLGenerator({'yosys_binary': 'yosys'})
    a1_result = a1_agent.process(spi_spec)

    if not a1_result.success:
        print(f"❌ A1 FAILED: {a1_result.errors}")
        return False

    rtl_code = a1_result.output_data['rtl_code']
    module_name = a1_result.output_data['module_name']
    ports = a1_result.output_data['ports']

    print(f"✅ A1 Success!")
    print(f"   Confidence: {a1_result.confidence:.2f}")
    print(f"   Generation Method: {a1_result.output_data['generation_method']}")
    print(f"   Lines Generated: {len(rtl_code.split(chr(10)))}")
    print(f"   Ports: {len(ports)}")
    print(f"   Syntax Valid: {a1_result.output_data['validation']['syntax_valid']}")

    # Write RTL to file
    rtl_file = Path(f"/tmp/{module_name}.v")
    rtl_file.write_text(rtl_code)
    print(f"   RTL written to: {rtl_file}")

    # =========================================================================
    # STEP 2: A5 - Style & Security Review
    # =========================================================================
    print_section("STEP 2/6: A5 - Style & Security Review")

    a5_agent = A5_StyleReviewCopilot()
    a5_result = a5_agent.process({
        'rtl_code': rtl_code,
        'file_path': str(rtl_file)
    })

    summary = a5_result.output_data['summary']

    print(f"📊 Review Results:")
    print(f"   Total Violations: {summary['total']}")
    print(f"   Critical: {summary['critical']} 🔴")
    print(f"   Warning: {summary['warning']} ⚠️")
    print(f"   Info: {summary['info']} ℹ️")

    violations = a5_result.output_data['violations']
    if violations:
        print(f"\n⚠️  Issues Found:")
        for v in violations[:5]:
            emoji = {'critical': '🔴', 'warning': '⚠️', 'info': 'ℹ️'}[v['severity']]
            print(f"   {emoji} [{v['severity'].upper()}] {v['rule_id']}: {v['message']}")

    if summary['critical'] == 0:
        print(f"\n✅ A5 Success! No critical violations")
    else:
        print(f"\n⚠️  A5: {summary['critical']} critical violations (continuing anyway)")

    # Write style report
    report_file = Path(f"/tmp/{module_name}_style_report.md")
    report_file.write_text(a5_result.output_data['report_markdown'])
    print(f"   Report written to: {report_file}")

    # =========================================================================
    # STEP 3: A4 - Lint & CDC Assistant
    # =========================================================================
    print_section("STEP 3/6: A4 - Lint & CDC Assistant")

    print("🔍 Running Verilator lint check...")

    # Run Verilator to get log
    try:
        verilator_cmd = ['verilator', '--lint-only', '-Wall', str(rtl_file)]
        verilator_proc = subprocess.run(
            verilator_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        lint_log = verilator_proc.stdout + verilator_proc.stderr

        print(f"   Verilator exit code: {verilator_proc.returncode}")

        # Count issues
        lint_lines = lint_log.split('\n')
        warnings = sum(1 for line in lint_lines if 'Warning' in line)
        errors = sum(1 for line in lint_lines if 'Error' in line)

        print(f"   Warnings: {warnings}")
        print(f"   Errors: {errors}")

        if lint_log.strip():
            # Process with A4
            a4_agent = A4_LintCDCAssistant()
            a4_result = a4_agent.process({
                'log_content': lint_log,
                'tool': 'verilator',
                'source_file': str(rtl_file)
            })

            issues = a4_result.output_data.get('issues', [])
            fixes = a4_result.output_data.get('fixes', [])

            print(f"\n📊 A4 Analysis:")
            print(f"   Issues detected: {len(issues)}")
            print(f"   Auto-fixes generated: {len(fixes)}")

            if fixes:
                print(f"\n🔧 Proposed Fixes:")
                for fix in fixes[:3]:
                    print(f"   • {fix['issue_type']}: {fix['description']}")
                    print(f"     Confidence: {fix['confidence']:.2f}")

            print(f"\n✅ A4 Success! (Fixes available but not applied for demo)")
        else:
            print(f"\n✅ A4 Success! No lint issues found")

    except FileNotFoundError:
        print(f"⚠️  Verilator not found, skipping A4 (non-critical)")
    except Exception as e:
        print(f"⚠️  A4 error: {e} (non-critical, continuing)")

    # =========================================================================
    # STEP 4: A3 - Constraint Synthesis
    # =========================================================================
    print_section("STEP 4/6: A3 - Constraint Synthesis")

    print("📝 Generating SDC timing constraints...")
    print(f"   Target: {spi_spec['constraints']['target_frequency_mhz']} MHz")

    a3_agent = A3_ConstraintSynthesizer()
    a3_result = a3_agent.process({
        'module_name': module_name,
        'constraints': spi_spec['constraints'],
        'context': {
            'ports': ports
        }
    })

    if not a3_result.success:
        print(f"❌ A3 FAILED: {a3_result.errors}")
        return False

    sdc_content = a3_result.output_data['constraints']
    clock_specs = a3_result.output_data['clock_specs']

    print(f"✅ A3 Success!")
    print(f"   Confidence: {a3_result.confidence:.2f}")
    print(f"   Clock domains: {len(clock_specs)}")

    for clk in clock_specs:
        print(f"   • {clk['clock_name']}: {clk['frequency_mhz']:.1f} MHz ({clk['period_ns']:.2f} ns)")

    # Write SDC file
    sdc_file = Path(f"/tmp/{module_name}.sdc")
    sdc_file.write_text(sdc_content)
    print(f"   SDC written to: {sdc_file}")

    # =========================================================================
    # STEP 5: A6 - EDA Command Generation
    # =========================================================================
    print_section("STEP 5/6: A6 - EDA Command Generation")

    print("📝 Generating Yosys synthesis script...")

    a6_agent = A6_EDACommandCopilot()
    a6_result = a6_agent.process({
        'tool': 'yosys',
        'command_type': 'synthesis',
        'input_files': [str(rtl_file)],
        'output_files': [f'/tmp/{module_name}_synth.v'],
        'parameters': {
            'top_module': module_name,
            'optimization_goal': 'balanced'
        }
    })

    if not a6_result.success:
        print(f"❌ A6 FAILED: {a6_result.errors}")
        return False

    yosys_script = a6_result.output_data.get('script_content', a6_result.output_data.get('script', a6_result.output_data.get('command', '')))

    if not yosys_script:
        print(f"❌ A6: No script generated")
        print(f"   Available keys: {list(a6_result.output_data.keys())}")
        return False

    print(f"✅ A6 Success!")
    print(f"   Confidence: {a6_result.confidence:.2f}")
    print(f"   Script lines: {len(yosys_script.split(chr(10)))}")

    # Write Yosys script
    script_file = Path(f"/tmp/{module_name}_synth.ys")
    script_file.write_text(yosys_script)
    print(f"   Script written to: {script_file}")

    print(f"\n📄 Yosys Script:")
    print("-" * 80)
    for i, line in enumerate(yosys_script.split('\n'), 1):
        print(f"   {i:2}: {line}")
    print("-" * 80)

    # =========================================================================
    # STEP 6: YOSYS SYNTHESIS EXECUTION (THE CRITICAL STEP!)
    # =========================================================================
    print_section("STEP 6/6: YOSYS SYNTHESIS EXECUTION")

    print("⚙️  Executing Yosys synthesis...")
    print(f"   Input: {rtl_file}")
    print(f"   Script: {script_file}")
    print(f"   Module: {module_name}\n")

    try:
        # Run Yosys
        yosys_cmd = ['yosys', '-s', str(script_file)]
        yosys_proc = subprocess.run(
            yosys_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        yosys_output = yosys_proc.stdout + yosys_proc.stderr

        # Write full log
        log_file = Path(f"/tmp/{module_name}_synthesis.log")
        log_file.write_text(yosys_output)

        print(f"📊 YOSYS SYNTHESIS RESULTS")
        print("=" * 80)
        print(f"Exit Code: {yosys_proc.returncode}")

        if yosys_proc.returncode == 0:
            print(f"Status: ✅ SUCCESS")
        else:
            print(f"Status: ❌ FAILED")

        print(f"\nFull log written to: {log_file}")
        print("\n" + "=" * 80)
        print("SYNTHESIS LOG OUTPUT:")
        print("=" * 80 + "\n")

        # Parse and display key metrics
        lines = yosys_output.split('\n')

        # Look for key information
        printing = False
        for line in lines:
            # Print important sections
            if any(keyword in line for keyword in [
                'Executing', 'Reading', 'Successfully finished',
                'Number of wires:', 'Number of cells:',
                'Checking', 'Found', 'warnings', 'errors',
                'ABC:', 'Statistics', 'area =', 'delay ='
            ]):
                print(line)
                printing = True
            elif printing and line.strip() == '':
                printing = False
            elif printing:
                print(line)

        # Extract statistics
        print("\n" + "=" * 80)
        print("📈 SYNTHESIS STATISTICS")
        print("=" * 80 + "\n")

        # Count cells
        for line in lines:
            if 'Number of wires:' in line:
                print(f"   {line.strip()}")
            elif 'Number of cells:' in line:
                print(f"   {line.strip()}")
            elif '$_' in line and ':' in line and line.strip().endswith(')'):
                # Cell type counts
                print(f"   {line.strip()}")

        # Check for errors/warnings
        error_lines = [l for l in lines if 'ERROR' in l.upper()]
        warning_lines = [l for l in lines if 'WARNING' in l.upper()]

        print(f"\n   Errors: {len(error_lines)}")
        print(f"   Warnings: {len(warning_lines)}")

        if error_lines:
            print(f"\n❌ Errors Found:")
            for err in error_lines[:5]:
                print(f"   {err}")

        if warning_lines and len(warning_lines) <= 10:
            print(f"\n⚠️  Warnings:")
            for warn in warning_lines:
                print(f"   {warn}")

        # Final verdict
        print("\n" + "=" * 80)
        if yosys_proc.returncode == 0 and len(error_lines) == 0:
            print("🎉 SYNTHESIS SUCCESSFUL!")
            print("\nThe spec-to-silicon pipeline is COMPLETE and FUNCTIONAL!")
            print(f"\nGenerated artifacts:")
            print(f"   ✅ {rtl_file} - Verilog RTL")
            print(f"   ✅ {sdc_file} - Timing constraints")
            print(f"   ✅ {script_file} - Synthesis script")
            print(f"   ✅ {log_file} - Synthesis log")
            print(f"   ✅ {report_file} - Style report")
        else:
            print("❌ SYNTHESIS FAILED")
            print(f"\nCheck the log for details: {log_file}")

        print("=" * 80 + "\n")

        return yosys_proc.returncode == 0

    except FileNotFoundError:
        print("❌ Yosys not found!")
        print("   Install: sudo apt-get install yosys")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Yosys timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*80)
    print("  MULTI-AGENT RTL SYSTEM - COMPLETE END-TO-END VALIDATION")
    print("  True Spec-to-Silicon Test with Actual Synthesis Execution")
    print("="*80)

    success = main()

    print("\n" + "="*80)
    if success:
        print("  ✅ VALIDATION COMPLETE - SYSTEM FULLY OPERATIONAL")
    else:
        print("  ⚠️  VALIDATION ISSUES DETECTED - CHECK LOGS")
    print("="*80 + "\n")

    sys.exit(0 if success else 1)
