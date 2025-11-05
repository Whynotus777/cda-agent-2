#!/usr/bin/env python3
"""
A1 V3 (Fine-tuned Mixtral) Battle Test

Test the rtl-generator-v1-mixtral model on SPI_MASTER_001 and run through
the complete 6-agent pipeline. Compare results to A1 V2 (Planner & Composer).
"""

import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from core.rtl_agents import (
    A5_StyleReviewCopilot,
    A4_LintCDCAssistant,
    A3_ConstraintSynthesizer,
    A6_EDACommandCopilot
)


class A1_V3_MixtralGenerator:
    """A1 V3 - Fine-tuned Mixtral RTL Generator"""

    def __init__(self, adapter_path: str):
        """Initialize with fine-tuned adapter"""
        print("\n🔧 Loading rtl-generator-v1-mixtral...")
        print(f"   Adapter: {adapter_path}")

        # BitsAndBytes config for 4-bit inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load base Mixtral
        print("   Loading base Mixtral-8x7B...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Load fine-tuned adapter
        print("   Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("   ✅ Model loaded successfully")

    def generate_rtl(self, spec: Dict[str, Any]) -> str:
        """Generate RTL from specification"""
        module_name = spec.get('module_name', 'generated_module')
        specification = spec.get('specification', '')
        parameters = spec.get('parameters', {})

        # Create prompt in Mixtral-Instruct format
        param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])

        prompt = f"""[INST] Generate a Verilog module called '{module_name}'.

Specification: {specification}

Parameters: {param_str}

Generate complete, synthesizable Verilog RTL code with proper module declaration, ports, and implementation. [/INST]"""

        print(f"\n📝 Generating RTL with Mixtral...")
        print(f"   Prompt length: {len(prompt)} chars")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generation_time = time.time() - start_time

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract RTL code (after [/INST])
        if "[/INST]" in full_response:
            rtl_code = full_response.split("[/INST]")[-1].strip()
        else:
            rtl_code = full_response

        # Clean up any markdown code blocks
        if "```verilog" in rtl_code:
            rtl_code = rtl_code.split("```verilog")[1].split("```")[0].strip()
        elif "```" in rtl_code:
            rtl_code = rtl_code.split("```")[1].split("```")[0].strip()

        print(f"   Generated {len(rtl_code)} chars in {generation_time:.2f}s")

        return rtl_code


def run_full_pipeline(rtl_code: str, module_name: str) -> Dict[str, Any]:
    """Run generated RTL through full 6-agent pipeline"""

    results = {
        'a5_style': None,
        'a4_lint': None,
        'a3_constraints': None,
        'a6_synthesis': None,
        'yosys_execution': None
    }

    # Save RTL to temp file
    rtl_file = Path(f'/tmp/{module_name}.v')
    rtl_file.write_text(rtl_code)

    print("\n" + "="*80)
    print("  RUNNING FULL 6-AGENT PIPELINE")
    print("="*80)

    # Step 1: A5 - Style Review
    print("\n📋 Step 1: A5 - Style Review")
    try:
        a5 = A5_StyleReviewCopilot({'yosys_binary': 'yosys'})
        a5_result = a5.process({
            'rtl_files': [str(rtl_file)],
            'coding_standards': 'industry'
        })
        results['a5_style'] = {
            'success': a5_result.success,
            'violations': len(a5_result.output_data.get('violations', [])),
            'critical': sum(1 for v in a5_result.output_data.get('violations', [])
                          if v.get('severity') == 'critical')
        }
        print(f"   Success: {a5_result.success}")
        print(f"   Violations: {results['a5_style']['violations']} "
              f"({results['a5_style']['critical']} critical)")
    except Exception as e:
        results['a5_style'] = {'error': str(e)}
        print(f"   ❌ Error: {e}")

    # Step 2: A4 - Lint & CDC
    print("\n🔍 Step 2: A4 - Lint & CDC")
    try:
        a4 = A4_LintCDCAssistant({'yosys_binary': 'yosys'})
        a4_result = a4.process({
            'rtl_files': [str(rtl_file)],
            'check_type': 'lint'
        })
        results['a4_lint'] = {
            'success': a4_result.success,
            'issues': len(a4_result.output_data.get('issues', [])),
            'fixed': a4_result.output_data.get('auto_fixed', 0)
        }
        print(f"   Success: {a4_result.success}")
        print(f"   Issues: {results['a4_lint']['issues']}")
        print(f"   Auto-fixed: {results['a4_lint']['fixed']}")
    except Exception as e:
        results['a4_lint'] = {'error': str(e)}
        print(f"   ❌ Error: {e}")

    # Step 3: A3 - Constraints
    print("\n⏱️  Step 3: A3 - Constraint Synthesis")
    try:
        a3 = A3_ConstraintSynthesizer({})
        a3_result = a3.process({
            'rtl_files': [str(rtl_file)],
            'target_frequency_mhz': 100.0,
            'io_delays': {}
        })
        results['a3_constraints'] = {
            'success': a3_result.success,
            'sdc_lines': len(a3_result.output_data.get('sdc_content', '').split('\n'))
        }
        print(f"   Success: {a3_result.success}")
        print(f"   SDC lines: {results['a3_constraints']['sdc_lines']}")
    except Exception as e:
        results['a3_constraints'] = {'error': str(e)}
        print(f"   ❌ Error: {e}")

    # Step 4: A6 - EDA Commands
    print("\n🔧 Step 4: A6 - Generate Synthesis Script")
    try:
        a6 = A6_EDACommandCopilot({'yosys_binary': 'yosys'})
        a6_result = a6.process({
            'tool': 'yosys',
            'command_type': 'synthesis',
            'input_files': [str(rtl_file)],
            'output_files': [f'/tmp/{module_name}_synth.v'],
            'parameters': {
                'top_module': module_name,
                'optimization_goal': 'balanced'
            }
        })
        results['a6_synthesis'] = {
            'success': a6_result.success,
            'script_lines': len(a6_result.output_data.get('script_content', '').split('\n'))
        }
        print(f"   Success: {a6_result.success}")
        print(f"   Script lines: {results['a6_synthesis']['script_lines']}")

        # Step 5: Execute Yosys
        if a6_result.success:
            print("\n⚙️  Step 5: Yosys Synthesis Execution")
            script_content = a6_result.output_data.get('script_content', '')
            script_file = Path('/tmp/yosys_synth.ys')
            script_file.write_text(script_content)

            import subprocess
            yosys_result = subprocess.run(
                ['yosys', '-s', str(script_file)],
                capture_output=True,
                text=True,
                timeout=120
            )

            results['yosys_execution'] = {
                'exit_code': yosys_result.returncode,
                'success': yosys_result.returncode == 0,
                'errors': yosys_result.stderr.count('ERROR:'),
                'warnings': yosys_result.stderr.count('Warning:')
            }
            print(f"   Exit code: {yosys_result.returncode}")
            print(f"   Errors: {results['yosys_execution']['errors']}")
            print(f"   Warnings: {results['yosys_execution']['warnings']}")
    except Exception as e:
        results['a6_synthesis'] = {'error': str(e)}
        print(f"   ❌ Error: {e}")

    return results


def main():
    """Main test execution"""

    print("="*80)
    print("  A1 V3 (FINE-TUNED MIXTRAL) BATTLE TEST")
    print("="*80)

    # Same spec used to test A1 V2
    spi_spec = {
        "module_name": "SPI_MASTER_001",
        "specification": """SPI Master controller with configurable clock polarity, phase,
        and data width (8/16/32-bit). Supports full-duplex operation, programmable clock
        divider (divide by 2 to 256), and FIFO buffers (8-deep TX/RX). Includes busy
        status flag and interrupt generation on transfer complete.""",
        "parameters": {
            "data_width": 8,
            "fifo_depth": 8,
            "max_clock_div": 256
        },
        "constraints": {
            "clock_period_ns": 10.0,
            "target_frequency_mhz": 100.0
        }
    }

    print("\n📝 Test Specification:")
    print(f"   Module: {spi_spec['module_name']}")
    print(f"   Data Width: {spi_spec['parameters']['data_width']}-bit")
    print(f"   FIFO Depth: {spi_spec['parameters']['fifo_depth']}")
    print(f"   Spec: {spi_spec['specification'][:120]}...")

    # Initialize A1 V3
    adapter_path = "models/mixtral_rtl/run_20251029_223754/final_model"
    project_root = Path(__file__).parent
    full_adapter_path = project_root / adapter_path

    if not full_adapter_path.exists():
        print(f"\n❌ Error: Adapter not found at {full_adapter_path}")
        print("   Did the training complete successfully?")
        return False

    generator = A1_V3_MixtralGenerator(str(full_adapter_path))

    # Generate RTL
    start_time = time.time()
    rtl_code = generator.generate_rtl(spi_spec)
    generation_time = time.time() - start_time

    # Analyze generated code
    lines = rtl_code.split('\n')
    non_comment_lines = [l for l in lines if l.strip() and not l.strip().startswith('//')]

    print("\n" + "="*80)
    print("  A1 V3 GENERATION RESULTS")
    print("="*80)

    print(f"\n📊 Generation Metrics:")
    print(f"   Total lines: {len(lines)}")
    print(f"   Non-comment lines: {len(non_comment_lines)}")
    print(f"   Characters: {len(rtl_code)}")
    print(f"   Generation time: {generation_time:.2f}s")

    # Check for key SPI components
    has_module = 'module' in rtl_code.lower()
    has_spi_keywords = any(kw in rtl_code.lower() for kw in ['spi', 'mosi', 'miso', 'sclk', 'cs'])
    has_fifo = 'fifo' in rtl_code.lower()
    has_state_machine = any(kw in rtl_code.lower() for kw in ['state', 'fsm'])
    has_always = 'always' in rtl_code
    has_ports = any(kw in rtl_code for kw in ['input', 'output'])

    print(f"\n🔍 Code Analysis:")
    print(f"   Has module declaration: {has_module}")
    print(f"   Has ports: {has_ports}")
    print(f"   Has SPI keywords: {has_spi_keywords}")
    print(f"   Has FIFO logic: {has_fifo}")
    print(f"   Has state machine: {has_state_machine}")
    print(f"   Has behavioral logic: {has_always}")

    # Show generated code
    print(f"\n📄 Generated RTL Code (first 100 lines):")
    print("="*80)
    for i, line in enumerate(lines[:100], 1):
        print(f"  {i:3}: {line}")
    if len(lines) > 100:
        print(f"\n  ... and {len(lines) - 100} more lines")
    print("="*80)

    # Save to file
    output_file = Path('/tmp/SPI_MASTER_001_V3.v')
    output_file.write_text(rtl_code)
    print(f"\n💾 RTL saved to: {output_file}")

    # Run through pipeline
    pipeline_results = run_full_pipeline(rtl_code, spi_spec['module_name'])

    # Final comparison report
    print("\n" + "="*80)
    print("  COMPARISON: A1 V2 (PLANNER) vs A1 V3 (MIXTRAL)")
    print("="*80)

    print("\n┌────────────────────────┬──────────────────┬──────────────────┐")
    print("│ Metric                 │ A1 V2 (Planner)  │ A1 V3 (Mixtral) │")
    print("├────────────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Lines Generated        │ 318              │ {len(lines):<16} │")
    print(f"│ Method                 │ hierarchical     │ LLM generation   │")
    print(f"│ Has Module             │ Yes              │ {str(has_module):<16} │")
    print(f"│ Has Ports              │ 47               │ {'Yes' if has_ports else 'No':<16} │")
    print(f"│ SPI Keywords           │ Yes              │ {str(has_spi_keywords):<16} │")
    print(f"│ FIFO Logic             │ Yes              │ {str(has_fifo):<16} │")
    print(f"│ State Machine          │ Yes              │ {str(has_state_machine):<16} │")
    print(f"│ Generation Time        │ ~37ms            │ {generation_time:.2f}s{' '*(13-len(f'{generation_time:.2f}s'))} │")
    print("└────────────────────────┴──────────────────┴──────────────────┘")

    # Pipeline results
    print("\n┌────────────────────────┬──────────────────┬──────────────────┐")
    print("│ Pipeline Step          │ Status           │ Details          │")
    print("├────────────────────────┼──────────────────┼──────────────────┤")

    a5 = pipeline_results.get('a5_style', {})
    print(f"│ A5 (Style Review)      │ {'✅ Pass' if a5.get('success') else '❌ Fail':<16} │ {a5.get('violations', 'N/A')} violations{' '*(6-len(str(a5.get('violations', 'N/A'))))} │")

    a4 = pipeline_results.get('a4_lint', {})
    print(f"│ A4 (Lint & CDC)        │ {'✅ Pass' if a4.get('success') else '❌ Fail':<16} │ {a4.get('issues', 'N/A')} issues{' '*(10-len(str(a4.get('issues', 'N/A'))))} │")

    a3 = pipeline_results.get('a3_constraints', {})
    print(f"│ A3 (Constraints)       │ {'✅ Pass' if a3.get('success') else '❌ Fail':<16} │ {a3.get('sdc_lines', 'N/A')} SDC lines{' '*(7-len(str(a3.get('sdc_lines', 'N/A'))))} │")

    yosys = pipeline_results.get('yosys_execution', {})
    print(f"│ Yosys Synthesis        │ {'✅ Pass' if yosys and yosys.get('success') else '❌ Fail':<16} │ Exit: {yosys.get('exit_code', 'N/A') if yosys else 'N/A'}{' '*(9-len(str(yosys.get('exit_code', 'N/A') if yosys else 'N/A')))} │")
    print("└────────────────────────┴──────────────────┴──────────────────┘")

    # Verdict
    print("\n📋 VERDICT:")

    quality_score = 0
    if len(lines) >= 50:
        quality_score += 1
    if has_module and has_ports:
        quality_score += 1
    if has_spi_keywords:
        quality_score += 1
    if has_fifo:
        quality_score += 1
    if has_state_machine:
        quality_score += 1
    if yosys and yosys.get('success'):
        quality_score += 2

    if quality_score >= 6:
        print("   ✅ EXCELLENT: A1 V3 (Mixtral) generated high-quality RTL")
        print("   🎉 Fine-tuned model demonstrates strong RTL generation capability")
    elif quality_score >= 4:
        print("   ⚠️  GOOD: A1 V3 generated functional RTL with some limitations")
        print("   💡 Model shows promise but may need refinement")
    else:
        print("   ❌ NEEDS IMPROVEMENT: A1 V3 output quality below expectations")
        print("   💡 Consider additional training or prompt engineering")

    print(f"   Quality Score: {quality_score}/7")

    print("\n" + "="*80)
    print("  A1 V3 BATTLE TEST COMPLETE")
    print("="*80)

    return quality_score >= 4


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
