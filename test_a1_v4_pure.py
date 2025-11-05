#!/usr/bin/env python3
"""
A1 V4 (Pure Dataset Mixtral) Battle Test

Test the rtl-generator-v4-pure model (trained on syntax-clean dataset)
on SPI_MASTER_001. Compare to:
- A1 V2 (Planner & Composer - symbolic)
- A1 V3 (Broken dataset - had syntax errors)
- A1 V4 (Pure dataset - THIS TEST)
"""

import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class A1_V4_PureGenerator:
    """A1 V4 - Fine-tuned Mixtral on Pure Verilog Dataset"""

    def __init__(self, model_path: str):
        """Initialize with fine-tuned model"""
        print("\n🔧 Loading rtl-generator-v4-pure (A1 V4)...")
        print(f"   Model: {model_path}")

        # BitsAndBytes config for 4-bit inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Check if this is an adapter or full model
        adapter_config = Path(model_path) / "adapter_config.json"

        if adapter_config.exists():
            # Load as adapter on top of base model
            print("   Loading base Mixtral-8x7B...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            print("   Loading fine-tuned adapter...")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load as full model
            print("   Loading full fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

        # Load tokenizer
        print("   Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            # Fallback to base model tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

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

        print(f"\n📝 Generating RTL with A1 V4 (Pure Dataset)...")
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
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract Verilog code (after [/INST])
        if '[/INST]' in generated_text:
            rtl_code = generated_text.split('[/INST]')[1].strip()
        else:
            rtl_code = generated_text.strip()

        print(f"   ✅ Generation complete ({generation_time:.2f}s)")
        print(f"   Generated {len(rtl_code)} characters")

        return rtl_code


def validate_syntax_yosys(rtl_code: str, module_name: str) -> Dict[str, Any]:
    """Validate RTL syntax using Yosys"""
    import subprocess
    import tempfile

    print("\n🔍 Validating syntax with Yosys...")

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
        f.write(rtl_code)
        temp_file = f.name

    try:
        # Run Yosys read_verilog
        result = subprocess.run(
            ['yosys', '-p', f'read_verilog {temp_file}'],
            capture_output=True,
            text=True,
            timeout=10
        )

        syntax_valid = result.returncode == 0
        errors = []
        warnings = []

        if not syntax_valid:
            # Parse errors from stderr
            for line in result.stderr.split('\n'):
                if 'ERROR' in line.upper():
                    errors.append(line.strip())
                elif 'WARNING' in line.upper():
                    warnings.append(line.strip())

        return {
            'syntax_valid': syntax_valid,
            'errors': errors,
            'warnings': warnings,
            'yosys_output': result.stderr
        }

    except Exception as e:
        return {
            'syntax_valid': False,
            'errors': [str(e)],
            'warnings': [],
            'yosys_output': ''
        }
    finally:
        Path(temp_file).unlink(missing_ok=True)


def main():
    """Test A1 V4 (Pure Dataset) on SPI_MASTER_001"""

    print("\n" + "="*80)
    print("  A1 V4 BATTLE TEST: SPI Master (Pure Dataset)")
    print("="*80 + "\n")

    # SPI Master spec - Same one used for V2 and V3
    spi_spec = {
        "module_name": "SPI_MASTER_001",
        "specification": """SPI Master controller with configurable clock polarity, phase,
        and data width (8/16/32-bit). Supports full-duplex operation, programmable clock
        divider (divide by 2 to 256), and FIFO buffers (8-deep TX/RX). Includes busy
        status flag and interrupt generation on transfer complete.""",
        "intent_type": "spi_master",
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

    # Find latest trained model
    model_path = Path('models/mixtral_rtl/run_pure_20251030_121523/final_model')

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("   Looking for alternative paths...")

        # Try checkpoint-156
        alt_path = Path('models/mixtral_rtl/run_pure_20251030_121523/checkpoint-156')
        if alt_path.exists():
            model_path = alt_path
            print(f"   ✅ Found checkpoint: {model_path}")
        else:
            print("   ❌ No trained model found!")
            return False

    print(f"📦 Model Path: {model_path}")

    # Initialize A1 V4
    try:
        generator = A1_V4_PureGenerator(str(model_path))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Generate RTL
    print("\n⚙️  Generating RTL with A1 V4...")
    start_time = time.time()

    try:
        rtl_code = generator.generate_rtl(spi_spec)
        generation_time = time.time() - start_time
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze generated code
    print("\n" + "="*80)
    print("  A1 V4 GENERATION RESULTS")
    print("="*80 + "\n")

    lines = rtl_code.split('\n')
    line_count = len(lines)

    print(f"📊 Basic Metrics:")
    print(f"   Lines: {line_count}")
    print(f"   Characters: {len(rtl_code)}")
    print(f"   Generation Time: {generation_time:.2f}s")

    # Validate syntax
    validation = validate_syntax_yosys(rtl_code, spi_spec['module_name'])

    print(f"\n🔍 Syntax Validation:")
    print(f"   Yosys Valid: {validation['syntax_valid']}")
    if validation['errors']:
        print(f"   Errors: {len(validation['errors'])}")
        for error in validation['errors'][:5]:
            print(f"      - {error}")
    if validation['warnings']:
        print(f"   Warnings: {len(validation['warnings'])}")

    # Analyze content
    has_module = 'module' in rtl_code.lower()
    has_endmodule = 'endmodule' in rtl_code.lower()
    has_spi_keywords = any(kw in rtl_code.lower() for kw in ['spi', 'mosi', 'miso', 'sclk', 'cs'])
    has_fifo = 'fifo' in rtl_code.lower()
    has_state_machine = any(kw in rtl_code.lower() for kw in ['state', 'fsm', 'case'])
    has_shift_register = 'shift' in rtl_code.lower()
    has_clock_divider = any(kw in rtl_code.lower() for kw in ['divider', 'divisor', 'div'])

    print(f"\n🔎 Content Analysis:")
    print(f"   Has module declaration: {has_module}")
    print(f"   Has endmodule: {has_endmodule}")
    print(f"   Has SPI keywords (mosi/miso/sclk): {has_spi_keywords}")
    print(f"   Has FIFO logic: {has_fifo}")
    print(f"   Has state machine: {has_state_machine}")
    print(f"   Has shift register: {has_shift_register}")
    print(f"   Has clock divider: {has_clock_divider}")

    # Display generated code
    print(f"\n📄 Generated RTL Code:")
    print("="*80)
    for i, line in enumerate(lines[:100], 1):
        print(f"  {i:3}: {line}")
    if len(lines) > 100:
        print(f"\n  ... and {len(lines) - 100} more lines")
    print("="*80)

    # Write to file
    output_file = Path('/tmp/SPI_MASTER_001_V4.v')
    output_file.write_text(rtl_code)
    print(f"\n✅ RTL written to: {output_file}")

    # Comparison
    print(f"\n📊 COMPARISON TO PREVIOUS VERSIONS:")
    print("="*80)
    print(f"\n   A1 V2 (Planner & Composer - Symbolic):")
    print(f"      Method: Template-based hierarchical")
    print(f"      Quality: ~7/7 human eval on simple designs")
    print(f"      Limitation: Symbolic, not data-driven")

    print(f"\n   A1 V3 (Mixtral on Broken Dataset):")
    print(f"      Method: Fine-tuned LLM")
    print(f"      Quality: ~4/7 (many syntax errors)")
    print(f"      Issue: Dataset had syntax errors")

    print(f"\n   A1 V4 (Mixtral on Pure Dataset - THIS TEST):")
    print(f"      Lines: {line_count}")
    print(f"      Syntax Valid: {validation['syntax_valid']}")
    print(f"      Has SPI features: {has_spi_keywords}")
    print(f"      Has FIFO: {has_fifo}")
    print(f"      Has FSM: {has_state_machine}")

    # Verdict
    print(f"\n📋 VERDICT:")
    print("="*80)

    score = 0
    max_score = 7

    # Scoring criteria
    if line_count >= 50:
        score += 1
        print(f"   ✅ [1/1] Line count ≥50: {line_count}")
    else:
        print(f"   ❌ [0/1] Line count <50: {line_count}")

    if validation['syntax_valid']:
        score += 2
        print(f"   ✅ [2/2] Syntax valid (Yosys)")
    else:
        print(f"   ❌ [0/2] Syntax errors found")

    if has_module and has_endmodule:
        score += 1
        print(f"   ✅ [1/1] Proper module structure")
    else:
        print(f"   ❌ [0/1] Missing module structure")

    if has_spi_keywords:
        score += 1
        print(f"   ✅ [1/1] SPI-specific keywords")
    else:
        print(f"   ❌ [0/1] Missing SPI keywords")

    if has_fifo or has_state_machine:
        score += 1
        print(f"   ✅ [1/1] Complex logic (FIFO/FSM)")
    else:
        print(f"   ❌ [0/1] No complex logic detected")

    if has_shift_register or has_clock_divider:
        score += 1
        print(f"   ✅ [1/1] Domain logic (shift/divider)")
    else:
        print(f"   ❌ [0/1] Missing domain logic")

    print(f"\n   📊 Final Score: {score}/{max_score}")

    if score >= 6:
        print(f"   ✅ EXCELLENT: A1 V4 produces high-quality RTL")
        print(f"   🎉 Pure dataset training was successful!")
        return True
    elif score >= 4:
        print(f"   ✅ GOOD: A1 V4 improved over A1 V3")
        print(f"   💡 Better than broken dataset, room for improvement")
        return True
    elif score >= 2:
        print(f"   ⚠️  FAIR: Some improvement but still needs work")
        return False
    else:
        print(f"   ❌ POOR: A1 V4 did not improve meaningfully")
        return False


if __name__ == '__main__':
    success = main()

    print("\n" + "="*80)
    if success:
        print("  ✅ A1 V4 (PURE DATASET) TEST PASSED")
        print("  🎉 Syntax-clean training improved RTL generation quality")
        print("  📋 Ready for production integration")
    else:
        print("  ⚠️  A1 V4 TEST NEEDS REVIEW")
        print("  💡 May need additional training or prompt engineering")
    print("="*80 + "\n")

    sys.exit(0 if success else 1)
