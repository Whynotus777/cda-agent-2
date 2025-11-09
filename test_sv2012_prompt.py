#!/usr/bin/env python3
"""Quick test: Does enforcing SystemVerilog 2012 syntax improve pass rates?"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load model (copy from benchmark_v5_4)
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ),
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "models/qwen_coder_rtl/latest")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Test FSM spec
fsm_spec = """3-state Moore FSM traffic light controller.
States: GREEN, YELLOW, RED
Transitions: GREEN→YELLOW after 30 cycles, YELLOW→RED after 5 cycles, RED→GREEN after 25 cycles
Outputs: green_light, yellow_light, red_light (one-hot encoded)
"""

print("\n" + "="*80)
print("TEST 1: OLD PROMPT (Verilog 2001)")
print("="*80)

old_prompt = f"Generate Verilog RTL for the following specification:\n\n{fsm_spec}\n\nProvide only the Verilog module code."
inputs_old = tokenizer(old_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs_old = model.generate(**inputs_old, max_new_tokens=1024, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)

rtl_old = tokenizer.decode(outputs_old[0], skip_special_tokens=True)[len(old_prompt):].strip()
print(rtl_old[:800])  # Show first 800 chars

print("\n" + "="*80)
print("TEST 2: NEW PROMPT (SystemVerilog 2012)")
print("="*80)

new_prompt = f"""Generate SystemVerilog (IEEE 1800-2012) RTL for the following specification.

REQUIRED SYNTAX:
- Use 'typedef enum' for state types
- Use 'always_ff @(posedge clk or negedge rst_n)' for sequential logic
- Use 'always_comb' for combinational logic
- Use 'logic' instead of 'reg'/'wire' where appropriate

Specification:
{fsm_spec}

Provide only the SystemVerilog module code."""

inputs_new = tokenizer(new_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs_new = model.generate(**inputs_new, max_new_tokens=1024, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)

rtl_new = tokenizer.decode(outputs_new[0], skip_special_tokens=True)[len(new_prompt):].strip()
print(rtl_new[:800])

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

import re
patterns_old = {
    "typedef enum": bool(re.search(r"typedef.*enum", rtl_old)),
    "always_ff": bool(re.search(r"always_ff", rtl_old)),
    "always_comb": bool(re.search(r"always_comb", rtl_old)),
    "logic": bool(re.search(r"\blogic\b", rtl_old)),
}

patterns_new = {
    "typedef enum": bool(re.search(r"typedef.*enum", rtl_new)),
    "always_ff": bool(re.search(r"always_ff", rtl_new)),
    "always_comb": bool(re.search(r"always_comb", rtl_new)),
    "logic": bool(re.search(r"\blogic\b", rtl_new)),
}

print("Old Prompt (Verilog 2001 features):")
for k, v in patterns_old.items():
    print(f"  {k}: {'✓' if v else '✗'}")

print("\nNew Prompt (SystemVerilog 2012 features):")
for k, v in patterns_new.items():
    print(f"  {k}: {'✓' if v else '✗'}")
