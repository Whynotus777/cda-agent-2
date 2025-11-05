#!/usr/bin/env python3
"""
Mission Test: Placement Specialist
Test integrated reasoning on real timing risk scenarios
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time

# Test prompts (increasing difficulty)
PROMPTS = [
    {
        "id": 1,
        "difficulty": "8/10",
        "scenario": "High wire delay dominance",
        "prompt": "I am analyzing a critical path timing report for a mobile SoC. The path starts at the CPU register file and ends at the memory controller. The total path delay is 500ps, but the delay breakdown shows that a single net, net_cpu_to_mem_data[31], contributes 430ps of that delay. The rest of the path consists of standard logic cells with minimal delay. What is the most likely root cause of this timing violation?"
    },
    {
        "id": 2,
        "difficulty": "9/10",
        "scenario": "Clock skew masking",
        "prompt": "The second critical path in my design is in the instruction decoder. The data path delay is high, but it is only failing setup timing by 35ps. I've noticed in the report that the clock path to the endpoint flip-flop has a latency of 110ps, while the clock path to the start point flip-flop is only 40ps. Given this information, what is the most probable true source of the timing failure?"
    },
    {
        "id": 3,
        "difficulty": "10/10",
        "scenario": "Clock domain crossing",
        "prompt": "The third critical path is a control signal that travels from the 2GHz main CPU clock domain to the 500MHz peripheral bus clock domain. The timing tool is reporting a catastrophic setup violation of -1500ps, and no amount of buffer insertion seems to fix it. What is the most likely architectural or constraints-related reason for this massive, unfixable timing error?"
    }
]

def load_specialist(model_path: str, base_model: str):
    """Load specialist model with QLoRA adapters"""

    print("="*80)
    print("LOADING SPECIALIST MODEL")
    print("="*80)
    print(f"Base: {base_model}")
    print(f"Adapters: {model_path}")
    print()

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # Load base model with 4-bit quantization
    print("Loading base model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    print("✓ Base model loaded")

    # Load LoRA adapters
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    print("✓ Specialist ready")
    print()

    return model, tokenizer

def test_specialist(model, tokenizer, prompts):
    """Run mission test prompts"""

    print("="*80)
    print("MISSION TEST: TIMING RISK ANALYSIS")
    print("="*80)
    print(f"Prompts: {len(prompts)}")
    print()

    results = []

    for p in prompts:
        print("="*80)
        print(f"SCENARIO {p['id']}: {p['scenario']} (Difficulty: {p['difficulty']})")
        print("="*80)
        print(f"\nQUESTION:")
        print(f"{p['prompt']}")
        print()
        print("-"*80)
        print("SPECIALIST RESPONSE:")
        print("-"*80)

        # Format prompt
        formatted = f"Q: {p['prompt']}\n\nA:"

        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        elapsed = time.time() - start_time

        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        if "A:" in generated:
            answer = generated.split("A:", 1)[1].strip()
            if "\nQ:" in answer:
                answer = answer.split("\nQ:")[0].strip()
        else:
            answer = generated

        print(answer)
        print()
        print(f"[Response time: {elapsed:.1f}s]")
        print()

        results.append({
            "scenario": p['scenario'],
            "difficulty": p['difficulty'],
            "prompt": p['prompt'],
            "response": answer,
            "time": elapsed
        })

    return results

def main():
    base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    adapter_path = "./models/wisdom-specialist-v3-mixtral"

    # Load model
    model, tokenizer = load_specialist(adapter_path, base_model)

    # Run tests
    results = test_specialist(model, tokenizer, PROMPTS)

    # Summary
    print("="*80)
    print("MISSION TEST COMPLETE")
    print("="*80)
    print(f"Scenarios tested: {len(results)}")
    print(f"Total response time: {sum(r['time'] for r in results):.1f}s")
    print()
    print("Results saved. Manual evaluation required to assess:")
    print("  1. Technical accuracy")
    print("  2. Integrated reasoning (synthesis across concepts)")
    print("  3. Actionable guidance")
    print("="*80)

if __name__ == "__main__":
    main()
