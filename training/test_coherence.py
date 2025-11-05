#!/usr/bin/env python3
"""
Sprint 2 Coherence Test: The Magic Test

Goal: Produce ONE coherent, insightful answer to a complex placement question.
Success: A single sentence an expert would find valuable.
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_coherence(model_path: str):
    """Test if apprentice can produce coherent, insightful output"""

    print("=" * 70)
    print("SPRINT 2: COHERENCE TEST")
    print("=" * 70)
    print(f"Loading apprentice from: {model_path}\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Device: {device}")
    print(f"Model loaded: gpt2-medium (354.8M parameters)\n")

    # Test question: Complex, multi-faceted placement challenge
    test_question = """For a 2 GHz out-of-order RISC-V processor at 28nm targeting mobile applications, what placement density should I use and why? Consider both timing and power constraints."""

    print("=" * 70)
    print("TEST QUESTION:")
    print("=" * 70)
    print(f"{test_question}\n")

    # Format prompt
    prompt = f"Q: {test_question}\n\nA:"

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    print("=" * 70)
    print("APPRENTICE RESPONSE:")
    print("=" * 70)

    # Generate with focused parameters for coherence
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.7,  # Lower for more focused output
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    if "A:" in generated:
        answer = generated.split("A:", 1)[1].strip()
        if "\nQ:" in answer:
            answer = answer.split("\nQ:")[0].strip()
    else:
        answer = generated

    print(f"{answer}\n")

    print("=" * 70)
    print("EVALUATION:")
    print("=" * 70)

    # Simple coherence checks
    word_count = len(answer.split())
    sentence_count = answer.count('.') + answer.count('!') + answer.count('?')

    # Domain concept detection
    placement_concepts = ['density', 'timing', 'power', 'buffer', 'white space',
                         'critical path', 'wirelength', 'congestion', 'utilization',
                         'area', 'frequency', 'leakage', 'dynamic', 'routing']

    detected_concepts = [c for c in placement_concepts if c.lower() in answer.lower()]

    print(f"Word count: {word_count}")
    print(f"Sentences: {sentence_count}")
    print(f"Placement concepts detected: {len(detected_concepts)}")
    print(f"Concepts: {detected_concepts}\n")

    # Coherence heuristics
    is_long_enough = word_count >= 30
    has_structure = sentence_count >= 2
    has_domain_knowledge = len(detected_concepts) >= 3
    not_repetitive = len(set(answer.split())) / len(answer.split()) > 0.6 if word_count > 0 else False

    print("Coherence indicators:")
    print(f"  {'✓' if is_long_enough else '✗'} Long enough (30+ words)")
    print(f"  {'✓' if has_structure else '✗'} Multi-sentence structure")
    print(f"  {'✓' if has_domain_knowledge else '✗'} Domain knowledge (3+ concepts)")
    print(f"  {'✓' if not_repetitive else '✗'} Not repetitive (60%+ unique words)")

    print("\n" + "=" * 70)

    if is_long_enough and has_structure and has_domain_knowledge and not_repetitive:
        print("VERDICT: ✓ COHERENT OUTPUT DETECTED")
        print("The apprentice has produced structured, domain-aware output.")
    elif has_domain_knowledge and is_long_enough:
        print("VERDICT: ⚠ PARTIAL COHERENCE")
        print("Domain knowledge present but structure may be weak.")
    else:
        print("VERDICT: ✗ INCOHERENT OUTPUT")
        print("Output lacks structure or domain knowledge.")

    print("=" * 70)

    return answer


if __name__ == "__main__":
    model_path = "./models/placement-apprentice-v2"

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    test_coherence(model_path)
