#!/usr/bin/env python3
"""
Expert-Level Question Test: Can the apprentice handle complex trade-off scenarios?
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Expert questions
QUESTIONS = [
    {
        "id": 1,
        "domain": "Trade-offs: Performance vs Power",
        "question": "A mobile SoC architect gives you an impossible mandate: a 15% performance improvement on the critical path, but with a non-negotiable, fixed power budget. Synthesis is already locked. Where, in the placement stage alone, do you find the leverage to solve this, and what is the inevitable price?"
    },
    {
        "id": 2,
        "domain": "Decision Making: Macro Placement",
        "question": "You have a design with 50 large SRAM macros and a high-speed data path that must snake between them. You can either pre-place the macros in a neat grid for power integrity or let the placer move them for better timing. Which do you choose and how do you justify it to the project lead?"
    },
    {
        "id": 3,
        "domain": "Failure Prediction: Routing",
        "question": "You've successfully placed a high-density (90% utilization) block and met all timing goals. What is the most likely reason the router will fail on this block, and what placement modification would you have made proactively to prevent that failure?"
    },
    {
        "id": 4,
        "domain": "Risk Awareness: Congestion",
        "question": "I choose to ignore congestion metrics during initial placement to achieve a 20ps better WNS (Worst Negative Slack). Convince me why this is a catastrophic mistake that will doom the project in the long run."
    },
    {
        "id": 5,
        "domain": "Transfer Learning: Novel Process",
        "question": "I have a new, experimental 3nm process technology. The transistors are incredibly fast, but the local interconnects are surprisingly resistive and slow, and there are new, complex DFM (Design for Manufacturability) rules that forbid placing certain cell patterns next to each other. None of your training data has seen this before. Based on your understanding of placement fundamentals, what is the single most important change you would make to a traditional placement algorithm to handle this new reality?"
    }
]


def test_expert_questions(model_path: str):
    """Test apprentice on expert-level questions"""

    print("=" * 80)
    print("EXPERT-LEVEL QUESTION TEST")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Questions: {len(QUESTIONS)}\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded\n")

    # Test each question
    for q in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION {q['id']}: {q['domain']}")
        print("=" * 80)
        print(f"\n{q['question']}\n")
        print("-" * 80)
        print("APPRENTICE RESPONSE:")
        print("-" * 80)

        # Format prompt
        prompt = f"Q: {q['question']}\n\nA:"

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
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

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    model_path = "./models/placement-apprentice-v2"

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        exit(1)

    test_expert_questions(model_path)
