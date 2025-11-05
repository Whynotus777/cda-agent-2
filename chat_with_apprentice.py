#!/usr/bin/env python3
"""
Simple Chat Interface for Placement Apprentice Model

This script provides an interactive command-line chat interface to talk with
the fine-tuned placement apprentice model trained on chip design knowledge.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from pathlib import Path
import sys

class ApprenticeChat:
    """Simple chat interface for the apprentice model"""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the chat interface.

        Args:
            model_path: Path to the trained model
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {self.model_path}...")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded successfully!")
        print(f"  Parameters: {self.model.num_parameters() / 1e6:.1f}M")
        print()

    def generate_response(self, question: str, max_new_tokens: int = 200) -> str:
        """
        Generate response to a question.

        Args:
            question: User's question
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        # Format as Q&A
        prompt = f"Q: {question}\n\nA:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (remove prompt)
        if "A:" in generated:
            answer = generated.split("A:", 1)[1].strip()
            # Stop at next Q: if present
            if "\nQ:" in answer:
                answer = answer.split("\nQ:")[0].strip()
        else:
            answer = generated

        return answer

    def chat(self):
        """Run interactive chat loop"""
        print("="*70)
        print("CHIP DESIGN PLACEMENT APPRENTICE - CHAT INTERFACE")
        print("="*70)
        print()
        print("I'm a specialized AI trained on chip design placement knowledge.")
        print("Ask me about placement algorithms, tools, techniques, and challenges!")
        print()
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to end the chat")
        print("  - Type 'help' for example questions")
        print()
        print("="*70)
        print()

        while True:
            try:
                # Get user input
                user_input = input("\n\033[1;34mYou:\033[0m ").strip()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! Thanks for chatting about chip design!")
                    break

                # Check for help
                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                # Generate response
                print("\n\033[1;32mApprentice:\033[0m ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for chatting about chip design!")
                break
            except Exception as e:
                print(f"\n\033[1;31mError:\033[0m {e}")
                print("Please try another question.")

    def show_help(self):
        """Show example questions"""
        print("\n" + "="*70)
        print("EXAMPLE QUESTIONS")
        print("="*70)

        examples = [
            "What is cell placement in chip design?",
            "How does DREAMPlace work?",
            "What's the difference between global and detailed placement?",
            "Why is placement density important?",
            "How do I reduce wirelength in placement?",
            "What are the main challenges in modern placement?",
            "Explain timing-driven placement",
            "What is legalization in placement?",
            "How does GPU acceleration help placement?",
            "What placement quality can OpenROAD achieve?"
        ]

        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")

        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with the Chip Design Placement Apprentice"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./training/models/placement-apprentice-v2',
        help='Path to the trained model directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run on (auto/cpu/cuda)'
    )
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='Single question mode (non-interactive)'
    )

    args = parser.parse_args()

    try:
        # Initialize chat
        chat = ApprenticeChat(args.model, args.device)

        # Single question mode
        if args.question:
            print(f"\nQuestion: {args.question}\n")
            response = chat.generate_response(args.question)
            print(f"Answer: {response}\n")
        else:
            # Interactive chat mode
            chat.chat()

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
