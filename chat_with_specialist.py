#!/usr/bin/env python3
"""
Enhanced Chat Interface for Chip Design Specialist Models

Supports both:
- Full fine-tuned models (GPT-2, etc.)
- LoRA adapters (Mixtral, Llama, etc.) using PEFT
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
from pathlib import Path
import sys
import json

class SpecialistChat:
    """Enhanced chat interface supporting both full models and LoRA adapters"""

    def __init__(self, model_path: str, device: str = "auto", load_in_4bit: bool = False):
        """
        Initialize the chat interface.

        Args:
            model_path: Path to the trained model or adapter
            device: Device to run on ('cpu', 'cuda', or 'auto')
            load_in_4bit: Use 4-bit quantization for large models
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

        # Check if this is a PEFT adapter
        adapter_config_path = self.model_path / "adapter_config.json"
        self.is_adapter = adapter_config_path.exists()

        if self.is_adapter:
            self._load_adapter_model(load_in_4bit)
        else:
            self._load_full_model()

        print(f"✓ Model loaded successfully!")
        print(f"  Type: {'LoRA Adapter' if self.is_adapter else 'Full Model'}")
        print(f"  Parameters: {self.model.num_parameters() / 1e9:.1f}B")
        print()

    def _load_adapter_model(self, load_in_4bit: bool):
        """Load a PEFT LoRA adapter model"""
        print("Detected LoRA adapter model")

        # Load adapter config to get base model
        with open(self.model_path / "adapter_config.json", 'r') as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path")
        print(f"Base model: {base_model_name}")

        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit and self.device == "cuda":
            print("Loading with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load base model
        print(f"Loading base model: {base_model_name}")
        print("(This may take a few minutes on first run...)")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load adapter
        print(f"Loading LoRA adapter from {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))

        if not load_in_4bit:
            self.model.to(self.device)

        self.model.eval()

    def _load_full_model(self):
        """Load a full fine-tuned model"""
        print("Loading full model...")

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

    def generate_response(self, question: str, max_new_tokens: int = 300) -> str:
        """
        Generate response to a question.

        Args:
            question: User's question
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        # Format prompt
        if self.is_adapter:
            # Use instruction format for Mixtral/Llama
            prompt = f"[INST] You are an expert in chip design and EDA tools. Answer the following question concisely and accurately.\n\nQuestion: {question} [/INST]"
        else:
            # Use Q&A format for GPT-2
            prompt = f"Q: {question}\n\nA:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move to device if not using device_map
        if not self.is_adapter or self.device == "cpu":
            inputs = inputs.to(self.device)

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

        # Extract answer
        if self.is_adapter:
            # For instruction models, get everything after [/INST]
            if "[/INST]" in generated:
                answer = generated.split("[/INST]", 1)[1].strip()
            else:
                answer = generated
        else:
            # For Q&A models
            if "A:" in generated:
                answer = generated.split("A:", 1)[1].strip()
                if "\nQ:" in answer:
                    answer = answer.split("\nQ:")[0].strip()
            else:
                answer = generated

        return answer

    def chat(self):
        """Run interactive chat loop"""
        print("="*70)
        print("CHIP DESIGN SPECIALIST - CHAT INTERFACE")
        print("="*70)
        print()
        print("I'm a specialized AI trained on chip design knowledge.")
        print("Ask me about placement, routing, timing, EDA tools, and more!")
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
                print("\n\033[1;32mSpecialist:\033[0m ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for chatting about chip design!")
                break
            except Exception as e:
                print(f"\n\033[1;31mError:\033[0m {e}")
                import traceback
                traceback.print_exc()
                print("Please try another question.")

    def show_help(self):
        """Show example questions"""
        print("\n" + "="*70)
        print("EXAMPLE QUESTIONS")
        print("="*70)

        examples = [
            "What is cell placement in chip design?",
            "How does DREAMPlace achieve GPU acceleration?",
            "Explain the difference between global and detailed placement",
            "Why is placement density important?",
            "How do I reduce wirelength in placement?",
            "What are the main challenges in modern placement?",
            "Explain timing-driven placement",
            "What is legalization in placement?",
            "How does GPU acceleration help placement?",
            "What placement quality can OpenROAD achieve?",
            "Why is wire delay so high on this net?",
            "How do I fix setup timing violations?"
        ]

        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")

        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with Chip Design Specialist Models"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./training/models/wisdom-specialist-v3-mixtral',
        help='Path to the model or adapter directory'
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
    parser.add_argument(
        '--4bit',
        action='store_true',
        help='Use 4-bit quantization (for large models on GPU)'
    )

    args = parser.parse_args()

    try:
        # Initialize chat
        chat = SpecialistChat(args.model, args.device, args.__dict__.get('4bit', False))

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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
