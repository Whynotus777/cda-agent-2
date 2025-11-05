#!/usr/bin/env python3
"""
Sprint 3: Specialist Corpus Training with QLoRA
Train timing analysis specialist on 10 deep examples using 4-bit QLoRA
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import argparse
from pathlib import Path

def load_specialist_corpus(corpus_path: str):
    """Load specialist corpus (PLACEMENT_REFORMED_V2.jsonl)"""

    print(f"Loading specialist corpus: {corpus_path}")

    examples = []
    with open(corpus_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # Format: Q: <prompt>\n\nA: <response>
            prompt = obj['prompt']
            response = obj['response']

            text = f"Q: {prompt}\n\nA: {response}"
            examples.append({'text': text})

    print(f"✓ Loaded {len(examples)} examples")
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='Path to specialist corpus')
    parser.add_argument('--output-dir', type=str, default='./models/placement-specialist-5090-v1')
    parser.add_argument('--base-model', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--use-qlora', type=bool, default=True)
    args = parser.parse_args()

    print("="*80)
    print("SPRINT 3: SPECIALIST TRAINING (QLoRA)")
    print("="*80)

    # Explicit confirmation for Mixtral
    if "Mixtral" in args.base_model or "mixtral" in args.base_model:
        print("🎯 Targeting Mixtral-8x7B with QLoRA")
        print("   (Mixture of Experts architecture - 46.7B params)")
        print()

    print(f"Base model: {args.base_model}")
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max length: {args.max_length}")
    print(f"QLoRA: {args.use_qlora}")
    print()

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()

    # Load corpus
    examples = load_specialist_corpus(args.corpus)
    dataset = Dataset.from_list(examples)

    # Split train/eval (80/20 on 10 examples = 8 train, 2 eval)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    print()

    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Tokenizer loaded")
    print()

    # QLoRA configuration (4-bit quantization)
    print("Configuring QLoRA (4-bit + LoRA adapters)...")
    print("✓ Configuration optimized for 32GB RTX 5090:")
    print("  - 4-bit NF4 quantization (reduces 46.7B params to ~12GB)")
    print("  - Double quantization for additional compression")
    print("  - BFloat16 compute for numerical stability")
    print()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with 4-bit quantization
    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha (scaling)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✓ Model loaded with QLoRA")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"  LoRA rank: {lora_config.r}")
    print(f"  LoRA alpha: {lora_config.lora_alpha}")
    print()

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding=False
        )

    print("Tokenizing corpus...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        remove_columns=['text'],
        desc="Tokenizing train"
    )

    tokenized_eval = eval_dataset.map(
        tokenize_function,
        remove_columns=['text'],
        desc="Tokenizing eval"
    )
    print("✓ Tokenization complete")
    print()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        save_steps=2,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=2,
        bf16=True,  # BF16 for QLoRA
        gradient_accumulation_steps=4,
        save_total_limit=2,
        report_to=[],
        optim="paged_adamw_8bit"  # Memory-efficient optimizer
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator
    )

    # Train
    print("="*80)
    print("TRAINING SPECIALIST (QLoRA)")
    print("="*80)
    print()

    trainer.train()

    # Save
    print()
    print("="*80)
    print("SAVING MODEL")
    print("="*80)

    # Save LoRA adapters
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✓ LoRA adapters saved to {args.output_dir}")
    print(f"  (Base model: {args.base_model})")

    # Final evaluation
    print()
    print("="*80)
    print("FINAL EVALUATION")
    print("="*80)
    eval_results = trainer.evaluate()

    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")

    print()
    print("="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print()
    print("To use the model:")
    print(f"  1. Load base model: {args.base_model}")
    print(f"  2. Load LoRA adapters from: {args.output_dir}")

if __name__ == "__main__":
    main()
