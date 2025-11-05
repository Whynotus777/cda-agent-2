#!/usr/bin/env python3
"""
Sprint 3: Specialist Corpus Training on Mistral-7B
Train timing analysis specialist on 10 deep examples
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

    print(f"Loaded {len(examples)} examples")
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='Path to specialist corpus')
    parser.add_argument('--output-dir', type=str, default='./models/placement-specialist-v1')
    parser.add_argument('--base-model', type=str, default='gradientai/Llama-3-8B-Instruct-Gradient-1048k')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=4096)  # 4k sufficient for specialist examples
    args = parser.parse_args()

    print("="*70)
    print("SPRINT 3: SPECIALIST CORPUS TRAINING")
    print("="*70)
    print(f"Base model: {args.base_model}")
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max length: {args.max_length}")
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

    # Load tokenizer and model
    print(f"Loading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 8-bit quantization configuration for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    ) if device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Enable gradient checkpointing to reduce memory
    model.gradient_checkpointing_enable()

    print(f"Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")
    print("8-bit quantization enabled")
    print("Gradient checkpointing enabled")
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
        fp16=False,  # Disabled for 8-bit quantization
        gradient_accumulation_steps=4,
        save_total_limit=2,
        report_to=[]
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
    print("="*70)
    print("TRAINING SPECIALIST")
    print("="*70)
    print()

    trainer.train()

    # Save
    print()
    print("="*70)
    print("SAVING MODEL")
    print("="*70)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✓ Model saved to {args.output_dir}")

    # Final evaluation
    print()
    print("="*70)
    print("FINAL EVALUATION")
    print("="*70)
    eval_results = trainer.evaluate()

    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")

    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
