#!/usr/bin/env python3
"""
Train Placement Apprentice - True Fine-Tuning on Gold 100

This script performs actual neural network fine-tuning on the placement corpus,
creating a specialized model that has internalized placement knowledge.

Usage:
    python3 train_placement_apprentice.py --corpus ../data/training/PLACEMENT_GOLD_STANDARD.jsonl
"""

import argparse
import json
import logging
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlacementApprenticeTrainer:
    """
    Fine-tunes a language model on placement knowledge using the Gold Standard corpus.

    The goal: Create a model that demonstrates specialized placement intelligence,
    not just memorization.
    """

    def __init__(
        self,
        base_model: str = "gpt2",
        corpus_path: str = "../data/training/PLACEMENT_GOLD_STANDARD.jsonl",
        output_dir: str = "./models/placement-apprentice-v1"
    ):
        """
        Initialize trainer.

        Args:
            base_model: Base model to fine-tune (gpt2, gpt2-medium, distilgpt2)
            corpus_path: Path to Gold 100 JSONL corpus
            output_dir: Where to save trained model
        """
        self.base_model = base_model
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing Placement Apprentice Trainer")
        logger.info(f"Base model: {base_model}")
        logger.info(f"Corpus: {corpus_path}")

        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training device: {self.device}")
        if self.device == "cpu":
            logger.warning("⚠️  Training on CPU - this will be slow. GPU recommended.")

    def load_corpus(self):
        """
        Load and validate Gold Standard corpus.

        Returns:
            List of training examples
        """
        logger.info(f"Loading corpus from {self.corpus_path}")

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")

        examples = []
        with open(self.corpus_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)

                    # Validate required fields
                    if 'prompt' not in entry or 'response' not in entry:
                        logger.warning(f"Line {line_num}: Missing prompt/response")
                        continue

                    # Format as Q&A pair for training
                    text = f"Q: {entry['prompt']}\n\nA: {entry['response']}"

                    examples.append({
                        'text': text,
                        'provenance_score': entry.get('provenance_score', 0),
                        'complexity_score': entry.get('complexity_score', 0),
                        'total_score': entry.get('total_score', 0)
                    })

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Line {line_num}: Unexpected error: {e}")

        logger.info(f"✓ Loaded {len(examples)} training examples")

        # Corpus statistics
        avg_provenance = np.mean([e['provenance_score'] for e in examples])
        avg_complexity = np.mean([e['complexity_score'] for e in examples])
        avg_total = np.mean([e['total_score'] for e in examples])

        logger.info(f"Corpus quality:")
        logger.info(f"  Avg provenance score: {avg_provenance:.1f}")
        logger.info(f"  Avg complexity score: {avg_complexity:.1f}")
        logger.info(f"  Avg total score: {avg_total:.1f}")

        return examples

    def prepare_dataset(self, examples, train_split=0.9):
        """
        Prepare HuggingFace dataset from examples.

        Args:
            examples: List of training examples
            train_split: Fraction for training (rest for validation)

        Returns:
            train_dataset, val_dataset
        """
        logger.info("Preparing datasets...")

        # Split train/val
        n_train = int(len(examples) * train_split)
        train_examples = examples[:n_train]
        val_examples = examples[n_train:]

        logger.info(f"Train examples: {len(train_examples)}")
        logger.info(f"Val examples: {len(val_examples)}")

        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict({
            'text': [e['text'] for e in train_examples]
        })

        val_dataset = Dataset.from_dict({
            'text': [e['text'] for e in val_examples]
        })

        return train_dataset, val_dataset

    def train(self, num_epochs=3, batch_size=4, learning_rate=5e-5):
        """
        Execute the training.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size (reduce if OOM)
            learning_rate: Learning rate for AdamW

        Returns:
            Trained model and tokenizer
        """
        logger.info("="*60)
        logger.info("BEGINNING APPRENTICE TRAINING")
        logger.info("="*60)

        # Load corpus
        examples = self.load_corpus()
        train_dataset, val_dataset = self.prepare_dataset(examples)

        # Load tokenizer and model
        logger.info(f"Loading {self.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Set pad token (GPT-2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.base_model)
        model.to(self.device)

        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )

        logger.info("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",  # Disable wandb, tensorboard
            fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
        )

        # Train!
        logger.info("🚀 Starting training...")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Total steps: {len(tokenized_train) // batch_size * num_epochs}")

        train_result = trainer.train()

        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        # Save training metrics
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)

        logger.info("="*60)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Final train loss: {train_result.metrics['train_loss']:.4f}")
        logger.info(f"Model saved to: {self.output_dir}")

        return model, tokenizer


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Placement Apprentice on Gold 100 corpus"
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default="../data/training/PLACEMENT_GOLD_STANDARD.jsonl",
        help='Path to Gold 100 JSONL corpus'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default="gpt2",
        choices=['gpt2', 'gpt2-medium', 'distilgpt2'],
        help='Base model to fine-tune'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="./models/placement-apprentice-v1",
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = PlacementApprenticeTrainer(
        base_model=args.base_model,
        corpus_path=args.corpus,
        output_dir=args.output_dir
    )

    # Train
    model, tokenizer = trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    logger.info("\n🎓 Apprentice is ready for testing!")
    logger.info(f"Load with: AutoModelForCausalLM.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
