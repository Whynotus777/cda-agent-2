#!/usr/bin/env python3
"""
Fine-tune 8B Model on Chip Design

Creates a chip design specialized model from llama3:8b using Ollama.

The fine-tuned model will have deep knowledge of:
- EDA tools (Yosys, OpenROAD, DREAMPlace)
- Verilog/SystemVerilog
- Chip design flow
- Optimization techniques
- Process technologies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import subprocess
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChipDesignFineTuner:
    """
    Fine-tune llama3:8b on chip design data using Ollama.
    """

    def __init__(
        self,
        training_data: str = "./data/training/chip_design_training.jsonl",
        base_model: str = "llama3:8b",
        output_model: str = "llama3:8b-chipdesign"
    ):
        self.training_data = Path(training_data)
        self.base_model = base_model
        self.output_model = output_model

        self.modelfile_path = Path("./training/Modelfile.chipdesign")

    def check_prerequisites(self) -> bool:
        """Check if everything is ready for fine-tuning"""
        logger.info("Checking prerequisites...")

        # Check training data exists
        if not self.training_data.exists():
            logger.error(f"Training data not found: {self.training_data}")
            logger.info("Run: python training/data_preparation/prepare_training_data.py")
            return False

        # Check Ollama is installed
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Ollama not found or not running")
                return False
        except FileNotFoundError:
            logger.error("Ollama not installed")
            return False

        # Check base model exists
        if self.base_model not in result.stdout:
            logger.error(f"Base model {self.base_model} not found")
            logger.info(f"Run: ollama pull {self.base_model}")
            return False

        # Count training examples
        with open(self.training_data, 'r') as f:
            num_examples = sum(1 for line in f)

        logger.info(f"✓ Training data: {num_examples} examples")
        logger.info(f"✓ Base model: {self.base_model}")
        logger.info(f"✓ Ollama ready")

        return True

    def create_modelfile(self):
        """
        Create Ollama Modelfile for fine-tuning.

        Format:
        FROM base_model
        ADAPTER ./path/to/training/data.jsonl
        PARAMETER temperature 0.7
        SYSTEM "You are an expert chip design assistant..."
        """
        logger.info("Creating Modelfile...")

        system_prompt = """You are an expert chip design and EDA assistant with deep knowledge of:
- VLSI design flow (specification, RTL design, synthesis, placement, routing, verification)
- EDA tools (Yosys, OpenROAD, DREAMPlace, Magic, OpenSTA)
- HDL languages (Verilog, SystemVerilog, VHDL)
- Process technologies (7nm, 5nm, advanced nodes)
- Design optimization (timing, power, area)
- CPU/SoC architectures (RISC-V, ARM, custom accelerators)

You provide accurate, detailed, technical answers grounded in chip design best practices."""

        modelfile_content = f"""FROM {self.base_model}

# Training data
ADAPTER {self.training_data.absolute()}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt
SYSTEM \"\"\"{ system_prompt}\"\"\"

# Template
TEMPLATE \"\"\"{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"
"""

        with open(self.modelfile_path, 'w') as f:
            f.write(modelfile_content)

        logger.info(f"✓ Modelfile created: {self.modelfile_path}")

    def finetune(self):
        """
        Run fine-tuning using Ollama.

        Note: Ollama's ADAPTER directive handles the actual fine-tuning.
        This creates a new model with the training data incorporated.
        """
        logger.info("="*60)
        logger.info(f"Fine-tuning {self.base_model} → {self.output_model}")
        logger.info("="*60)

        logger.info("\nThis will create a chip design specialized model.")
        logger.info("Training time depends on data size and hardware.")
        logger.info("\nStarting fine-tuning...\n")

        try:
            # Create model using Modelfile
            result = subprocess.run(
                ['ollama', 'create', self.output_model, '-f', str(self.modelfile_path)],
                capture_output=False,  # Show output in real-time
                text=True
            )

            if result.returncode == 0:
                logger.info("\n" + "="*60)
                logger.info("✅ Fine-tuning complete!")
                logger.info("="*60)
                logger.info(f"Model created: {self.output_model}")
                logger.info(f"\nTest it with:")
                logger.info(f"  ollama run {self.output_model} 'What is synthesis?'")
                return True
            else:
                logger.error("\n❌ Fine-tuning failed")
                return False

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False

    def test_model(self):
        """Test the fine-tuned model with chip design questions"""
        logger.info("\n" + "="*60)
        logger.info("Testing Fine-tuned Model")
        logger.info("="*60)

        test_questions = [
            "What is synthesis in chip design?",
            "How do I optimize for low power?",
            "Explain placement and routing",
            "What is the difference between setup and hold time?",
        ]

        for question in test_questions:
            logger.info(f"\nQ: {question}")

            try:
                result = subprocess.run(
                    ['ollama', 'run', self.output_model, question],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    response = result.stdout.strip()
                    preview = response[:200] + "..." if len(response) > 200 else response
                    logger.info(f"A: {preview}")
                else:
                    logger.warning("No response")

            except Exception as e:
                logger.error(f"Error testing: {e}")


def main():
    """Run fine-tuning pipeline"""
    logger.info("\n🚀 Chip Design Model Fine-tuning\n")

    tuner = ChipDesignFineTuner()

    # Check prerequisites
    if not tuner.check_prerequisites():
        logger.error("\n❌ Prerequisites not met. Please fix issues above.")
        return

    # Create Modelfile
    tuner.create_modelfile()

    # Confirm before proceeding
    logger.info("\n" + "="*60)
    logger.info("Ready to fine-tune!")
    logger.info("="*60)
    logger.info(f"Base model: {tuner.base_model}")
    logger.info(f"Training data: {tuner.training_data}")
    logger.info(f"Output model: {tuner.output_model}")

    response = input("\nProceed with fine-tuning? (y/n): ")

    if response.lower() != 'y':
        logger.info("Fine-tuning cancelled")
        return

    # Fine-tune
    success = tuner.finetune()

    if success:
        # Test
        tuner.test_model()

        logger.info("\n✅ All done!")
        logger.info(f"\nYour chip design model is ready: {tuner.output_model}")
        logger.info("\nUpdate your config to use this model:")
        logger.info(f"  model_name: '{tuner.output_model}'")


if __name__ == "__main__":
    main()
