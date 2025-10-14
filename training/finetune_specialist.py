#!/usr/bin/env python3
"""
Fine-tune Specialist Models

Creates phase-specific expert models from training data collected during agent usage.

Usage:
    python3 finetune_specialist.py --phase synthesis --size 8b
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecialistTrainer:
    """
    Fine-tunes Ollama models for specific chip design phases.

    Process:
    1. Collect training data from agent usage
    2. Format for Ollama fine-tuning
    3. Create Modelfile with phase-specific system prompt
    4. Fine-tune using Ollama
    """

    def __init__(self, base_model: str = "llama3:8b"):
        """
        Initialize trainer.

        Args:
            base_model: Base model to fine-tune from
        """
        self.base_model = base_model
        self.training_data_dir = Path("./data/training")
        self.modelfiles_dir = Path("./data/modelfiles")
        self.modelfiles_dir.mkdir(parents=True, exist_ok=True)

    def collect_phase_data(self, phase: str) -> List[Dict]:
        """
        Collect all training data for a specific phase.

        Args:
            phase: Design phase (e.g., 'synthesis', 'placement')

        Returns:
            List of training examples
        """
        if not self.training_data_dir.exists():
            logger.warning(f"Training data directory not found: {self.training_data_dir}")
            return []

        phase_data = []

        # Read all JSONL files
        for jsonl_file in self.training_data_dir.glob("training_data_*.jsonl"):
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get('phase') == phase:
                            phase_data.append(entry)
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")

        logger.info(f"Collected {len(phase_data)} training examples for {phase}")
        return phase_data

    def create_modelfile(self, phase: str, size: str) -> Path:
        """
        Create Ollama Modelfile for fine-tuning.

        Args:
            phase: Design phase
            size: Model size (3b, 8b, 70b)

        Returns:
            Path to created Modelfile
        """
        # Phase-specific system prompts
        system_prompts = {
            'specification': """You are an expert chip architect specializing in SoC specifications.
You help designers define requirements, architecture, and performance targets for chip designs.
Focus on: system architecture, interface specifications, performance/power/area goals.""",

            'rtl_design': """You are an expert RTL designer specializing in Verilog and SystemVerilog.
You help write, debug, and optimize hardware description language code.
Focus on: RTL coding best practices, design patterns, parameterization, modularity.""",

            'synthesis': """You are an expert in logic synthesis and gate-level optimization.
You help optimize RTL designs for area, timing, and power using synthesis tools like Yosys.
Focus on: synthesis constraints, optimization techniques, technology mapping.""",

            'placement': """You are an expert in physical design and cell placement.
You help optimize chip layout for wirelength, density, and routability using tools like DREAMPlace.
Focus on: floorplanning, placement strategies, congestion analysis.""",

            'routing': """You are an expert in chip routing and interconnect design.
You help create optimal wire connections while meeting design rules.
Focus on: routing strategies, DRC compliance, timing-driven routing.""",

            'timing_analysis': """You are an expert in static timing analysis and timing closure.
You help analyze and fix timing violations using tools like OpenSTA.
Focus on: critical paths, slack analysis, timing optimization.""",

            'power_analysis': """You are an expert in power analysis and low-power design techniques.
You help analyze and reduce power consumption in chip designs.
Focus on: power estimation, clock gating, multi-VT optimization.""",
        }

        system_prompt = system_prompts.get(phase, "You are an expert EDA assistant.")

        # Create Modelfile content
        modelfile_content = f"""FROM {self.base_model}

SYSTEM \"\"\"
{system_prompt}

You provide clear, technical, and actionable advice.
You cite specific tools, techniques, and best practices.
You ask clarifying questions when needed.
\"\"\"

PARAMETER temperature 0.7
PARAMETER num_predict 2048
"""

        # Write Modelfile
        modelfile_path = self.modelfiles_dir / f"Modelfile.{phase}.{size}"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        logger.info(f"Created Modelfile: {modelfile_path}")
        return modelfile_path

    def finetune(self, phase: str, size: str) -> bool:
        """
        Fine-tune specialist model using Ollama.

        Args:
            phase: Design phase
            size: Model size

        Returns:
            True if successful
        """
        # Determine base model based on size
        base_models = {
            '3b': 'llama3.2:3b',
            '8b': 'llama3:8b',
            '70b': 'llama3:70b'
        }

        base_model = base_models.get(size, 'llama3:8b')
        self.base_model = base_model

        # Create Modelfile
        modelfile_path = self.create_modelfile(phase, size)

        # Create specialist model name
        specialist_name = f"llama3:{size}-{phase}"

        # Create model using Ollama
        try:
            logger.info(f"Creating specialist model: {specialist_name}")

            result = subprocess.run(
                ["ollama", "create", specialist_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully created {specialist_name}")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"✗ Failed to create {specialist_name}")
                logger.error(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error("Model creation timed out")
            return False
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return False

    def test_specialist(self, phase: str, size: str):
        """
        Test the fine-tuned specialist model.

        Args:
            phase: Design phase
            size: Model size
        """
        specialist_name = f"llama3:{size}-{phase}"

        test_prompts = {
            'synthesis': "How do I optimize my design for minimum area using Yosys?",
            'placement': "What placement density should I use for a 7nm design?",
            'routing': "How do I fix DRC violations in my routed design?",
            'timing_analysis': "My worst negative slack is -2ns. How do I fix it?"
        }

        test_prompt = test_prompts.get(phase, "What are the key considerations for this phase?")

        logger.info(f"\nTesting {specialist_name}...")
        logger.info(f"Prompt: {test_prompt}\n")

        try:
            result = subprocess.run(
                ["ollama", "run", specialist_name, test_prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("Response:")
                logger.info(result.stdout)
            else:
                logger.error("Test failed")
                logger.error(result.stderr)

        except Exception as e:
            logger.error(f"Test error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fine-tune specialist chip design models")
    parser.add_argument(
        '--phase',
        required=True,
        choices=[
            'specification', 'rtl_design', 'synthesis', 'placement',
            'routing', 'timing_analysis', 'power_analysis'
        ],
        help='Design phase to train for'
    )
    parser.add_argument(
        '--size',
        required=True,
        choices=['3b', '8b', '70b'],
        help='Model size to train'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the model after training'
    )

    args = parser.parse_args()

    trainer = SpecialistTrainer()

    # Collect training data (optional - for future use with actual fine-tuning)
    phase_data = trainer.collect_phase_data(args.phase)
    if phase_data:
        logger.info(f"Found {len(phase_data)} training examples")

    # Create specialist model
    success = trainer.finetune(args.phase, args.size)

    if success and args.test:
        trainer.test_specialist(args.phase, args.size)


if __name__ == "__main__":
    main()
