#!/usr/bin/env python3
"""
Separate Training Data by Phase

Takes the general chip design training data and separates it into
phase-specific datasets for training specialist models.

Phases:
- triage: Conversational routing and intent understanding
- specification: Requirements and architecture
- rtl_design: Verilog/SystemVerilog coding
- synthesis: Logic synthesis and optimization
- placement: Physical placement and floorplanning
- routing: Routing and interconnect
- timing_analysis: Static timing analysis
- power_analysis: Power estimation and optimization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from pathlib import Path
from typing import Dict, List
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhaseSeparator:
    """Separate training data into phase-specific datasets"""

    def __init__(self, input_file: str = "./data/training/chip_design_training.jsonl"):
        self.input_file = Path(input_file)
        self.output_dir = self.input_file.parent / "phase_specific"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase keywords for classification
        self.phase_keywords = {
            'triage': [
                'help', 'how do i', 'what is', 'explain', 'can you', 'show me',
                'user intent', 'routing', 'conversation', 'question'
            ],
            'specification': [
                'specification', 'requirements', 'architecture', 'soc', 'system',
                'interface', 'performance target', 'design goals', 'use case'
            ],
            'rtl_design': [
                'verilog', 'systemverilog', 'rtl', 'module', 'always', 'assign',
                'reg', 'wire', 'input', 'output', 'hdl', 'coding', 'design pattern'
            ],
            'synthesis': [
                'synthesis', 'yosys', 'gate', 'technology mapping', 'optimization',
                'area', 'delay', 'logic', 'netlist', 'abc', 'flatten'
            ],
            'placement': [
                'placement', 'dreamplace', 'cell location', 'floorplan', 'wirelength',
                'hpwl', 'density', 'congestion', 'overlap', 'die area'
            ],
            'routing': [
                'routing', 'tritonroute', 'wire', 'via', 'interconnect', 'drc',
                'metal layer', 'track', 'global routing', 'detailed routing'
            ],
            'timing_analysis': [
                'timing', 'sta', 'opensta', 'setup', 'hold', 'slack', 'critical path',
                'wns', 'tns', 'clock', 'delay', 'transition'
            ],
            'power_analysis': [
                'power', 'leakage', 'dynamic power', 'static power', 'switching',
                'clock gating', 'power domain', 'voltage', 'current'
            ],
        }

        # Phase-specific data storage
        self.phase_data = {phase: [] for phase in self.phase_keywords.keys()}
        self.unclassified = []

    def classify_example(self, example: Dict) -> str:
        """
        Classify a training example into a phase.

        Returns:
            Phase name or 'unclassified'
        """
        prompt = example.get('prompt', '').lower()
        response = example.get('response', '').lower()
        combined_text = f"{prompt} {response}"

        # Check for explicit phase in metadata
        if 'metadata' in example and 'phase' in example['metadata']:
            return example['metadata']['phase']

        # Score each phase based on keyword matches
        phase_scores = {}
        for phase, keywords in self.phase_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                phase_scores[phase] = score

        # Return phase with highest score
        if phase_scores:
            return max(phase_scores, key=phase_scores.get)

        # Special case: Verilog code examples
        if 'module ' in response and 'endmodule' in response:
            if any(keyword in combined_text for keyword in ['synthesis', 'yosys', 'gate']):
                return 'synthesis'
            else:
                return 'rtl_design'

        return 'unclassified'

    def separate_data(self):
        """Read input file and separate into phase-specific datasets"""
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return

        logger.info(f"Reading training data from: {self.input_file}")

        example_count = 0
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    phase = self.classify_example(example)

                    if phase == 'unclassified':
                        self.unclassified.append(example)
                    else:
                        self.phase_data[phase].append(example)

                    example_count += 1

                    if example_count % 1000 == 0:
                        logger.info(f"Processed {example_count} examples...")

                except Exception as e:
                    logger.debug(f"Error processing line: {e}")

        logger.info(f"Total examples processed: {example_count}")

    def save_phase_datasets(self):
        """Save each phase's dataset to a separate file"""
        logger.info("\n" + "="*60)
        logger.info("Saving Phase-Specific Datasets")
        logger.info("="*60)

        for phase, examples in self.phase_data.items():
            if not examples:
                logger.info(f"  ⊙ {phase}: 0 examples (skipped)")
                continue

            output_file = self.output_dir / f"{phase}_training.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    # Add phase to metadata
                    if 'metadata' not in example:
                        example['metadata'] = {}
                    example['metadata']['phase'] = phase

                    f.write(json.dumps(example) + '\n')

            logger.info(f"  ✓ {phase}: {len(examples)} examples → {output_file.name}")

        # Save unclassified
        if self.unclassified:
            output_file = self.output_dir / "unclassified_training.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in self.unclassified:
                    f.write(json.dumps(example) + '\n')
            logger.info(f"  ⊙ unclassified: {len(self.unclassified)} examples → {output_file.name}")

    def generate_statistics(self) -> Dict:
        """Generate statistics about phase separation"""
        stats = {
            'total_examples': sum(len(examples) for examples in self.phase_data.values()) + len(self.unclassified),
            'by_phase': {phase: len(examples) for phase, examples in self.phase_data.items()},
            'unclassified': len(self.unclassified),
            'phase_percentages': {}
        }

        total = stats['total_examples']
        if total > 0:
            for phase, count in stats['by_phase'].items():
                stats['phase_percentages'][phase] = (count / total) * 100

        return stats

    def save_statistics(self):
        """Save statistics to JSON file"""
        stats = self.generate_statistics()

        stats_file = self.output_dir / "phase_separation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"\n  ✓ Statistics saved to: {stats_file.name}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Phase Separation Statistics")
        logger.info("="*60)
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"\nBy phase:")
        for phase, count in sorted(stats['by_phase'].items(), key=lambda x: x[1], reverse=True):
            pct = stats['phase_percentages'].get(phase, 0)
            logger.info(f"  {phase:20s}: {count:6d} ({pct:5.1f}%)")
        logger.info(f"  {'unclassified':20s}: {stats['unclassified']:6d}")


def main():
    """Run phase separation"""
    logger.info("="*60)
    logger.info("Phase-Specific Training Data Separation")
    logger.info("="*60)

    separator = PhaseSeparator()

    # Separate data
    separator.separate_data()

    # Save datasets
    separator.save_phase_datasets()

    # Save statistics
    separator.save_statistics()

    logger.info("\n" + "="*60)
    logger.info("✅ Phase separation complete!")
    logger.info("="*60)
    logger.info(f"\nPhase-specific datasets saved to: {separator.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review phase datasets in: data/training/phase_specific/")
    logger.info("2. Run: training/finetune_specialist.py --phase synthesis --size 8b")
    logger.info("3. Train each specialist model with its phase-specific dataset")


if __name__ == "__main__":
    main()
