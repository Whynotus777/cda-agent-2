#!/usr/bin/env python3
"""
Test Placement Apprentice - Evaluate Specialized Intelligence

Tests the trained apprentice on diverse placement questions it has never seen,
comparing against a baseline model to measure specialized learning.

The goal: Prove the apprentice has internalized placement knowledge, not just memorized.
"""

import argparse
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 10 Diverse Test Questions - Covering All Domains
TEST_QUESTIONS = [
    {
        "id": 1,
        "domain": "High-Performance CPU",
        "question": "For a 2 GHz out-of-order RISC-V processor at 28nm, what placement density should I target and why?",
        "expected_concepts": ["0.60-0.70", "low density", "timing closure", "buffer insertion", "critical path"]
    },
    {
        "id": 2,
        "domain": "Macro Placement",
        "question": "I have an SoC with 30 SRAM macros. Should I use automated macro placement or manual floorplanning?",
        "expected_concepts": ["manual", "10-50 macros", "ML-based", "Circuit Training", "floorplan"]
    },
    {
        "id": 3,
        "domain": "Power-Aware",
        "question": "What is the difference between clock gating and power gating in terms of power savings?",
        "expected_concepts": ["clock gating", "30-50%", "dynamic power", "power gating", "100×", "leakage"]
    },
    {
        "id": 4,
        "domain": "High-Density",
        "question": "Why does placement become significantly harder above 85% utilization?",
        "expected_concepts": ["routing congestion", "legalization", "overflow", "white space", "routing capacity"]
    },
    {
        "id": 5,
        "domain": "Routability",
        "question": "What is RUDY and how is it used during placement optimization?",
        "expected_concepts": ["RUDY", "congestion estimation", "wire density", "routing demand", "hotspots"]
    },
    {
        "id": 6,
        "domain": "Timing-Driven",
        "question": "How does multi-Vt cell assignment help with both timing and power?",
        "expected_concepts": ["LVt", "critical paths", "HVt", "non-critical", "leakage", "5×"]
    },
    {
        "id": 7,
        "domain": "Real-World",
        "question": "What placement quality can OpenROAD achieve compared to commercial tools?",
        "expected_concepts": ["90-95%", "8%", "600 tapeouts", "commercial", "open-source"]
    },
    {
        "id": 8,
        "domain": "Contest Benchmarks",
        "question": "What was the key innovation in NTUPlacerDR that won ISPD 2015?",
        "expected_concepts": ["routability", "cell inflation", "RUDY", "congestion", "overflow"]
    },
    {
        "id": 9,
        "domain": "Advanced Nodes",
        "question": "What makes placement at 7nm fundamentally different from 28nm?",
        "expected_concepts": ["fewer metal layers", "via resistance", "wire delay", "timing", "density lower"]
    },
    {
        "id": 10,
        "domain": "Production",
        "question": "How many place-route iterations should I budget for a production tapeout?",
        "expected_concepts": ["3-5", "iterations", "timing closure", "convergence", "95%"]
    }
]


class ApprenticeEvaluator:
    """
    Evaluates apprentice model against baseline to measure specialized learning.
    """

    def __init__(
        self,
        apprentice_path: str,
        baseline_model: str = "distilgpt2"
    ):
        """
        Initialize evaluator.

        Args:
            apprentice_path: Path to trained apprentice model
            baseline_model: Baseline model for comparison
        """
        self.apprentice_path = Path(apprentice_path)
        self.baseline_model = baseline_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

    def load_models(self):
        """Load apprentice and baseline models"""
        logger.info(f"Loading apprentice from {self.apprentice_path}")

        try:
            self.apprentice_tokenizer = AutoTokenizer.from_pretrained(str(self.apprentice_path))
            self.apprentice_model = AutoModelForCausalLM.from_pretrained(str(self.apprentice_path))
            self.apprentice_model.to(self.device)
            self.apprentice_model.eval()
            logger.info("✓ Apprentice loaded")
        except Exception as e:
            logger.error(f"Failed to load apprentice: {e}")
            raise

        logger.info(f"Loading baseline {self.baseline_model}")
        try:
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
            self.baseline_model_obj = AutoModelForCausalLM.from_pretrained(self.baseline_model)
            self.baseline_model_obj.to(self.device)
            self.baseline_model_obj.eval()
            logger.info("✓ Baseline loaded")
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            raise

    def generate_response(self, model, tokenizer, question: str, max_length=256):
        """
        Generate response from model.

        Args:
            model: Language model
            tokenizer: Tokenizer
            question: Input question
            max_length: Maximum generation length

        Returns:
            Generated text
        """
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Format as Q&A
        prompt = f"Q: {question}\n\nA:"

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,  # Don't pad single input
            truncation=True,
            max_length=256
        ).to(self.device)

        # Generate with adjusted parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=200,  # Generate up to 200 new tokens
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetition
                no_repeat_ngram_size=3
            )

        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (remove prompt)
        if "A:" in generated:
            answer = generated.split("A:", 1)[1].strip()
            # Stop at next Q: if present
            if "\nQ:" in answer:
                answer = answer.split("\nQ:")[0].strip()
        else:
            answer = generated

        return answer

    def evaluate_response(self, response: str, expected_concepts: list) -> dict:
        """
        Evaluate response quality.

        Args:
            response: Generated response
            expected_concepts: List of expected concepts/keywords

        Returns:
            Evaluation dict with scores
        """
        response_lower = response.lower()

        # Count concept matches
        matches = []
        for concept in expected_concepts:
            if concept.lower() in response_lower:
                matches.append(concept)

        concept_score = len(matches) / len(expected_concepts) if expected_concepts else 0

        # Length check (not too short, not too long)
        length = len(response.split())
        length_score = 1.0 if 50 <= length <= 300 else (0.5 if length < 50 else 0.7)

        # Overall score
        overall_score = (concept_score * 0.7) + (length_score * 0.3)

        return {
            "concept_coverage": concept_score,
            "matched_concepts": matches,
            "length": length,
            "length_score": length_score,
            "overall_score": overall_score
        }

    def run_test_suite(self):
        """
        Run complete test suite.

        Returns:
            Results dictionary
        """
        logger.info("="*70)
        logger.info("APPRENTICE TEST SUITE - 10 DIVERSE QUESTIONS")
        logger.info("="*70)

        results = {
            "apprentice": [],
            "baseline": [],
            "questions": TEST_QUESTIONS
        }

        for test_case in TEST_QUESTIONS:
            logger.info(f"\n{'='*70}")
            logger.info(f"Test {test_case['id']}: {test_case['domain']}")
            logger.info(f"Question: {test_case['question']}")
            logger.info(f"{'='*70}")

            # Generate from apprentice
            logger.info("\n🎓 APPRENTICE:")
            apprentice_response = self.generate_response(
                self.apprentice_model,
                self.apprentice_tokenizer,
                test_case['question']
            )
            logger.info(f"{apprentice_response}\n")

            # Evaluate apprentice
            apprentice_eval = self.evaluate_response(
                apprentice_response,
                test_case['expected_concepts']
            )
            logger.info(f"Apprentice score: {apprentice_eval['overall_score']:.2f}")
            logger.info(f"Concepts found: {apprentice_eval['matched_concepts']}")

            # Generate from baseline
            logger.info("\n📚 BASELINE:")
            baseline_response = self.generate_response(
                self.baseline_model_obj,
                self.baseline_tokenizer,
                test_case['question']
            )
            logger.info(f"{baseline_response}\n")

            # Evaluate baseline
            baseline_eval = self.evaluate_response(
                baseline_response,
                test_case['expected_concepts']
            )
            logger.info(f"Baseline score: {baseline_eval['overall_score']:.2f}")
            logger.info(f"Concepts found: {baseline_eval['matched_concepts']}")

            # Compare
            improvement = apprentice_eval['overall_score'] - baseline_eval['overall_score']
            logger.info(f"\n📊 IMPROVEMENT: {improvement:+.2f}")

            # Store results
            results["apprentice"].append({
                "question_id": test_case['id'],
                "response": apprentice_response,
                "evaluation": apprentice_eval
            })

            results["baseline"].append({
                "question_id": test_case['id'],
                "response": baseline_response,
                "evaluation": baseline_eval
            })

        return results

    def analyze_results(self, results: dict):
        """
        Analyze and report results.

        Args:
            results: Test results
        """
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS: SPECIALIZED INTELLIGENCE ASSESSMENT")
        logger.info("="*70)

        apprentice_scores = [r['evaluation']['overall_score'] for r in results['apprentice']]
        baseline_scores = [r['evaluation']['overall_score'] for r in results['baseline']]

        apprentice_avg = sum(apprentice_scores) / len(apprentice_scores)
        baseline_avg = sum(baseline_scores) / len(baseline_scores)

        apprentice_concepts = sum([r['evaluation']['concept_coverage'] for r in results['apprentice']]) / len(results['apprentice'])
        baseline_concepts = sum([r['evaluation']['concept_coverage'] for r in results['baseline']]) / len(results['baseline'])

        logger.info(f"\n📈 OVERALL SCORES:")
        logger.info(f"  Apprentice: {apprentice_avg:.3f}")
        logger.info(f"  Baseline:   {baseline_avg:.3f}")
        logger.info(f"  Improvement: {(apprentice_avg - baseline_avg):+.3f} ({((apprentice_avg - baseline_avg) / baseline_avg * 100):+.1f}%)")

        logger.info(f"\n🎯 CONCEPT COVERAGE:")
        logger.info(f"  Apprentice: {apprentice_concepts:.1%}")
        logger.info(f"  Baseline:   {baseline_concepts:.1%}")
        logger.info(f"  Improvement: {(apprentice_concepts - baseline_concepts):+.1%}")

        # Domain analysis
        logger.info(f"\n📊 DOMAIN-WISE PERFORMANCE:")
        for i, test_case in enumerate(results['questions']):
            app_score = results['apprentice'][i]['evaluation']['overall_score']
            base_score = results['baseline'][i]['evaluation']['overall_score']
            improvement = app_score - base_score

            status = "✓" if improvement > 0 else "✗"
            logger.info(f"  {status} {test_case['domain']}: {improvement:+.2f}")

        # Verdict
        logger.info(f"\n" + "="*70)
        logger.info("VERDICT:")

        if apprentice_avg > baseline_avg + 0.1:
            logger.info("✓ SPECIALIZED INTELLIGENCE DETECTED")
            logger.info("  The apprentice demonstrates measurable placement expertise")
            logger.info("  beyond the baseline model. It has learned domain knowledge.")
        elif apprentice_avg > baseline_avg:
            logger.info("⚠ MARGINAL IMPROVEMENT")
            logger.info("  The apprentice shows slight improvement but needs more training")
            logger.info("  or higher quality examples to demonstrate strong specialization.")
        else:
            logger.info("✗ NO SPECIALIZATION")
            logger.info("  The apprentice does not outperform baseline. Training may")
            logger.info("  have failed or the corpus lacks sufficient signal.")

        logger.info("="*70)

        return {
            "apprentice_avg": apprentice_avg,
            "baseline_avg": baseline_avg,
            "improvement": apprentice_avg - baseline_avg,
            "apprentice_concepts": apprentice_concepts,
            "baseline_concepts": baseline_concepts,
            "verdict": "specialized" if apprentice_avg > baseline_avg + 0.1 else ("marginal" if apprentice_avg > baseline_avg else "no_specialization")
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test Placement Apprentice on diverse questions"
    )
    parser.add_argument(
        '--apprentice',
        type=str,
        required=True,
        help='Path to trained apprentice model'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default="distilgpt2",
        help='Baseline model for comparison'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="./test_results.json",
        help='Output file for results'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ApprenticeEvaluator(
        apprentice_path=args.apprentice,
        baseline_model=args.baseline
    )

    # Load models
    evaluator.load_models()

    # Run tests
    results = evaluator.run_test_suite()

    # Analyze
    summary = evaluator.analyze_results(results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            "summary": summary,
            "detailed_results": results
        }, f, indent=2)

    logger.info(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
