#!/usr/bin/env python3
"""
Add hierarchy metadata (L1/L2/L3) to legacy datasets using the
HierarchicalClassifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from classify_hierarchical_v5_2 import HierarchicalClassifier


def annotate_dataset(input_path: Path, output_path: Path) -> dict:
    """Annotate dataset with hierarchy fields and write to output."""
    classifier = HierarchicalClassifier()
    records = []

    with input_path.open() as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    annotated = []
    for ex in records:
        classification = classifier.classify(ex)
        hierarchy = {
            "l1": classification["l1"],
            "l2": classification["l2"],
            "l3": classification["l3"],
            "l1_confidence": classification.get("l1_confidence", 1.0),
            "l2_confidence": classification.get("l2_confidence", 1.0),
            "l3_confidence": classification.get("l3_confidence", 0.9),
            "method": classification.get("method", "heuristic"),
        }
        ex["hierarchy"] = hierarchy
        ex["l1"] = hierarchy["l1"]
        ex["l2"] = hierarchy["l2"]
        ex["l3"] = hierarchy["l3"]
        metadata = ex.get("metadata") or {}
        metadata.setdefault("hierarchy_method", hierarchy["method"])
        ex["metadata"] = metadata
        annotated.append(ex)

    with output_path.open("w") as outfile:
        for ex in annotated:
            outfile.write(json.dumps(ex) + "\n")

    counts = {}
    for ex in annotated:
        counts[ex["hierarchy"]["l2"]] = counts.get(ex["hierarchy"]["l2"], 0) + 1

    return {"total": len(annotated), "l2_counts": counts}


def main():
    parser = argparse.ArgumentParser(description="Add hierarchy metadata to dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    args = parser.parse_args()

    stats = annotate_dataset(args.input, args.output)
    print(f"Annotated {stats['total']} examples from {args.input} -> {args.output}")
    print("L2 distribution:")
    for l2, count in sorted(stats["l2_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {l2:15s}: {count}")


if __name__ == "__main__":
    main()

