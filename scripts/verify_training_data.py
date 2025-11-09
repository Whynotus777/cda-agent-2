#!/usr/bin/env python3
"""
Verify Existing Training Data with Semantic Grader
Filters RTL training examples using Yosys + Verilator verification
"""
from pathlib import Path
import json
import sys
from datetime import datetime
from rtl_verifier import RTLVerifier

def verify_dataset(dataset_path: Path, output_path: Path, score_threshold: float = 0.7):
    """
    Verify all examples in dataset and filter by score

    Args:
        dataset_path: Path to input JSONL dataset
        output_path: Path to save filtered dataset
        score_threshold: Minimum score to keep (0.0-1.0)
    """
    verifier = RTLVerifier()

    if not verifier.yosys_available:
        print("❌ ERROR: Yosys not available. Cannot verify RTL code.")
        sys.exit(1)

    print("="*80)
    print("  RTL TRAINING DATA VERIFICATION")
    print("="*80)
    print(f"\n📂 Dataset: {dataset_path}")
    print(f"📂 Output: {output_path}")
    print(f"⚖️  Score threshold: {score_threshold}")
    print(f"🔧 Verifier tools: Yosys {'✓' if verifier.yosys_available else '✗'}, Verilator {'✓' if verifier.verilator_available else '✗'}")

    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    # Load and verify all examples
    good_examples = []
    bad_examples = []
    verification_logs = []

    print(f"\n🔍 Verifying examples...")
    start_time = datetime.now()

    with dataset_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                ex = json.loads(line)

                # Extract RTL code and spec
                rtl_code = ex.get('output', '')
                spec = ex.get('instruction', '') + ' ' + ex.get('input', '')

                # Skip if no RTL code
                if not rtl_code or 'module' not in rtl_code:
                    bad_examples.append({
                        'example': ex,
                        'result': None,
                        'reason': 'No RTL code found'
                    })
                    continue

                # Run verifier
                result = verifier.verify(
                    rtl_code=rtl_code,
                    spec=spec
                )

                # Log verification result
                log_entry = {
                    'line_num': line_num,
                    'score': result.score,
                    'syntax_valid': result.syntax_valid,
                    'synthesis_success': result.synthesis_success,
                    'all_io_used': result.all_io_used,
                    'errors': result.errors,
                    'warnings': result.warnings[:3] if result.warnings else [],  # Limit warnings
                    'instruction': ex.get('instruction', '')[:100]  # First 100 chars
                }
                verification_logs.append(log_entry)

                # Filter by score
                if result.score >= score_threshold:
                    good_examples.append({
                        'example': ex,
                        'result': result
                    })
                else:
                    bad_examples.append({
                        'example': ex,
                        'result': result,
                        'reason': f"Score {result.score:.2f} < {score_threshold}"
                    })

                # Progress update
                if line_num % 100 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = line_num / elapsed if elapsed > 0 else 0
                    print(f"  [{line_num:4d}] Good: {len(good_examples):4d} | Bad: {len(bad_examples):4d} | Rate: {rate:.1f} ex/s")

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Line {line_num}: JSON decode error - {e}")
                continue
            except Exception as e:
                print(f"  ⚠️  Line {line_num}: Verification error - {e}")
                bad_examples.append({
                    'example': ex if 'ex' in locals() else {},
                    'result': None,
                    'reason': f"Exception: {str(e)}"
                })
                continue

    elapsed_time = (datetime.now() - start_time).total_seconds()
    total_examples = len(good_examples) + len(bad_examples)

    print("\n" + "="*80)
    print("  VERIFICATION RESULTS")
    print("="*80)
    print(f"\n✅ Good examples (score ≥ {score_threshold}): {len(good_examples)} ({len(good_examples)/total_examples*100:.1f}%)")
    print(f"❌ Bad examples (score < {score_threshold}): {len(bad_examples)} ({len(bad_examples)/total_examples*100:.1f}%)")
    print(f"⏱️  Total time: {elapsed_time:.1f}s ({total_examples/elapsed_time:.1f} ex/s)")

    # Save filtered dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for item in good_examples:
            f.write(json.dumps(item['example'], ensure_ascii=False) + '\n')

    print(f"\n💾 Saved verified dataset: {output_path}")

    # Save verification logs
    log_path = output_path.parent / f"verification_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with log_path.open('w', encoding='utf-8') as f:
        for log in verification_logs:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

    print(f"📋 Saved verification logs: {log_path}")

    # Print score distribution
    print("\n📊 Score Distribution:")
    score_ranges = {
        '0.9-1.0': 0,
        '0.8-0.9': 0,
        '0.7-0.8': 0,
        '0.6-0.7': 0,
        '0.5-0.6': 0,
        '<0.5': 0
    }

    for item in good_examples + bad_examples:
        if item['result'] and item['result'].score is not None:
            score = item['result'].score
            if score >= 0.9:
                score_ranges['0.9-1.0'] += 1
            elif score >= 0.8:
                score_ranges['0.8-0.9'] += 1
            elif score >= 0.7:
                score_ranges['0.7-0.8'] += 1
            elif score >= 0.6:
                score_ranges['0.6-0.7'] += 1
            elif score >= 0.5:
                score_ranges['0.5-0.6'] += 1
            else:
                score_ranges['<0.5'] += 1

    for range_name, count in score_ranges.items():
        pct = count / total_examples * 100 if total_examples > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {range_name}: {count:4d} ({pct:5.1f}%) {bar}")

    # Print common failure reasons
    print("\n🔍 Top Failure Reasons:")
    failure_reasons = {}
    for item in bad_examples[:100]:  # Check first 100 bad examples
        reason = item.get('reason', 'Unknown')
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • {reason}: {count} examples")

    print("\n" + "="*80)
    print("  ✅ VERIFICATION COMPLETE")
    print("="*80 + "\n")

    return len(good_examples), len(bad_examples)


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / 'data' / 'rtl_comprehensive_training.jsonl'
    output_path = project_root / 'data' / 'rtl_verified_training.jsonl'

    # Run verification with 0.95 threshold (STRICT - only gold standard examples)
    good_count, bad_count = verify_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        score_threshold=0.95
    )

    print(f"\n📈 Next Step: Combine with diagnostic corpus")
    print(f"   Run: python3 scripts/combine_verified_datasets.py")
