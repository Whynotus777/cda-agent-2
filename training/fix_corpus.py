#!/usr/bin/env python3
"""
Fix JSON errors in PLACEMENT_GOLD_STANDARD.jsonl
Repairs lines 126, 127, 151
"""

import json
import sys
from pathlib import Path

def fix_corpus(corpus_path: str):
    """Fix JSON errors in corpus"""

    print(f"Fixing corpus: {corpus_path}")

    corpus_file = Path(corpus_path)
    backup_file = corpus_file.with_suffix('.jsonl.backup')

    # Backup original
    print(f"Creating backup: {backup_file}")
    with open(corpus_file, 'r') as f:
        content = f.read()
    with open(backup_file, 'w') as f:
        f.write(content)

    # Read all lines
    lines = []
    errors = []
    fixed = []

    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Try to parse
                obj = json.loads(line)
                lines.append(json.dumps(obj))  # Re-serialize cleanly
            except json.JSONDecodeError as e:
                errors.append(i)
                print(f"Line {i}: JSON error - {e}")

                # Attempt automatic fix
                fixed_line = None

                # Common fixes
                # 1. Control characters - remove
                cleaned = ''.join(char for char in line if char.isprintable() or char in '\n\r\t')

                # 2. Try parsing cleaned version
                try:
                    obj = json.loads(cleaned)
                    fixed_line = json.dumps(obj)
                    fixed.append(i)
                    print(f"  → Fixed: Removed control characters")
                except:
                    pass

                # 3. If still broken, try escaping quotes in response field
                if fixed_line is None:
                    try:
                        # Find response field and escape quotes
                        import re
                        # This is a heuristic fix
                        if '"response":' in line:
                            # Try to fix malformed quotes
                            fixed_candidate = line.replace('Assum"', 'Assume')
                            obj = json.loads(fixed_candidate)
                            fixed_line = json.dumps(obj)
                            fixed.append(i)
                            print(f"  → Fixed: Corrected quote placement")
                    except:
                        pass

                if fixed_line:
                    lines.append(fixed_line)
                else:
                    print(f"  → Could not auto-fix, skipping line {i}")

    # Write fixed corpus
    print(f"\nWriting fixed corpus...")
    with open(corpus_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"\n{'='*70}")
    print(f"CORPUS FIX SUMMARY")
    print(f"{'='*70}")
    print(f"Total lines processed: {len(lines) + len(errors)}")
    print(f"Errors found: {len(errors)}")
    print(f"Fixed automatically: {len(fixed)}")
    print(f"Skipped (could not fix): {len(errors) - len(fixed)}")
    print(f"Final corpus size: {len(lines)} examples")
    print(f"{'='*70}")

    if len(errors) - len(fixed) > 0:
        print(f"\n⚠ Warning: {len(errors) - len(fixed)} lines could not be fixed")
        print(f"Manual inspection required for lines: {[e for e in errors if e not in fixed]}")
        return False

    print(f"\n✓ All errors fixed successfully")
    print(f"✓ Backup saved to: {backup_file}")
    return True

if __name__ == "__main__":
    corpus_path = "../data/training/PLACEMENT_GOLD_STANDARD.jsonl"

    if not Path(corpus_path).exists():
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)

    success = fix_corpus(corpus_path)
    sys.exit(0 if success else 1)
