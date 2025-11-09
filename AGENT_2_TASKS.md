# Agent 2 - Local Tasks for Data Quality Resolution
*Priority: Critical | No network access required*

## Context

Agent 1 discovered a **critical categorization bug** in V5.1 dataset merge:
- Bug: Script looked for "prompt" field, but V4/V5 data uses "instruction"
- Impact: All 1453 V4/V5 examples were marked as "complex" → destroyed FSM training signal
- Fix: Script updated to check both fields
- **Corrected distribution would be: 288 FSMs (36%) vs what V5.1 actually got**

---

## 🔴 Priority 1: Analyze What V5.1 Actually Trained On

### Task 1.1: Inspect OLD V5.1 Dataset Categories
**File**: `data/rtl_behavioral_v5_1.jsonl`

```bash
cd ~/cda-agent-2C1
python3 - <<'PY'
import json
from collections import Counter

categories = Counter()
sources = Counter()

with open('data/rtl_behavioral_v5_1.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        categories[ex.get('category', 'unknown')] += 1
        sources[ex.get('source', 'unknown')] += 1

print("=== V5.1 ACTUAL TRAINING DATA ===")
print(f"Total: {sum(categories.values())} examples\n")
print("Categories:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    pct = count / sum(categories.values()) * 100
    print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")

print("\nSources:")
for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    pct = count / sum(sources.values()) * 100
    print(f"  {src:20s}: {count:4d} ({pct:5.1f}%)")
PY
```

**Expected Output**: Likely shows massive "complex" bias that explains FSM collapse

**Deliverable**: Save output to `reports/v5_1_actual_composition.txt`

---

### Task 1.2: Compare OLD vs CORRECTED Merge
Run the **corrected** merge script (already fixed) and compare:

```bash
cd ~/cda-agent-2C1
python3 scripts/merge_v5_1_hybrid_dataset.py > reports/v5_1_corrected_composition.txt 2>&1

# Generate side-by-side comparison
python3 - <<'PY'
import json

def analyze_dataset(path):
    from collections import Counter
    categories = Counter()
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            categories[ex.get('category', 'unknown')] += 1
    return categories

old = analyze_dataset('data/rtl_behavioral_v5_1.jsonl')
# Note: corrected dataset will be at same path after re-run
# For now, manually note the difference

print("COMPARISON: OLD V5.1 vs CORRECTED")
print("="*50)
print(f"{'Category':<20} | {'OLD':<10} | {'CORRECTED':<10}")
print("-"*50)
# You'll need to fill in corrected numbers after re-merge
PY
```

**Deliverable**: Document in `reports/V5_1_CATEGORY_BUG_IMPACT.md`

---

## 🟡 Priority 2: Dataset Quality Audit Scripts

### Task 2.1: Create V5.1 Audit Script
**Goal**: Automated analysis tool for any JSONL dataset

**File to create**: `scripts/audit_dataset_quality.py`

```python
#!/usr/bin/env python3
"""
Audit dataset quality - categories, duplicates, field completeness
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

def audit_dataset(jsonl_path: Path):
    categories = Counter()
    sources = Counter()
    field_coverage = defaultdict(int)
    duplicates = defaultdict(list)
    total = 0

    with jsonl_path.open() as f:
        for idx, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
                total += 1

                # Category distribution
                categories[ex.get('category', 'unknown')] += 1
                sources[ex.get('source', 'unknown')] += 1

                # Field completeness
                for field in ['instruction', 'output', 'category', 'source']:
                    if ex.get(field):
                        field_coverage[field] += 1

                # Duplicate detection (by output hash)
                code = ex.get('output', '')
                code_hash = hash(code[:200])  # First 200 chars
                duplicates[code_hash].append(idx)

            except json.JSONDecodeError as e:
                print(f"Line {idx}: JSON error - {e}", file=sys.stderr)

    # Report
    print(f"=== DATASET QUALITY AUDIT ===")
    print(f"File: {jsonl_path}")
    print(f"Total: {total} examples\n")

    print("Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")

    print("\nSources:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {src:20s}: {count:4d} ({pct:5.1f}%)")

    print("\nField Coverage:")
    for field, count in sorted(field_coverage.items()):
        pct = count / total * 100
        print(f"  {field:20s}: {count:4d} ({pct:5.1f}%)")

    print("\nDuplicates:")
    dup_groups = [indices for indices in duplicates.values() if len(indices) > 1]
    print(f"  Found {len(dup_groups)} potential duplicate groups")
    print(f"  Total duplicated examples: {sum(len(g) - 1 for g in dup_groups)}")

    # FSM-specific check
    fsm_count = categories.get('fsm', 0)
    if fsm_count == 0:
        print("\n⚠️  WARNING: NO FSM EXAMPLES FOUND!")
    elif fsm_count / total < 0.15:
        print(f"\n⚠️  WARNING: Low FSM representation ({fsm_count/total*100:.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset.jsonl>")
        sys.exit(1)

    audit_dataset(Path(sys.argv[1]))
```

**Test it**:
```bash
chmod +x scripts/audit_dataset_quality.py
python3 scripts/audit_dataset_quality.py data/rtl_behavioral_v5_1.jsonl > reports/v5_1_audit.txt
python3 scripts/audit_dataset_quality.py data/rtl_behavioral_v4.jsonl > reports/v4_audit.txt
python3 scripts/audit_dataset_quality.py data/rtl_behavioral_v5.jsonl > reports/v5_audit.txt
```

**Deliverable**: Working audit script + 3 audit reports

---

### Task 2.2: Validate 56 Claude FSM "Gold" Examples
**Goal**: Confirm our "gold standard" is actually gold

```bash
cd ~/cda-agent-2C1

# Count Claude FSMs by verification status
python3 - <<'PY'
import json
from pathlib import Path

gold_dir = Path('data/claude_fsm_gold_v2')
if not gold_dir.exists():
    print(f"ERROR: {gold_dir} not found!")
    exit(1)

verified = 0
unverified = 0
compile_fail = 0

for jsonl_file in gold_dir.glob('*.jsonl'):
    with jsonl_file.open() as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('verified'):
                verified += 1
            elif ex.get('compile_success') == False:
                compile_fail += 1
            else:
                unverified += 1

print(f"=== CLAUDE FSM GOLD STANDARD AUDIT ===")
print(f"Verified:    {verified}")
print(f"Unverified:  {unverified}")
print(f"Compile Fail: {compile_fail}")
print(f"Total:       {verified + unverified + compile_fail}")

if verified < 50:
    print(f"\n⚠️  WARNING: Only {verified}/56 examples verified!")
PY
```

**Deliverable**: Report in `reports/claude_fsm_gold_quality.txt`

---

## 🟢 Priority 3: Data Labeling Improvements

### Task 3.1: Enhance Categorization Logic
**File**: `scripts/merge_v5_1_hybrid_dataset.py` (already fixed, but add improvements)

**Suggested enhancements** (for Agent 2 to implement):

```python
def categorize_example(ex: dict) -> str:
    """Enhanced categorization with better keyword coverage"""
    # Get all text fields
    prompt = ex.get("prompt", "").lower()
    instruction = ex.get("instruction", "").lower()
    spec = ex.get("spec", "").lower()
    output = ex.get("output", "")[:500].lower()  # Check code too!
    text = prompt + " " + instruction + " " + spec + " " + output

    # FSM patterns (EXPANDED)
    fsm_keywords = [
        "fsm", "finite state", "state machine", "moore", "mealy",
        "state transition", "state diagram", "current_state", "next_state",
        "case (state)", "case(state)"  # Code-level detection
    ]
    if any(kw in text for kw in fsm_keywords):
        return "fsm"

    # Counter patterns (more specific)
    counter_keywords = ["counter", "count up", "count down", "increment", "decrement"]
    if any(kw in text for kw in counter_keywords):
        return "simple_module"

    # Register/Flip-flop
    register_keywords = ["register", "flip-flop", "d flip", "shift register"]
    if any(kw in text for kw in register_keywords):
        return "simple_module"

    # Arithmetic
    arithmetic_keywords = ["alu", "arithmetic", "multiply", "divide", "add", "subtract"]
    if any(kw in text for kw in arithmetic_keywords):
        return "arithmetic"

    # Memory
    memory_keywords = ["fifo", "memory", "ram", "rom", "register file", "buffer"]
    if any(kw in text for kw in memory_keywords):
        return "memory"

    # Mux/Demux/Encoder/Decoder
    simple_keywords = ["mux", "demux", "encoder", "decoder", "selector"]
    if any(kw in text for kw in simple_keywords):
        return "simple_module"

    return "complex"
```

**Test**: Re-run merge and verify improved categorization

---

### Task 3.2: Add Category to Claude FSM Gold Dataset
The 56 Claude FSMs likely don't have `category` field set. Fix this:

```python
#!/usr/bin/env python3
"""Add 'category': 'fsm' to all Claude gold examples"""
import json
from pathlib import Path

gold_dir = Path('data/claude_fsm_gold_v2')
for jsonl_file in gold_dir.glob('*.jsonl'):
    updated_lines = []
    with jsonl_file.open() as f:
        for line in f:
            ex = json.loads(line)
            ex['category'] = 'fsm'  # Explicitly label as FSM
            ex['source'] = 'claude_v2_gold'
            updated_lines.append(json.dumps(ex))

    # Write back
    with jsonl_file.open('w') as f:
        f.write('\n'.join(updated_lines) + '\n')

    print(f"Updated {jsonl_file}: added category='fsm'")
```

---

## 🔵 Priority 4: Formal Verification Test

### Task 4.1: Test SymbiYosys on Sample FSMs
Agent 2 set up formal verification infrastructure. Test it on a few Claude FSMs:

```bash
cd ~/cda-agent-2C1

# Pick 3 Claude FSMs to test
ls data/claude_fsm_gold_v2/*.v | head -3 > /tmp/test_fsms.txt

# For each, try formal verification
while read rtl_file; do
    echo "Testing: $rtl_file"

    # Extract module name
    module=$(grep '^module' "$rtl_file" | head -1 | awk '{print $2}' | sed 's/(.*//')

    # Create work dir
    work_dir="/tmp/formal_test_$(basename $rtl_file .v)"
    mkdir -p "$work_dir"

    # Try running formal_runner (if sby is installed)
    python3 - <<PY
from pathlib import Path
from scripts.formal_runner import run_formal

rtl = Path("$rtl_file")
top = "$module"
work = Path("$work_dir")

try:
    result = run_formal([rtl], top, work)
    print(f"Result: passed={result.passed}, properties={result.properties_checked}")
except Exception as e:
    print(f"Error: {e}")
PY

done < /tmp/test_fsms.txt
```

**Expected**: Some FSMs may pass, others may fail (that's data!)

**Deliverable**: Report which FSMs pass formal verification in `reports/formal_verification_sample.txt`

---

## 📋 Summary Checklist

- [ ] Task 1.1: Analyze OLD V5.1 actual categories → `reports/v5_1_actual_composition.txt`
- [ ] Task 1.2: Compare OLD vs CORRECTED → `reports/V5_1_CATEGORY_BUG_IMPACT.md`
- [ ] Task 2.1: Create audit script → `scripts/audit_dataset_quality.py` + 3 reports
- [ ] Task 2.2: Validate 56 Claude FSMs → `reports/claude_fsm_gold_quality.txt`
- [ ] Task 3.1: Enhance categorization logic in merge script
- [ ] Task 3.2: Add category labels to Claude FSM gold dataset
- [ ] Task 4.1: Test formal verification on sample FSMs → `reports/formal_verification_sample.txt`

---

## 🎯 Success Criteria

By completing these tasks, Agent 2 will provide:

1. **Root cause confirmation**: Exact breakdown of what V5.1 trained on (likely ~750 "complex" + 56 FSM)
2. **Quality baseline**: Audit reports for V3/V4/V5/V5.1 showing data quality metrics
3. **Gold standard validation**: Confirmation that 56 Claude FSMs are actually verified
4. **Improved tooling**: Reusable audit + categorization scripts for future datasets
5. **Formal verification pilot**: Evidence that formal properties can validate FSM quality

**Outcome**: Agent 1 can make informed decision on V5.2 strategy (retrain V5.1 with corrected data? Use external datasets? Both?)

---

## Notes

- All tasks are **local** - no network access required
- Most tasks are Python analysis scripts - fast to run
- Reports go in `reports/` directory
- If SymbiYosys (sby) is not installed, Task 4.1 can be skipped for now

---

*Agent 1 has completed: Licensing research (VeriThoughts ✅, CVDP ✅, SecFSM ❌ not released). Full report: `docs/DATASET_LICENSING_REPORT.md`*
