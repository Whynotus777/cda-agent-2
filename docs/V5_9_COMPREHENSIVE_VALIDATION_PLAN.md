# V5.9 Comprehensive Validation & Data Acquisition Plan

## CRITICAL FINDINGS FROM V5.8 ANALYSIS

### Overall Regression (61.2% from baseline)
V5.8 ultimate clean showed regression across the board, NOT just FSM. This indicates **dataset-wide quality issues**.

### Existing Data Quality (912 examples in V5.8 ultimate clean):
```
ShiftRegister: 296 examples
Arithmetic:    218 examples
Counter:       102 examples
Controller:     98 examples
Edge:           84 examples
FSM:            73 examples (30% pass rate - still poor)
Clocking:       25 examples
Memory:         14 examples
Protocol:        2 examples
```

**Problem**: We validated ONLY FSMs (2.4% pass rate from V5.x), but never validated the other 839 examples!

## PHASE 1: VALIDATE ALL EXISTING DATA

### Step 1.1: Universal Validation Framework
Create `scripts/validate_all_hierarchies.py` that validates:

1. **Syntax** (iverilog/verilator)
2. **Simulation** (runs without errors)
3. **Functional Correctness** (output matches specification)
4. **Hierarchy-Specific Rules**:
   - **FSM**: State machine requirements (already implemented in `validate_fsm_strict`)
   - **Counter**: Count sequences, enable/reset behavior
   - **ShiftRegister**: Shift operations, parallel load
   - **Arithmetic**: Correct mathematical operations
   - **Controller**: Pipeline control signals
   - **Edge**: Edge detection accuracy
   - **Clocking**: Clock division ratios
   - **Memory**: Read/write operations
   - **Protocol**: Protocol compliance

### Step 1.2: Validate V5.8 Ultimate Clean
Run universal validator on all 912 examples:
```bash
python3 scripts/validate_all_hierarchies.py \
  --dataset data/rtl_behavioral_v5_8_ultimate_clean.jsonl \
  --output data/v5_8_validation_report.json
```

Expected outcome: Identify which hierarchies have quality issues causing the overall regression.

### Step 1.3: Validate V3/V4 Datasets
V5.8 ultimate clean includes FSMs from V3 (5) and V4 (18), but we never validated the non-FSM examples from those datasets.

```bash
# Validate ALL examples from V3
python3 scripts/validate_all_hierarchies.py \
  --dataset data/rtl_behavioral_v3.jsonl \
  --output data/v3_validation_report.json

# Validate ALL examples from V4
python3 scripts/validate_all_hierarchies.py \
  --dataset data/rtl_behavioral_v4.jsonl \
  --output data/v4_validation_report.json
```

## PHASE 2: ACQUIRE GOLD STANDARD DATASETS

### Tier 1: Gold Standard (Formally Verified / Comprehensive Test Suites)

#### 2.1. CVDP (Comprehensive Verilog Design Problems) - HIGHEST PRIORITY
- **Source**: https://github.com/os-data/CVDP
- **Quality**: 783 human-authored problems with cocotb testbenches
- **Relevance**: Handshake-heavy logic, protocol interfaces, arbiters
- **Action**: Clone, extract, validate each module
- **Expected Yield**: ~700+ validated examples

#### 2.2. PULP Platform (ETH Zurich)
- **Source**: https://github.com/pulp-platform
- **Quality**: Silicon-proven, taped-out RISC-V processor IP
- **Relevance**: AXI, APB, protocol controllers with correct handshake FSMs
- **Action**: Extract protocol shim layers + testbenches
- **Expected Yield**: ~50-100 protocol FSMs

#### 2.3. OpenPOWER Cores
- **Source**: https://github.com/openpower-cores
- **Quality**: Formally verified (mathematical proof of correctness)
- **Relevance**: A2I/A2O AXI-to-PLB bus bridges (deadlock-free)
- **Action**: Extract verified handshake FSMs
- **Expected Yield**: ~10-20 formally verified FSMs

### Tier 2: Production Proven (Widely Used, Battle-Tested)

#### 2.4. Alexforencic's verilog-ip
- **Source**: https://github.com/alexforencich/verilog-{ethernet,axis,pcie}
- **Quality**: Most popular Verilog IP collection on GitHub
- **Relevance**: AXI, UART, SPI, I2C with testbenches
- **Action**: Extract modules, run through strict validator
- **Expected Yield**: ~100-200 validated examples

#### 2.5. OpenCores (Wishbone)
- **Source**: https://opencores.org
- **Quality**: Classic open-source IP (quality varies)
- **Relevance**: Wishbone B4-compliant handshake-based bus modules
- **Action**: Extract Wishbone modules, strict validation required
- **Expected Yield**: ~50-100 validated examples (after filtering)

## PHASE 3: CREATE V5.9 COMPREHENSIVE DATASET

### Step 3.1: Combine All Validated Data
```
V3 validated:              X examples
V4 validated:              Y examples
V5.8 ultimate clean validated:  Z examples
CVDP:                     ~700 examples
PULP Platform:            ~50-100 examples
OpenPOWER:                ~10-20 examples
Alexforencich verilog-ip: ~100-200 examples
OpenCores Wishbone:       ~50-100 examples
-------------------------------------------
TOTAL:                    ~1,000-1,200 VALIDATED examples
```

### Step 3.2: Ensure Proper Metadata Tagging
Each example must have:
```json
{
  "instruction": "...",
  "output": "...",
  "hierarchy": {
    "l1": "Behavioral",
    "l2": "FSM|Counter|ShiftRegister|...",
    "l3": "Moore|Mealy|Handshake|Up|Down|..."
  },
  "metadata": {
    "id": "unique_id",
    "source": "CVDP|PULP|OpenPOWER|V5.8|...",
    "dialect": "sv2005|sv2012|v2001",
    "validated": true,
    "validation_passed": true,
    "quality_tier": "gold|silver",
    "date_added": "2025-11-08"
  }
}
```

### Step 3.3: Balance Dataset by Hierarchy
Ensure balanced representation:
- FSM: 200-300 examples (focus on handshake)
- Counter: 100-150 examples
- ShiftRegister: 150-200 examples
- Arithmetic: 150-200 examples
- Controller: 100-150 examples
- Edge: 80-100 examples
- Protocol: 100-150 examples (NEW - from CVDP/PULP)
- Clocking: 50-80 examples
- Memory: 50-80 examples

## PHASE 4: TRAIN & BENCHMARK V5.9

### Step 4.1: Train V5.9 Model
```bash
python3 scripts/train_qwen_coder_qlora.py \
  --dataset data/rtl_behavioral_v5_9_comprehensive.jsonl \
  2>&1 | tee training_v5_9.log
```

### Step 4.2: Benchmark V5.9
```bash
python3 scripts/benchmark_v5_4.py \
  --model models/qwen_coder_rtl/run_rtl_behavioral_v5_9_*/final_model \
  --runs 5 \
  2>&1 | tee benchmark_v5_9_results.log
```

### Expected Improvements:
- **Overall**: 61.2% → 75%+ (high-quality validated data)
- **FSM**: 30.0% → 60%+ (gold standard handshake FSMs)
- **Counter**: Should maintain/improve
- **ShiftRegister**: Should maintain/improve
- **Arithmetic**: Should maintain/improve

## SUCCESS CRITERIA

1. ✅ All existing data validated with strict checks
2. ✅ CVDP dataset integrated (~700 examples)
3. ✅ PULP/OpenPOWER handshake FSMs integrated (~60-120 examples)
4. ✅ Alexforencich/OpenCores modules integrated (~150-300 examples)
5. ✅ V5.9 dataset: 1,000-1,200 validated examples
6. ✅ V5.9 benchmark: Overall >75%, FSM >60%

## TIMELINE

- **Phase 1** (Validate existing): 2-4 hours
- **Phase 2** (Acquire gold data): 4-8 hours
- **Phase 3** (Create V5.9): 1-2 hours
- **Phase 4** (Train & benchmark): 3-5 hours

**Total**: 10-19 hours (can be parallelized)

## NEXT STEPS

1. Implement `scripts/validate_all_hierarchies.py`
2. Run validation on V5.8 ultimate clean
3. Clone and process CVDP (highest priority)
4. Clone and process PULP Platform
5. Continue with remaining sources
6. Create V5.9 comprehensive dataset
7. Train and benchmark
