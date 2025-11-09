# CDA-AGENT-2C1 CODEBASE CLEANUP ANALYSIS
**Date:** 2025-11-09
**Current Version:** V5.9
**Mission:** AI Agent for Chip Design → Foundational Model → Custom Chip Manufacturing

---

## EXECUTIVE SUMMARY

**Current State:** V5.9 operational with Qwen2.5-Coder fine-tuning, but buried under **182GB+ of deprecated artifacts**

**Key Issues Identified:**
- ❌ **A7 Testbench Generator NOT integrated in training/inference** (your primary concern)
- ⚠️ 68 log files cluttering root directory
- ⚠️ 182GB Mixtral models (superseded, no longer used)
- ⚠️ Multiple dataset versions (v2-v5.8) when only v5.9 is current
- ⚠️ 13,689 .pyc cache files scattered throughout
- ⚠️ Unused multi-agent conversational infrastructure
- ⚠️ Duplicate VeriThoughts external datasets

**Space Recovery Potential:** 182GB+ (mostly Mixtral models)

---

## CRITICAL FINDING: A7 ADAPTER NOT IN USE ⚠️

### Current Status
The A7 Professor testbench generator exists but is **NOT integrated** into the training or evaluation pipeline:

**Evidence:**
```python
# scripts/benchmark_v5_4.py (lines 42-44)
try:
    from core.verification.testbench_generator import A7_TestbenchGenerator
except ImportError:
    A7_TestbenchGenerator = None  # ← Import exists but NOT used

# Actual inference code (lines 235-246)
def run_test(model, tokenizer, spec_entry: Dict, run_id: int) -> Dict:
    prompt = f"Generate Verilog RTL..."
    outputs = model.generate(**inputs, ...)  # ← Standard generation, NO A7
```

**Impact:** Your V5.9 model is NOT benefiting from A7's testbench generation capabilities. This is a **missed optimization opportunity**.

**Recommendation:**
1. Integrate A7 into testbench generation phase
2. Use A7 for dataset validation/augmentation
3. Consider A7-generated testbenches as training signal

---

## CURRENT ACTIVE STACK (Keep These)

### Core Training Infrastructure ✅
```
scripts/train_qwen_coder_qlora.py          [PRIMARY TRAINING]
scripts/benchmark_v5_4.py                   [V5.4+ BENCHMARKING]
scripts/create_v5_9_dataset.py             [V5.9 DATASET BUILDER]
scripts/analyze_fsm_success_patterns.py    [FSM ANALYSIS]
core/verification/testbench_generator.py   [A7 GENERATOR - NOT USED YET]
```

### Active Datasets ✅
```
data/rtl_behavioral_v5_9.jsonl             [CURRENT - 357 samples, 510KB]
data/fsm_gold_v5_8.jsonl                   [FSM GOLD STANDARD]
data/fsm_comprehensive_validated.jsonl    [VALIDATED FSMs]
```

### Active Models ✅
```
models/qwen_coder_rtl/run_rtl_behavioral_v5_9_final/    [CURRENT V5.9 MODEL]
models/qwen_coder_rtl/                                  [19GB total]
```

### Core RTL Agents ✅
```
core/rtl_agents/a1_spec_to_rtl.py         [PRIMARY RTL GEN]
core/rtl_agents/a1_llm_generator.py
core/rtl_agents/a2_boilerplate_gen.py
core/rtl_agents/a3_constraint_synth.py
core/rtl_agents/a4_lint_cdc.py
core/rtl_agents/a5_style_review.py
core/rtl_agents/a6_eda_command.py
core/rtl_agents/base_agent.py
```

### Validation Infrastructure ✅
```
core/simulation_engine/
core/validation/
core/verification/
scripts/run_iverilog_simulation.py
scripts/rtl_verifier.py
scripts/formal_runner.py
```

---

## DEPRECATED ARTIFACTS (Move to garbage/)

### 1. Root Log Files (68 files, ~5MB)
```bash
benchmark_v2*.log                   (5 files - old benchmarks)
benchmark_v3*.log                   (3 files)
benchmark_v4*.log                   (2 files)
benchmark_v5_1-5_8*.log            (9 files - superseded by v5.9)
training_behavioral_v2-v5_8.log    (13 files - superseded)
claude_smoke_test*.log             (3 files)
generate_v4-v5_1*.log              (3 files)
fsm_gold_v5_8*.log                 (4 files)
validation logs                     (3 files)
... and 22 more log files
```
**Action:** Move all to `garbage/logs/`

### 2. Deprecated Datasets (~25MB)
```bash
# Keep ONLY v5.9, archive rest
rtl_behavioral_v2.jsonl                          (3.6M)
rtl_behavioral_v3.jsonl + tagged variants        (4 files, ~800K)
rtl_behavioral_v4.jsonl + tagged variants        (4 files, ~2.3M)
rtl_behavioral_v5.jsonl                          (413K)
rtl_behavioral_v5_1.jsonl + backup               (2 files, ~2.8M)
rtl_behavioral_v5_2*.jsonl                       (3 files, ~4M)
rtl_behavioral_v5_3*.jsonl                       (2 files, ~2.4M)
rtl_behavioral_v5_4.jsonl                        (1.4M)
rtl_behavioral_v5_5.jsonl                        (1.1M)
rtl_behavioral_v5_6.jsonl                        (1.9M)
rtl_behavioral_v5_7.jsonl                        (1.9M)
rtl_behavioral_v5_8*.jsonl                       (4 files, ~4.8M)

# FSM intermediate artifacts
fsm_clean_v5_4.jsonl
fsm_failed_repair_v5_4.jsonl
fsm_repaired_v5_4.jsonl
fsm_synthetic_*_v5_6.jsonl                       (3 files)
```
**Action:** Move to `garbage/datasets/v{2,3,4,5,5_1,...,5_8}/`

### 3. Deprecated Training Scripts (~40 files)
```bash
# Old model trainers
scripts/train_mixtral_qlora.py                   [Mixtral superseded]
scripts/train_llama31_qlora.py                   [LLaMA abandoned]

# Old benchmarks
scripts/benchmark_behavioral_model*.py           [4 files, pre-v4]
scripts/benchmark_v4*.py                         [2 files]
scripts/benchmark_v5*.py                         [8 files, v5.0-5.3]
scripts/benchmark_with_simulation.py

# Old dataset builders
scripts/generate_v4_dataset.py
scripts/generate_v5_1_dataset.py
scripts/create_v5_[2-8]*.py                      [7 files]
scripts/build_v5_[3-4]*.py                       [2 files]

# Old analysis
scripts/analyze_v[4-5]*.py                       [5 files]
scripts/extract_v[3-4]*.py                       [2 files]
scripts/watch_v4_progress.py
scripts/fsm_forensics_v5_2.py
```
**Action:** Move to `garbage/scripts/{training,benchmarking,dataset_builders,analysis}/`

### 4. Unused Multi-Agent Infrastructure
```bash
# Conversational agent system (not in current mission)
api/                                             [4 files]
  ├── decision_manager.py
  ├── models.py
  ├── pipeline.py
  └── __init__.py

core/conversational/                             [9 files]
  ├── action_executor.py
  ├── conversation_manager.py
  ├── intent_parser.py
  ├── llm_interface.py
  ├── model_router.py
  ├── phase_router.py
  ├── specialist_router.py
  ├── triage_router.py
  └── __init__.py

core/rl_optimizer/                               [6 files]
  ├── actions.py
  ├── environment.py
  ├── ppo_agent.py
  ├── reward.py
  ├── rl_agent.py
  └── __init__.py

react_api/server.py

# Related files
chat_with_apprentice.py
chat_with_specialist.py
demo_conversational_agent.py
launch_backend_prod.sh
launch_react_api.sh
launch_mixtral_chat.sh
launch_apprentice_chat.sh
```
**Action:** Move to `garbage/conversational_system/`

### 5. Deprecated Models (182GB!) 💾
```bash
models/mixtral_base/         [178GB - HUGE!]
models/mixtral_rtl/          [3.5GB]
models/llama31_rtl/          [12KB - tiny, abandoned]
```
**Action:**
1. **CRITICAL:** Backup Mixtral offsite if needed
2. Move to `garbage/models/` or delete
3. **Immediate space recovery: 182GB**

### 6. Old Test Files (25 files in root)
```bash
test_a1_*.py                     (5 files)
test_a2_*.py                     (3 files)
test_a3_agent.py
test_a4_agent.py
test_a5_agent.py
test_a6_agent.py
test_agent_triage.py
test_api_cli.py
test_cli.py
test_conversational_flow.py
test_design_parser.py
test_eda_pipeline.py
test_end_to_end_integration.py
test_fastapi_integration.py
test_ppo_agent.py
test_rag_client.py
test_rl_environment.py
test_spi_master_corrected.py
```
**Action:** Move to `garbage/tests/`

### 7. Duplicate External Datasets
```bash
external_datasets/               [18MB]
  ├── axi/
  └── cvdp_benchmark/

# DUPLICATE OF:
data/external/verithoughts/      [9.1M]
```
**Action:** Move `external_datasets/` to `garbage/duplicates/`

### 8. Old Training Infrastructure
```bash
training/                        [entire directory]
  ├── finetune_specialist.py
  ├── finetune_8b_chipdesign.py
  ├── train_placement_apprentice.py
  ├── train_specialist*.py (multiple)
  ├── data_preparation/ (7 files)
  └── test_* files (5 files)

expanded_training/               [123MB - v3 artifacts]
```
**Action:** Move to `garbage/training_deprecated/`

### 9. Python Cache (13,689 files, ~50MB)
```bash
__pycache__/ directories throughout codebase
*.pyc files
```
**Action:** Delete all with cleanup script

### 10. Old Test Runs
```bash
data/runs/run_20251030_*         [Oct 30 experiments]
data/runs/run_20251031_*         [Oct 31 experiments]
data/runs/[UUID-only]            [Generic test runs]
```
**Action:** Move to `garbage/test_runs/october_november/`

---

## CLEANUP EXECUTION PLAN

### 🟢 Phase 1: Safe Cleanup (Execute Immediately)
**Risk:** NONE | **Time:** 5 minutes | **Space:** ~60MB

```bash
# 1. Create garbage directory structure
mkdir -p garbage/{logs,datasets,scripts,tests,models,conversational,misc,duplicates}

# 2. Move root log files (68 files)
mv *.log garbage/logs/ 2>/dev/null

# 3. Clean Python cache (13,689 files)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 4. Move old test files from root
mv test_*.py garbage/tests/ 2>/dev/null

# 5. Move duplicate external datasets
mv external_datasets garbage/duplicates/ 2>/dev/null

# 6. Move old markdown docs
mv A1_*.md AGENT_2_TASKS.md BEHAVIORAL_V3_PLAN.md garbage/docs/ 2>/dev/null
```

### 🟡 Phase 2: Dataset Cleanup (Week 1)
**Risk:** LOW | **Time:** 10 minutes | **Space:** ~25MB

```bash
# Move all datasets except v5.9 and gold standards
cd data/
mkdir -p ../garbage/datasets/{v2,v3,v4,v5,v5_1,v5_2,v5_3,v5_4,v5_5,v5_6,v5_7,v5_8,fsm_old}

# V2-V4
mv rtl_behavioral_v2.jsonl ../garbage/datasets/v2/
mv rtl_behavioral_v3*.jsonl ../garbage/datasets/v3/
mv rtl_behavioral_v4*.jsonl ../garbage/datasets/v4/

# V5.0-5.8 (keep only v5.9!)
mv rtl_behavioral_v5.jsonl ../garbage/datasets/v5/
mv rtl_behavioral_v5_1*.jsonl ../garbage/datasets/v5_1/
mv rtl_behavioral_v5_2*.jsonl ../garbage/datasets/v5_2/
mv rtl_behavioral_v5_3*.jsonl ../garbage/datasets/v5_3/
mv rtl_behavioral_v5_4*.jsonl ../garbage/datasets/v5_4/
mv rtl_behavioral_v5_5*.jsonl ../garbage/datasets/v5_5/
mv rtl_behavioral_v5_6*.jsonl ../garbage/datasets/v5_6/
mv rtl_behavioral_v5_7*.jsonl ../garbage/datasets/v5_7/
mv rtl_behavioral_v5_8*.jsonl ../garbage/datasets/v5_8/

# FSM intermediate files
mv fsm_*_v5_[4-6]*.jsonl ../garbage/datasets/fsm_old/
mv verification_log_*.jsonl ../garbage/datasets/fsm_old/
mv needs_manual_review.jsonl ../garbage/datasets/fsm_old/

cd ..
```

### 🟡 Phase 3: Script Cleanup (Week 1)
**Risk:** LOW | **Time:** 15 minutes | **Space:** minimal

```bash
cd scripts/
mkdir -p ../garbage/scripts/{training,benchmarking,dataset_builders,analysis}

# Old trainers
mv train_mixtral_qlora.py train_llama31_qlora.py ../garbage/scripts/training/

# Old benchmarks
mv benchmark_behavioral_model*.py ../garbage/scripts/benchmarking/
mv benchmark_v4*.py benchmark_v5.py benchmark_v5_1.py benchmark_v5_2.py benchmark_v5_3.py ../garbage/scripts/benchmarking/
mv benchmark_with_simulation.py ../garbage/scripts/benchmarking/

# Old dataset builders (keep create_v5_9_dataset.py!)
mv generate_v4_dataset.py generate_v5_1_dataset.py ../garbage/scripts/dataset_builders/
mv create_v5_dataset.py create_v5_2*.py build_v5_3*.py build_v5_4*.py ../garbage/scripts/dataset_builders/
mv create_v5_[5-8]*.py generate_fsm_gold_v5_8.py ../garbage/scripts/dataset_builders/

# Old analysis
mv analyze_v4*.py analyze_v5_frozen.py ../garbage/scripts/analysis/
mv extract_v3*.py extract_v4*.py watch_v4_progress.py ../garbage/scripts/analysis/
mv fsm_forensics_v5_2.py ../garbage/scripts/analysis/

cd ..
```

### 🟠 Phase 4: Infrastructure Cleanup (Week 2)
**Risk:** MEDIUM | **Time:** 10 minutes | **Space:** ~150MB

```bash
# Move conversational system
mv api core/conversational core/rl_optimizer react_api garbage/conversational/

# Move UI/demo
mv ui demo placement_demo.py demo_*.py chat_*.py garbage/misc/

# Move launch scripts for deprecated systems
mv launch_backend_prod.sh launch_react_api.sh launch_mixtral_chat.sh launch_apprentice_chat.sh garbage/misc/

# Move old training infrastructure
mv training expanded_training garbage/training_deprecated/

# Move old data
mv data/training data/rag garbage/misc/
```

### 🔴 Phase 5: Model Cleanup (CRITICAL - 182GB Recovery)
**Risk:** HIGH | **Time:** 1-2 hours | **Space:** 182GB

```bash
# ⚠️ WARNING: Backup Mixtral models offsite BEFORE deleting
# They are 178GB and may have research value

# Option A: Archive offsite first
# rclone copy models/mixtral_base/ remote:archive/mixtral_base/
# rclone copy models/mixtral_rtl/ remote:archive/mixtral_rtl/

# Option B: Move to garbage (can delete later)
mkdir -p garbage/models/{mixtral,llama}
mv models/mixtral_base models/mixtral_rtl garbage/models/mixtral/
rm -rf models/llama31_rtl  # Only 12KB, clearly abandoned

# ⚠️ VERIFY qwen models are intact before proceeding
ls -lh models/qwen_coder_rtl/
```

### 🟡 Phase 6: Test Runs Cleanup (Week 2)
**Risk:** LOW | **Time:** 5 minutes | **Space:** ~1.5MB

```bash
cd data/runs/
mkdir -p ../../garbage/test_runs/october_november

# Move old UUID-based runs
mv *-*-*-*-* ../../garbage/test_runs/october_november/ 2>/dev/null

# Move October/early November runs
mv run_20251030_* run_20251031_* ../../garbage/test_runs/october_november/ 2>/dev/null
mv run_202511040* run_202511050* ../../garbage/test_runs/october_november/ 2>/dev/null

cd ../..
```

---

## POST-CLEANUP DIRECTORY STRUCTURE

```
cda-agent-2C1/
├── core/
│   ├── rtl_agents/           ✅ A1-A6 agents
│   ├── schemas/              ✅ JSON schemas
│   ├── simulation_engine/    ✅ Synthesis, timing, power
│   ├── validation/           ✅ Validators
│   ├── verification/         ✅ A7 testbench generator
│   └── world_model/          ✅ Design state, tech lib
├── scripts/
│   ├── train_qwen_coder_qlora.py          ✅ CURRENT
│   ├── create_v5_9_dataset.py             ✅ CURRENT
│   ├── benchmark_v5_4.py                  ✅ CURRENT
│   ├── run_iverilog_simulation.py         ✅
│   ├── rtl_verifier.py                    ✅
│   ├── formal_runner.py                   ✅
│   └── analyze_fsm_success_patterns.py    ✅
├── data/
│   ├── rtl_behavioral_v5_9.jsonl          ✅ CURRENT (357 samples)
│   ├── fsm_gold_v5_8.jsonl                ✅ GOLD
│   ├── fsm_comprehensive_validated.jsonl  ✅ VALIDATED
│   ├── external/verithoughts/             ✅
│   └── runs/                              ✅ (recent only)
├── models/
│   └── qwen_coder_rtl/                    ✅ CURRENT (19GB)
├── formal/
│   └── properties/                        ✅
├── utils/                                 ✅
├── logs/                                  📁 NEW (future logs)
├── tests/                                 📁 (proper tests)
├── requirements.txt                       ✅
├── README.md                              ✅
├── CLEANUP_REPORT.md                      📄 THIS FILE
└── garbage/                               🗑️ ALL DEPRECATED CODE
    ├── logs/                              (68 log files)
    ├── datasets/                          (~25MB v2-v5.8)
    ├── scripts/                           (~40 scripts)
    ├── tests/                             (25 test files)
    ├── models/mixtral/                    (182GB)
    ├── conversational/                    (multi-agent system)
    ├── training_deprecated/               (old training infra)
    └── misc/                              (UI, demos, duplicates)
```

---

## CRITICAL NEXT STEPS

### 1. Integrate A7 Testbench Generator 🚨
**Priority:** IMMEDIATE

The A7 adapter exists but is NOT being used. Integration options:

**Option A: A7 for Training Data Validation**
```python
# In scripts/create_v5_9_dataset.py
from core.verification.testbench_generator import A7_TestbenchGenerator

a7 = A7_TestbenchGenerator()
for example in dataset:
    testbench = a7.generate_testbench(example['spec'])
    # Validate RTL against A7-generated testbench
    # Only include if passes
```

**Option B: A7 for Benchmark Testbenches**
```python
# In scripts/benchmark_v5_4.py
def run_test(model, tokenizer, spec_entry: Dict, run_id: int) -> Dict:
    # Generate RTL
    rtl_code = model.generate(...)

    # Generate testbench with A7
    testbench = A7_TestbenchGenerator().generate(spec_entry["spec"], rtl_code)

    # Run simulation with A7 testbench
    result = run_simulation(rtl_code, testbench)
    return result
```

**Option C: A7 as Training Signal**
```python
# Use A7 testbench quality as reward signal
# High-quality A7 testbenches = better training examples
```

### 2. Update Documentation
```bash
# Update README.md to reflect:
- Current V5.9 status
- A7 integration roadmap
- Cleanup completion status
- Mission: RTL generation → Foundational model → Custom chips
```

### 3. Create .gitignore Updates
```bash
echo "*.log" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "garbage/" >> .gitignore
echo "logs/" >> .gitignore
```

### 4. Version Control Snapshot
```bash
git tag -a v5.9-pre-cleanup -m "State before major cleanup"
git add .gitignore CLEANUP_REPORT.md
git commit -m "Add cleanup report and update gitignore"
```

---

## VALIDATION CHECKLIST

After cleanup, verify:

- [ ] Can train V5.9: `python3 scripts/train_qwen_coder_qlora.py --dataset data/rtl_behavioral_v5_9.jsonl`
- [ ] Can benchmark: `python3 scripts/benchmark_v5_4.py --model models/qwen_coder_rtl/run_rtl_behavioral_v5_9_final/final_model --runs 5`
- [ ] Can run A7 testbench generation: `python3 -c "from core.verification.testbench_generator import A7_TestbenchGenerator; print('OK')"`
- [ ] Can run formal verification: `python3 scripts/formal_runner.py`
- [ ] No broken imports in core/rtl_agents/
- [ ] README.md reflects current mission
- [ ] requirements.txt is accurate
- [ ] Git status shows clean working tree
- [ ] Disk space recovered (check with `df -h`)

---

## SPACE RECOVERY SUMMARY

| Category | Size | Phase | Risk |
|----------|------|-------|------|
| Root log files | ~5MB | Phase 1 | None |
| Python cache | ~50MB | Phase 1 | None |
| Old datasets | ~25MB | Phase 2 | Low |
| Test runs | ~1.5MB | Phase 6 | Low |
| Training infra | ~150MB | Phase 4 | Medium |
| **Mixtral models** | **182GB** | **Phase 5** | **HIGH** |
| **TOTAL** | **~182GB** | | |

---

## RECOMMENDATIONS

### Immediate (Today)
1. ✅ Execute Phase 1 cleanup (safe, no risk)
2. 🚨 **Investigate A7 integration** (your primary concern)
3. ✅ Create .gitignore updates

### This Week
4. ✅ Execute Phase 2-3 cleanup (datasets, scripts)
5. ✅ Update README.md with current mission status
6. ✅ Test V5.9 model after cleanup

### Next Week
7. ✅ Execute Phase 4 cleanup (infrastructure)
8. ✅ Execute Phase 6 cleanup (test runs)
9. 🔴 **DEFER Phase 5** until disk space critical (182GB Mixtral)

### Long Term
10. Integrate A7 into training pipeline
11. Reorganize scripts/ by function (training/, validation/, generation/)
12. Create proper tests/ directory with updated tests
13. Implement logging to logs/ instead of root

---

## CONCLUSION

Your codebase is **functionally operational** but buried under **182GB of legacy artifacts**. The **A7 testbench generator exists but is not being leveraged**, which is a critical missed opportunity for your chip design mission.

**Recommended Action Order:**
1. Investigate A7 integration (addresses your concern about "not leveraging A7")
2. Execute Phase 1-3 cleanup (safe, immediate benefit)
3. Test V5.9 functionality
4. Plan A7 integration strategy
5. Defer Mixtral cleanup until disk space critical

**Risk Assessment:**
- Phase 1-3: ✅ SAFE (can execute immediately)
- Phase 4: ⚠️ MODERATE (test after execution)
- Phase 5: 🔴 HIGH (backup Mixtral first!)

---

**Generated:** 2025-11-09
**For questions, see:** `/home/quantumc1/cda-agent-2C1/CLEANUP_REPORT.md`
