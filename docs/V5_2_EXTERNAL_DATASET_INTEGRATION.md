# V5.2 External Dataset Integration Plan
*Date: 2025-11-06*
*Strategy: Leverage VeriThoughts + CVDP instead of generating Claude FSMs*

---

## Why External Datasets Over Claude FSM Generation

| Metric | Claude FSM Gen (Old Plan) | External Datasets (New Plan) |
|--------|---------------------------|------------------------------|
| **Cost** | $30-50 API calls | **$0** |
| **Time** | 3-4 hours generation | **30-60 min download+convert** |
| **Quality** | Simulation-verified | **Formal verification** (VeriThoughts) |
| **Quantity** | 150 FSMs | **300-450 FSMs** |
| **Diversity** | Single source (Claude) | **Multiple sources** (VeriThoughts, CVDP, V4/V5) |
| **Licensing** | N/A | BSD 3-Clause, CC BY 4.0 ✅ |

**Decision**: Use external datasets (Agent 2's licensing research)

---

## V5.2 Final Composition

### Target Distribution

| Source | Examples | % | Quality Tier | Notes |
|--------|----------|---|--------------|-------|
| **VeriThoughts FSMs** | 200-300 | 25-35% | Tier 1 (Gold) | Formal verification labels |
| **CVDP FSMs** | 100-150 | 12-18% | Tier 1 (Gold) | CocoTB testbenches |
| **V4/V5 Clean** | 350-450 | 45-55% | Tier 2 (Silver) | Detoxified, 100% simple modules |
| **Total** | **650-900** | 100% | Multi-source | FSM: 35-50% |

### Key Improvements Over V5.1

- **FSM representation**: 300-450 FSMs (35-50%) vs 30 FSMs (3.7% in V5.1)
- **Quality**: Formal verification + CocoTB tests vs unverified V4/V5 FSMs
- **Diversity**: 3 sources vs 2 sources (more robust training)
- **No toxic data**: 236 bad V4/V5 FSMs surgically removed

---

## Integration Pipeline

### Phase 1: Download Raw Datasets ✅

**VeriThoughts** (BSD 3-Clause):
```bash
# Option A: GitHub clone
cd data/external/verithoughts
git clone https://github.com/wilyub/VeriThoughts.git
cd VeriThoughts && git checkout main

# Option B: HuggingFace (if available)
python3 -c "from datasets import load_dataset; ds = load_dataset('wilyub/verithoughts'); ds.save_to_disk('data/external/verithoughts/raw')"
```

**CVDP** (CC BY 4.0 + Apache 2.0):
```bash
cd data/external/cvdp
python3 -c "from datasets import load_dataset; ds = load_dataset('nvidia/cvdp-benchmark-dataset'); ds.save_to_disk('data/external/cvdp/raw')"
```

### Phase 2: Convert to JSONL ⏳

**Conversion Scripts**:
- `scripts/convert_verithoughts.py` - Extract FSMs, add formal verification labels
- `scripts/convert_cvdp.py` - Extract RTL problems, parse CocoTB tests

**Output Format** (our JSONL schema):
```json
{
  "instruction": "Design a 3-bit Gray code counter...",
  "output": "module gray_counter(...);",
  "category": "fsm",
  "source": "verithoughts",
  "license": "BSD-3-Clause",
  "verification": "formal",
  "formal_pass": true,
  "dataset_version": "v2025.05",
  "attribution": "VeriThoughts (MIT/wilyub)"
}
```

### Phase 3: Filter & Curate ⏳

**VeriThoughts Filtering**:
- Extract FSM-related examples (keyword search: "fsm", "state machine", "moore", "mealy")
- Prioritize examples with `formal_pass: true`
- Target: 200-300 FSMs

**CVDP Filtering**:
- Extract "rtl_generation" category
- Filter for FSM problems
- Parse CocoTB testbenches (Python → metadata)
- Target: 100-150 FSMs

### Phase 4: Merge V5.2 Dataset ⏳

Run `scripts/merge_v5_2_external_dataset.py`:
- Load 300-450 external FSMs
- Load 350-450 clean V4/V5 simple modules
- Shuffle & save to `data/rtl_behavioral_v5_2.jsonl`

### Phase 5: Train V5.2 ⏳

```bash
python3 scripts/train_qwen_coder_qlora.py  # 3-5 minutes
python3 scripts/benchmark_v5_2.py          # 2 minutes
```

**Expected Results**:
- Overall functional: 88-92% (vs 78% V5.1, 86% V4/V5)
- FSM functional: 70-85% (vs 0% V5.1)
- Simple modules: 95-100% maintained

---

## Metadata Schema Requirements

Per Agent 2's licensing doc, all external examples must include:

```python
{
  "source": "verithoughts|cvdp|secfsm|veriprefer",
  "license": "BSD-3-Clause|CC-BY-4.0|Apache-2.0",
  "attribution": "Original work by...",
  "dataset_version": "v1.0",
  "verification": "formal|simulation|cocotb|none",
  "formal_pass": true|false|null,
  "testbench_type": "systemverilog|cocotb|none"
}
```

### Attribution Strings

**VeriThoughts** (BSD 3-Clause):
```
"attribution": "VeriThoughts dataset (BSD 3-Clause, github.com/wilyub/VeriThoughts)"
```

**CVDP** (CC BY 4.0):
```
"attribution": "Derived from NVIDIA CVDP dataset (CC BY 4.0, HuggingFace: nvidia/cvdp-benchmark-dataset)"
```

---

## File Locations

### Raw Data
- `data/external/verithoughts/raw/` - VeriThoughts raw files
- `data/external/verithoughts/LICENSE` - BSD 3-Clause license
- `data/external/cvdp/raw/` - CVDP raw HuggingFace dataset
- `data/external/cvdp/LICENSE` - CC BY 4.0 license text

### Converted JSONL
- `data/external/verithoughts/verithoughts_fsm.jsonl` - Filtered FSMs
- `data/external/cvdp/cvdp_fsm.jsonl` - Filtered FSM problems

### Final Merged Dataset
- `data/rtl_behavioral_v5_2.jsonl` - Final V5.2 training dataset

---

## Success Criteria

**Dataset Quality**:
- ✅ 300-450 formally verified FSMs
- ✅ 350-450 proven simple modules (100% V5.1 performers)
- ✅ No toxic data (236 V4/V5 FSMs removed)
- ✅ Proper licensing attribution

**Training Results** (predicted):
- Overall functional: 88-92%
- FSM functional: 70-85%
- Simple modules: 95-100%

**Legal Compliance**:
- ✅ All sources properly attributed
- ✅ Licenses compatible with commercial use
- ✅ Provenance tracked in metadata

---

## Next Steps

1. ⏳ **Download VeriThoughts** - Clone GitHub repo or download from HuggingFace
2. ⏳ **Download CVDP** - Load from HuggingFace (nvidia/cvdp-benchmark-dataset)
3. ⏳ **Create conversion scripts** - `convert_verithoughts.py`, `convert_cvdp.py`
4. ⏳ **Run conversion** - Extract FSMs, add metadata
5. ⏳ **Update merge script** - Modify `merge_v5_2_hybrid_dataset.py` to use external datasets
6. ⏳ **Merge V5.2 dataset** - Combine all sources
7. ⏳ **Train V5.2** - QLoRA training (3-5 minutes)
8. ⏳ **Benchmark V5.2** - Validate results

**Estimated Total Time**: 1-2 hours (vs 3-4 hours for Claude FSM generation)

---

*This plan supersedes the original "generate 150 Claude FSMs" strategy and leverages Agent 2's external dataset licensing research.*
