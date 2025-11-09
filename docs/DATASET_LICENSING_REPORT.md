# External Dataset Licensing Report
*Updated: 2025-11-06*

## Summary

| Dataset | Status | License | Can Train? | Notes |
|---------|--------|---------|------------|-------|
| **VeriThoughts** | ✅ Available | BSD 3-Clause | **YES** | 20K Verilog samples with formal verification |
| **CVDP** | ✅ Available | CC BY 4.0 + Apache 2.0 | **YES** | 783 problems with cocotb testbenches |
| **SecFSM** | ❌ Not Released | TBD | **NO** | Code pending paper acceptance |
| **VeriPrefer** | 🔍 Research Needed | TBD | TBD | Not yet investigated |

---

## 1. VeriThoughts (2025)

### Overview
- **Paper**: ["VeriThoughts: Enabling Automated Verilog Code Generation using Reasoning and Formal Verification"](https://arxiv.org/html/2505.20302v1)
- **GitHub**: [github.com/wilyub/VeriThoughts](https://github.com/wilyub/VeriThoughts)
- **HuggingFace**: Available for download
- **Size**: 20,000 Verilog samples with reasoning traces and formal verification labels

### License
**BSD 3-Clause** (permissive, commercial use allowed)
- Built on MetRex dataset (also BSD 3-Clause)
- No attribution restrictions beyond standard BSD terms

### Content
- Verilog RTL code pairs (original + generated)
- Prompts describing each design
- Reasoning traces used in generation
- Functional equivalence labels from formal verification

### Integration Plan
✅ **APPROVED FOR TRAINING**
1. Clone repo: `git clone https://github.com/wilyub/VeriThoughts`
2. Download from HuggingFace as backup
3. Parse into JSONL format matching our schema
4. Tag examples with `source: verithoughts`, `verified: true`

---

## 2. CVDP (2024)

### Overview
- **Paper**: ["Comprehensive Verilog Design Problems"](https://arxiv.org/abs/2506.14074)
- **GitHub**: [github.com/NVlabs/cvdp_benchmark](https://github.com/NVlabs/cvdp_benchmark)
- **HuggingFace**: [nvidia/cvdp-benchmark-dataset](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset)
- **Size**: 783 human-authored problems across 13 categories

### License
**Dual License** (both permissive):
- Data: **CC BY 4.0** (attribution required)
- Code: **Apache 2.0** (permissive)
- Some derivative works: MIT, BSD-2-Clause (all compatible)

### Content
- RTL generation problems
- Design verification tasks
- Debugging scenarios
- Assertion creation
- Technical comprehension
- CocoTB testbenches (Python-based)

### Integration Plan
✅ **APPROVED FOR TRAINING**
1. Download from HuggingFace: `datasets/nvidia/cvdp-benchmark-dataset`
2. Extract RTL problems + testbenches
3. Convert cocotb Python tests → SystemVerilog testbenches (or keep Python)
4. Tag with `source: cvdp`, `testbench_type: cocotb`
5. Add attribution note in dataset metadata

---

## 3. SecFSM (2025)

### Overview
- **Paper**: ["SecFSM: Knowledge Graph-Guided Verilog Code Generation for Secure FSMs"](https://arxiv.org/html/2508.12910v1)
- **Status**: 🚫 **NOT YET PUBLIC**
- **Expected**: Code/data release upon paper acceptance

### License
**Unknown** - repo not published yet

### Content (from paper)
- Collected from academic datasets, papers, industrial cases
- 25 security test cases evaluated
- FSM Security Knowledge Graph (FSKG)
- Focus on hardware security vulnerabilities

### Action Plan
⏳ **MONITOR FOR RELEASE**
1. Check arXiv paper for updates
2. Search for GitHub repo monthly
3. Once released, review license terms
4. If permissive, integrate as high-priority FSM gold standard

---

## 4. VeriPrefer (2023)

### Overview
- **Paper**: Referenced in Agent 2's docs as RL-based dataset
- **Status**: 🔍 Not yet researched

### License
**TBD** - needs web search

### Action Plan
Agent 1 to research and update this section

---

## Legal Compliance

### Attribution Requirements
- **VeriThoughts**: Standard BSD credit in LICENSE file
- **CVDP**: Must include "Derived from NVIDIA CVDP dataset" in metadata
- **CocoTB**: BSD license credit (if using testbenches)

### Metadata Tracking
All external examples must include:
```json
{
  "source": "verithoughts|cvdp|secfsm|veriprefer",
  "license": "BSD-3-Clause|CC-BY-4.0|Apache-2.0",
  "attribution": "Original work by...",
  "dataset_version": "v1.0"
}
```

### Commercial Use
✅ **All approved datasets allow commercial training**
- BSD 3-Clause: Yes
- CC BY 4.0: Yes (with attribution)
- Apache 2.0: Yes

---

## Next Steps

### Immediate (This Week)
1. ✅ Download VeriThoughts from GitHub/HuggingFace
2. ✅ Download CVDP from HuggingFace
3. ⏳ Research VeriPrefer licensing
4. ⏳ Monitor SecFSM release status

### Integration (Next Week)
1. Write `convert_verithoughts.py` → JSONL
2. Write `convert_cvdp.py` → JSONL
3. Validate formal verification labels
4. Merge into `data/external/` with proper attribution

### Training
Once converted, these datasets can be merged with V4/V5 data for V5.2/V6 training with proper provenance tracking.
