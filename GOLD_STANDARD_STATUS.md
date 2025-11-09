# Gold Standard Placement Corpus - Status Report

**Target: 100 Examples (Gold 100 Sprint)**
**Current: 35 Examples (35% complete)**
**Average Quality Score: 95+**

## Sprint 1 Progress: AGGRESSIVE EXECUTION

### Phase 1: Fabricated Chips & Tapeouts (Target: 100)
**Current: 35/100 (35% of phase)**

**Recent additions (Session 2):**
- Ariane/CVA6 (GF 22FDX, 1.7 GHz)
- SweRV EH1/EH2 (28nm/16nm)
- Shakti IRIS-LV, Moushik, RiseCreek, RIMO (180nm/22nm)
- OR1200 (TSMC 0.18µm)
- Hybro framework (DAC 2024)
- Differentiable Timing-Driven (DAC 2022)
- SimPL, FastPlace 3.0, mPL6, Capo algorithms

#### Tier 1: Fabricated Commercial (Score 100+)
- ✓ OpenTitan Earl Grey (GF 22nm, 2024) - 145 pts
- ✓ OpenTitan 34-block integration - 140 pts
- ✓ Ibex RISC-V core - 125 pts

#### Tier 2: Academic Tapeouts (Score 80-120)
- ✓ BlackParrot multicore (GF 12nm FinFET) - 120 pts
- ✓ Google TPU v1 (28nm, 92 TOPS) - 120 pts
- ✓ Pulpissimo 8-core cluster (65nm) - 120 pts
- ✓ Rocket Chip/BOOM (16nm TSMC) - 115 pts
- ✓ Eyeriss v2 accelerator (65nm) - 115 pts
- ✓ CV32E40P CORE-V (GF 22FDX) - 110 pts
- ✓ OpenROAD 600+ tapeouts - 105 pts
- ✓ OpenLane 40+ SkyWater designs - 100 pts
- ✓ PicoRV32 (X-Fab 180nm, TSMC 130nm) - 95 pts

#### Tier 3: FPGA-Validated (Score 60-95)
- ✓ Nyuzi GPU processor - 95 pts
- ✓ SERV smallest RISC-V (2.1 kGE) - 80 pts
- ✓ VexRiscv configurable - 80 pts

#### Tier 4: Academic Papers (Score 40-75)
- ✓ ICCAD 2023 3D placement - 75 pts
- ✓ ISPD 2015 contest (NTUPlacerDR) - 70 pts
- ✓ ICCAD 2014-2015 timing-driven - 70 pts
- ✓ ISPD 2005 contest (APlace) - 65 pts
- ✓ VPR FPGA placement - 65 pts
- ✓ RePlAce benchmarks - 60 pts
- ✓ ICCAD 2024 MBFF optimization - 60 pts
- ✓ DREAMPlace ISPD 2015 - 55 pts

### Remaining Phases (Target: 477 examples)

**Phase 2: FPGA-Validated Designs (50 examples)**
- More RISC-V cores with FPGA results
- Custom accelerators (crypto, DSP, compression)
- Memory controllers and interfaces
- Network processors

**Phase 3: Academic Papers with Measurements (100 examples)**
- DAC papers (1990-2024)
- ICCAD papers (optimization techniques)
- ISPD papers (routing-driven, congestion-aware)
- DATE papers (European perspective)
- ASPDAC papers (Asian contributions)

**Phase 4: EDA Tool Techniques (100 examples)**
- Analytical placement methods
- Simulated annealing variations
- Partitioning-based approaches
- Machine learning for placement
- Legalization algorithms

**Phase 5: Benchmarks & Contests (150 examples)**
- ISPD 2006, 2013, 2014, 2016 contests
- DAC 2012 routability contest
- ICCAD 2012, 2013, 2016-2022 contests
- TAU timing contests
- Individual benchmark results (adaptec, bigblue, etc.)

**Phase 6: Parametric & Scenario Variations (100 examples)**
- Density variations (0.5-0.9) with detailed tradeoffs
- Timing weight variations
- Iteration count studies
- Design scenario types (high-perf, low-power, area-constrained)
- Error cases and debugging

## Coverage Analysis

### ✓ Achieved Coverage

**Processors:**
- OpenTitan, Ibex, Pulpissimo, Rocket/BOOM, CV32E40P, BlackParrot, PicoRV32, SERV, VexRiscv, Nyuzi

**Accelerators:**
- TPU v1, Eyeriss v2

**Tools & Algorithms:**
- DREAMPlace, RePlAce, APlace, NTUPlacerDR, VPR, OpenROAD, OpenLane

**Contests:**
- ISPD 2005, 2015
- ICCAD 2014, 2015, 2023, 2024

### 🎯 Priority Targets (Next 50 examples)

**More Accelerators (15):**
- TPU v2, v3 architecture details
- AWS Inferentia
- Cerebras WSE placement strategies
- NVIDIA tensor cores
- AMD Instinct accelerators
- Edge TPU, Neural Engine
- Custom DNN accelerators from academia

**More Processors (15):**
- Ariane/CVA6 (ETH Zurich)
- SweRV cores (Western Digital)
- Shakti (IIT Madras)
- BOOM variants (superscalar configs)
- Leon3 (Gaisler)
- OR1200 OpenRISC
- Amber ARM-compatible

**More Placement Algorithms (10):**
- FastPlace series (FastPlace 1.0, 2.0, 3.0)
- mPL series (mPL5, mPL6)
- Capo (multilevel partitioning)
- Dragon (min-cut based)
- SimPL (force-directed)

**More Contest Results (10):**
- ISPD 2006 details
- ICCAD 2012, 2013 results
- DAC 2012 routability results
- ISPD 2014 discrete gate sizing

## Path to 500: Execution Strategy

### Weeks 1-2: Mine Academic Literature (150 examples)
- Systematic search through DAC/ICCAD/ISPD proceedings
- Extract examples from 50+ papers
- Focus on papers with measured silicon/FPGA results
- Target: 3 examples per paper average

### Weeks 2-3: Benchmark & Contest Deep Dive (100 examples)
- Download all ISPD/ICCAD contest problem descriptions
- Extract winning strategies from top 3 teams per contest
- Document benchmark characteristics
- Create comparison tables

### Week 3-4: Tool Techniques Encyclopedia (100 examples)
- Study each major placement algorithm family
- Document algorithmic innovations
- Compare performance across benchmarks
- Extract lessons learned

### Week 4-5: Parametric Generation (100 examples)
- Systematic parameter sweeps with explanations
- Scenario-based examples (design types)
- Troubleshooting guide (common errors)
- Best practices and heuristics

### Week 5: Quality Control & Curation (27 examples to backfill)
- Review all examples for accuracy
- Ensure provenance scores are correct
- Rank by total score
- Select top 500 (will have ~527 by then)

## Quality Metrics

**Current Corpus Quality:**
- Provenance distribution:
  - Tier 1 (100+ pts): 3 examples (13%)
  - Tier 2 (80-120 pts): 9 examples (39%)
  - Tier 3 (60-79 pts): 5 examples (22%)
  - Tier 4 (40-59 pts): 6 examples (26%)

- Verification status:
  - Fabricated: 12 examples (52%)
  - Academic silicon: 1 example (4%)
  - FPGA validated: 3 examples (13%)
  - Academic benchmark: 5 examples (22%)
  - Contest validated: 2 examples (9%)

**Target Distribution (500 examples):**
- Tier 1: 50 examples (10%) - Focus on quality over quantity
- Tier 2: 150 examples (30%) - Academic tapeouts, major tools
- Tier 3: 150 examples (30%) - FPGA-validated, synthesis results
- Tier 4: 150 examples (30%) - Papers, contests, benchmarks

## Success Criteria

A trained specialist model with this corpus should be able to:

✓ **Answer fabrication questions:** "How was OpenTitan placed?" → Detailed, accurate answer with real results
✓ **Compare tools:** "DREAMPlace vs RePlAce?" → Quantitative comparison with benchmarks
✓ **Debug problems:** "15% routing overflow?" → Root cause + specific solution
✓ **Recommend strategies:** "Place DSP with macros?" → Concrete placement approach
✓ **Cite provenance:** "Based on..." → Reference to OpenTitan/ISPD/paper

**Quantitative Target:** Specialist should score 30%+ higher than baseline llama3:8b on placement question benchmark.

## Repository Structure

```
data/training/
├── PLACEMENT_GOLD_STANDARD.jsonl        # Current corpus (23 examples)
├── placement_massive_corpus.jsonl       # Earlier synthetic corpus
└── verilog_training_data.jsonl         # General chip design

training/data_preparation/
├── curate_gold_standard.py              # Main curator script
├── generate_placement_corpus.py         # Earlier generator
└── scale_placement_corpus.py            # Parametric generator
```

## Next Session Priorities

1. **Continue Phase 1** (fabricated chips): Target 30 more examples
   - Find more open-source tapeouts
   - Mine OpenROAD/OpenLane project gallery
   - Search for industry case studies (if publicly available)

2. **Start Phase 3** (academic papers): Begin systematic literature review
   - Set up automated search queries
   - Target DAC 2020-2024 recent papers first
   - Extract examples with concrete numbers

3. **Parallel work on Phase 5** (benchmarks): Low-hanging fruit
   - ISPD benchmark suite documentation
   - Contest problem statements (publicly available)
   - Create comparison tables

**Estimated completion:** 5-6 focused sessions at current pace (75-100 examples per session if mining papers aggressively).
