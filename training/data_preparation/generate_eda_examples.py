#!/usr/bin/env python3
"""
Generate EDA Tool-Specific Training Examples

Creates synthetic training examples for EDA tools from documentation
and templates. This augments the Verilog-heavy dataset with tool-specific
knowledge for synthesis, placement, routing, and timing analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAExampleGenerator:
    """Generate synthetic training examples for EDA tools"""

    def __init__(self):
        self.examples = []

    def generate_synthesis_examples(self) -> List[Dict]:
        """Generate Yosys synthesis training examples"""
        examples = [
            {
                'prompt': 'How do I synthesize a Verilog design with Yosys?',
                'response': '''To synthesize a Verilog design with Yosys:

1. Read the Verilog file:
   ```
   read_verilog design.v
   ```

2. Synthesize the design:
   ```
   synth -top top_module
   ```

3. Technology mapping (if using a library):
   ```
   dfflibmap -liberty tech_library.lib
   abc -liberty tech_library.lib
   ```

4. Write the output netlist:
   ```
   write_verilog synth_output.v
   ```

Complete example:
```tcl
yosys -p "read_verilog design.v; synth -top top_module; abc -liberty tech.lib; write_verilog output.v"
```''',
                'phase': 'synthesis',
                'source': 'synthetic',
                'tool': 'yosys'
            },
            {
                'prompt': 'What optimization goals can I use in Yosys synthesis?',
                'response': '''Yosys supports several optimization goals:

1. **Area Optimization** (-auto-top):
   - Minimizes gate count and chip area
   - Use: `synth -auto-top`
   - Best for area-constrained designs

2. **Speed Optimization**:
   - Reduces critical path delay
   - Use ABC with timing optimization
   - Command: `abc -fast`

3. **Power Optimization**:
   - Reduces switching activity
   - Use power-aware synthesis
   - Combine with clock gating

4. **Balanced** (default):
   - Balances area and timing
   - Use: `synth -top module_name`

Example for area optimization:
```tcl
read_verilog design.v
synth -top counter -flatten
opt -full
abc -liberty tech.lib
opt_clean
write_verilog -noattr optimized.v
```''',
                'phase': 'synthesis',
                'source': 'synthetic',
                'tool': 'yosys'
            },
            {
                'prompt': 'How do I check synthesis results in Yosys?',
                'response': '''To check synthesis results in Yosys:

1. **Cell Count Statistics**:
   ```
   stat
   ```
   Shows: number of cells, wires, flip-flops

2. **Check for Issues**:
   ```
   check
   ```
   Verifies design integrity

3. **View Hierarchy**:
   ```
   ls
   ```
   Lists all modules

4. **Show Design**:
   ```
   show
   ```
   Generates graphical view (requires dot)

5. **Cell Types**:
   ```
   stat -tech
   ```
   Shows technology-specific statistics

Example output:
```
Number of cells:           123
  AND2:                     45
  OR2:                      32
  DFF:                      10
  ...
```''',
                'phase': 'synthesis',
                'source': 'synthetic',
                'tool': 'yosys'
            },
        ]
        return examples

    def generate_placement_examples(self) -> List[Dict]:
        """Generate DREAMPlace training examples"""
        examples = [
            {
                'prompt': 'How do I run placement with DREAMPlace?',
                'response': '''To run placement with DREAMPlace:

1. **Prepare Input Files**:
   - Netlist (Verilog or BOOKSHELF format)
   - DEF file with floorplan
   - LEF file with cell/technology info

2. **Create Configuration JSON**:
```json
{
  "aux_input": "design.aux",
  "gpu": 1,
  "num_threads": 8,
  "target_density": 0.7,
  "global_place_flag": 1,
  "detailed_place_flag": 1
}
```

3. **Run Placement**:
```python
import dreamplace

placer = dreamplace.Placer(config_file="config.json")
result = placer.place()
```

4. **Check Results**:
   - HPWL (half-perimeter wirelength)
   - Overflow
   - Density distribution

Output will show:
- Final HPWL
- Runtime
- GPU utilization''',
                'phase': 'placement',
                'source': 'synthetic',
                'tool': 'dreamplace'
            },
            {
                'prompt': 'What parameters affect placement quality in DREAMPlace?',
                'response': '''Key DREAMPlace placement parameters:

1. **target_density** (0.0-1.0):
   - How densely to pack cells
   - Lower = more whitespace, better routability
   - Higher = less area, potential congestion
   - Typical: 0.6-0.8

2. **gamma** (wirelength smoothing):
   - Smoothing parameter for HPWL
   - Higher = smoother optimization
   - Range: 0.5-5.0

3. **num_bins_x/y**:
   - Grid resolution for density
   - More bins = finer control
   - Typical: 512x512 or 1024x1024

4. **global_place_stages**:
   - Number of global placement iterations
   - More stages = better convergence
   - Typical: 3-5

5. **learning_rate**:
   - Optimization step size
   - Too high = instability
   - Too low = slow convergence
   - Adaptive recommended

Example configuration:
```json
{
  "target_density": 0.7,
  "gamma": 2.0,
  "num_bins_x": 512,
  "num_bins_y": 512,
  "global_place_stages": 4
}
```''',
                'phase': 'placement',
                'source': 'synthetic',
                'tool': 'dreamplace'
            },
            {
                'prompt': 'How do I reduce wirelength in placement?',
                'response': '''Strategies to reduce wirelength in placement:

1. **Adjust Target Density**:
   - Lower density gives more room for optimization
   - Try: target_density = 0.6-0.7

2. **Enable Timing-Driven Mode**:
   ```json
   {
     "timing_driven": true,
     "timing_weight": 0.3
   }
   ```

3. **Use Multi-Stage Placement**:
   - Global placement first
   - Then detailed placement
   - Refine iteratively

4. **Optimize Net Weighting**:
   - Weight critical nets higher
   - Reduces delay on critical paths

5. **Adjust Wirelength Model**:
   - Use weighted HPWL
   - Consider routing congestion

6. **Clustering**:
   - Group related cells
   - Place clusters together

DREAMPlace automatically optimizes wirelength through:
- Gradient-based optimization
- GPU acceleration
- Density-aware smoothing

Monitor HPWL during placement:
```
Iteration 0: HPWL = 1,234,567
Iteration 10: HPWL = 987,654
Iteration 20: HPWL = 856,432
...
```

Lower HPWL = better placement quality.''',
                'phase': 'placement',
                'source': 'synthetic',
                'tool': 'dreamplace'
            },
        ]
        return examples

    def generate_routing_examples(self) -> List[Dict]:
        """Generate TritonRoute training examples"""
        examples = [
            {
                'prompt': 'How do I run detailed routing with TritonRoute?',
                'response': '''To run detailed routing with TritonRoute:

1. **Prepare Input Files**:
   - Placed DEF file
   - LEF files (technology + standard cells)
   - Routing guide (from global router)

2. **Create Parameter File** (route.param):
```
lef:tech.lef
lef:cells.lef
def:placed.def
guide:route.guide
output:routed.def
threads:8
verbose:1
```

3. **Run TritonRoute**:
```bash
TritonRoute route.param
```

4. **Check Results**:
   - Total wirelength
   - Number of vias
   - DRC violations

Output includes:
- Detailed routing statistics
- Via usage per layer
- DRC violation report
- Runtime metrics

TritonRoute performs:
- Track assignment
- Panel routing
- Via optimization
- DRC fixing''',
                'phase': 'routing',
                'source': 'synthetic',
                'tool': 'tritonroute'
            },
            {
                'prompt': 'How do I fix DRC violations after routing?',
                'response': '''To fix DRC (Design Rule Check) violations after routing:

1. **Identify Violations**:
   ```bash
   # TritonRoute reports violations in log
   grep "DRC" routing.log
   ```

2. **Common DRC Violations**:
   - Min spacing violations
   - Min width violations
   - Short circuits
   - Off-grid routing

3. **Fixing Strategies**:

   a) **Adjust Routing Parameters**:
      - Increase track spacing
      - Use wider wires for power
      - Add more routing layers

   b) **Reroute Specific Nets**:
      - Identify problematic nets
      - Rip-up and re-route
      - Use different layers

   c) **ECO (Engineering Change Order)**:
      - Manual fixes for critical violations
      - Use DEF editing tools

4. **Prevention**:
   - Start with good placement (low congestion)
   - Use appropriate target density
   - Enable DRC-aware routing

5. **Verification**:
```bash
# Run DRC checker
magic -noconsole -dnull <<EOF
load routed.def
drc check
quit
EOF
```

Iterate until DRC-clean:
```
Initial: 150 violations
After fix 1: 45 violations
After fix 2: 5 violations
After fix 3: 0 violations (clean!)
```''',
                'phase': 'routing',
                'source': 'synthetic',
                'tool': 'tritonroute'
            },
        ]
        return examples

    def generate_timing_examples(self) -> List[Dict]:
        """Generate OpenSTA training examples"""
        examples = [
            {
                'prompt': 'How do I run static timing analysis with OpenSTA?',
                'response': '''To run static timing analysis with OpenSTA:

1. **Prepare Input Files**:
   - Netlist (gate-level Verilog)
   - Liberty (.lib) files
   - SDC constraints file
   - SPEF (parasitic extraction, optional)

2. **Create TCL Script** (sta.tcl):
```tcl
# Read liberty files
read_liberty fast_corner.lib
read_liberty slow_corner.lib

# Read netlist
read_verilog netlist.v

# Link design
link_design top_module

# Read constraints
read_sdc constraints.sdc

# Read parasitics (if available)
read_spef parasitics.spef

# Report timing
report_checks -path_delay min_max
report_tns
report_wns
report_worst_slack

# Report specific paths
report_checks -from input_reg -to output_reg
```

3. **Run OpenSTA**:
```bash
sta sta.tcl
```

4. **Interpret Results**:
   - WNS (Worst Negative Slack): < 0 = violation
   - TNS (Total Negative Slack): sum of all violations
   - Slack: positive = meeting timing, negative = violation

Example output:
```
Startpoint: data_reg (rising edge-triggered flip-flop)
Endpoint: output_reg (rising edge-triggered flip-flop)
Path Group: clk
Path Type: max

  Delay    Time   Description
----------------------------------------------------------
  0.00    0.00   clock clk (rise edge)
  0.50    0.50   clock network delay
  0.10    0.60   data_reg/CK (DFF)
  0.25    0.85   data_reg/Q (DFF)
  1.45    2.30   U45/A (AND2)
  0.32    2.62   U45/Y (AND2)
  0.18    2.80   output_reg/D (DFF)
          2.80   data arrival time

  10.00   10.00  clock clk (rise edge)
  0.50    10.50  clock network delay
  -0.20   10.30  clock uncertainty
  0.00    10.30  output_reg/CK (DFF)
  -0.15   10.15  library setup time
          10.15  data required time
----------------------------------------------------------
          10.15  data required time
          -2.80  data arrival time
----------------------------------------------------------
           7.35  slack (MET)
```''',
                'phase': 'timing_analysis',
                'source': 'synthetic',
                'tool': 'opensta'
            },
            {
                'prompt': 'How do I fix setup time violations?',
                'response': '''To fix setup time violations:

1. **Identify Critical Paths**:
```tcl
report_checks -path_delay max -slack_max 0.0 -n 20
```

2. **Common Fixes**:

   a) **Optimize Logic**:
      - Reduce logic depth
      - Use faster cells (higher drive strength)
      - Remove redundant gates

   b) **Buffer Insertion**:
      - Add buffers to reduce delay
      - Break long wires
      ```
      # In synthesis
      insert_buffer -net long_net
      ```

   c) **Cell Sizing**:
      - Upsize cells on critical path
      - Use higher VT (threshold voltage) cells
      ```
      size_cell critical_gate AND2_X4
      ```

   d) **Retiming**:
      - Move registers to balance paths
      - Use synthesis tool retiming

   e) **Clock Tree Optimization**:
      - Balance clock skew
      - Reduce clock network delay

3. **Iterative Process**:
```
Initial WNS: -1.5ns
After upsizing: -0.8ns
After buffering: -0.3ns
After retiming: +0.2ns (MET!)
```

4. **Verify Fix**:
```tcl
report_checks -from start_reg -to end_reg
report_slack
```

5. **ECO Flow**:
```tcl
# Make changes
size_cell U123 AND2_X8

# Update timing
update_timing

# Re-check
report_checks
```

Remember:
- Fixing setup may worsen hold
- Check both setup and hold timing
- Consider power impact of upsizing''',
                'phase': 'timing_analysis',
                'source': 'synthetic',
                'tool': 'opensta'
            },
            {
                'prompt': 'What is the difference between setup and hold time violations?',
                'response': '''Setup vs Hold Time Violations:

**Setup Time Violation:**
- Data must arrive BEFORE clock edge
- Checked with: max delay analysis
- Violation: data arrives too late
- Slack = required_time - arrival_time
- If slack < 0: VIOLATION

Example:
```
Clock period: 10ns
Data arrives at: 9.8ns
Setup time required: 0.5ns
Required arrival: 10ns - 0.5ns = 9.5ns
Slack: 9.5ns - 9.8ns = -0.3ns (VIOLATION!)
```

**Hold Time Violation:**
- Data must stay stable AFTER clock edge
- Checked with: min delay analysis
- Violation: data changes too quickly
- Slack = arrival_time - required_time
- If slack < 0: VIOLATION

Example:
```
Hold time required: 0.2ns
Data arrives at: 0.1ns (too fast!)
Required: 0.2ns
Slack: 0.1ns - 0.2ns = -0.1ns (VIOLATION!)
```

**Key Differences:**

| Aspect | Setup | Hold |
|--------|-------|------|
| Timing | Before clock | After clock |
| Path type | Long paths | Short paths |
| Fix | Speed up | Slow down |
| Method | Upsize, buffer | Add delay, downsize |

**Checking Both:**
```tcl
# Setup (max delay)
report_checks -path_delay max

# Hold (min delay)
report_checks -path_delay min

# Both together
report_checks -path_delay min_max
```

**Typical Fixes:**
- Setup: Faster cells, better placement
- Hold: Add delay cells, buffers

Important: Fixing one can impact the other!''',
                'phase': 'timing_analysis',
                'source': 'synthetic',
                'tool': 'opensta'
            },
        ]
        return examples

    def generate_power_examples(self) -> List[Dict]:
        """Generate power analysis training examples"""
        examples = [
            {
                'prompt': 'How do I reduce power consumption in a chip design?',
                'response': '''Strategies to reduce power consumption:

**1. Clock Gating:**
```verilog
// Disable clock when logic is idle
wire gated_clk = clk & enable;

always @(posedge gated_clk) begin
    // Logic only runs when enabled
    data_reg <= data_in;
end
```

**2. Multi-Threshold Voltage (Multi-VT):**
- High-VT: Low leakage, slow (non-critical paths)
- Standard-VT: Balanced
- Low-VT: Fast, high leakage (critical paths only)

**3. Power Gating:**
- Shut down unused blocks completely
- Use power switches
```
# In constraints
set_power_gating domain1 -state off
```

**4. Dynamic Voltage/Frequency Scaling (DVFS):**
- Lower voltage when high performance not needed
- Reduce frequency during idle

**5. Design Optimization:**
- Reduce switching activity
- Minimize capacitance
- Short interconnects
- Efficient FSM encoding

**6. Technology Selection:**
- Smaller process node (but higher leakage!)
- SOI (Silicon-On-Insulator)
- FinFET technology

**Power Components:**
```
Total Power = Dynamic + Static

Dynamic = α × C × V² × f
  α = switching activity
  C = capacitance
  V = voltage
  f = frequency

Static = Leakage × V
```

**Power Analysis:**
```tcl
# In synthesis tool
report_power
read_saif activity.saif
update_power
report_power -hier
```

**Expected Savings:**
- Clock gating: 20-40% dynamic power
- Multi-VT: 15-30% leakage power
- Power gating: 50-90% block power
- DVFS: 30-70% (mode dependent)''',
                'phase': 'power_analysis',
                'source': 'synthetic',
                'tool': 'general'
            },
        ]
        return examples

    def generate_all_examples(self):
        """Generate all EDA tool examples"""
        logger.info("Generating EDA tool-specific examples...")

        self.examples.extend(self.generate_synthesis_examples())
        logger.info(f"  ✓ Synthesis: {len([e for e in self.examples if e['phase'] == 'synthesis'])} examples")

        self.examples.extend(self.generate_placement_examples())
        logger.info(f"  ✓ Placement: {len([e for e in self.examples if e['phase'] == 'placement'])} examples")

        self.examples.extend(self.generate_routing_examples())
        logger.info(f"  ✓ Routing: {len([e for e in self.examples if e['phase'] == 'routing'])} examples")

        self.examples.extend(self.generate_timing_examples())
        logger.info(f"  ✓ Timing: {len([e for e in self.examples if e['phase'] == 'timing_analysis'])} examples")

        self.examples.extend(self.generate_power_examples())
        logger.info(f"  ✓ Power: {len([e for e in self.examples if e['phase'] == 'power_analysis'])} examples")

        logger.info(f"\nTotal: {len(self.examples)} EDA tool examples")

    def save_examples(self, output_dir: str = "./data/training/phase_specific"):
        """Save examples to phase-specific files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("\nAppending to phase-specific datasets...")

        # Group by phase
        by_phase = {}
        for example in self.examples:
            phase = example['phase']
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(example)

        # Append to existing files
        for phase, examples in by_phase.items():
            output_file = output_path / f"{phase}_training.jsonl"

            with open(output_file, 'a', encoding='utf-8') as f:
                for example in examples:
                    training_record = {
                        'prompt': example['prompt'],
                        'response': example['response'],
                        'metadata': {
                            'phase': example['phase'],
                            'source': example['source'],
                            'tool': example.get('tool', 'general')
                        }
                    }
                    f.write(json.dumps(training_record) + '\n')

            logger.info(f"  ✓ Appended {len(examples)} examples to {output_file.name}")


def main():
    """Generate and save EDA examples"""
    logger.info("="*60)
    logger.info("EDA Tool-Specific Training Data Generation")
    logger.info("="*60)

    generator = EDAExampleGenerator()

    # Generate examples
    generator.generate_all_examples()

    # Save to files
    generator.save_examples()

    logger.info("\n" + "="*60)
    logger.info("✅ EDA example generation complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Review augmented datasets in: data/training/phase_specific/")
    logger.info("2. Run: training/finetune_specialist.py --phase synthesis --size 8b")
    logger.info("3. Train specialist models with enhanced datasets")


if __name__ == "__main__":
    main()
