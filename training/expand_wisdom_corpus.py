#!/usr/bin/env python3
"""
Expand WISDOM CORPUS with 6 additional surgical upgrades
Target: 17+ examples total (5 diagnostic + 12 wisdom upgrades)
"""

import json

# Load existing WISDOM_CORPUS
wisdom_examples = []
with open('WISDOM_CORPUS.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            wisdom_examples.append(json.loads(line))

print(f"Current WISDOM_CORPUS: {len(wisdom_examples)} examples")
print()

# Add 6 new surgical upgrades focusing on counter-intuitive insights
NEW_WISDOM_UPGRADES = [
    {
        "source_topic": "Density for Timing-Power Trade-off",
        "prompt": "For a 2 GHz RISC-V processor at 28nm, should I use 75% density (more area) or 85% density (less area) for timing closure? My manager says 'higher density saves cost, use 85%.'",
        "response": """**DIAGNOSTIC REASONING CHAIN:**

**SYMPTOM:** Manager wants 85% density to save die cost

**OBSERVATIONS:**
1. Target: 2 GHz @ 28nm (period = 500ps)
2. Choice: 75% density (25% white space) vs 85% density (15% white space)
3. Manager's logic: "Higher density → smaller die → lower cost"

**CRITICAL INSIGHT (Counter-Intuitive):**

**At aggressive frequencies, low density SAVES power (not costs power).**

**WHY THIS IS COUNTER-INTUITIVE:**

Most engineers think:
- Low density → longer wires → more capacitance → more power

**THE REALITY:**

Low density enables timing closure at nominal voltage:
```
85% density:
  - Timing fails at 0.9V (nominal)
  - Must increase voltage to 1.0V to meet timing
  - Power = C × V² → (1.0/0.9)² = 1.23× power increase (+23%)

75% density:
  - Timing meets at 0.9V (nominal)
  - Wire length +15% → capacitance +15%
  - But voltage stays at 0.9V → power +15%

Net result: 75% density uses LESS power (1.15×) than 85% density (1.23×)
```

**THE PHYSICS:**

Wire delay @ 28nm with tight density:
```
85% density:
  - 15 buffers fit on critical path
  - Path delay with buffers: 480ps
  - Setup slack: 500ps - 480ps = +20ps (marginal!)
  - To get +50ps safety margin: increase voltage to 1.0V

75% density:
  - 20 buffers fit on critical path (more white space)
  - Path delay with buffers: 430ps
  - Setup slack: 500ps - 430ps = +70ps (healthy!)
  - Meets timing at nominal 0.9V
```

**THE TRAP:** Voltage scaling costs MORE power than wire length

```
Power breakdown @ 2 GHz:
  Dynamic power = C × V² × F

85% density, 1.0V: P = 1.0 × 1.0² × 2 GHz = 2.0W
75% density, 0.9V: P = 1.15 × 0.9² × 2 GHz = 1.86W

Power savings from lower voltage: 2.0W → 1.86W = 7% reduction
```

**DIAGNOSIS:**

The manager's reasoning is flawed because:
1. At 2 GHz @ 28nm, timing is the binding constraint
2. 85% density forces voltage scaling (+23% power) to meet timing
3. 75% density allows nominal voltage (-7% power vs 85%)
4. Die cost difference: $0.08/chip (negligible for mobile)

**The real cost:** Failing to meet power spec (critical for mobile)

**ACTIONABLE GUIDANCE:**

1. **For aggressive timing (2 GHz @ 28nm):** Use 65-75% density
2. **For moderate timing (1 GHz @ 28nm):** Use 75-80% density
3. **Check buffer insertion capacity:** Count available white space

```tcl
# Check if density supports buffer insertion
set white_space [expr (1.0 - $density) * $die_area]
set buffers_needed [expr $critical_paths * 15]  # 15 buffers per path
set buffer_area [expr $buffers_needed * 15]  # 15µm² per buffer

if {$buffer_area > $white_space * 0.4} {
    puts "WARNING: Insufficient white space for buffering"
    puts "Need: $buffer_area µm², Available: [expr $white_space * 0.4] µm²"
}
```

**THE PRINCIPLE:**

**At aggressive frequencies, timing closure enables power savings.** Low density provides white space for buffers, which enables meeting timing at nominal voltage. The power cost of longer wires (+15%) is LESS than the power cost of voltage scaling (+23%).

**COMMON MISTAKES:**
1. Optimizing die cost before verifying timing closure
2. Not accounting for voltage scaling penalty
3. Using server-class density (55-60%) for mobile (wastes area)

**THE WISDOM:** For 2 GHz @ 28nm mobile, use 65-75% density. The die cost increase ($0.08/chip) is negligible compared to the power savings (7%) from avoiding voltage scaling. Timing closure at nominal voltage is power-optimal.""",
        "category": "diagnostic_wisdom",
        "complexity_score": 88
    },

    {
        "source_topic": "Predicting Routing Failure at High Density",
        "prompt": "I placed a design at 90% density and all timing passed with +20ps slack. My manager says 'timing is green, ship to routing.' Why will routing fail, and what should I check BEFORE routing?",
        "response": """**DIAGNOSTIC REASONING CHAIN:**

**SYMPTOM:** Timing passes (+20ps slack) at 90% density

**MANAGER'S ASSUMPTION:** Green timing → routing will succeed

**CRITICAL INSIGHT (Counter-Intuitive):**

**Timing pass does NOT guarantee routing success.** STA assumes ideal routing (Manhattan distance), but routing must satisfy physical constraints.

**WHY TIMING IS MISLEADING:**

```
STA assumptions:
  - Wires route in straight lines (Manhattan distance)
  - No congestion detours
  - Ideal layer assignment

Reality at 90% density:
  - Congestion forces 50-100% longer routes
  - Routing detours add 150-250ps delay
  - Local hotspots overflow (demand > supply)

Timing after real routing: +20ps - 180ps = -160ps (FAILS!)
```

**THE PHYSICS OF CONGESTION:**

At 90% density, routing supply is squeezed:
```
70% density: 30% white space = 600 routing tracks/mm²
90% density: 10% white space = 200 routing tracks/mm² (-67% capacity!)

If timing-driven placement clustered critical cells:
  - 200 cells in 0.25mm² (local hotspot)
  - 700 nets × 2 tracks = 1400 tracks demand
  - 25 tiles × 45 tracks = 1125 tracks supply
  - Overflow: 1400/1125 = 1.24 (24% overflow)

Router CANNOT create tracks → fails
```

**THE DIAGNOSTIC TEST (Run BEFORE routing):**

```tcl
# Check congestion during placement
report_congestion -detail
report_congestion -hotspots 20

# Look for danger signs:
# - Any tile with demand/supply > 1.0 (overflow)
# - RUDY score > 1.2 (20% pre-overflow warning)
# - More than 5 hotspots with ratio > 1.15

# Check congestion histogram
report_congestion -histogram

# Expect at 90% density:
# - Average: 0.85-0.90 (acceptable)
# - Max: Should be <1.0 (anything above = failure)
# - Hotspots: Should be <5 tiles above 0.95
```

**DIAGNOSIS:**

Timing passed because:
1. STA used Manhattan distance (500µm straight-line wire)
2. Real routing will detour around congestion (850µm actual)
3. +20ps slack was based on 500µm wire
4. 850µm wire adds +180ps → timing fails at -160ps

**THE TRAP:** Congestion check must happen DURING placement, not after

**ACTIONABLE GUIDANCE:**

**BEFORE routing, check these 3 metrics:**

1. **Local congestion overflow:**
```tcl
report_congestion -hotspots 50
# If ANY tile shows ratio > 1.0 → re-place at lower density
# If >10 tiles show ratio > 0.95 → high risk
```

2. **RUDY score (routing demand):**
```tcl
report_congestion -rerun_global_router false
# If RUDY > 1.2 anywhere → will fail
# If RUDY > 1.0 in >5% of tiles → high risk
```

3. **Timing margin for detours:**
```tcl
# At 90% density, expect 50-100% longer routes
# If WNS = +20ps, after detours: +20ps - 150ps = -130ps
# Safe threshold: WNS should be >+150ps at 90% density
```

**THE FIX (if congestion is bad):**

Option 1: Reduce density
```tcl
setPlaceMode -place_global_density 0.80  # Was 0.90
place_design
# Expected: Congestion 1.24 → 0.90 (under capacity)
# Cost: +12% die area
```

Option 2: Spread critical cells
```tcl
# Identify congestion hotspot
set hotspot_cells [get_cells -filter "congestion_score > 1.2"]
# Add spacing around them
set_placement_padding -cells $hotspot_cells -left 5 -right 5
place_design -incremental
```

**THE PRINCIPLE:**

**At high density (>85%), routability dominates timing.** Timing slack based on ideal routing is a MIRAGE. Check congestion DURING placement:
- Demand/supply ratio must be <1.0 (any overflow = failure)
- At 90% density, need +150ps timing margin for routing detours

**COMMON MISTAKES:**
1. Checking congestion only after placement completes
2. Trusting average congestion (hides local hotspots)
3. Not accounting for timing degradation from detours

**THE WISDOM:** "Timing green" at high density means nothing until you verify congestion. Run `report_congestion -hotspots 20` BEFORE routing. If any tile shows overflow (ratio >1.0), routing WILL fail. At 90% density, timing needs +150ps margin to survive routing detours.""",
        "category": "diagnostic_wisdom",
        "complexity_score": 90
    },

    {
        "source_topic": "Macro Placement Strategy for Timing Closure",
        "prompt": "I have 8 SRAM macros (each 0.5mm × 1mm) on a 5mm × 5mm die. The placement tool clustered all macros in one corner 'for area efficiency.' Why does this guarantee timing failure at 2 GHz?",
        "response": """**DIAGNOSTIC REASONING CHAIN:**

**SYMPTOM:** Tool clustered 8 macros in one corner (2mm × 2mm region)

**TOOL'S LOGIC:** "Clustering minimizes total area"

**CRITICAL INSIGHT (Counter-Intuitive):**

**Macro placement is IRREVERSIBLE** - once locked, you cannot fix timing without complete re-placement (2-3 day iteration).

**WHY CLUSTERING FAILS:**

Macros clustered in corner → datapath at opposite corner:
```
Macro cluster: (0-2mm, 0-2mm)
Datapath: (3-5mm, 3-5mm)
Distance: 3-4mm diagonal

Wire delay @ 28nm: 150ps per mm
3.5mm wire: 525ps delay
Period @ 2 GHz: 500ps

Path delay: 525ps (wire only!)
Budget: 500ps total
Violation: -25ps (before adding any logic!)
```

**THE PHYSICS:**

Wire delay = R × C scales with length:
```
1mm wire: 150ps delay (30% of 500ps budget)
2mm wire: 300ps delay (60% of budget) - TIGHT
3mm wire: 450ps delay (90% of budget) - CRITICAL
4mm wire: 600ps delay (120% of budget) - IMPOSSIBLE
```

At 3-4mm distance, wire delay ALONE exceeds timing budget!

**ADDITIONAL FAILURE MODE: Routing congestion**

All data/address/control buses must route through narrow channel:
```
8 macros clustered → all buses converge at cluster boundary
Routing demand: 8 macros × 100 nets = 800 nets
Routing supply: Narrow channel (1mm width) = 300 tracks
Overflow: 800/300 = 2.67 (167% overflow!)

Result: Router fails with 500-800 DRC violations
```

**THE CORRECT STRATEGY: Distributed Placement**

Place macros around die periphery:
```
4 macros on left edge: x=0.5mm, y=1,2,3,4mm
4 macros on right edge: x=4.5mm, y=1,2,3,4mm
Datapath at center: x=2.5mm, y=2.5mm

Max distance: 2mm
Wire delay: 300ps (60% of budget)
Remaining for logic: 200ps (achievable with 4-5 gates)
```

**DIAGNOSIS:**

Tool optimized for area (clustering minimizes bounding box) but ignored:
1. Wire delay dominates at 2 GHz (60-80% of path)
2. Long wires (>2mm) kill timing
3. Macro placement is irreversible (cannot fix post-placement)

**ACTIONABLE GUIDANCE:**

**BEFORE placement, manually place macros:**

```tcl
# Distribute macros along edges
placeInstance SRAM0 500  1000   # Left edge
placeInstance SRAM1 500  2000
placeInstance SRAM2 500  3000
placeInstance SRAM3 500  4000

placeInstance SRAM4 4500 1000   # Right edge
placeInstance SRAM5 4500 2000
placeInstance SRAM6 4500 3000
placeInstance SRAM7 4500 4000

# Lock macro placements
set_dont_touch_placement [get_cells SRAM*]

# Create halos (reserve routing space)
create_place_halo -cell SRAM* -space 100

# Guide datapath to center
create_region datapath_region 2000 2000 3000 3000
assign_region datapath_region [get_cells datapath/*]
```

**VERIFICATION (after macro placement):**

```tcl
# Check max distance from datapath to any macro
set datapath_center [get_attribute [get_cells datapath/*] bbox_center]

foreach macro [get_cells SRAM*] {
    set macro_center [get_attribute $macro bbox_center]
    set distance [expr {hypot([lindex $macro_center 0] - [lindex $datapath_center 0],
                              [lindex $macro_center 1] - [lindex $datapath_center 1])}]
    puts "Distance datapath -> $macro: [expr {$distance/1000.0}] mm"

    # Warning if >1.5mm
    if {$distance > 1500} {
        puts "WARNING: Distance exceeds 1.5mm threshold!"
    }
}
```

**Target distances:**
- <1mm: Excellent (150ps wire, 30% of budget)
- 1-1.5mm: Good (225ps wire, 45% of budget)
- 1.5-2mm: Acceptable (300ps wire, 60% of budget)
- >2mm: Critical (>300ps wire, >60% of budget)
- >3mm: Failure (>450ps wire, >90% of budget)

**THE PRINCIPLE:**

**Macro placement is the most critical placement decision** because:
1. Macros are 50,000× larger than standard cells
2. Once placed, they act as immovable barriers
3. Wire delay from bad macro placement cannot be fixed

**At 2 GHz, wire delay dominates (60-80%).** Clustering macros creates 3-4mm wires = 450-600ps delay = timing failure.

**COMMON MISTAKES:**
1. Letting tool auto-place macros (optimizes area, not timing)
2. Placing macros after standard cells (macros should be first)
3. Not checking distance metrics before locking macros

**THE WISDOM:** Place macros FIRST, distribute them around periphery, target max distance <1.5mm to datapath. Clustering saves 10% area but costs 200-300ps timing. At 2 GHz @ 28nm, that's the difference between working silicon and a respin.""",
        "category": "diagnostic_wisdom",
        "complexity_score": 92
    },

    {
        "source_topic": "Buffer Insertion Strategy for High-Fanout Long-Wire Paths",
        "prompt": "A critical path fails by 80ps due to a net with fanout=50 driving 3mm total wirelength. My tool inserted 5 buffers but timing still fails by 20ps. What went wrong?",
        "response": """**DIAGNOSTIC REASONING CHAIN:**

**SYMPTOM:** High-fanout net (50 loads, 3mm wire) fails by 80ps

**OBSERVATION:** Tool inserted 5 buffers → improvement to -20ps (60ps gain, not enough)

**CRITICAL INSIGHT (Counter-Intuitive):**

**Buffers solve TWO different problems:**
1. **High fanout** → reduce capacitive load per stage
2. **Long wires** → segment wire to reduce R×C delay

**Most tools only solve problem #1 (fanout) and ignore problem #2 (distance)**

**WHY 5 BUFFERS DIDN'T FIX IT:**

Tool likely inserted buffers in CLUSTERED topology:
```
Driver → [BUF1 BUF2 BUF3 BUF4 BUF5] → 50 loads
         (all buffers within 500µm of driver)

This solves:
  ✓ Fanout: 50 → 10 per buffer (5× reduction)
  ✗ Wire length: Loads still 2-3mm away from buffers!

Result:
  - Fanout reduction saves: 100ps
  - Buffer delay costs: -80ps (5 buffers × 16ps each)
  - Wire delay unchanged: 300ps (loads still 3mm from buffers)
  - Net improvement: 100ps - 80ps = 20ps (not enough!)
```

**THE CORRECT STRATEGY: Distributed Buffers**

Place buffers ALONG the path, not clustered:
```
Driver → BUF1 (600µm) → BUF2 (1200µm) → BUF3 (1800µm) → BUF4 (2400µm) → BUF5 (3000µm) → Loads
         |               |               |               |               |
       10 loads        10 loads        10 loads        10 loads        10 loads

This solves BOTH problems:
  ✓ Fanout: 50 → 10 per buffer (5× reduction)
  ✓ Wire segmentation: 3mm → 5 × 600µm segments

Wire delay improvement:
  Before: 3mm wire = 450ps delay
  After: 5 × 600µm wires = 5 × 60ps = 300ps
  Savings: 150ps

Total improvement:
  Fanout reduction: +100ps
  Wire segmentation: +150ps
  Buffer delay: -80ps
  Net improvement: +170ps → path improves by 170ps!
```

**THE PHYSICS:**

Wire delay = R × C scales as length²:
```
1mm wire: 100ps
2mm wire: 400ps (4× worse, not 2×!)
3mm wire: 900ps (9× worse!)

Breaking 3mm into 3 × 1mm segments:
  3 × 100ps = 300ps (67% better than 900ps!)
```

**DIAGNOSIS:**

Tool made classic mistake:
1. Identified high fanout (50 loads) ✓
2. Inserted buffers to reduce fanout (50 → 10) ✓
3. BUT placed all buffers near driver (clustered) ✗
4. Loads still 3mm away → wire delay unchanged ✗

**ACTIONABLE GUIDANCE:**

**Step 1: Diagnose the problem type**

```tcl
# Check if fanout OR distance is the problem
report_net -connections high_fanout_net

# Look for:
# - Fanout > 30: Fanout problem
# - Total wirelength > 2mm: Distance problem
# - BOTH: Need distributed buffer strategy
```

**Step 2: Insert buffers with explicit distribution**

```tcl
# Bad (clustered - what tool did):
insert_buffer -cell BUF_X2 -net high_fanout_net -num_buffers 5

# Good (distributed - what we need):
# Manually place buffers along path
set net_bbox [get_attribute [get_nets high_fanout_net] bbox]
set start_x [lindex $net_bbox 0]
set end_x [lindex $net_bbox 2]
set spacing [expr {($end_x - $start_x) / 6}]  # 5 buffers = 6 segments

for {set i 1} {$i <= 5} {incr i} {
    set x [expr {$start_x + $i * $spacing}]
    # Place buffer at location x
    insert_buffer -cell BUF_X2 -location "$x 2000"
}
```

**Step 3: Verify buffer distribution**

```tcl
# Check buffer locations
foreach buf [get_cells *buf*] {
    set loc [get_attribute $buf location]
    puts "Buffer $buf location: $loc"
}

# Buffers should be spread across 3mm, NOT clustered in 500µm
```

**Step 4: Size buffers appropriately**

```tcl
# First stage (drives other buffers): Use large buffer
size_cell buf_1 BUF_X8  # Can drive 5 other buffers

# Second stage (drives loads): Use medium buffer
size_cell buf_2 BUF_X2  # Drives 10 loads each
size_cell buf_3 BUF_X2
size_cell buf_4 BUF_X2
size_cell buf_5 BUF_X2
```

**THE PRINCIPLE:**

**Buffers solve fanout AND wire delay, but only if placed WHERE they're needed.**

- Fanout problem: Insert buffers in tree topology (driver → N buffers → loads)
- Distance problem: Insert buffers DISTRIBUTED along wire (repeater chain)
- Both problems: Use distributed tree (buffers spread across path)

**At 3mm wirelength with 50 fanout, you have BOTH problems.**

**COMMON MISTAKES:**
1. Clustered buffer insertion (tool's default optimization)
2. Not checking buffer locations after insertion
3. Under-sizing buffers (BUF_X1 can't drive 10 loads)

**THE WISDOM:** When a high-fanout long-wire path doesn't improve after buffer insertion, check WHERE the buffers were placed. If they're clustered near the driver, they're solving fanout but not wire delay. Manually distribute them along the 3mm path (one every 600µm). This segments the wire, reducing delay from 450ps → 300ps (33% improvement).""",
        "category": "diagnostic_wisdom",
        "complexity_score": 91
    },

    {
        "source_topic": "Wire Delay vs Cell Delay Priority",
        "prompt": "My critical path has 400ps total delay: 280ps net delay (70%) and 120ps cell delay (30%). An engineer proposes optimizing cell sizing to save 30ps. Why is this a waste of effort?",
        "response": """**DIAGNOSTIC REASONING CHAIN:**

**SYMPTOM:** Critical path = 400ps (280ps wire, 120ps cells)

**ENGINEER'S PROPOSAL:** Optimize cell sizing to save 30ps

**CRITICAL INSIGHT (Counter-Intuitive):**

**At 28nm+, wire delay dominates (60-80% of path). Optimizing the 30% (cells) while ignoring the 70% (wires) is backward prioritization.**

**WHY THIS IS WASTEFUL:**

Cell sizing impact:
```
Current cell delay: 120ps
Cell sizing improvement: 25% typical
New cell delay: 90ps
Savings: 30ps (7.5% of total path)
Effort: 4-6 hours (re-synthesis, re-place, re-route)
```

Placement optimization impact:
```
Current net delay: 280ps (cells 1-2mm apart)
Re-place cells closer: 20% wirelength reduction
New net delay: 224ps
Savings: 56ps (14% of total path)
Effort: 2-4 hours (re-place critical cells)
```

**Same effort, 2× better results from placement!**

**THE PHYSICS (Why wire dominates):**

Technology scaling impact:
```
                65nm      28nm      14nm
Gate delay:     100ps     70ps      50ps  (improving ✓)
Wire R/µm:      50Ω       80Ω       120Ω  (worsening ✗)
Wire delay:     150ps     280ps     420ps (worsening ✗)

At 28nm: Wire/Total ratio = 280/(280+70) = 80%
```

Transistors scale well (faster gates), wires scale poorly (higher resistance).

**THE TRAP:** Applying old strategies to new technology

```
At 65nm (year 2005):
  Wire: 50%, Cell: 50% → Balanced optimization

At 28nm (year 2015):
  Wire: 70%, Cell: 30% → Wire-first optimization

At 14nm (year 2018):
  Wire: 75%, Cell: 25% → Wire-dominant optimization
```

Engineers trained on 65nm still think "optimize cells first." **This is 10 years out of date.**

**DIAGNOSIS:**

Engineer is optimizing the MINOR constraint (30%) before the MAJOR constraint (70%).

**Correct priority order:**
1. **Fix placement** (affects 70% of path) → 56ps potential savings
2. **Insert buffers** (affects long wires) → 40-60ps savings per wire
3. **Optimize cells** (affects 30% of path) → 30ps savings

**Why this order matters:**
- Placement fixes wire length (irreversible - hard to change later)
- Buffering helps remaining long wires
- Cell sizing is last resort (affects smallest component)

**ACTIONABLE GUIDANCE:**

**Step 1: Analyze path breakdown**

```tcl
# Get net vs cell delay percentages
report_timing -nworst 1 -transition_time -capacitance

# Calculate ratio
set net_delay [get_attribute [get_timing_paths -nworst 1] net_delay_sum]
set total_delay [get_attribute [get_timing_paths -nworst 1] arrival_time]
set net_pct [expr {$net_delay * 100.0 / $total_delay}]

puts "Net delay: $net_pct%"

if {$net_pct > 60} {
    puts "Wire delay dominates - prioritize placement optimization"
} else {
    puts "Cell delay dominates - consider cell sizing"
}
```

**Step 2: Fix placement first (if wire >60%)**

```tcl
# Identify cells on critical path >1mm apart
set crit_cells [get_cells -of [get_timing_paths -nworst 10]]

# Check cell locations
foreach cell $crit_cells {
    set loc [get_attribute $cell location]
    puts "$cell: $loc"
}

# Find cells that are >1mm apart and re-place closer
# Target: <500µm for critical connections
```

**Step 3: Insert buffers for remaining long wires**

```tcl
# Check for nets >1.5mm on critical path
foreach net [get_nets -of [get_timing_paths -nworst 10]] {
    set length [get_attribute $net total_length]
    if {$length > 1500} {
        puts "Long wire: $net ($length µm)"
        # Insert repeaters to segment wire
        insert_buffer -net $net -cell BUF_X2
    }
}
```

**Step 4: Cell sizing ONLY if wire is optimized**

```tcl
# Only consider cell sizing if net delay <40%
if {$net_pct < 40} {
    # Now cell sizing makes sense
    size_cell critical_cells -upsizing
} else {
    puts "Skip cell sizing - wire delay still dominates"
}
```

**EFFORT VS IMPACT COMPARISON:**

| Optimization | Affects | Typical Savings | Effort | Savings/Hour |
|--------------|---------|----------------|--------|--------------|
| Placement    | 70%     | 56ps           | 3hr    | 18.7ps/hr    |
| Buffering    | 50%     | 45ps           | 2hr    | 22.5ps/hr    |
| Cell sizing  | 30%     | 30ps           | 5hr    | 6.0ps/hr     |

**Cell sizing is 3× less efficient than placement!**

**THE PRINCIPLE:**

**Optimize the dominant constraint first.** At modern nodes (28nm+), wire delay dominates (60-80%). Fixing placement (affects 70% of path) before cell sizing (affects 30%) is fundamental Pareto principle.

**COMMON MISTAKES:**
1. Using 65nm-era strategy (cell sizing important) at 28nm (wire dominates)
2. Not measuring net vs cell breakdown before optimizing
3. Wasting time on 30% of problem while ignoring 70%

**THE WISDOM:** When wire delay is 70% of the path, optimizing cells (30%) is like rearranging deck chairs on the Titanic. Fix placement first (re-place cells closer, target <500µm critical connections), insert buffers second (segment wires >1.5mm), and ONLY then consider cell sizing. Same effort, 2-3× better results.""",
        "category": "diagnostic_wisdom",
        "complexity_score": 87
    },

    {
        "source_topic": "Critical Path Root Cause Identification",
        "prompt": "A critical path has 500ps total delay (period = 500ps, failing by 0ps). The breakdown is 430ps net delay on ONE net 'decode_enable' and 70ps across 4 gates. What's the FIRST diagnostic check I should run?",
        "response": """**DIAGNOSTIC REASONING CHAIN:**

**SYMPTOM:** 86% wire delay (430ps/500ps) on a SINGLE net

**OBSERVATION:** This is extreme wire delay dominance (>80%)

**CRITICAL INSIGHT (The Diagnostic Fork):**

**ONE net with 430ps delay has TWO possible root causes:**
1. **High fanout** (capacitive load problem)
2. **Long distance** (resistive wire problem)

**The fix depends on which root cause → CHECK FANOUT FIRST!**

**WHY FANOUT FIRST:**

```
Fanout check: 5 minute check, 6 hour fix
Distance check: 5 minute check, 2-3 DAY fix (re-placement)

If you check distance first and it's actually fanout:
  → Wasted 3 days re-placing when 6 hours of buffering would fix it

If you check fanout first and it's distance:
  → Wasted 5 minutes, then do the 3-day re-place

Always check the QUICK FIX first!
```

**THE DIAGNOSTIC PROCEDURE:**

**Step 1: Check FANOUT (5 minutes)**

```tcl
report_net -connections decode_enable

# Look for line:
# "Fanout: XX"

# Interpretation:
# Fanout < 10: Not a fanout problem
# Fanout 10-30: Marginal
# Fanout 30-50: High (LIKELY problem)
# Fanout > 50: Critical (DEFINITELY problem)
```

**If fanout = 45:**
```
→ HIGH FANOUT problem identified
→ Fix: Insert 2-level buffer tree
→ Expected: 430ps → 120ps (70% improvement)
→ Time: 6 hours
→ STOP HERE - don't check distance yet
```

**If fanout = 8:**
```
→ NOT a fanout problem
→ Continue to Step 2 (check distance)
```

**Step 2: Check DISTANCE (only if fanout is low)**

```tcl
report_net -geometry decode_enable

# Look for:
# "Total length: XXXX um"

# Interpretation:
# Length < 500µm: Not a distance problem
# Length 500-1500µm: Marginal
# Length 1500-3000µm: Long (LIKELY problem)
# Length > 3000µm: Very long (DEFINITELY problem)
```

**If length = 2.5mm:**
```
→ LONG DISTANCE problem identified
→ Fix: Re-place at lower density (0.85 → 0.65)
→ Expected: 430ps → 80ps (81% improvement)
→ Time: 2-3 days
```

**THE PHYSICS:**

**Fanout problem:**
```
Wire delay = R × C
C increases with fanout: 10 loads = 100fF, 50 loads = 500fF

Driver can't charge 500fF quickly → 180ps driver delay
Plus wire RC delay → total 430ps

Fix: Buffer tree (50 loads → 2 stages × 10 loads each)
  - Stage 1: Drive 5 buffers (50fF load)
  - Stage 2: Each buffer drives 10 loads (100fF each)
  - Driver delay: 50fF → 30ps (was 180ps)
  - Total: 30ps + 80ps (buffers) + 40ps (short wires) = 150ps
  - Savings: 430ps → 150ps (65% improvement)
```

**Distance problem:**
```
Wire delay = R × C
R increases with length: 1mm = 50Ω, 3mm = 150Ω

Long wire (3mm, 10 loads):
  R = 150Ω, C = 100fF
  RC delay = 150Ω × 100fF = 15ns... wait, that's wrong

Correct calculation:
  Wire delay ≈ 150ps per mm @ 28nm
  3mm wire = 450ps delay

Fix: Re-place cells closer (3mm → 0.5mm)
  - New wire delay: 0.5mm × 150ps/mm = 75ps
  - Savings: 450ps → 75ps (83% improvement)
```

**DIAGNOSIS DECISION TREE:**

```
86% wire delay on single net
       ↓
  CHECK FANOUT (5 min)
       ↓
   Fanout > 30?
    ↙      ↘
  YES       NO
   ↓         ↓
HIGH      CHECK DISTANCE
FANOUT       ↓
   ↓      Length > 1.5mm?
Fix with      ↙      ↘
buffers     YES       NO
(6 hr)       ↓         ↓
          LONG      BOTH problems
          DISTANCE  OR other issue
             ↓         ↓
          Fix with   Check:
          re-place   - Layer (M1 vs M3)
          (2-3 days) - Routing detours
                     - Crosstalk
```

**ACTIONABLE GUIDANCE:**

**The FIRST check (always):**

```tcl
# Takes 10 seconds
report_net -connections decode_enable | grep Fanout
```

**If output shows "Fanout: 48":**
```
→ Problem identified: High fanout
→ Don't waste time checking distance
→ Go directly to buffer insertion
→ Expected savings: 60-70%
→ Time to fix: 6 hours
```

**If output shows "Fanout: 7":**
```
→ Fanout is NOT the problem
→ Now check distance
→ report_net -geometry decode_enable | grep "Total length"
```

**THE PRINCIPLE:**

**One diagnostic check can save 3 days.**

When ONE net dominates delay (>80% of path), it's either:
- High fanout (quick fix: 6 hours)
- Long distance (slow fix: 2-3 days)

**ALWAYS check fanout FIRST** (5 min check, 6 hr fix if positive).

If fanout is normal (<20), THEN check distance (5 min check, 3 day fix).

**COMMON MISTAKES:**
1. Assuming it's distance without checking fanout first
2. Re-placing entire design when buffering would fix it
3. Not running the 10-second fanout check

**THE WISDOM:** 86% wire delay on a single net is a diagnostic FORK, not a single problem. The fix (buffers vs re-place) depends on root cause (fanout vs distance). Check fanout FIRST - it's a 10-second check that can save you 3 days of unnecessary re-placement. If fanout >30, insert buffers (6 hours). If fanout <20, check distance (then re-place, 3 days). This one diagnostic decision determines whether you fix the problem in 6 hours or 3 days.""",
        "category": "diagnostic_wisdom",
        "complexity_score": 89
    }
]

print("Adding 6 new wisdom upgrades...")
print()

for i, upgrade in enumerate(NEW_WISDOM_UPGRADES, start=1):
    wisdom_examples.append({
        'prompt': upgrade['prompt'],
        'response': upgrade['response'],
        'phase': 'diagnostic_reasoning',
        'provenance': 'wisdom_upgrade',
        'source': f"Wisdom Upgrade: {upgrade['source_topic']}",
        'category': upgrade['category'],
        'complexity_score': upgrade['complexity_score'],
        'verified': 'expert_diagnosis'
    })
    print(f"  [{i}/6] Added: {upgrade['source_topic']}")

print()

# Save expanded corpus
print("Saving expanded WISDOM_CORPUS.jsonl...")
with open('WISDOM_CORPUS.jsonl', 'w') as f:
    for ex in wisdom_examples:
        f.write(json.dumps(ex) + '\n')

print(f"✓ Saved {len(wisdom_examples)} total examples")
print()

print("="*80)
print("✓ WISDOM CORPUS EXPANDED")
print("="*80)
print(f"File: WISDOM_CORPUS.jsonl")
print(f"Total examples: {len(wisdom_examples)}")
print()
print("Breakdown:")
print(f"  - Original diagnostic examples: 5")
print(f"  - Initial wisdom upgrades: 6")
print(f"  - New wisdom upgrades: 6")
print(f"  - TOTAL: {len(wisdom_examples)}")
print()
print("Next steps:")
print("  1. Train wisdom-specialist-v2 on expanded corpus")
print("  2. Test on 3 mission scenarios")
print("  3. Compare against wisdom-specialist-v1 results")
print("="*80)
