#!/usr/bin/env python3
"""
Generate FSM Gold Standard for V5.8 (Proactive + Reactive Quality Control)

STRATEGY - DUAL QUALITY CONTROL:
================================================================================
1. PROACTIVE: Enhanced prompts with:
   - Few-shot examples (show "answer key")
   - Explicit rules (common mistakes to avoid)
   - Templates (proper FSM structure scaffolding)

2. REACTIVE: Strict validation with:
   - Verilator -Wall -Werror (no warnings)
   - FSM structure checks
   - Specification compliance
   - Functional correctness (testbench simulation)

TARGET: 150 FSMs with ≥80% pass rate (vs 7.3% in V5.6)
================================================================================
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import anthropic
from anthropic import AsyncAnthropic

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.validation import validate_fsm_strict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fsm_gold_v5_8_generation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# FEW-SHOT EXAMPLE (High-quality FSM that passed strict validation)
# ==============================================================================
FEW_SHOT_EXAMPLE = """
[EXAMPLE OF HIGH-QUALITY MOORE FSM]

Specification: "[HDL:verilog2001] Moore FSM with 3 states outputting different binary values (00, 01, 10)"

Solution:
```verilog
module moore_fsm_3state (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [1:0] out
);

    // State encoding
    parameter IDLE  = 2'b00;
    parameter STATE1 = 2'b01;
    parameter STATE2 = 2'b10;

    reg [1:0] current_state, next_state;

    // State transition logic (sequential)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end

    // Next state logic (combinational)
    always @(*) begin
        case (current_state)
            IDLE: begin
                if (enable)
                    next_state = STATE1;
                else
                    next_state = IDLE;
            end

            STATE1: begin
                next_state = STATE2;
            end

            STATE2: begin
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end

    // Output logic (Moore: output depends ONLY on current state)
    always @(*) begin
        case (current_state)
            IDLE:   out = 2'b00;
            STATE1: out = 2'b01;
            STATE2: out = 2'b10;
            default: out = 2'b00;
        endcase
    end

endmodule
```

Note the key characteristics:
- Separate current_state and next_state registers
- Sequential always block for state transitions
- Combinational always block for next state logic
- Moore FSM: outputs depend ONLY on current state
- Proper default cases for safety
- Clear comments explaining logic
"""


# ==============================================================================
# RULE-BASED GUIDANCE (Common Failures to Avoid)
# ==============================================================================
STRICT_RULES = """
CRITICAL RULES - YOU MUST FOLLOW THESE TO PASS VALIDATION:
===========================================================

1. SYNTAX RULES:
   - File MUST end with a newline character (POSIX compliance)
   - No warnings allowed (code must compile with -Werror)
   - No syntax errors of any kind
   - Use proper Verilog/SystemVerilog syntax for the specified dialect

2. FSM STRUCTURE RULES:
   - MUST declare `current_state` register
   - MUST declare `next_state` register/wire
   - MUST have sequential always block: always @(posedge clk or negedge rst_n)
   - MUST have combinational always block for next state logic
   - MUST have proper reset logic (active-low rst_n)

3. MOORE FSM SPECIFIC:
   - Outputs depend ONLY on current_state
   - Output logic must be combinational (always @(*))
   - Never use inputs directly in output assignment

4. MEALY FSM SPECIFIC:
   - Outputs depend on BOTH current_state AND inputs
   - Output logic references both state and input signals

5. HANDSHAKE FSM SPECIFIC:
   - MUST include `req` and `ack` signals as ports
   - MUST implement proper handshake protocol
   - States should reflect handshake phases (IDLE, REQ, WAIT_ACK, etc.)

6. NAMING RULES:
   - DO NOT use Verilog reserved keywords as signal names
   - Avoid: disable, task, function, begin, end, etc.
   - If spec mentions these, append _sig (e.g., disable_sig)

7. ASSIGNMENT RULES:
   - Use non-blocking (<=) in sequential always blocks
   - Use blocking (=) in combinational always blocks
   - Never mix blocking/non-blocking for same variable

8. DEFAULT CASES:
   - Always include default case in case statements
   - Prevents synthesis issues and latches
"""


# ==============================================================================
# FSM SPECIFICATIONS
# ==============================================================================
def generate_fsm_specs(count: int = 150) -> List[Dict]:
    """Generate diverse FSM specifications for V5.8"""

    moore_specs = [
        "Moore FSM 3-state traffic light: RED->YELLOW->GREEN cycle, outputs color based on state only",
        "Moore FSM 4-state binary counter: counts 00->01->10->11->00, output is state value",
        "Moore FSM sequence generator: outputs pattern 1,0,1,1 cyclically based on state",
        "Moore FSM one-hot encoder: 4 states with one-hot outputs (0001, 0010, 0100, 1000)",
        "Moore FSM BCD counter: counts 0-9 with BCD output depending only on current state",
        "Moore FSM vending machine states: IDLE, COLLECT, DISPENSE, CHANGE with status outputs",
        "Moore FSM state indicator: 3 states outputting their state number (0, 1, 2)",
        "Moore FSM Gray code counter: 4 states with Gray code outputs based on state",
        "Moore FSM modulo-3 counter: outputs remainder (0,1,2) based on current state only",
        "Moore FSM alarm system: ARMED, TRIGGERED, ALARM states with status output from state",
    ]

    mealy_specs = [
        "Mealy FSM edge detector: outputs '1' on rising edge of input signal",
        "Mealy FSM sequence detector '101': outputs '1' when pattern detected",
        "Mealy FSM parity checker: outputs running parity based on state and input",
        "Mealy FSM serial adder: outputs sum bit based on state (carry) and input bits",
        "Mealy FSM pattern matcher '1011': outputs '1' on complete match",
        "Mealy FSM debouncer: outputs stable signal after consistent input samples",
        "Mealy FSM XOR accumulator: output depends on accumulated state and current input",
        "Mealy FSM frequency divider: output toggles based on state and input transitions",
        "Mealy FSM serial comparator: outputs '1' if A > B based on state and inputs",
        "Mealy FSM packet detector: outputs '1' when start/end pattern detected based on input",
    ]

    handshake_specs = [
        "4-phase handshake FSM: IDLE->REQ_ASSERT->WAIT_ACK->COMPLETE cycle with req/ack signals",
        "2-phase handshake FSM: IDLE->REQUEST->ACKNOWLEDGE->IDLE with valid/ready protocol",
        "3-state req/ack protocol: IDLE->REQ_SENT->WAIT_ACK->IDLE with req and ack signals",
        "AXI-lite handshake: IDLE->VALID->WAIT_READY->DONE with valid/ready handshake",
        "Request-grant arbiter: IDLE->REQ->GRANT->COMPLETE with req/ack handshaking",
        "Producer-consumer handshake: IDLE->PRODUCE->WAIT_CONSUME->COMPLETE with req/ack",
        "Credit-based flow control: IDLE->SEND->WAIT_CREDIT->RECEIVE->IDLE with req/ack",
        "Handshake with retry: IDLE->REQ->WAIT_ACK->TIMEOUT->RETRY with req/ack signals",
        "Simple 2-way handshake: IDLE->REQ_HIGH->WAIT_ACK_HIGH->DONE with req and ack",
        "Master-slave handshake: IDLE->MASTER_REQ->WAIT_ACK->SLAVE_ACK->COMPLETE with req/ack",
    ]

    specs = []
    target_per_type = count // 3

    # Generate Moore FSMs
    for i in range(target_per_type):
        spec = moore_specs[i % len(moore_specs)]
        specs.append({
            'id': f'moore_{i:04d}',
            'type': 'Moore',
            'spec': spec,
            'dialect': 'verilog2001'
        })

    # Generate Mealy FSMs
    for i in range(target_per_type):
        spec = mealy_specs[i % len(mealy_specs)]
        specs.append({
            'id': f'mealy_{i:04d}',
            'type': 'Mealy',
            'spec': spec,
            'dialect': 'verilog2001'
        })

    # Generate Handshake FSMs
    remaining = count - len(specs)
    for i in range(remaining):
        spec = handshake_specs[i % len(handshake_specs)]
        specs.append({
            'id': f'handshake_{i:04d}',
            'type': 'Handshake',
            'spec': spec,
            'dialect': 'verilog2001'
        })

    return specs


# ==============================================================================
# ENHANCED PROMPT GENERATION (Proactive Quality Control)
# ==============================================================================
def create_enhanced_prompt(fsm_type: str, spec: str, dialect: str) -> str:
    """Create enhanced prompt with few-shot example and strict rules"""

    return f"""You are an expert Verilog/SystemVerilog RTL designer. Your task is to generate PRODUCTION-READY, STRICTLY VALIDATED HDL code.

{FEW_SHOT_EXAMPLE}

{STRICT_RULES}

================================================================================
YOUR TASK
================================================================================

FSM Type: {fsm_type}
Dialect: {dialect}
Specification: [HDL:{dialect}] {spec}

Using the EXACT SAME style, structure, and quality as the example above, generate a complete {fsm_type} FSM module that:
1. Follows ALL the strict rules listed above
2. Uses proper FSM structure (current_state, next_state, sequential/combinational blocks)
3. Implements the specification correctly
4. Compiles with ZERO warnings (Verilator -Werror compliant)
5. Has proper comments explaining the logic
6. Ends with a newline character

IMPORTANT:
- For {fsm_type} FSMs, remember:
  {"- Outputs depend ONLY on current_state (not inputs)" if fsm_type == "Moore" else ""}
  {"- Outputs depend on BOTH current_state AND inputs" if fsm_type == "Mealy" else ""}
  {"- Must include req and ack signals for handshake protocol" if fsm_type == "Handshake" else ""}

Generate ONLY the Verilog module code. Do not include any explanation or markdown formatting.
"""


# ==============================================================================
# GENERATION WITH VALIDATION LOOP
# ==============================================================================
async def generate_fsm_with_validation(
    client: AsyncAnthropic,
    spec_dict: Dict,
    max_retries: int = 2
) -> Optional[Dict]:
    """Generate FSM with strict validation, retry on failure"""

    fsm_id = spec_dict['id']
    fsm_type = spec_dict['type']
    spec = spec_dict['spec']
    dialect = spec_dict['dialect']

    logger.info(f"Generating {fsm_id} ({fsm_type})...")

    for attempt in range(max_retries):
        try:
            # PROACTIVE: Use enhanced prompt
            prompt = create_enhanced_prompt(fsm_type, spec, dialect)

            # Generate code
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            code = response.content[0].text.strip()

            # Extract code from markdown blocks if present
            if '```' in code:
                import re
                # Match ```verilog, ```systemverilog, ```sv, or just ```
                match = re.search(r'```(?:verilog|systemverilog|sv)?\s*\n(.*?)```', code, re.DOTALL)
                if match:
                    code = match.group(1).strip()

            # Ensure code ends with newline (POSIX compliance)
            if not code.endswith('\n'):
                code += '\n'

            # REACTIVE: Strict validation
            logger.info(f"  Validating {fsm_id} (attempt {attempt + 1}/{max_retries})...")
            result = validate_fsm_strict(
                code=code,
                instruction=spec,
                fsm_type=fsm_type,
                dialect=dialect
            )

            if result.passed:
                logger.info(f"  ✓ {fsm_id} PASSED all validation checks")
                return {
                    'instruction': f"[HDL:{dialect}] {spec}",
                    'output': code,
                    'hierarchy': {
                        'l1': 'Sequential',
                        'l2': 'FSM',
                        'l3': fsm_type
                    },
                    'category': 'FSM',
                    'metadata': {
                        'id': fsm_id,
                        'fsm_type': fsm_type,
                        'dialect': dialect,
                        'strict_validated': True,
                        'validation_passed': True,
                        'generated_v5_8': True,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            else:
                logger.warning(f"  ✗ {fsm_id} FAILED validation (attempt {attempt + 1}/{max_retries})")
                logger.warning(f"    Errors: {result.errors[:2]}")  # Show first 2 errors

                if attempt < max_retries - 1:
                    logger.info(f"    Retrying with error feedback...")
                    # Could add error feedback to prompt here for retry

        except Exception as e:
            logger.error(f"  ✗ {fsm_id} generation error (attempt {attempt + 1}/{max_retries}): {e}")

    logger.error(f"  ✗ {fsm_id} FAILED after {max_retries} attempts")
    return None


# ==============================================================================
# MAIN GENERATION ORCHESTRATION
# ==============================================================================
async def generate_gold_standard(
    num_fsms: int = 150,
    max_concurrent: int = 5,
    smoke_test: bool = False
):
    """Generate FSM gold standard with strict validation"""

    if smoke_test:
        num_fsms = 6  # 2 of each type for smoke test
        logger.info("SMOKE TEST MODE: Generating 6 FSMs (2 Moore, 2 Mealy, 2 Handshake)")

    logger.info("="*80)
    logger.info("FSM GOLD STANDARD GENERATION V5.8")
    logger.info("="*80)
    logger.info(f"Target: {num_fsms} high-quality FSMs")
    logger.info(f"Strategy: Proactive prompts + Reactive validation")
    logger.info(f"Max concurrent: {max_concurrent}")
    logger.info("="*80)

    # Initialize Claude client
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        return

    client = AsyncAnthropic(api_key=api_key)

    # Generate specifications
    specs = generate_fsm_specs(num_fsms)
    logger.info(f"Generated {len(specs)} FSM specifications")

    # Generation tracking
    passed = []
    failed = []
    attempts = 0

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_semaphore(spec):
        async with semaphore:
            nonlocal attempts
            attempts += 1
            return await generate_fsm_with_validation(client, spec)

    # Generate all FSMs concurrently
    logger.info("\nStarting generation...")
    tasks = [generate_with_semaphore(spec) for spec in specs]
    results = await asyncio.gather(*tasks)

    # Collect results
    for result in results:
        if result is not None:
            passed.append(result)
        else:
            failed.append(None)

    # Statistics
    pass_rate = len(passed) / attempts * 100 if attempts > 0 else 0

    logger.info("\n" + "="*80)
    logger.info("GENERATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total attempts: {attempts}")
    logger.info(f"Passed:         {len(passed)} ({pass_rate:.1f}%)")
    logger.info(f"Failed:         {len(failed)}")
    logger.info("="*80)

    # Assessment
    if pass_rate >= 80:
        logger.info("✓ EXCELLENT: Pass rate ≥80% - dataset is production ready")
    elif pass_rate >= 50:
        logger.info("⚠ ACCEPTABLE: Pass rate ≥50% - dataset is usable")
    elif pass_rate >= 20:
        logger.info("⚠ POOR: Pass rate ≥20% - consider regenerating")
    else:
        logger.info("✗ UNACCEPTABLE: Pass rate <20% - prompts need improvement")

    # Save results
    if passed:
        output_file = Path('data/fsm_gold_v5_8.jsonl')
        with open(output_file, 'w') as f:
            for example in passed:
                f.write(json.dumps(example) + '\n')
        logger.info(f"\n✓ Saved {len(passed)} high-quality FSMs to: {output_file}")
    else:
        logger.error("\n✗ No FSMs passed validation!")

    return len(passed), len(failed), pass_rate


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FSM Gold Standard V5.8")
    parser.add_argument('--num-fsms', type=int, default=150, help='Number of FSMs to generate')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Max concurrent generations')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke test (6 FSMs)')
    args = parser.parse_args()

    # Run generation
    asyncio.run(generate_gold_standard(
        num_fsms=args.num_fsms,
        max_concurrent=args.max_concurrent,
        smoke_test=args.smoke_test
    ))
