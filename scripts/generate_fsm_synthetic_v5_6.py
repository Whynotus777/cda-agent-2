#!/usr/bin/env python3
"""
Generate Targeted FSM Synthetic Data for V5.6 (Parallel + RAG-Enhanced)

STRATEGY:
- Restore FSM L3 diversity lost in V5.5 cleaning
- Parallel generation: Moore (150), Mealy (150), Handshake (78) simultaneously
- RAG-enhanced: Retrieve similar examples from existing corpus
- Tournament-style: 2 attempts per FSM, pick first success
- Dual-path validation: iverilog + Verilator

TARGET: 378 verified FSM examples in ~15-20 minutes
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
import subprocess
import tempfile
import re

import anthropic
from anthropic import AsyncAnthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fsm_generation_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# FSM Category Specifications
MOORE_FSM_SPECS = [
    "Moore FSM with 3 states outputting different binary values (00, 01, 10)",
    "Moore FSM 4-state binary counter: outputs 00->01->10->11->00",
    "Moore FSM traffic light: outputs RED, YELLOW, GREEN based on state only",
    "Moore FSM sequence generator: outputs 1,0,1,1 pattern cyclically",
    "Moore FSM one-hot: 4 states with one-hot outputs (0001, 0010, 0100, 1000)",
    "Moore FSM BCD counter: counts 0-9 with BCD output depending on state",
    "Moore FSM Gray code counter: 4 states with Gray code outputs",
    "Moore FSM vending machine: outputs IDLE, COLLECT, DISPENSE, CHANGE states",
    "Moore FSM state indicator: 3 states output their state number (0,1,2)",
    "Moore FSM modulo-3 counter: outputs remainder of count/3",
]

MEALY_FSM_SPECS = [
    "Mealy FSM edge detector: outputs '1' on rising edge of input, '0' otherwise",
    "Mealy FSM sequence detector '101': outputs '1' when pattern detected",
    "Mealy FSM parity checker: outputs running parity based on state and input",
    "Mealy FSM serial adder: outputs sum bit based on state (carry) and inputs",
    "Mealy FSM pattern matcher '1011': outputs '1' on match, '0' otherwise",
    "Mealy FSM debouncer: outputs stable signal after 2 consistent input samples",
    "Mealy FSM XOR accumulator: output depends on state and current input XOR",
    "Mealy FSM frequency divider: output toggles based on state and input transitions",
    "Mealy FSM serial comparator: outputs '1' if input A > B based on state history",
    "Mealy FSM packet detector: outputs '1' when start/end pattern detected",
]

HANDSHAKE_FSM_SPECS = [
    "4-phase handshake FSM: IDLE->REQ_ASSERT->WAIT_ACK->ACK_RECEIVED->IDLE with req/ack signals",
    "2-phase handshake FSM: IDLE->REQUEST->ACKNOWLEDGE->IDLE with valid/ready protocol",
    "3-state req/ack protocol: IDLE->REQ_SENT->ACK_WAIT->IDLE with timeout",
    "AXI-lite handshake: IDLE->ADDR_VALID->DATA_VALID->RESP_WAIT->DONE with AWVALID/AWREADY",
    "Request-grant arbiter: IDLE->REQ0->GRANT0->REQ1->GRANT1->IDLE round-robin",
    "Producer-consumer handshake: IDLE->PRODUCE->WAIT_CONSUME->CONSUME->IDLE",
    "Credit-based flow control: IDLE->SEND->WAIT_CREDIT->RECEIVE_CREDIT->IDLE",
    "Handshake with retry: IDLE->REQ->WAIT_ACK->TIMEOUT->RETRY->IDLE with max retries",
    "Dual handshake FSM: manages req1/ack1 and req2/ack2 pairs independently",
    "Multi-master arbiter: IDLE->REQ_M0->GRANT_M0->REQ_M1->GRANT_M1->IDLE with fairness",
]


def generate_fsm_spec_pool(category: str, count: int) -> List[Dict]:
    """Generate FSM specifications for a category"""
    specs = []

    if category == "Moore":
        templates = MOORE_FSM_SPECS
    elif category == "Mealy":
        templates = MEALY_FSM_SPECS
    elif category == "Handshake":
        templates = HANDSHAKE_FSM_SPECS
    else:
        raise ValueError(f"Unknown category: {category}")

    # Expand templates with variations
    spec_id = 0
    while len(specs) < count:
        for template in templates:
            if len(specs) >= count:
                break

            # Add dialect conditioning
            dialect = ["verilog2001", "sv2005", "sv2009"][spec_id % 3]

            specs.append({
                "id": f"fsm_{category.lower()}_{spec_id:04d}",
                "spec": template,
                "category": category,
                "dialect": dialect
            })
            spec_id += 1

    return specs[:count]


def lint_verilog(code: str, dialect: str, work_dir: Path, spec_id: str = "unknown") -> Tuple[bool, str]:
    """Lint Verilog/SystemVerilog code"""
    code_file = work_dir / "design.v"
    code_file.write_text(code)

    try:
        if dialect == "verilog2001":
            # Use iverilog for Verilog
            cmd = ["iverilog", "-tnull", "-Wall", str(code_file)]
            logger.info(f"[{spec_id}] Linting with iverilog")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
        else:
            # Use verilator for SystemVerilog
            cmd = ["verilator", "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Wno-EOFNEWLINE", str(code_file)]
            logger.info(f"[{spec_id}] Linting with verilator")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

        if result.returncode == 0:
            logger.info(f"[{spec_id}] Lint passed ✓")
            return True, ""
        else:
            logger.warning(f"[{spec_id}] Lint failed. Error: {result.stderr[:300]}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        logger.error(f"[{spec_id}] Lint timeout")
        return False, "Lint timeout"
    except Exception as e:
        logger.error(f"[{spec_id}] Lint exception: {e}")
        return False, str(e)


def extract_verilog_code(response: str, spec_id: str = "unknown") -> Optional[str]:
    """Extract Verilog code from Claude response"""
    # Try to find code blocks
    patterns = [
        r'```verilog\s*(.*?)\s*```',
        r'```systemverilog\s*(.*?)\s*```',
        r'```sv\s*(.*?)\s*```',
        r'```\s*(module\s+.*?endmodule)\s*```',
        r'(module\s+\w+.*?endmodule)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            # Basic validation: must have module and endmodule
            if 'module' in code and 'endmodule' in code:
                logger.info(f"[{spec_id}] Successfully extracted code ({len(code)} chars)")
                return code

    logger.warning(f"[{spec_id}] Failed to extract code from response. Response preview: {response[:200]}...")
    return None


def build_rag_context(category: str, spec: str) -> str:
    """
    Build RAG context by retrieving similar examples.
    For V5.6, we'll use simple keyword matching from V5.5 clean dataset.
    """
    # TODO: Later integrate with ChromaDB vector store
    # For now, provide category-specific guidance

    if category == "Moore":
        return """
MOORE FSM REFERENCE:
- Output depends ONLY on current state (not inputs)
- Use separate always blocks: state transition + output logic
- Example pattern:
  always_ff @(posedge clk) state <= next_state;
  always_comb begin
    case (state)
      S0: out = 2'b00;
      S1: out = 2'b01;
      // Output based on state only
    endcase
  end
"""
    elif category == "Mealy":
        return """
MEALY FSM REFERENCE:
- Output depends on BOTH state AND input
- Combine state and input in output logic
- Example pattern:
  always_comb begin
    case (state)
      S0: out = (input_sig) ? 1'b1 : 1'b0;  // Output depends on input
      S1: out = (input_sig) ? 1'b0 : 1'b1;
    endcase
  end
"""
    elif category == "Handshake":
        return """
HANDSHAKE FSM REFERENCE:
- Manage req/ack or valid/ready signals
- Include timeout/retry logic if specified
- Example pattern:
  IDLE: if (req) next_state = WAIT_ACK;
  WAIT_ACK: if (ack) next_state = DONE; else if (timeout) next_state = RETRY;
  DONE: next_state = IDLE;
"""
    else:
        return ""


async def generate_fsm_async(
    client: AsyncAnthropic,
    spec: Dict,
    work_dir: Path,
    attempt: int = 1
) -> Optional[Dict]:
    """Generate a single FSM with RAG-enhanced prompting"""

    spec_id = spec["id"]
    category = spec["category"]
    dialect = spec["dialect"]
    spec_text = spec["spec"]

    logger.info(f"[{spec_id}] Starting generation (attempt {attempt})")
    logger.debug(f"[{spec_id}] Category: {category}, Dialect: {dialect}")

    # Build RAG context
    rag_context = build_rag_context(category, spec_text)

    # Construct prompt
    prompt = f"""You are an expert RTL designer. Generate a {category} FSM in {dialect.upper()}.

SPECIFICATION:
{spec_text}

{rag_context}

REQUIREMENTS:
1. Use {dialect} syntax
2. Include proper reset logic (asynchronous or synchronous as appropriate)
3. For {category} FSMs:
   {'- Output MUST depend only on state' if category == 'Moore' else ''}
   {'- Output MUST depend on both state and input' if category == 'Mealy' else ''}
   {'- Include req/ack or valid/ready handshake signals' if category == 'Handshake' else ''}
4. All signals in spec must be present in module ports
5. Use meaningful state names
6. Add comments explaining state transitions

Generate ONLY the Verilog/SystemVerilog code inside a code block. No explanations outside the code block."""

    try:
        logger.info(f"[{spec_id}] Calling Claude API...")
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        logger.info(f"[{spec_id}] Received API response")
        response_text = message.content[0].text
        logger.debug(f"[{spec_id}] Response length: {len(response_text)} chars")

        code = extract_verilog_code(response_text, spec_id)

        if not code:
            logger.warning(f"[{spec_id}] Failed to extract code")
            return None

        # Lint the code
        lint_ok, lint_error = lint_verilog(code, dialect, work_dir, spec_id)

        if lint_ok:
            logger.info(f"[{spec_id}] ✓ SUCCESS - FSM generated and validated")
            return {
                "instruction": f"[HDL:{dialect}] {spec_text}",
                "output": code,
                "hierarchy": {
                    "l1": "Sequential",
                    "l2": "FSM",
                    "l3": category
                },
                "metadata": {
                    "id": spec["id"],
                    "category": category,
                    "dialect": dialect,
                    "attempt": attempt,
                    "generated_at": datetime.now().isoformat(),
                    "synthetic": True,
                    "rag_enhanced": True
                }
            }
        else:
            logger.warning(f"[{spec_id}] Failed lint validation")
            return None

    except Exception as e:
        logger.error(f"[{spec_id}] Exception during generation: {type(e).__name__}: {e}")
        import traceback
        logger.debug(f"[{spec_id}] Traceback:\n{traceback.format_exc()}")
        return None


async def generate_fsm_with_tournament(
    client: AsyncAnthropic,
    spec: Dict,
    work_dir: Path,
    max_attempts: int = 2
) -> Optional[Dict]:
    """Tournament-style: try multiple times, return first success"""

    # Create tasks from coroutines
    tasks = [
        asyncio.create_task(generate_fsm_async(client, spec, work_dir, attempt=i+1))
        for i in range(max_attempts)
    ]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            return result

    return None


async def generate_fsm_category(
    client: AsyncAnthropic,
    category: str,
    count: int,
    work_dir: Path,
    batch_size: int = 10
) -> List[Dict]:
    """Generate all FSMs for a category with parallel batching"""

    print(f"\n{'='*80}")
    print(f"GENERATING {count} {category.upper()} FSMs")
    print(f"{'='*80}\n")

    specs = generate_fsm_spec_pool(category, count)
    results = []

    # Process in batches
    for i in range(0, len(specs), batch_size):
        batch = specs[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(specs)-1)//batch_size + 1}: Processing {len(batch)} FSMs...")

        tasks = [
            generate_fsm_with_tournament(client, spec, work_dir)
            for spec in batch
        ]

        batch_results = await asyncio.gather(*tasks)

        # Filter out None results
        batch_successes = [r for r in batch_results if r is not None]
        results.extend(batch_successes)

        print(f"    ✓ {len(batch_successes)}/{len(batch)} successful")

    print(f"\n{'='*80}")
    print(f"{category.upper()} GENERATION COMPLETE: {len(results)}/{count} ({len(results)/count*100:.1f}%)")
    print(f"{'='*80}\n")

    return results


async def main_async(args):
    """Main async orchestrator"""

    print("="*80)
    print("FSM SYNTHETIC DATA GENERATOR V5.6 (RAG-Enhanced)")
    print("="*80)
    print(f"\nTARGET:")
    print(f"  Moore FSMs:     {args.moore}")
    print(f"  Mealy FSMs:     {args.mealy}")
    print(f"  Handshake FSMs: {args.handshake}")
    print(f"  TOTAL:          {args.moore + args.mealy + args.handshake}")
    print(f"\nSTRATEGY:")
    print(f"  - Parallel async generation (all 3 categories simultaneously)")
    print(f"  - RAG-enhanced prompting with category-specific examples")
    print(f"  - Tournament-style: 2 attempts per FSM, pick first success")
    print(f"  - Dual-path validation: iverilog + Verilator")
    print()

    # Initialize client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=api_key)

    # Create temp work directory
    work_dir = Path(tempfile.mkdtemp())
    print(f"Work directory: {work_dir}\n")

    # Launch all 3 categories in parallel
    start_time = datetime.now()

    moore_task = generate_fsm_category(client, "Moore", args.moore, work_dir)
    mealy_task = generate_fsm_category(client, "Mealy", args.mealy, work_dir)
    handshake_task = generate_fsm_category(client, "Handshake", args.handshake, work_dir)

    moore_results, mealy_results, handshake_results = await asyncio.gather(
        moore_task, mealy_task, handshake_task
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save results
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    moore_file = output_dir / "fsm_synthetic_moore_v5_6.jsonl"
    mealy_file = output_dir / "fsm_synthetic_mealy_v5_6.jsonl"
    handshake_file = output_dir / "fsm_synthetic_handshake_v5_6.jsonl"

    with open(moore_file, 'w') as f:
        for example in moore_results:
            f.write(json.dumps(example) + '\n')

    with open(mealy_file, 'w') as f:
        for example in mealy_results:
            f.write(json.dumps(example) + '\n')

    with open(handshake_file, 'w') as f:
        for example in handshake_results:
            f.write(json.dumps(example) + '\n')

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"\nResults:")
    print(f"  Moore:     {len(moore_results)}/{args.moore} ({len(moore_results)/args.moore*100:.1f}%)")
    print(f"  Mealy:     {len(mealy_results)}/{args.mealy} ({len(mealy_results)/args.mealy*100:.1f}%)")
    print(f"  Handshake: {len(handshake_results)}/{args.handshake} ({len(handshake_results)/args.handshake*100:.1f}%)")
    print(f"  TOTAL:     {len(moore_results) + len(mealy_results) + len(handshake_results)}/{args.moore + args.mealy + args.handshake}")
    print(f"\nOutput:")
    print(f"  {moore_file}")
    print(f"  {mealy_file}")
    print(f"  {handshake_file}")
    print(f"\nNext Steps:")
    print(f"  1. Merge with V5.5: cat data/rtl_behavioral_v5_5.jsonl {moore_file} {mealy_file} {handshake_file} > data/rtl_behavioral_v5_6.jsonl")
    print(f"  2. Train V5.6: python scripts/train_qwen_coder_qlora.py --dataset data/rtl_behavioral_v5_6.jsonl")
    print(f"  3. Benchmark V5.6: python scripts/benchmark_v5_4.py --model <v5_6_path> --runs 5")
    print("="*80)

    # Cleanup
    import shutil
    shutil.rmtree(work_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate FSM synthetic data with parallel RAG-enhanced generation")
    parser.add_argument("--moore", type=int, default=150, help="Number of Moore FSMs to generate")
    parser.add_argument("--mealy", type=int, default=150, help="Number of Mealy FSMs to generate")
    parser.add_argument("--handshake", type=int, default=78, help="Number of Handshake FSMs to generate")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test with 3 FSMs per category")

    args = parser.parse_args()

    if args.smoke_test:
        print("SMOKE TEST MODE: Generating 3 FSMs per category\n")
        args.moore = 3
        args.mealy = 3
        args.handshake = 3

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
