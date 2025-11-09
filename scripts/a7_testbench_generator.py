#!/usr/bin/env python3
"""
A7 LLM Testbench Generator for Gold-Standard RTL Verification

Uses Claude (or other LLM) to generate comprehensive, state-aware testbenches
for complex RTL designs (especially FSMs). This solves the template testbench
limitation that caused V4's FSM regression.

Key Innovation: LLM understands FSM semantics and can generate testbenches that:
- Test all state transitions
- Include edge cases
- Add assertions for illegal states
- Verify complex state sequences
"""

import os
import anthropic
from typing import Dict, Optional, Tuple
from pathlib import Path
import re

class A7TestbenchGenerator:
    """
    A7 LLM-based Testbench Generator

    Uses Claude Sonnet 4 to generate comprehensive SystemVerilog testbenches
    that far exceed simple template-based approaches.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize A7 testbench generator.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: claude-sonnet-4)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate_fsm_testbench(
        self,
        spec: str,
        rtl_code: str,
        category: str = "fsm"
    ) -> str:
        """
        Generate comprehensive FSM testbench using A7 LLM.

        Args:
            spec: Natural language specification of the FSM
            rtl_code: Verilog/SystemVerilog RTL implementation
            category: FSM category (fsm_2state, fsm_3state, etc.)

        Returns:
            Complete SystemVerilog testbench code
        """

        prompt = self._build_fsm_testbench_prompt(spec, rtl_code, category)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,  # Deterministic for testbench generation
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract testbench code from response
            testbench_code = self._extract_code(response.content[0].text)

            return testbench_code

        except Exception as e:
            raise RuntimeError(f"A7 testbench generation failed: {e}")

    def _build_fsm_testbench_prompt(self, spec: str, rtl_code: str, category: str) -> str:
        """Build comprehensive prompt for FSM testbench generation."""

        return f"""You are an expert RTL verification engineer specializing in FSM verification.

Generate a comprehensive SystemVerilog testbench for the following FSM design.

**Specification:**
{spec}

**RTL Implementation:**
```systemverilog
{rtl_code}
```

**Category:** {category}

**Testbench Requirements:**

1. **State Coverage:**
   - Test ALL states (not just happy path)
   - Ensure every state is reachable
   - Verify outputs in each state

2. **Transition Coverage:**
   - Test ALL valid state transitions
   - Test edge cases (rapid input changes, held inputs)
   - Verify transition conditions

3. **Reset Behavior:**
   - Test reset in IDLE state
   - Test reset during active states
   - Verify FSM returns to correct initial state

4. **Edge Cases:**
   - Back-to-back transitions
   - Invalid input combinations (if applicable)
   - Boundary conditions

5. **Assertions:**
   - Add assertions for illegal states (if applicable)
   - Verify state encoding
   - Check for state machine deadlock

6. **Pass/Fail Indication:**
   - Clear error messages for each failure
   - Final line MUST be exactly "TEST PASSED" or "TEST FAILED"
   - Use $display for debugging output

**Testbench Structure:**
```systemverilog
module testbench;
    // Clock and reset
    reg clk = 0;
    reg rst_n;

    // DUT inputs
    reg [inputs...];

    // DUT outputs
    wire [outputs...];

    // Test variables
    integer errors = 0;

    // Instantiate DUT
    <module_name> dut (
        .clk(clk),
        .rst_n(rst_n),
        ...
    );

    // Clock generation (10ns period)
    always #5 clk = ~clk;

    // Main test sequence
    initial begin
        // Initialize
        rst_n = 0;
        [inputs] = 0;
        #20;
        rst_n = 1;
        #10;

        // Test Case 1: Reset behavior
        // ...

        // Test Case 2: State transitions
        // ...

        // Test Case 3: Edge cases
        // ...

        // Final result
        #100;
        if (errors == 0) begin
            $display("TEST PASSED");
        end else begin
            $display("TEST FAILED");
            $display("%0d errors detected", errors);
        end

        $finish;
    end

    // Helper task for checking outputs
    task check_output;
        input [expected_values...];
        begin
            if (actual != expected) begin
                $display("ERROR at time %0t: Expected X, got Y", $time);
                errors = errors + 1;
            end
        end
    endtask

endmodule
```

**CRITICAL:**
- Generate COMPLETE, executable testbench
- Test MORE than just the happy path
- Include timing (#delays) between test steps
- Final line MUST be "TEST PASSED" or "TEST FAILED"
- Be thorough - this testbench determines if RTL is training-worthy

Generate the complete testbench now:"""

    def _extract_code(self, response_text: str) -> str:
        """Extract SystemVerilog code from LLM response."""

        # Try to find code block
        if "```systemverilog" in response_text:
            code = response_text.split("```systemverilog")[1].split("```")[0]
        elif "```verilog" in response_text:
            code = response_text.split("```verilog")[1].split("```")[0]
        elif "```" in response_text:
            # Generic code block
            code = response_text.split("```")[1].split("```")[0]
        else:
            # No code block markers, take everything
            code = response_text

        return code.strip()

    def generate_generic_testbench(
        self,
        spec: str,
        rtl_code: str,
        category: str
    ) -> str:
        """
        Generate testbench for non-FSM designs.

        Can be used for counters, shift registers, etc. if template
        testbenches are insufficient.
        """

        prompt = f"""Generate a comprehensive SystemVerilog testbench for:

**Specification:** {spec}

**RTL Code:**
```systemverilog
{rtl_code}
```

**Category:** {category}

Requirements:
- Test all functionality
- Include edge cases
- Clear pass/fail with "TEST PASSED" or "TEST FAILED"
- Use $display for errors

Generate complete, executable testbench:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3072,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            return self._extract_code(response.content[0].text)

        except Exception as e:
            raise RuntimeError(f"A7 testbench generation failed: {e}")


def test_a7_generator():
    """Test A7 generator with a simple FSM example."""

    print("="*80)
    print("  TESTING A7 TESTBENCH GENERATOR")
    print("="*80)
    print()

    # Example FSM
    spec = "Create a 2-state FSM that transitions from IDLE to ACTIVE on start signal, back to IDLE on done signal."

    rtl_code = """module simple_fsm (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic done,
    output logic busy
);
    typedef enum logic {
        IDLE = 1'b0,
        ACTIVE = 1'b1
    } state_t;

    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    always_comb begin
        next_state = state;
        busy = (state == ACTIVE);

        case (state)
            IDLE: if (start) next_state = ACTIVE;
            ACTIVE: if (done) next_state = IDLE;
        endcase
    end
endmodule"""

    try:
        generator = A7TestbenchGenerator()
        print("✅ A7 generator initialized")
        print()
        print("🔧 Generating testbench for simple 2-state FSM...")
        print()

        testbench = generator.generate_fsm_testbench(spec, rtl_code, "fsm_2state")

        print("✅ Testbench generated!")
        print()
        print("📝 Generated testbench (first 500 chars):")
        print("-" * 80)
        print(testbench[:500])
        print("...")
        print("-" * 80)
        print()
        print(f"Total length: {len(testbench)} characters")
        print()

        # Save for inspection
        output_file = Path("test_a7_testbench.sv")
        output_file.write_text(testbench)
        print(f"💾 Full testbench saved to: {output_file}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    print()
    print("="*80)
    print("  ✅ A7 GENERATOR TEST COMPLETE")
    print("="*80)

    return True


if __name__ == "__main__":
    test_a7_generator()
