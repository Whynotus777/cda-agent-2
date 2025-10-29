#!/usr/bin/env python3
"""
Test script for A5 - Style & Review Copilot

Tests style enforcement, security rules, and report generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rtl_agents import A5_StyleReviewCopilot


def test_clean_code():
    """Test style checking on clean, compliant code"""
    print("\n" + "="*70)
    print("TEST 1: Clean Code (No Violations)")
    print("="*70)

    agent = A5_StyleReviewCopilot()

    # Clean RTL code following all rules
    clean_rtl = """
module counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [7:0] count
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 8'h00;
        end else if (enable) begin
            count <= count + 1;
        end
    end

endmodule
"""

    input_data = {
        'rtl_code': clean_rtl,
        'file_path': 'clean_counter.v'
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Execution Time: {result.execution_time_ms:.2f}ms")

    summary = result.output_data.get('summary', {})

    print(f"\n--- Violation Summary ---")
    print(f"  Total: {summary.get('total', 0)}")
    print(f"  Critical: {summary.get('critical', 0)}")
    print(f"  Warning: {summary.get('warning', 0)}")
    print(f"  Info: {summary.get('info', 0)}")

    # Success if no critical violations
    critical_count = summary.get('critical', 0)

    if critical_count == 0:
        print(f"\n✅ Phase 6 Target: 0 critical violations achieved")
        return True
    else:
        print(f"\n❌ Phase 6 Target: {critical_count} critical violations found")
        return False


def test_naming_violations():
    """Test naming convention enforcement"""
    print("\n" + "="*70)
    print("TEST 2: Naming Convention Violations")
    print("="*70)

    agent = A5_StyleReviewCopilot()

    # RTL with naming violations
    bad_naming_rtl = """
module BadModule (
    input wire CLK,           // Should be 'clk'
    input wire Reset,         // Should be 'rst_n' for active-low
    input wire DataIn,        // Should be snake_case
    output reg DataOut
);

    parameter MaxCount = 100;  // Should be MAX_COUNT

    always @(posedge CLK) begin
        if (Reset) begin
            DataOut <= 0;
        end
    end

endmodule
"""

    input_data = {
        'rtl_code': bad_naming_rtl,
        'file_path': 'bad_naming.v'
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    violations = result.output_data.get('violations', [])
    summary = result.output_data.get('summary', {})

    print(f"\n--- Violations Detected: {len(violations)} ---")

    naming_violations = [v for v in violations if v['category'] == 'naming']
    print(f"  Naming violations: {len(naming_violations)}")

    for v in naming_violations[:3]:  # Show first 3
        print(f"    [{v['severity']}] {v['rule_id']}: {v['message']} (line {v['line']})")

    # Should detect naming violations
    return len(naming_violations) > 0


def test_clock_reset_rules():
    """Test clock and reset domain checking"""
    print("\n" + "="*70)
    print("TEST 3: Clock & Reset Rules")
    print("="*70)

    agent = A5_StyleReviewCopilot()

    # RTL with clock/reset issues
    clock_reset_rtl = """
module clk_test (
    input wire clk,
    input wire rst_n,
    input wire data_in,
    output reg data_out
);

    // Good: explicit posedge
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 0;
        end else begin
            data_out <= data_in;
        end
    end

    // Warning: @(*) for combinational
    reg temp;
    always @(*) begin
        temp = data_in & 1'b1;
    end

endmodule
"""

    input_data = {
        'rtl_code': clock_reset_rtl,
        'file_path': 'clock_reset.v'
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    violations = result.output_data.get('violations', [])
    summary = result.output_data.get('summary', {})

    clock_violations = [v for v in violations if v['category'] == 'clock']
    reset_violations = [v for v in violations if v['category'] == 'reset']

    print(f"\n--- Clock/Reset Violations ---")
    print(f"  Clock: {len(clock_violations)}")
    print(f"  Reset: {len(reset_violations)}")

    for v in clock_violations + reset_violations:
        print(f"    [{v['severity']}] {v['rule_id']}: {v['message']}")

    return result.success


def test_security_rules():
    """Test security rule detection"""
    print("\n" + "="*70)
    print("TEST 4: Security Rules")
    print("="*70)

    agent = A5_StyleReviewCopilot()

    # RTL with security concerns
    security_rtl = """
module crypto_engine (
    input wire clk,
    input wire rst_n,
    input wire [127:0] key,        // Sensitive: encryption key
    input wire [127:0] data,
    output reg [127:0] encrypted
);

    // TODO: Add key expansion
    // FIXME: Implement AES rounds

    reg [127:0] secret_buffer;  // Sensitive: secret storage

    always @(posedge clk) begin
        if (!rst_n) begin
            encrypted <= 0;
            secret_buffer <= 0;
        end else begin
            encrypted <= data ^ key;  // Simple XOR (placeholder)
        end
    end

endmodule
"""

    input_data = {
        'rtl_code': security_rtl,
        'file_path': 'crypto.v'
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    violations = result.output_data.get('violations', [])
    summary = result.output_data.get('summary', {})

    security_violations = [v for v in violations if v['category'] == 'security']

    print(f"\n--- Security Violations: {len(security_violations)} ---")

    for v in security_violations:
        print(f"    [{v['severity']}] {v['rule_id']}: {v['message']} (line {v['line']})")
        if v.get('suggestion'):
            print(f"      Suggestion: {v['suggestion']}")

    # Should detect 'key' and 'secret' keywords + TODO/FIXME
    return len(security_violations) >= 3


def test_style_best_practices():
    """Test style guidelines and best practices"""
    print("\n" + "="*70)
    print("TEST 5: Style & Best Practices")
    print("="*70)

    agent = A5_StyleReviewCopilot()

    # RTL with style issues
    style_rtl = """
module style_test (
    input wire clk,
    input wire rst_n,
    input wire [7:0] data_in,
    output reg [7:0] data_out
);

    // Using old-style always instead of always_ff
    always @(posedge clk) begin
        if (!rst_n) begin
            data_out <= 0;
        end else
            data_out <= data_in;  // Missing begin/end
    end

    // Debug statement
    initial begin
        $display("Module initialized");
    end

endmodule
"""

    input_data = {
        'rtl_code': style_rtl,
        'file_path': 'style_test.v'
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    violations = result.output_data.get('violations', [])
    summary = result.output_data.get('summary', {})

    style_violations = [v for v in violations if v['category'] == 'style']
    bp_violations = [v for v in violations if v['category'] == 'best_practice']

    print(f"\n--- Style Issues ---")
    print(f"  Style: {len(style_violations)}")
    print(f"  Best Practice: {len(bp_violations)}")

    for v in style_violations + bp_violations:
        print(f"    [{v['severity']}] {v['rule_id']}: {v['message']}")

    return result.success


def test_markdown_report():
    """Test markdown report generation"""
    print("\n" + "="*70)
    print("TEST 6: Markdown Report Generation")
    print("="*70)

    agent = A5_StyleReviewCopilot()

    # Sample RTL with mixed violations
    sample_rtl = """
module sample (
    input wire CLK,
    input wire reset,
    input wire [7:0] key,
    output reg [7:0] out
);

    // TODO: Complete implementation

    always @(posedge CLK) begin
        if (reset) begin
            out <= 0;
        end
    end

endmodule
"""

    input_data = {
        'rtl_code': sample_rtl,
        'file_path': 'sample.v'
    }

    result = agent.process(input_data)

    print(f"\n✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")

    report_md = result.output_data.get('report_markdown', '')

    print(f"\n--- Report Length: {len(report_md)} chars ---")

    # Check report sections
    has_header = '# Style & Security Review Report' in report_md
    has_summary = '## Summary' in report_md
    has_violations = 'Violations' in report_md
    has_compliance = '## Compliance Status' in report_md

    print(f"\n--- Report Sections ---")
    print(f"  Header: {has_header}")
    print(f"  Summary: {has_summary}")
    print(f"  Violations: {has_violations}")
    print(f"  Compliance: {has_compliance}")

    print(f"\n--- Report Preview (first 800 chars) ---")
    print(report_md[:800])

    return has_header and has_summary and has_violations and has_compliance


def test_violation_rate():
    """Test violation detection rate across multiple samples"""
    print("\n" + "="*70)
    print("TEST 7: Violation Detection Rate")
    print("="*70)

    agent = A5_StyleReviewCopilot()

    test_cases = [
        {
            'name': 'Clock naming',
            'code': 'input wire CLK;',
            'should_detect': True
        },
        {
            'name': 'Reset naming',
            'code': 'input wire reset;',
            'should_detect': True
        },
        {
            'name': 'Sensitive data',
            'code': 'reg [127:0] password;',
            'should_detect': True
        },
        {
            'name': 'TODO comment',
            'code': '// TODO: Fix this',
            'should_detect': True
        },
        {
            'name': 'Clean code',
            'code': 'input wire clk; input wire rst_n;',
            'should_detect': False
        }
    ]

    results = []

    for test_case in test_cases:
        rtl = f"""
module test (
    {test_case['code']}
);
endmodule
"""

        input_data = {'rtl_code': rtl}
        result = agent.process(input_data)

        summary = result.output_data.get('summary', {})
        violations_found = summary.get('total', 0) > 0

        expected = test_case['should_detect']
        matched = (violations_found == expected)

        results.append((test_case['name'], matched, violations_found, expected))

        status = "✅" if matched else "❌"
        print(f"  {status} {test_case['name']:20} - Found: {violations_found}, Expected: {expected}")

    success_count = sum(1 for _, matched, _, _ in results if matched)
    total = len(results)

    detection_rate = (success_count / total * 100) if total > 0 else 0

    print(f"\n✓ Detection Rate: {detection_rate:.1f}%")

    # Target: ≥80% detection accuracy
    if detection_rate >= 80:
        print(f"\n🎉 Detection Target Achieved: {detection_rate:.1f}% ≥ 80%")
        return True
    else:
        print(f"\n⚠️  Detection Target: {detection_rate:.1f}% < 80%")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("A5 STYLE & REVIEW COPILOT - TEST SUITE")
    print("="*70)

    tests = [
        ("Clean Code (0 Critical Violations)", test_clean_code),
        ("Naming Convention Violations", test_naming_violations),
        ("Clock & Reset Rules", test_clock_reset_rules),
        ("Security Rules", test_security_rules),
        ("Style & Best Practices", test_style_best_practices),
        ("Markdown Report Generation", test_markdown_report),
        ("Violation Detection Rate", test_violation_rate)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    # Phase 6 target: 0 critical violations on clean code
    if success_rate >= 85:
        print(f"\n🎉 Phase 6 Complete: {success_rate:.1f}% success rate")
        print("✅ Target: 0 critical violations on compliant code")
    else:
        print(f"\n⚠️  Phase 6 Status: {success_rate:.1f}%")

    return passed_count == total_count


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
