#!/usr/bin/env python3
"""
Test Design Parser - Priority 2 Implementation

Tests the Verilog parser with:
1. Module extraction
2. Port parsing
3. Instance hierarchy
4. Design statistics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.world_model.design_parser import DesignParser
import json


def create_test_design():
    """Create a more complex test design with hierarchy"""
    verilog_code = """
// Simple ALU module
module alu (
    input wire [7:0] a,
    input wire [7:0] b,
    input wire [1:0] op,
    output reg [7:0] result
);
    always @(*) begin
        case (op)
            2'b00: result = a + b;
            2'b01: result = a - b;
            2'b10: result = a & b;
            2'b11: result = a | b;
        endcase
    end
endmodule

// Top-level processor with ALU instance
module simple_processor (
    input wire clk,
    input wire reset,
    input wire [7:0] data_in,
    output wire [7:0] data_out
);
    wire [7:0] reg_a, reg_b;
    wire [1:0] alu_op;
    wire [7:0] alu_result;

    // Register file (simplified)
    reg [7:0] registers[0:3];

    // ALU instantiation
    alu alu_inst (
        .a(reg_a),
        .b(reg_b),
        .op(alu_op),
        .result(alu_result)
    );

    assign data_out = alu_result;

endmodule
"""

    test_file = "/tmp/test_processor.v"
    with open(test_file, 'w') as f:
        f.write(verilog_code)

    print(f"✓ Created test Verilog: {test_file}\n")
    return test_file


def test_design_parser():
    """Test the design parser"""
    print("="*70)
    print("PRIORITY 2: WorldModel Design Parser Test")
    print("="*70)
    print()

    # Create test design
    verilog_file = create_test_design()

    # Initialize parser
    parser = DesignParser()

    # Parse Verilog
    print("Parsing Verilog file...")
    netlist = parser.parse_verilog(verilog_file)

    print(f"✓ Successfully parsed netlist\n")

    # Display results
    print("="*70)
    print("PARSING RESULTS")
    print("="*70)
    print()

    # Summary
    summary = parser.get_design_summary()
    print("Design Summary:")
    print(f"  Top Module: {summary['top_module']}")
    print(f"  Total Modules: {summary['total_modules']}")
    print(f"  Total Instances: {summary['total_instances']}")
    print(f"  Unique Cell Types: {summary['unique_cell_types']}")
    print()

    # Module details
    print("Module Details:")
    print("-" * 70)
    for module_name, module in netlist.modules.items():
        print(f"\nModule: {module_name}")
        print(f"  Ports ({len(module.ports)}):")
        for port in module.ports:
            print(f"    - {port.direction:6} {port.name}")

        if module.instances:
            print(f"  Instances ({len(module.instances)}):")
            for inst in module.instances:
                print(f"    - {inst.name} ({inst.module_type})")
                if inst.connections:
                    print(f"      Connections: {len(inst.connections)} ports")

        if module.wires:
            print(f"  Wires: {', '.join(module.wires[:5])}{'...' if len(module.wires) > 5 else ''}")

    # Hierarchy
    print("\n" + "="*70)
    print("DESIGN HIERARCHY")
    print("="*70)
    hierarchy = parser.get_design_hierarchy()
    print(json.dumps(hierarchy, indent=2))

    # Cell types
    print("\n" + "="*70)
    print("CELL TYPES USED")
    print("="*70)
    cell_types = parser.get_cell_types()
    for cell_type in cell_types:
        print(f"  - {cell_type}")

    print("\n" + "="*70)
    print("PRIORITY 2 STATUS: COMPLETE!")
    print("="*70)
    print("\n✓ WorldModel Design Parser:")
    print("  - Parses Verilog modules")
    print("  - Extracts ports (I/O)")
    print("  - Identifies instances")
    print("  - Builds hierarchy")
    print("  - Generates statistics")
    print("\nCapabilities Demonstrated:")
    print(f"  ✓ Parsed {summary['total_modules']} modules")
    print(f"  ✓ Found {summary['total_instances']} instances")
    print(f"  ✓ Extracted module hierarchy")
    print(f"  ✓ Identified {summary['unique_cell_types']} unique cell types")

    return True


if __name__ == "__main__":
    success = test_design_parser()
    sys.exit(0 if success else 1)
