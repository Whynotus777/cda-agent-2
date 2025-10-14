"""
Synthesis Engine

Interfaces with Yosys to perform RTL synthesis.
Converts Verilog/SystemVerilog to gate-level netlist.
"""

import subprocess
import os
import re
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SynthesisEngine:
    """
    Manages synthesis using Yosys open-source synthesis tool.

    Yosys converts RTL (Register Transfer Level) Verilog into a
    gate-level netlist using cells from the technology library.
    """

    def __init__(self, tech_library_path: Optional[str] = None):
        """
        Initialize synthesis engine.

        Args:
            tech_library_path: Path to technology library (.lib file)
        """
        self.tech_library_path = tech_library_path
        self.yosys_binary = "yosys"  # Assumes yosys is in PATH

        # Verify Yosys installation
        self._verify_yosys()

        logger.info("Initialized SynthesisEngine")

    def _verify_yosys(self):
        """Verify that Yosys is installed and accessible"""
        try:
            result = subprocess.run(
                [self.yosys_binary, "-V"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.info(f"Yosys version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Yosys not found or not working: {e}")
            raise RuntimeError(
                "Yosys not found. Install with: sudo apt-get install yosys"
            )

    def synthesize(
        self,
        rtl_files: List[str],
        top_module: str,
        output_netlist: str,
        optimization_goal: str = "balanced"
    ) -> Dict:
        """
        Run synthesis on RTL design.

        Args:
            rtl_files: List of Verilog/SystemVerilog source files
            top_module: Name of top-level module
            output_netlist: Path for output gate-level netlist
            optimization_goal: 'speed', 'area', 'power', or 'balanced'

        Returns:
            Dictionary with synthesis results and statistics
        """
        logger.info(f"Starting synthesis: top={top_module}, goal={optimization_goal}")

        # Create Yosys script
        script_path = self._create_synthesis_script(
            rtl_files=rtl_files,
            top_module=top_module,
            output_netlist=output_netlist,
            optimization_goal=optimization_goal
        )

        # Run Yosys
        result = self._run_yosys(script_path)

        # Parse results
        stats = self._parse_synthesis_results(result.stdout)

        logger.info(f"Synthesis complete: {stats.get('cell_count', 0)} cells")
        return stats

    def _create_synthesis_script(
        self,
        rtl_files: List[str],
        top_module: str,
        output_netlist: str,
        optimization_goal: str
    ) -> str:
        """
        Generate Yosys synthesis script.

        Returns:
            Path to generated script file
        """
        script_lines = []

        # Read RTL files
        for rtl_file in rtl_files:
            script_lines.append(f"read_verilog {rtl_file}")

        # Set hierarchy
        script_lines.append(f"hierarchy -check -top {top_module}")

        # Perform synthesis passes based on optimization goal
        if optimization_goal == "speed":
            # High-performance synthesis
            script_lines.extend([
                "proc",
                "opt -full",
                "fsm",
                "opt",
                "memory",
                "opt",
            ])
        elif optimization_goal == "area":
            # Area-optimized synthesis
            script_lines.extend([
                "proc",
                "opt -full",
                "fsm",
                "opt -full",
                "memory",
                "opt -full",
                "techmap",
                "opt -fast",
            ])
        elif optimization_goal == "power":
            # Low-power synthesis
            script_lines.extend([
                "proc",
                "opt",
                "fsm",
                "opt",
                "memory",
                "opt",
            ])
        else:  # balanced
            script_lines.extend([
                "proc",
                "opt",
                "fsm",
                "opt",
                "memory",
                "opt",
            ])

        # Technology mapping
        if self.tech_library_path and os.path.exists(self.tech_library_path):
            # Map to technology library
            script_lines.append(f"dfflibmap -liberty {self.tech_library_path}")
            script_lines.append(f"abc -liberty {self.tech_library_path}")
        else:
            # Generic synthesis (no specific tech library)
            script_lines.append("techmap")
            script_lines.append("abc -g AND,OR,XOR")

        # Clean up
        script_lines.append("clean")

        # Generate statistics
        script_lines.append("stat")

        # Write output netlist
        script_lines.append(f"write_verilog {output_netlist}")

        # Write script to file
        script_path = "/tmp/yosys_synthesis.ys"
        with open(script_path, 'w') as f:
            f.write("\n".join(script_lines))

        logger.debug(f"Generated Yosys script: {script_path}")
        return script_path

    def _run_yosys(self, script_path: str) -> subprocess.CompletedProcess:
        """
        Execute Yosys with the given script.

        Args:
            script_path: Path to Yosys script

        Returns:
            Completed process with stdout/stderr
        """
        try:
            result = subprocess.run(
                [self.yosys_binary, "-s", script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"Yosys failed: {result.stderr}")
                raise RuntimeError(f"Synthesis failed: {result.stderr}")

            return result

        except subprocess.TimeoutExpired:
            logger.error("Yosys synthesis timed out")
            raise RuntimeError("Synthesis timed out after 5 minutes")

    def _parse_synthesis_results(self, stdout: str) -> Dict:
        """
        Parse Yosys output to extract synthesis statistics.

        Args:
            stdout: Yosys stdout output

        Returns:
            Dictionary of synthesis statistics
        """
        stats = {
            'cell_count': 0,
            'net_count': 0,
            'gate_count': 0,
            'flip_flop_count': 0,
            'area': 0.0
        }

        # Parse cell count
        cell_match = re.search(r'Number of cells:\s+(\d+)', stdout)
        if cell_match:
            stats['cell_count'] = int(cell_match.group(1))

        # Parse wire/net count
        wire_match = re.search(r'Number of wires:\s+(\d+)', stdout)
        if wire_match:
            stats['net_count'] = int(wire_match.group(1))

        # Count specific cell types
        # AND, OR, XOR, etc.
        gate_match = re.findall(r'\$_(AND|OR|XOR|NOT|NAND|NOR)\s+(\d+)', stdout)
        stats['gate_count'] = sum(int(count) for _, count in gate_match)

        # Flip-flops
        dff_match = re.findall(r'(\$_DFF_\w+|DFF\w*)\s+(\d+)', stdout)
        stats['flip_flop_count'] = sum(int(count) for _, count in dff_match)

        # Area estimate (if available from tech library)
        area_match = re.search(r'Chip area.*?(\d+\.?\d*)', stdout)
        if area_match:
            stats['area'] = float(area_match.group(1))

        return stats

    def optimize_netlist(
        self,
        input_netlist: str,
        output_netlist: str,
        optimization_passes: int = 3
    ) -> Dict:
        """
        Perform additional optimization passes on existing netlist.

        Args:
            input_netlist: Input gate-level netlist
            output_netlist: Output optimized netlist
            optimization_passes: Number of optimization iterations

        Returns:
            Optimization statistics
        """
        logger.info(f"Optimizing netlist: {optimization_passes} passes")

        script_lines = [
            f"read_verilog {input_netlist}",
            "hierarchy -check",
        ]

        # Multiple optimization passes
        for _ in range(optimization_passes):
            script_lines.extend([
                "opt -full",
                "clean",
            ])

        script_lines.extend([
            "stat",
            f"write_verilog {output_netlist}"
        ])

        script_path = "/tmp/yosys_optimize.ys"
        with open(script_path, 'w') as f:
            f.write("\n".join(script_lines))

        result = self._run_yosys(script_path)
        stats = self._parse_synthesis_results(result.stdout)

        return stats

    def generate_technology_mapped_netlist(
        self,
        rtl_files: List[str],
        top_module: str,
        lib_file: str,
        output_netlist: str
    ) -> Dict:
        """
        Synthesize with specific technology library for accurate results.

        Args:
            rtl_files: Verilog source files
            top_module: Top module name
            lib_file: Liberty .lib file
            output_netlist: Output netlist path

        Returns:
            Synthesis statistics
        """
        logger.info(f"Technology mapping with library: {lib_file}")

        self.tech_library_path = lib_file

        return self.synthesize(
            rtl_files=rtl_files,
            top_module=top_module,
            output_netlist=output_netlist,
            optimization_goal="balanced"
        )

    def estimate_gate_count(self, rtl_files: List[str], top_module: str) -> int:
        """
        Quick estimate of gate count without full synthesis.

        Args:
            rtl_files: Verilog source files
            top_module: Top module name

        Returns:
            Estimated gate count
        """
        # Run minimal synthesis for quick estimate
        temp_netlist = "/tmp/estimate_netlist.v"

        stats = self.synthesize(
            rtl_files=rtl_files,
            top_module=top_module,
            output_netlist=temp_netlist,
            optimization_goal="balanced"
        )

        # Clean up
        if os.path.exists(temp_netlist):
            os.remove(temp_netlist)

        return stats.get('cell_count', 0)
