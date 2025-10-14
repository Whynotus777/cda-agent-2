"""
Routing Engine

Interfaces with TritonRoute for detailed routing.
"""

import subprocess
import os
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RoutingEngine:
    """
    Manages routing using TritonRoute open-source detailed router.

    TritonRoute performs detailed routing to create physical wire connections
    between all placed cells while meeting design rules.
    """

    def __init__(self, tritonroute_path: Optional[str] = None):
        """
        Initialize routing engine.

        Args:
            tritonroute_path: Path to TritonRoute binary
        """
        self.tritonroute_binary = tritonroute_path or "TritonRoute"

        # Verify installation
        self._verify_tritonroute()

        logger.info("Initialized RoutingEngine")

    def _verify_tritonroute(self):
        """Verify TritonRoute installation"""
        try:
            result = subprocess.run(
                [self.tritonroute_binary, "-h"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.info("TritonRoute found")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"TritonRoute not found: {e}")

    def route(
        self,
        def_file: str,
        lef_file: str,
        guide_file: Optional[str],
        output_def: str,
        routing_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run detailed routing.

        Args:
            def_file: Input DEF with placement
            lef_file: LEF file with technology info
            guide_file: Optional routing guide from global router
            output_def: Output DEF with routing
            routing_params: Optional routing parameters

        Returns:
            Routing statistics
        """
        logger.info(f"Starting routing: {def_file}")

        params = routing_params or {}

        # Create TritonRoute parameter file
        param_file = self._create_routing_params(
            def_file=def_file,
            lef_file=lef_file,
            guide_file=guide_file,
            output_def=output_def,
            params=params
        )

        # Run TritonRoute
        result = self._run_tritonroute(param_file)

        # Parse results
        stats = self._parse_routing_results(result)

        logger.info(f"Routing complete: {stats.get('total_wirelength', 0):.2f} um")
        return stats

    def _create_routing_params(
        self,
        def_file: str,
        lef_file: str,
        guide_file: Optional[str],
        output_def: str,
        params: Dict
    ) -> str:
        """Create TritonRoute parameter file"""
        param_lines = [
            f"lef:{lef_file}",
            f"def:{def_file}",
            f"output:{output_def}",
            f"threads:{params.get('threads', 8)}",
        ]

        if guide_file:
            param_lines.append(f"guide:{guide_file}")

        # DRC parameters
        if 'drc_cost' in params:
            param_lines.append(f"drCost:{params['drc_cost']}")

        # Write to file
        param_file = "/tmp/tritonroute_params.txt"
        with open(param_file, 'w') as f:
            f.write("\n".join(param_lines))

        return param_file

    def _run_tritonroute(self, param_file: str) -> subprocess.CompletedProcess:
        """Execute TritonRoute"""
        try:
            result = subprocess.run(
                [self.tritonroute_binary, param_file],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            return result

        except subprocess.TimeoutExpired:
            logger.error("TritonRoute timed out")
            raise RuntimeError("Routing timed out")
        except FileNotFoundError:
            logger.error("TritonRoute binary not found")
            raise RuntimeError("TritonRoute not installed")

    def _parse_routing_results(self, result: subprocess.CompletedProcess) -> Dict:
        """Parse routing output"""
        stats = {
            'total_wirelength': 0.0,
            'via_count': {},
            'drc_violations': 0,
            'runtime': 0.0,
            'success': result.returncode == 0
        }

        stdout = result.stdout

        # Parse total wirelength
        wl_match = re.search(r'total\s+wirelength[:\s]+(\d+\.?\d*)', stdout, re.IGNORECASE)
        if wl_match:
            stats['total_wirelength'] = float(wl_match.group(1))

        # Parse via counts
        via_matches = re.findall(r'(VIA\d+)[:\s]+(\d+)', stdout)
        for via_type, count in via_matches:
            stats['via_count'][via_type] = int(count)

        # Parse DRC violations
        drc_match = re.search(r'DRC\s+violations[:\s]+(\d+)', stdout, re.IGNORECASE)
        if drc_match:
            stats['drc_violations'] = int(drc_match.group(1))

        # Parse runtime
        runtime_match = re.search(r'runtime[:\s]+(\d+\.?\d*)', stdout, re.IGNORECASE)
        if runtime_match:
            stats['runtime'] = float(runtime_match.group(1))

        return stats

    def calculate_wirelength_from_def(self, def_file: str) -> float:
        """
        Calculate total wirelength from routed DEF file.

        Args:
            def_file: DEF file with routing information

        Returns:
            Total wirelength in micrometers
        """
        total_length = 0.0

        try:
            with open(def_file, 'r') as f:
                lines = f.readlines()

            # Parse NETS section
            in_nets = False
            for line in lines:
                if line.strip().startswith('NETS'):
                    in_nets = True
                    continue
                elif line.strip().startswith('END NETS'):
                    break

                if in_nets and 'ROUTED' in line:
                    # Parse routing segments and calculate length
                    # Simplified - actual DEF routing is more complex
                    length = self._estimate_segment_length(line)
                    total_length += length

        except Exception as e:
            logger.error(f"Failed to parse DEF for wirelength: {e}")

        return total_length

    def _estimate_segment_length(self, routing_line: str) -> float:
        """Estimate wire segment length from DEF routing line"""
        # Simplified estimation
        # In practice, would parse full routing path
        coords = re.findall(r'\(\s*(\d+)\s+(\d+)\s*\)', routing_line)

        if len(coords) < 2:
            return 0.0

        total = 0.0
        for i in range(len(coords) - 1):
            x1, y1 = int(coords[i][0]), int(coords[i][1])
            x2, y2 = int(coords[i + 1][0]), int(coords[i + 1][1])

            # Manhattan distance
            total += abs(x2 - x1) + abs(y2 - y1)

        # Convert from DEF units to micrometers (typically 1 DEF unit = 1 nm)
        return total / 1000.0  # nm to um
