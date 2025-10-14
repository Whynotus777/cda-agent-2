"""
Timing Analysis Module

Interfaces with OpenSTA (Open Static Timing Analysis) for timing verification.
"""

import subprocess
import os
import re
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TimingAnalyzer:
    """
    Manages static timing analysis using OpenSTA.

    OpenSTA verifies that the design meets all timing constraints
    (setup time, hold time, clock requirements).
    """

    def __init__(self, opensta_path: Optional[str] = None):
        """
        Initialize timing analyzer.

        Args:
            opensta_path: Path to OpenSTA binary (default: 'sta' in PATH)
        """
        self.opensta_binary = opensta_path or "sta"

        # Verify OpenSTA installation
        self._verify_opensta()

        logger.info("Initialized TimingAnalyzer")

    def _verify_opensta(self):
        """Verify OpenSTA installation"""
        try:
            result = subprocess.run(
                [self.opensta_binary, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.info(f"OpenSTA found: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"OpenSTA not found: {e}")

    def analyze_timing(
        self,
        netlist_file: str,
        sdc_file: str,
        lib_files: List[str],
        spef_file: Optional[str] = None
    ) -> Dict:
        """
        Run static timing analysis.

        Args:
            netlist_file: Gate-level netlist
            sdc_file: SDC constraints file
            lib_files: Liberty timing library files
            spef_file: Optional SPEF file with parasitic RC extraction

        Returns:
            Timing analysis results
        """
        logger.info(f"Running timing analysis on {netlist_file}")

        # Create OpenSTA script
        script_path = self._create_sta_script(
            netlist_file=netlist_file,
            sdc_file=sdc_file,
            lib_files=lib_files,
            spef_file=spef_file
        )

        # Run OpenSTA
        result = self._run_opensta(script_path)

        # Parse results
        timing_results = self._parse_timing_results(result.stdout)

        logger.info(
            f"Timing analysis complete: WNS={timing_results.get('wns', 0):.3f}ns"
        )

        return timing_results

    def _create_sta_script(
        self,
        netlist_file: str,
        sdc_file: str,
        lib_files: List[str],
        spef_file: Optional[str]
    ) -> str:
        """Create OpenSTA TCL script"""
        script_lines = []

        # Read liberty libraries
        for lib_file in lib_files:
            script_lines.append(f"read_liberty {lib_file}")

        # Read netlist
        script_lines.append(f"read_verilog {netlist_file}")

        # Link design
        script_lines.append("link_design [get_cells *]")

        # Read constraints
        script_lines.append(f"read_sdc {sdc_file}")

        # Read parasitics if available
        if spef_file and os.path.exists(spef_file):
            script_lines.append(f"read_spef {spef_file}")

        # Run timing analysis
        script_lines.extend([
            "report_checks -path_delay min_max -format full_clock_expanded",
            "report_tns",
            "report_wns",
            "report_worst_slack -max",
            "report_worst_slack -min",
            "report_checks -path_delay max -group_count 10",
        ])

        # Write script
        script_path = "/tmp/opensta_script.tcl"
        with open(script_path, 'w') as f:
            f.write("\n".join(script_lines))

        logger.debug(f"Created OpenSTA script: {script_path}")
        return script_path

    def _run_opensta(self, script_path: str) -> subprocess.CompletedProcess:
        """Execute OpenSTA"""
        try:
            result = subprocess.run(
                [self.opensta_binary, "-f", script_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            return result

        except subprocess.TimeoutExpired:
            logger.error("OpenSTA timed out")
            raise RuntimeError("Timing analysis timed out")
        except FileNotFoundError:
            logger.error("OpenSTA binary not found")
            raise RuntimeError("OpenSTA not installed")

    def _parse_timing_results(self, stdout: str) -> Dict:
        """Parse OpenSTA output"""
        results = {
            'wns': None,  # Worst Negative Slack
            'tns': None,  # Total Negative Slack
            'worst_path_delay': None,
            'setup_violations': 0,
            'hold_violations': 0,
            'critical_paths': []
        }

        # Parse WNS (Worst Negative Slack)
        wns_match = re.search(r'wns\s+([-\d.]+)', stdout, re.IGNORECASE)
        if wns_match:
            results['wns'] = float(wns_match.group(1))

        # Parse TNS (Total Negative Slack)
        tns_match = re.search(r'tns\s+([-\d.]+)', stdout, re.IGNORECASE)
        if tns_match:
            results['tns'] = float(tns_match.group(1))

        # Parse worst path delay
        delay_match = re.search(r'data\s+arrival\s+time\s+([-\d.]+)', stdout)
        if delay_match:
            results['worst_path_delay'] = float(delay_match.group(1))

        # Count setup violations
        setup_violations = re.findall(r'VIOLATED.*setup', stdout, re.IGNORECASE)
        results['setup_violations'] = len(setup_violations)

        # Count hold violations
        hold_violations = re.findall(r'VIOLATED.*hold', stdout, re.IGNORECASE)
        results['hold_violations'] = len(hold_violations)

        # Extract critical paths
        results['critical_paths'] = self._extract_critical_paths(stdout)

        return results

    def _extract_critical_paths(self, stdout: str) -> List[Dict]:
        """Extract critical path information"""
        paths = []

        # Simple extraction - find path summaries
        path_pattern = r'Startpoint:\s+(\S+).*?Endpoint:\s+(\S+).*?slack\s+([-\d.]+)'

        for match in re.finditer(path_pattern, stdout, re.DOTALL):
            paths.append({
                'startpoint': match.group(1),
                'endpoint': match.group(2),
                'slack': float(match.group(3))
            })

        return paths[:10]  # Return top 10 critical paths

    def check_timing_constraints(
        self,
        netlist_file: str,
        sdc_file: str,
        lib_files: List[str]
    ) -> bool:
        """
        Quick check if timing constraints are met.

        Args:
            netlist_file: Gate-level netlist
            sdc_file: Constraints file
            lib_files: Liberty files

        Returns:
            True if all timing constraints met, False otherwise
        """
        results = self.analyze_timing(netlist_file, sdc_file, lib_files)

        wns = results.get('wns', -999)
        setup_violations = results.get('setup_violations', 999)
        hold_violations = results.get('hold_violations', 999)

        return wns >= 0 and setup_violations == 0 and hold_violations == 0

    def find_critical_cells(
        self,
        netlist_file: str,
        sdc_file: str,
        lib_files: List[str]
    ) -> List[str]:
        """
        Identify cells on critical timing paths.

        Args:
            netlist_file: Gate-level netlist
            sdc_file: Constraints file
            lib_files: Liberty files

        Returns:
            List of cell names on critical paths
        """
        results = self.analyze_timing(netlist_file, sdc_file, lib_files)

        critical_cells = []
        for path in results.get('critical_paths', []):
            if 'startpoint' in path:
                critical_cells.append(path['startpoint'])
            if 'endpoint' in path:
                critical_cells.append(path['endpoint'])

        return list(set(critical_cells))  # Remove duplicates

    def estimate_max_frequency(
        self,
        netlist_file: str,
        sdc_file: str,
        lib_files: List[str]
    ) -> float:
        """
        Estimate maximum achievable clock frequency.

        Args:
            netlist_file: Gate-level netlist
            sdc_file: Constraints file
            lib_files: Liberty files

        Returns:
            Maximum frequency in MHz
        """
        results = self.analyze_timing(netlist_file, sdc_file, lib_files)

        worst_delay = results.get('worst_path_delay', 0)

        if worst_delay <= 0:
            return 0.0

        # Max frequency = 1 / worst_delay (convert ns to MHz)
        max_freq_mhz = 1000.0 / worst_delay

        return max_freq_mhz
