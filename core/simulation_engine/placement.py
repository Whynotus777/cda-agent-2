"""
Placement Engine

Interfaces with DREAMPlace for GPU-accelerated placement.
Handles global and detailed placement of standard cells.
"""

import subprocess
import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PlacementEngine:
    """
    Manages placement using DREAMPlace GPU-accelerated placer.

    DREAMPlace uses deep learning and GPU acceleration to perform
    fast, high-quality placement of millions of cells.
    """

    def __init__(self, dreamplace_path: Optional[str] = None):
        """
        Initialize placement engine.

        Args:
            dreamplace_path: Path to DREAMPlace installation
                           (defaults to searching common locations)
        """
        self.dreamplace_path = dreamplace_path or self._find_dreamplace()
        self.dreamplace_script = os.path.join(
            self.dreamplace_path, "dreamplace", "Placer.py"
        )

        # Verify DREAMPlace installation
        self._verify_dreamplace()

        logger.info(f"Initialized PlacementEngine with DREAMPlace at {self.dreamplace_path}")

    def _find_dreamplace(self) -> str:
        """Attempt to find DREAMPlace installation"""
        possible_paths = [
            "/usr/local/DREAMPlace",
            os.path.expanduser("~/DREAMPlace"),
            os.path.expanduser("~/dreamplace"),
            "/opt/DREAMPlace",
            "./DREAMPlace",
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                logger.info(f"Found DREAMPlace at: {path}")
                return path

        logger.warning("DREAMPlace not found in standard locations")
        return "/usr/local/DREAMPlace"  # Default guess

    def _verify_dreamplace(self):
        """Verify DREAMPlace installation"""
        if not os.path.exists(self.dreamplace_script):
            logger.warning(
                f"DREAMPlace Placer.py not found at {self.dreamplace_script}. "
                "Placement may fail."
            )

    def place(
        self,
        netlist_file: str,
        def_file: str,
        output_def: str,
        placement_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run placement on design.

        Args:
            netlist_file: Gate-level netlist (.v)
            def_file: DEF file with floorplan and initial placement
            output_def: Output DEF file with final placement
            placement_params: Optional placement parameters

        Returns:
            Placement statistics and metrics
        """
        logger.info(f"Starting placement: {netlist_file}")

        # Create DREAMPlace configuration
        config = self._create_placement_config(
            netlist_file=netlist_file,
            def_file=def_file,
            output_def=output_def,
            params=placement_params or {}
        )

        # Run DREAMPlace
        result = self._run_dreamplace(config)

        # Parse results
        stats = self._parse_placement_results(result, output_def)

        logger.info(f"Placement complete: HPWL={stats.get('hpwl', 0):.2f}")
        return stats

    def _create_placement_config(
        self,
        netlist_file: str,
        def_file: str,
        output_def: str,
        params: Dict
    ) -> str:
        """
        Create DREAMPlace JSON configuration file.

        Args:
            netlist_file: Netlist path
            def_file: Input DEF path
            output_def: Output DEF path
            params: Placement parameters

        Returns:
            Path to generated config file
        """
        # Default parameters
        config = {
            "aux_input": netlist_file,
            "def_input": def_file,
            "verilog_input": netlist_file,
            "result_dir": os.path.dirname(output_def),

            # GPU settings
            "gpu": 1,  # Enable GPU
            "num_threads": 8,

            # Placement density
            "target_density": params.get('target_density', 0.7),
            "density_weight": params.get('density_weight', 8.0),

            # Wirelength optimization
            "wirelength_weight": params.get('wirelength_weight', 1.0),

            # Global placement iterations
            "global_place_iterations": params.get('global_place_iterations', 2000),

            # Legalization and detailed placement
            "legalize_flag": 1,
            "detailed_place_flag": 1,

            # Optimization flags
            "routability_opt_flag": params.get('routability_opt', 1),
            "timing_opt_flag": params.get('timing_opt', 0),

            # Random seed for reproducibility
            "random_seed": params.get('random_seed', 1000),
        }

        # Write config
        config_path = "/tmp/dreamplace_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.debug(f"Created DREAMPlace config: {config_path}")
        return config_path

    def _run_dreamplace(self, config_path: str) -> Dict:
        """
        Execute DREAMPlace placer.

        Args:
            config_path: Path to configuration JSON

        Returns:
            Dictionary with stdout, stderr, and return code
        """
        if not os.path.exists(self.dreamplace_script):
            logger.error("DREAMPlace not properly installed")
            raise RuntimeError(
                f"DREAMPlace Placer.py not found at {self.dreamplace_script}. "
                "Please install DREAMPlace."
            )

        try:
            result = subprocess.run(
                ["python3", self.dreamplace_script, config_path],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=self.dreamplace_path
            )

            if result.returncode != 0:
                logger.error(f"DREAMPlace failed: {result.stderr}")
                # Don't raise exception, return error info
                return {
                    'success': False,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }

            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            logger.error("DREAMPlace timed out")
            raise RuntimeError("Placement timed out after 1 hour")
        except Exception as e:
            logger.error(f"DREAMPlace execution failed: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e)
            }

    def _parse_placement_results(self, result: Dict, output_def: str) -> Dict:
        """
        Parse DREAMPlace output to extract placement metrics.

        Args:
            result: DREAMPlace execution result
            output_def: Path to output DEF file

        Returns:
            Placement statistics
        """
        import re

        stats = {
            'success': result.get('success', False),
            'hpwl': 0.0,  # Half-Perimeter Wire Length
            'overflow': 0.0,
            'max_density': 0.0,
            'runtime': 0.0,
            'cell_count': 0,
        }

        if not result.get('success'):
            return stats

        stdout = result.get('stdout', '')

        # Parse HPWL (Half-Perimeter Wire Length)
        hpwl_match = re.search(r'HPWL[:\s]+(\d+\.?\d*)', stdout)
        if hpwl_match:
            stats['hpwl'] = float(hpwl_match.group(1))

        # Parse overflow (routing congestion metric)
        overflow_match = re.search(r'overflow[:\s]+(\d+\.?\d*)', stdout)
        if overflow_match:
            stats['overflow'] = float(overflow_match.group(1))

        # Parse density
        density_match = re.search(r'density[:\s]+(\d+\.?\d*)', stdout)
        if density_match:
            stats['max_density'] = float(density_match.group(1))

        # Parse runtime
        runtime_match = re.search(r'total runtime[:\s]+(\d+\.?\d*)', stdout, re.IGNORECASE)
        if runtime_match:
            stats['runtime'] = float(runtime_match.group(1))

        # Count cells from DEF file if it exists
        if os.path.exists(output_def):
            stats['cell_count'] = self._count_cells_in_def(output_def)

        return stats

    def _count_cells_in_def(self, def_file: str) -> int:
        """Count number of placed cells in DEF file"""
        try:
            with open(def_file, 'r') as f:
                content = f.read()

            # Count COMPONENTS section
            import re
            match = re.search(r'COMPONENTS\s+(\d+)', content)
            if match:
                return int(match.group(1))

        except Exception as e:
            logger.warning(f"Could not parse DEF file: {e}")

        return 0

    def optimize_placement(
        self,
        current_def: str,
        output_def: str,
        optimization_focus: str = "wirelength"
    ) -> Dict:
        """
        Re-run placement with different parameters for optimization.

        Args:
            current_def: Current placement DEF
            output_def: Optimized output DEF
            optimization_focus: 'wirelength', 'density', or 'routability'

        Returns:
            Optimized placement statistics
        """
        logger.info(f"Optimizing placement for: {optimization_focus}")

        # Adjust parameters based on optimization focus
        if optimization_focus == "wirelength":
            params = {
                'wirelength_weight': 2.0,
                'density_weight': 4.0,
                'global_place_iterations': 3000,
            }
        elif optimization_focus == "density":
            params = {
                'target_density': 0.65,
                'density_weight': 12.0,
                'wirelength_weight': 1.0,
            }
        elif optimization_focus == "routability":
            params = {
                'routability_opt': 1,
                'target_density': 0.6,
                'density_weight': 10.0,
            }
        else:
            params = {}

        # Extract netlist from DEF comments or use original
        # (In practice, would track this from initial placement)
        netlist_file = "/tmp/netlist.v"  # Placeholder

        return self.place(
            netlist_file=netlist_file,
            def_file=current_def,
            output_def=output_def,
            placement_params=params
        )

    def get_cell_positions(self, def_file: str) -> Dict[str, Tuple[float, float]]:
        """
        Extract cell positions from DEF file.

        Args:
            def_file: DEF file path

        Returns:
            Dictionary mapping cell name to (x, y) coordinates
        """
        positions = {}

        try:
            with open(def_file, 'r') as f:
                lines = f.readlines()

            # Parse COMPONENTS section
            in_components = False
            for line in lines:
                if line.strip().startswith('COMPONENTS'):
                    in_components = True
                    continue
                elif line.strip().startswith('END COMPONENTS'):
                    break

                if in_components and line.strip().startswith('-'):
                    # Parse component line
                    # Format: - cell_name gate_type + PLACED ( x y ) orientation ;
                    parts = line.split()
                    if len(parts) >= 7 and 'PLACED' in parts:
                        cell_name = parts[1]
                        try:
                            x_idx = parts.index('(') + 1
                            x = float(parts[x_idx])
                            y = float(parts[x_idx + 1])
                            positions[cell_name] = (x, y)
                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"Failed to parse DEF file: {e}")

        return positions

    def calculate_hpwl(
        self,
        cell_positions: Dict[str, Tuple[float, float]],
        netlist_connections: List[List[str]]
    ) -> float:
        """
        Calculate Half-Perimeter Wire Length.

        Args:
            cell_positions: Cell name -> (x, y) mapping
            netlist_connections: List of nets, each net is list of connected cells

        Returns:
            Total HPWL
        """
        total_hpwl = 0.0

        for net in netlist_connections:
            if len(net) < 2:
                continue

            # Get positions of all cells in this net
            positions = []
            for cell in net:
                if cell in cell_positions:
                    positions.append(cell_positions[cell])

            if len(positions) < 2:
                continue

            # Calculate bounding box
            xs = [pos[0] for pos in positions]
            ys = [pos[1] for pos in positions]

            hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            total_hpwl += hpwl

        return total_hpwl
