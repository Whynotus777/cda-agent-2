"""
Power Analysis Module

Estimates power consumption (static and dynamic) for the chip design.
"""

import os
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PowerAnalyzer:
    """
    Analyzes power consumption of chip design.

    Calculates:
    - Static power (leakage)
    - Dynamic power (switching)
    - Total power consumption
    """

    def __init__(self, tech_library=None):
        """Initialize power analyzer

        Args:
            tech_library: Optional TechLibrary instance to reuse parsed power data
        """
        self.tech_library = tech_library
        logger.info("Initialized PowerAnalyzer")

    def analyze_power(
        self,
        netlist_file: str,
        sdc_file: str,
        lib_files: List[str],
        activity_file: Optional[str] = None,
        voltage: float = 1.0,
        temperature: float = 25.0
    ) -> Dict:
        """
        Estimate power consumption.

        Args:
            netlist_file: Gate-level netlist
            sdc_file: SDC constraints (for clock info)
            lib_files: Liberty files with power models
            activity_file: Optional VCD/SAIF with switching activity
            voltage: Operating voltage (V)
            temperature: Operating temperature (C)

        Returns:
            Power analysis results
        """
        logger.info(f"Analyzing power for {netlist_file}")

        # Parse netlist to get cell list
        cells = self._parse_netlist_cells(netlist_file)

        # Parse lib files for power data or reuse from TechLibrary
        if self.tech_library is not None:
            cell_power_data = self.tech_library.get_power_map()
        else:
            cell_power_data = self._parse_lib_power(lib_files)

        # Get switching activity (or use defaults)
        activity = self._get_switching_activity(activity_file, cells)

        # Calculate static power
        static_power = self._calculate_static_power(
            cells, cell_power_data, voltage, temperature
        )

        # Calculate dynamic power
        dynamic_power = self._calculate_dynamic_power(
            cells, cell_power_data, activity, voltage
        )

        # Total power
        total_power = static_power + dynamic_power

        results = {
            'total_power': total_power,  # mW
            'static_power': static_power,  # mW (leakage)
            'dynamic_power': dynamic_power,  # mW (switching)
            'internal_power': dynamic_power * 0.6,  # Estimate
            'switching_power': dynamic_power * 0.4,  # Estimate
            'cell_count': len(cells),
            'voltage': voltage,
            'temperature': temperature
        }

        logger.info(f"Power analysis complete: {total_power:.2f} mW")
        return results

    def _parse_netlist_cells(self, netlist_file: str) -> List[Dict]:
        """Extract cell instances from netlist"""
        cells = []

        try:
            with open(netlist_file, 'r') as f:
                content = f.read()

            # Find module instantiations
            # Format: cell_type instance_name ( ... );
            inst_pattern = r'(\w+)\s+(\w+)\s*\([^;]+\);'

            for match in re.finditer(inst_pattern, content):
                cell_type = match.group(1)
                instance_name = match.group(2)

                # Filter out wire declarations and keywords
                if cell_type not in ['wire', 'input', 'output', 'module', 'endmodule', 'assign']:
                    cells.append({
                        'name': instance_name,
                        'type': cell_type
                    })

        except Exception as e:
            logger.error(f"Failed to parse netlist: {e}")

        return cells

    def _parse_lib_power(self, lib_files: List[str]) -> Dict[str, Dict]:
        """
        Extract power information from Liberty files.

        Returns:
            Dict mapping cell type to power characteristics
        """
        power_data = {}

        for lib_file in lib_files:
            if not os.path.exists(lib_file):
                continue

            try:
                with open(lib_file, 'r') as f:
                    content = f.read()

                # Parse cell blocks
                cell_pattern = r'cell\s*\(\s*"?(\w+)"?\s*\)\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}'

                for match in re.finditer(cell_pattern, content, re.DOTALL):
                    cell_name = match.group(1)
                    cell_content = match.group(2)

                    # Extract leakage power
                    leakage_match = re.search(
                        r'leakage_power.*?value\s*:\s*([\d.]+)',
                        cell_content,
                        re.DOTALL
                    )
                    leakage = float(leakage_match.group(1)) if leakage_match else 1.0

                    # Extract internal power (energy per transition)
                    energy_match = re.search(
                        r'energy\s*:\s*([\d.]+)',
                        cell_content
                    )
                    energy = float(energy_match.group(1)) if energy_match else 0.5

                    power_data[cell_name] = {
                        'leakage_power': leakage,  # nW
                        'switching_energy': energy  # pJ
                    }

            except Exception as e:
                logger.warning(f"Failed to parse {lib_file}: {e}")

        return power_data

    def _get_switching_activity(
        self,
        activity_file: Optional[str],
        cells: List[Dict]
    ) -> Dict[str, float]:
        """
        Get switching activity for each cell.

        Args:
            activity_file: Optional VCD/SAIF file with activity data
            cells: List of cell instances

        Returns:
            Dict mapping cell name to toggle rate (MHz)
        """
        activity = {}

        if activity_file and os.path.exists(activity_file):
            # Parse activity file (VCD or SAIF format)
            # TODO: Implement full VCD/SAIF parser
            logger.info(f"Loading activity from {activity_file}")
        else:
            # Use default activity estimates
            # Clock tree: high activity
            # Data path: medium activity
            # Control logic: low activity

            for cell in cells:
                cell_name = cell['name']
                cell_type = cell['type'].upper()

                if 'CLK' in cell_name or 'CLOCK' in cell_name:
                    # Clock tree cells toggle at clock frequency
                    activity[cell_name] = 1000.0  # 1 GHz
                elif 'DFF' in cell_type or 'FLIP' in cell_type:
                    # Flip-flops toggle at ~10% clock rate
                    activity[cell_name] = 100.0  # 100 MHz
                elif 'BUF' in cell_type or 'INV' in cell_type:
                    # Buffers: medium activity
                    activity[cell_name] = 50.0  # 50 MHz
                else:
                    # Combinational logic: low activity
                    activity[cell_name] = 20.0  # 20 MHz

        return activity

    def _calculate_static_power(
        self,
        cells: List[Dict],
        cell_power_data: Dict[str, Dict],
        voltage: float,
        temperature: float
    ) -> float:
        """
        Calculate total static (leakage) power.

        Args:
            cells: List of cell instances
            cell_power_data: Power data from lib files
            voltage: Operating voltage
            temperature: Operating temperature

        Returns:
            Total static power in mW
        """
        total_leakage = 0.0  # nW

        for cell in cells:
            cell_type = cell['type']

            if cell_type in cell_power_data:
                leakage = cell_power_data[cell_type]['leakage_power']
            else:
                # Default estimate
                leakage = 1.0  # nW

            # Temperature scaling (rough approximation)
            # Leakage doubles every 10C
            temp_scale = 2.0 ** ((temperature - 25) / 10)

            # Voltage scaling (exponential)
            voltage_scale = voltage ** 2

            total_leakage += leakage * temp_scale * voltage_scale

        # Convert nW to mW
        return total_leakage / 1e6

    def _calculate_dynamic_power(
        self,
        cells: List[Dict],
        cell_power_data: Dict[str, Dict],
        activity: Dict[str, float],
        voltage: float
    ) -> float:
        """
        Calculate total dynamic (switching) power.

        Args:
            cells: List of cell instances
            cell_power_data: Power data from lib files
            activity: Switching activity per cell
            voltage: Operating voltage

        Returns:
            Total dynamic power in mW
        """
        total_dynamic = 0.0  # uW

        for cell in cells:
            cell_name = cell['name']
            cell_type = cell['type']

            # Get switching energy
            if cell_type in cell_power_data:
                energy = cell_power_data[cell_type]['switching_energy']  # pJ
            else:
                energy = 0.5  # pJ (default)

            # Get toggle rate
            toggle_rate = activity.get(cell_name, 20.0)  # MHz

            # Power = Energy * Frequency
            # pJ * MHz = uW
            cell_power = energy * toggle_rate

            # Voltage scaling (quadratic)
            voltage_scale = voltage ** 2

            total_dynamic += cell_power * voltage_scale

        # Convert uW to mW
        return total_dynamic / 1000.0

    def estimate_power_at_frequency(
        self,
        netlist_file: str,
        lib_files: List[str],
        clock_frequency_mhz: float,
        voltage: float = 1.0
    ) -> Dict:
        """
        Estimate power consumption at a specific clock frequency.

        Args:
            netlist_file: Gate-level netlist
            lib_files: Liberty files
            clock_frequency_mhz: Target clock frequency
            voltage: Operating voltage

        Returns:
            Power estimates
        """
        logger.info(f"Estimating power at {clock_frequency_mhz} MHz")

        # Scale activity based on clock frequency
        cells = self._parse_netlist_cells(netlist_file)
        activity = {}

        for cell in cells:
            cell_name = cell['name']
            cell_type = cell['type'].upper()

            if 'CLK' in cell_name:
                activity[cell_name] = clock_frequency_mhz
            elif 'DFF' in cell_type:
                activity[cell_name] = clock_frequency_mhz * 0.1
            else:
                activity[cell_name] = clock_frequency_mhz * 0.02

        # Run power analysis with scaled activity
        # (Create a dummy SDC file)
        dummy_sdc = "/tmp/dummy.sdc"
        with open(dummy_sdc, 'w') as f:
            f.write(f"create_clock -period {1000/clock_frequency_mhz} clk\n")

        return self.analyze_power(
            netlist_file=netlist_file,
            sdc_file=dummy_sdc,
            lib_files=lib_files,
            activity_file=None,
            voltage=voltage
        )

    def find_high_power_cells(
        self,
        netlist_file: str,
        lib_files: List[str],
        top_n: int = 10
    ) -> List[Dict]:
        """
        Identify cells consuming the most power.

        Args:
            netlist_file: Gate-level netlist
            lib_files: Liberty files
            top_n: Number of top power consumers to return

        Returns:
            List of high-power cells with their power consumption
        """
        cells = self._parse_netlist_cells(netlist_file)
        cell_power_data = self._parse_lib_power(lib_files)
        activity = self._get_switching_activity(None, cells)

        # Calculate power for each cell
        cell_powers = []

        for cell in cells:
            cell_name = cell['name']
            cell_type = cell['type']

            if cell_type not in cell_power_data:
                continue

            # Static power
            static = cell_power_data[cell_type]['leakage_power'] / 1e6  # nW to mW

            # Dynamic power
            energy = cell_power_data[cell_type]['switching_energy']  # pJ
            toggle = activity.get(cell_name, 20.0)  # MHz
            dynamic = (energy * toggle) / 1000.0  # mW

            total = static + dynamic

            cell_powers.append({
                'name': cell_name,
                'type': cell_type,
                'power': total,
                'static_power': static,
                'dynamic_power': dynamic
            })

        # Sort by power and return top N
        cell_powers.sort(key=lambda x: x['power'], reverse=True)
        return cell_powers[:top_n]
