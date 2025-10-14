"""
Technology Library Parser

Reads and parses standard cell libraries (.lib, .lef files) to understand:
- Available gates (AND, OR, NAND, NOR, flip-flops, etc.)
- Timing characteristics (delay, setup/hold times)
- Power consumption (static and dynamic)
- Physical dimensions and constraints
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CellTiming:
    """Timing characteristics of a standard cell"""
    cell_name: str
    delay_rise: float  # Rising edge delay (ns)
    delay_fall: float  # Falling edge delay (ns)
    setup_time: float  # Setup time requirement (ns)
    hold_time: float   # Hold time requirement (ns)
    transition_rise: float  # Rise transition time (ns)
    transition_fall: float  # Fall transition time (ns)


@dataclass
class CellPower:
    """Power characteristics of a standard cell"""
    cell_name: str
    static_power: float  # Leakage power (nW)
    dynamic_power: float  # Switching power per transition (pJ)
    internal_power: float  # Internal power (pJ)


@dataclass
class CellPhysical:
    """Physical characteristics of a standard cell"""
    cell_name: str
    width: float  # Cell width (um)
    height: float  # Cell height (um)
    area: float  # Total area (um^2)
    pins: Dict[str, tuple]  # Pin name -> (x, y) coordinates


@dataclass
class StandardCell:
    """Complete representation of a standard cell"""
    name: str
    cell_type: str  # 'combinational', 'sequential', 'buffer', etc.
    function: str  # Boolean function (e.g., "A & B" for AND)
    timing: CellTiming
    power: CellPower
    physical: CellPhysical
    drive_strength: int  # Relative drive strength (1, 2, 4, 8, etc.)


class TechLibrary:
    """
    Manages technology library information for a specific process node.

    Parses .lib files (Liberty format) and .lef files (Library Exchange Format)
    to extract all necessary information about available standard cells.
    """

    def __init__(self, process_node: str):
        """
        Initialize technology library.

        Args:
            process_node: Process technology (e.g., '7nm', '12nm', '28nm')
        """
        self.process_node = process_node
        self.cells: Dict[str, StandardCell] = {}
        self.voltage: Optional[float] = None
        self.temperature: Optional[float] = None
        self.process_corner: str = "typical"  # typical, slow, fast
        self._power_map_cache: Dict[str, Dict[str, float]] = {}

        logger.info(f"Initialized TechLibrary for {process_node}")

    def load_liberty_file(self, lib_file_path: str):
        """
        Load and parse Liberty (.lib) file.

        Args:
            lib_file_path: Path to .lib file

        Liberty files contain timing and power information in a structured format.
        """
        logger.info(f"Loading Liberty file: {lib_file_path}")

        try:
            with open(lib_file_path, 'r') as f:
                content = f.read()

            # Parse library header for global parameters
            self._parse_library_header(content)

            # Parse individual cell definitions
            self._parse_cells(content)

            logger.info(f"Successfully loaded {len(self.cells)} cells from {lib_file_path}")
            # Build power map cache for reuse by other modules
            self._build_power_map_cache()

        except FileNotFoundError:
            logger.error(f"Liberty file not found: {lib_file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse Liberty file: {e}")
            raise

    def _parse_library_header(self, content: str):
        """Extract global library parameters"""
        # Parse voltage
        voltage_match = re.search(r'voltage\s*:\s*([\d.]+)', content)
        if voltage_match:
            self.voltage = float(voltage_match.group(1))

        # Parse temperature
        temp_match = re.search(r'temperature\s*:\s*([\d.]+)', content)
        if temp_match:
            self.temperature = float(temp_match.group(1))

        # Parse process corner
        corner_match = re.search(r'process\s*:\s*"?(\w+)"?', content)
        if corner_match:
            self.process_corner = corner_match.group(1)

    def _parse_cells(self, content: str):
        """
        Parse all cell definitions from Liberty file.

        TODO: Implement full Liberty parser (complex format)
        For now, using simplified parsing
        """
        # Find all cell blocks: cell (cell_name) { ... }
        cell_pattern = r'cell\s*\(\s*"?(\w+)"?\s*\)\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}'

        for match in re.finditer(cell_pattern, content, re.DOTALL):
            cell_name = match.group(1)
            cell_content = match.group(2)

            try:
                cell = self._parse_single_cell(cell_name, cell_content)
                self.cells[cell_name] = cell
            except Exception as e:
                logger.warning(f"Failed to parse cell {cell_name}: {e}")

    def _parse_single_cell(self, cell_name: str, content: str) -> StandardCell:
        """
        Parse a single cell definition.

        TODO: Implement comprehensive parsing for all Liberty attributes
        """
        # Extract cell function
        func_match = re.search(r'function\s*:\s*"([^"]+)"', content)
        function = func_match.group(1) if func_match else "unknown"

        # Determine cell type
        cell_type = self._infer_cell_type(cell_name, function)

        # Parse timing (simplified - actual Liberty has complex timing tables)
        timing = self._parse_timing(content)

        # Parse power
        power = self._parse_power(content)

        # Parse physical attributes
        physical = self._parse_physical(cell_name, content)

        # Infer drive strength from cell name (e.g., AND2_X2 has strength 2)
        drive_strength = self._infer_drive_strength(cell_name)

        return StandardCell(
            name=cell_name,
            cell_type=cell_type,
            function=function,
            timing=timing,
            power=power,
            physical=physical,
            drive_strength=drive_strength
        )

    def _build_power_map_cache(self):
        """Build a simple name->power characteristics map for reuse.

        This avoids reparsing Liberty in other modules.
        """
        power_map: Dict[str, Dict[str, float]] = {}
        for name, cell in self.cells.items():
            power_map[name] = {
                'leakage_power': float(cell.power.static_power),
                'switching_energy': float(cell.power.dynamic_power)
            }
        self._power_map_cache = power_map

    def get_power_map(self) -> Dict[str, Dict[str, float]]:
        """Return cached power characteristics per cell name."""
        if not self._power_map_cache and self.cells:
            self._build_power_map_cache()
        return self._power_map_cache

    def _infer_cell_type(self, name: str, function: str) -> str:
        """Infer cell type from name and function"""
        name_upper = name.upper()

        if 'DFF' in name_upper or 'DFFR' in name_upper or 'FLIP' in name_upper:
            return 'sequential'
        elif 'BUF' in name_upper or 'INV' in name_upper:
            return 'buffer'
        elif 'LATCH' in name_upper:
            return 'latch'
        else:
            return 'combinational'

    def _parse_timing(self, content: str) -> CellTiming:
        """Parse timing information (simplified)"""
        # TODO: Parse full timing tables
        # For now, extract scalar values

        def extract_float(pattern, default=0.1):
            match = re.search(pattern, content)
            return float(match.group(1)) if match else default

        return CellTiming(
            cell_name="",
            delay_rise=extract_float(r'cell_rise.*value.*?(\d+\.?\d*)', 0.1),
            delay_fall=extract_float(r'cell_fall.*value.*?(\d+\.?\d*)', 0.1),
            setup_time=extract_float(r'setup.*value.*?(\d+\.?\d*)', 0.05),
            hold_time=extract_float(r'hold.*value.*?(\d+\.?\d*)', 0.05),
            transition_rise=extract_float(r'rise_transition.*value.*?(\d+\.?\d*)', 0.05),
            transition_fall=extract_float(r'fall_transition.*value.*?(\d+\.?\d*)', 0.05)
        )

    def _parse_power(self, content: str) -> CellPower:
        """Parse power information"""
        def extract_float(pattern, default=1.0):
            match = re.search(pattern, content)
            return float(match.group(1)) if match else default

        return CellPower(
            cell_name="",
            static_power=extract_float(r'leakage_power.*?(\d+\.?\d*)', 1.0),
            dynamic_power=extract_float(r'energy.*?(\d+\.?\d*)', 0.5),
            internal_power=extract_float(r'internal_power.*?(\d+\.?\d*)', 0.3)
        )

    def _parse_physical(self, cell_name: str, content: str) -> CellPhysical:
        """Parse physical attributes"""
        def extract_float(pattern, default=1.0):
            match = re.search(pattern, content)
            return float(match.group(1)) if match else default

        area = extract_float(r'area\s*:\s*(\d+\.?\d*)', 1.0)

        return CellPhysical(
            cell_name=cell_name,
            width=area ** 0.5,  # Approximate
            height=area ** 0.5,
            area=area,
            pins={}  # TODO: Parse pin locations from LEF
        )

    def _infer_drive_strength(self, cell_name: str) -> int:
        """Extract drive strength from cell name (e.g., AND2_X4 -> 4)"""
        match = re.search(r'_X(\d+)', cell_name, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    def load_lef_file(self, lef_file_path: str):
        """
        Load and parse LEF (Library Exchange Format) file.

        Args:
            lef_file_path: Path to .lef file

        LEF files contain physical/geometric information about cells.
        """
        logger.info(f"Loading LEF file: {lef_file_path}")
        # TODO: Implement LEF parser
        # LEF contains detailed pin locations, layer information, etc.

    def get_cell(self, cell_name: str) -> Optional[StandardCell]:
        """Get standard cell by name"""
        return self.cells.get(cell_name)

    def get_cells_by_type(self, cell_type: str) -> List[StandardCell]:
        """Get all cells of a specific type"""
        return [cell for cell in self.cells.values() if cell.cell_type == cell_type]

    def get_cells_by_function(self, function_pattern: str) -> List[StandardCell]:
        """Get cells matching a function pattern (e.g., all AND gates)"""
        return [
            cell for cell in self.cells.values()
            if function_pattern.lower() in cell.function.lower()
        ]

    def find_equivalent_cells(self, cell_name: str) -> List[StandardCell]:
        """
        Find functionally equivalent cells with different drive strengths.

        Useful for optimization: swap cell with higher/lower drive strength.
        """
        if cell_name not in self.cells:
            return []

        base_cell = self.cells[cell_name]
        base_name = re.sub(r'_X\d+', '', cell_name)

        equivalent = [
            cell for cell in self.cells.values()
            if re.sub(r'_X\d+', '', cell.name) == base_name
            and cell.function == base_cell.function
        ]

        return sorted(equivalent, key=lambda c: c.drive_strength)

    def estimate_cell_delay(self, cell_name: str, load_capacitance: float) -> float:
        """
        Estimate cell delay given output load.

        Args:
            cell_name: Name of the cell
            load_capacitance: Output load capacitance (fF)

        Returns:
            Estimated delay (ns)

        TODO: Use actual timing tables for accurate calculation
        """
        cell = self.get_cell(cell_name)
        if not cell:
            return 0.0

        # Simplified linear delay model: delay = intrinsic + load * slope
        intrinsic_delay = (cell.timing.delay_rise + cell.timing.delay_fall) / 2
        load_sensitivity = 0.01  # ns per fF (approximate)

        return intrinsic_delay + load_capacitance * load_sensitivity

    def estimate_cell_power(self, cell_name: str, toggle_rate: float) -> float:
        """
        Estimate cell power consumption.

        Args:
            cell_name: Name of the cell
            toggle_rate: Switching frequency (MHz)

        Returns:
            Power consumption (uW)
        """
        cell = self.get_cell(cell_name)
        if not cell:
            return 0.0

        # Power = static + dynamic * frequency
        static_power_uw = cell.power.static_power / 1000  # nW to uW
        dynamic_power_uw = cell.power.dynamic_power * toggle_rate / 1000  # pJ * MHz -> uW

        return static_power_uw + dynamic_power_uw

    def get_library_summary(self) -> Dict:
        """Get summary statistics of the library"""
        return {
            'process_node': self.process_node,
            'voltage': self.voltage,
            'temperature': self.temperature,
            'process_corner': self.process_corner,
            'total_cells': len(self.cells),
            'combinational_cells': len(self.get_cells_by_type('combinational')),
            'sequential_cells': len(self.get_cells_by_type('sequential')),
            'buffer_cells': len(self.get_cells_by_type('buffer')),
        }
