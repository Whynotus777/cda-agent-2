"""
Design Parser Module

Parses chip design files in various formats:
- Verilog/SystemVerilog (RTL and gate-level netlists)
- LEF/DEF (physical layout)
- SDC (Synopsys Design Constraints - timing constraints)
- SPEF (parasitic extraction)
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class Port:
    """Design port (input/output)"""
    name: str
    direction: str  # 'input', 'output', 'inout'
    width: int = 1  # Bit width


@dataclass
class Module:
    """Verilog module representation"""
    name: str
    ports: List[Port] = field(default_factory=list)
    instances: List['Instance'] = field(default_factory=list)
    wires: List[str] = field(default_factory=list)
    parameters: Dict[str, any] = field(default_factory=dict)


@dataclass
class Instance:
    """Module instance (cell instantiation)"""
    name: str
    module_type: str  # Cell type or module name
    connections: Dict[str, str] = field(default_factory=dict)  # Port -> net mapping


@dataclass
class Constraint:
    """Timing or design constraint"""
    constraint_type: str  # 'clock', 'input_delay', 'output_delay', 'max_delay', etc.
    target: str  # Clock name, port name, etc.
    value: float  # Constraint value
    parameters: Dict[str, any] = field(default_factory=dict)


@dataclass
class DesignNetlist:
    """Complete netlist representation"""
    top_module: str
    modules: Dict[str, Module] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    clock_period: Optional[float] = None
    clock_uncertainty: float = 0.0


class DesignParser:
    """
    Parses various chip design file formats.

    Supports:
    - Verilog (.v)
    - SystemVerilog (.sv)
    - Gate-level netlists
    - SDC constraints (.sdc)
    - DEF files (.def) for physical design
    """

    def __init__(self):
        """Initialize design parser"""
        self.netlist: Optional[DesignNetlist] = None
        logger.info("Initialized DesignParser")

    def parse_verilog(self, verilog_file: str) -> DesignNetlist:
        """
        Parse Verilog/SystemVerilog file.

        Args:
            verilog_file: Path to .v or .sv file

        Returns:
            Parsed netlist structure
        """
        logger.info(f"Parsing Verilog file: {verilog_file}")

        try:
            with open(verilog_file, 'r') as f:
                content = f.read()

            # Remove comments
            content = self._remove_comments(content)

            # Parse modules
            modules = self._parse_modules(content)

            # Determine top module (last module or specified)
            top_module = list(modules.keys())[-1] if modules else None

            self.netlist = DesignNetlist(
                top_module=top_module,
                modules=modules
            )

            logger.info(f"Successfully parsed {len(modules)} modules")
            return self.netlist

        except FileNotFoundError:
            logger.error(f"Verilog file not found: {verilog_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse Verilog: {e}")
            raise

    def _remove_comments(self, content: str) -> str:
        """Remove Verilog comments (// and /* */)"""
        # Remove single-line comments
        content = re.sub(r'//.*?\n', '\n', content)
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content

    def _parse_modules(self, content: str) -> Dict[str, Module]:
        """Parse all module definitions"""
        modules = {}

        # Find all module definitions
        module_pattern = r'module\s+(\w+)\s*(?:#\s*\([^)]*\))?\s*\(([^;]*?)\);(.*?)endmodule'

        for match in re.finditer(module_pattern, content, re.DOTALL):
            module_name = match.group(1)
            port_list = match.group(2)
            module_body = match.group(3)

            module = Module(name=module_name)

            # Parse ports
            module.ports = self._parse_ports(port_list, module_body)

            # Parse instances
            module.instances = self._parse_instances(module_body)

            # Parse wires
            module.wires = self._parse_wires(module_body)

            modules[module_name] = module
            logger.debug(f"Parsed module: {module_name} with {len(module.ports)} ports")

        return modules

    def _parse_ports(self, port_list: str, module_body: str) -> List[Port]:
        """Parse module ports"""
        ports = []

        # Parse ANSI-style port declarations (in port list)
        ansi_pattern = r'(input|output|inout)\s+(?:wire|reg)?\s*(?:\[.*?\])?\s*(\w+)'

        for match in re.finditer(ansi_pattern, port_list):
            direction = match.group(1)
            name = match.group(2)
            ports.append(Port(name=name, direction=direction))

        # Parse non-ANSI style (in module body)
        if not ports:
            # Extract port names from port list
            port_names = [name.strip() for name in port_list.split(',') if name.strip()]

            # Find their directions in module body
            for name in port_names:
                direction_match = re.search(
                    rf'(input|output|inout)\s+(?:wire|reg)?\s*(?:\[.*?\])?\s*{name}\b',
                    module_body
                )
                if direction_match:
                    direction = direction_match.group(1)
                    ports.append(Port(name=name, direction=direction))

        return ports

    def _parse_instances(self, module_body: str) -> List[Instance]:
        """Parse cell/module instances"""
        instances = []

        # Pattern: module_type instance_name ( .port(net), ... );
        instance_pattern = r'(\w+)\s+(\w+)\s*\(([^;]+)\);'

        for match in re.finditer(instance_pattern, module_body):
            module_type = match.group(1)
            instance_name = match.group(2)
            connections_str = match.group(3)

            # Skip wire declarations and other non-instance statements
            if module_type in ['wire', 'reg', 'input', 'output', 'assign']:
                continue

            # Parse connections
            connections = self._parse_connections(connections_str)

            instances.append(Instance(
                name=instance_name,
                module_type=module_type,
                connections=connections
            ))

        return instances

    def _parse_connections(self, connections_str: str) -> Dict[str, str]:
        """Parse port connections (.port(net))"""
        connections = {}

        # Pattern: .port_name(net_name)
        conn_pattern = r'\.(\w+)\s*\(\s*(\w+)\s*\)'

        for match in re.finditer(conn_pattern, connections_str):
            port_name = match.group(1)
            net_name = match.group(2)
            connections[port_name] = net_name

        return connections

    def _parse_wires(self, module_body: str) -> List[str]:
        """Parse wire declarations"""
        wires = []

        # Pattern: wire [width] name1, name2, ...;
        wire_pattern = r'wire\s+(?:\[.*?\])?\s*([\w,\s]+);'

        for match in re.finditer(wire_pattern, module_body):
            wire_names = match.group(1)
            names = [n.strip() for n in wire_names.split(',')]
            wires.extend(names)

        return wires

    def parse_sdc(self, sdc_file: str):
        """
        Parse SDC (Synopsys Design Constraints) file.

        Args:
            sdc_file: Path to .sdc file

        SDC files contain timing constraints like clock definitions,
        input/output delays, false paths, etc.
        """
        logger.info(f"Parsing SDC file: {sdc_file}")

        try:
            with open(sdc_file, 'r') as f:
                lines = f.readlines()

            if not self.netlist:
                self.netlist = DesignNetlist(top_module="unknown")

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse create_clock
                if line.startswith('create_clock'):
                    self._parse_create_clock(line)

                # Parse set_input_delay
                elif line.startswith('set_input_delay'):
                    self._parse_input_delay(line)

                # Parse set_output_delay
                elif line.startswith('set_output_delay'):
                    self._parse_output_delay(line)

                # Parse set_max_delay
                elif line.startswith('set_max_delay'):
                    self._parse_max_delay(line)

            logger.info(f"Parsed {len(self.netlist.constraints)} constraints")

        except FileNotFoundError:
            logger.error(f"SDC file not found: {sdc_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse SDC: {e}")
            raise

    def _parse_create_clock(self, line: str):
        """Parse create_clock constraint"""
        # create_clock -period 10 -name clk [get_ports clk]
        period_match = re.search(r'-period\s+([\d.]+)', line)
        name_match = re.search(r'-name\s+(\w+)', line)

        if period_match:
            period = float(period_match.group(1))
            clock_name = name_match.group(1) if name_match else 'clk'

            self.netlist.clock_period = period

            self.netlist.constraints.append(Constraint(
                constraint_type='clock',
                target=clock_name,
                value=period
            ))

    def _parse_input_delay(self, line: str):
        """Parse set_input_delay constraint"""
        # set_input_delay -clock clk -max 2.0 [get_ports data_in]
        delay_match = re.search(r'-max\s+([\d.]+)', line)
        clock_match = re.search(r'-clock\s+(\w+)', line)
        port_match = re.search(r'\[get_ports\s+(\w+)\]', line)

        if delay_match and port_match:
            delay = float(delay_match.group(1))
            port = port_match.group(1)
            clock = clock_match.group(1) if clock_match else 'default'

            self.netlist.constraints.append(Constraint(
                constraint_type='input_delay',
                target=port,
                value=delay,
                parameters={'clock': clock}
            ))

    def _parse_output_delay(self, line: str):
        """Parse set_output_delay constraint"""
        delay_match = re.search(r'-max\s+([\d.]+)', line)
        port_match = re.search(r'\[get_ports\s+(\w+)\]', line)

        if delay_match and port_match:
            delay = float(delay_match.group(1))
            port = port_match.group(1)

            self.netlist.constraints.append(Constraint(
                constraint_type='output_delay',
                target=port,
                value=delay
            ))

    def _parse_max_delay(self, line: str):
        """Parse set_max_delay constraint"""
        delay_match = re.search(r'set_max_delay\s+([\d.]+)', line)

        if delay_match:
            delay = float(delay_match.group(1))

            self.netlist.constraints.append(Constraint(
                constraint_type='max_delay',
                target='all',
                value=delay
            ))

    def parse_def(self, def_file: str):
        """
        Parse DEF (Design Exchange Format) file.

        Args:
            def_file: Path to .def file

        DEF files contain physical design information like component placement,
        routing, pins, etc.

        TODO: Implement DEF parser
        """
        logger.info(f"Parsing DEF file: {def_file}")
        # DEF parsing is complex - placeholder for now

    def get_module(self, module_name: str) -> Optional[Module]:
        """Get module by name"""
        if not self.netlist:
            return None
        return self.netlist.modules.get(module_name)

    def get_top_module(self) -> Optional[Module]:
        """Get top-level module"""
        if not self.netlist:
            return None
        return self.netlist.modules.get(self.netlist.top_module)

    def get_instance_count(self) -> int:
        """Get total number of instances across all modules"""
        if not self.netlist:
            return 0

        total = 0
        for module in self.netlist.modules.values():
            total += len(module.instances)
        return total

    def get_cell_types(self) -> Set[str]:
        """Get set of all cell types used in the design"""
        if not self.netlist:
            return set()

        cell_types = set()
        for module in self.netlist.modules.values():
            for instance in module.instances:
                cell_types.add(instance.module_type)

        return cell_types

    def get_design_hierarchy(self) -> Dict:
        """Get hierarchical structure of the design"""
        if not self.netlist or not self.netlist.top_module:
            return {}

        def build_hierarchy(module_name: str, visited: Set[str]) -> Dict:
            if module_name in visited:
                return {'name': module_name, 'children': [], 'circular': True}

            visited.add(module_name)
            module = self.get_module(module_name)

            if not module:
                return {'name': module_name, 'children': []}

            children = []
            for instance in module.instances:
                if instance.module_type in self.netlist.modules:
                    child_hierarchy = build_hierarchy(instance.module_type, visited.copy())
                    children.append(child_hierarchy)

            return {
                'name': module_name,
                'instance_count': len(module.instances),
                'children': children
            }

        return build_hierarchy(self.netlist.top_module, set())

    def get_design_summary(self) -> Dict:
        """Get summary of the parsed design"""
        if not self.netlist:
            return {}

        return {
            'top_module': self.netlist.top_module,
            'total_modules': len(self.netlist.modules),
            'total_instances': self.get_instance_count(),
            'unique_cell_types': len(self.get_cell_types()),
            'clock_period': self.netlist.clock_period,
            'total_constraints': len(self.netlist.constraints)
        }
