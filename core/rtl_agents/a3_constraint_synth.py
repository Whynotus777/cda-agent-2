"""
A3 - Constraint Synthesizer

Generates SDC timing constraints from design intent and timing specifications.
Target: ≥70% STA clean on first run.
"""

import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

from .base_agent import BaseAgent, AgentOutput

logger = logging.getLogger(__name__)


class A3_ConstraintSynthesizer(BaseAgent):
    """
    Constraint Synthesizer - Generates SDC timing constraints.

    Capabilities:
    - Parse timing specs from design intent
    - Generate SDC clock definitions
    - Generate I/O delay constraints
    - Generate timing exceptions (false paths, multicycle)
    - Validate with OpenSTA
    - Calculate WNS/TNS
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize A3 agent.

        Args:
            config: Configuration dict
        """
        super().__init__(
            agent_id="A3",
            agent_name="Constraint Synthesizer",
            config=config
        )

        self.opensta_binary = config.get('opensta_binary', 'sta') if config else 'sta'

        logger.info("A3 Constraint Synthesizer initialized")

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Generate SDC constraints from design intent.

        Args:
            input_data: Dict conforming to design_intent schema with timing specs

        Returns:
            AgentOutput with generated constraints and validation results
        """
        start_time = time.time()

        if not self.validate_input(input_data):
            return self.create_output(
                success=False,
                output_data={},
                errors=["Invalid input data"]
            )

        design_name = input_data.get('module_name', 'design')
        constraints_spec = input_data.get('constraints', {})
        context = input_data.get('context', {})

        logger.info(f"A3 generating constraints for {design_name}")

        # Parse timing specifications
        clock_specs = self._parse_clock_specs(constraints_spec, context)
        io_delays = self._parse_io_delays(constraints_spec, context)
        timing_exceptions = self._parse_timing_exceptions(constraints_spec, context)

        # Generate SDC content
        sdc_content = self._generate_sdc(
            design_name=design_name,
            clock_specs=clock_specs,
            io_delays=io_delays,
            timing_exceptions=timing_exceptions
        )

        # Validate if netlist provided
        validation = {'sta_pass': None, 'wns': None, 'tns': None, 'errors': []}

        netlist_file = context.get('netlist_file')
        lib_files = context.get('lib_files', [])

        if netlist_file and lib_files:
            validation = self._validate_constraints(
                sdc_content=sdc_content,
                netlist_file=netlist_file,
                lib_files=lib_files,
                design_name=design_name
            )

        execution_time = (time.time() - start_time) * 1000

        output_data = {
            'constraint_id': str(uuid.uuid4()),
            'format': 'sdc',
            'constraints': sdc_content,
            'design_name': design_name,
            'clock_specs': clock_specs,
            'io_delays': io_delays,
            'timing_exceptions': timing_exceptions,
            'validation': validation,
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'generator_version': '1.0',
                'target_tool': 'OpenSTA'
            }
        }

        # Success if constraints generated and (no validation or STA passes)
        success = True
        errors = []
        warnings = []

        if validation['sta_pass'] is not None:
            if not validation['sta_pass']:
                errors.extend(validation.get('errors', []))
                success = False
            elif validation.get('wns', 0) < 0:
                warnings.append(f"Timing violations: WNS={validation['wns']:.3f}ns")

        return self.create_output(
            success=success,
            output_data=output_data,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            metadata={'clock_count': len(clock_specs)}
        )

    def _parse_clock_specs(
        self,
        constraints: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse clock specifications from constraints.

        Returns:
            List of clock spec dicts
        """
        clock_specs = []

        # Extract clock period/frequency
        clock_period_ns = constraints.get('clock_period_ns')
        target_freq_mhz = constraints.get('target_frequency_mhz')

        if clock_period_ns:
            freq_mhz = 1000.0 / clock_period_ns
        elif target_freq_mhz:
            clock_period_ns = 1000.0 / target_freq_mhz
            freq_mhz = target_freq_mhz
        else:
            # Default: 100 MHz
            clock_period_ns = 10.0
            freq_mhz = 100.0

        # Primary clock
        clock_specs.append({
            'clock_name': 'clk',
            'period_ns': clock_period_ns,
            'frequency_mhz': freq_mhz,
            'source_pin': 'clk',
            'duty_cycle': 0.5
        })

        # Additional clocks from context
        additional_clocks = context.get('clock_domains', [])
        for clk_info in additional_clocks:
            clock_specs.append({
                'clock_name': clk_info.get('name', 'clk_aux'),
                'period_ns': clk_info.get('period_ns', 10.0),
                'frequency_mhz': clk_info.get('frequency_mhz', 100.0),
                'source_pin': clk_info.get('pin', 'clk_aux'),
                'duty_cycle': clk_info.get('duty_cycle', 0.5)
            })

        return clock_specs

    def _parse_io_delays(
        self,
        constraints: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse I/O delay specifications.

        Returns:
            List of I/O delay dicts
        """
        io_delays = []

        # Default I/O delays (conservative)
        default_input_delay = constraints.get('default_input_delay_ns', 2.0)
        default_output_delay = constraints.get('default_output_delay_ns', 2.0)

        # Get port list from context
        ports = context.get('ports', [])

        for port in ports:
            if port.get('name') in ['clk', 'rst_n', 'rst', 'reset']:
                continue  # Skip clock and reset

            direction = port.get('direction', 'input')

            if direction == 'input':
                io_delays.append({
                    'port_name': port['name'],
                    'direction': 'input',
                    'delay_ns': default_input_delay,
                    'clock_ref': 'clk'
                })
            elif direction == 'output':
                io_delays.append({
                    'port_name': port['name'],
                    'direction': 'output',
                    'delay_ns': default_output_delay,
                    'clock_ref': 'clk'
                })

        return io_delays

    def _parse_timing_exceptions(
        self,
        constraints: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse timing exceptions (false paths, multicycle).

        Returns:
            List of timing exception dicts
        """
        exceptions = []

        # False paths from context
        false_paths = context.get('false_paths', [])
        for fp in false_paths:
            exceptions.append({
                'exception_type': 'false_path',
                'from_pin': fp.get('from', '*'),
                'to_pin': fp.get('to', '*')
            })

        # Multicycle paths
        multicycle_paths = context.get('multicycle_paths', [])
        for mp in multicycle_paths:
            exceptions.append({
                'exception_type': 'multicycle',
                'from_pin': mp.get('from', '*'),
                'to_pin': mp.get('to', '*'),
                'value': mp.get('cycles', 2)
            })

        return exceptions

    def _generate_sdc(
        self,
        design_name: str,
        clock_specs: List[Dict],
        io_delays: List[Dict],
        timing_exceptions: List[Dict]
    ) -> str:
        """
        Generate SDC file content.

        Returns:
            SDC file content as string
        """
        lines = []

        # Header
        lines.append("# SDC Timing Constraints")
        lines.append(f"# Generated by A3 Constraint Synthesizer")
        lines.append(f"# Design: {design_name}")
        lines.append(f"# Generated: {datetime.utcnow().isoformat()}")
        lines.append("")

        # Clock definitions
        lines.append("# Clock Definitions")
        for clk in clock_specs:
            lines.append(f"create_clock -name {clk['clock_name']} "
                        f"-period {clk['period_ns']:.3f} "
                        f"[get_ports {clk['source_pin']}]")

            # Add duty cycle if not 50%
            if abs(clk['duty_cycle'] - 0.5) > 0.01:
                lines.append(f"set_clock_duty_cycle {clk['clock_name']} {clk['duty_cycle']}")

        lines.append("")

        # Clock uncertainty (jitter + skew)
        lines.append("# Clock Uncertainty")
        for clk in clock_specs:
            uncertainty = clk['period_ns'] * 0.05  # 5% of period
            lines.append(f"set_clock_uncertainty {uncertainty:.3f} [get_clocks {clk['clock_name']}]")

        lines.append("")

        # I/O delays
        if io_delays:
            lines.append("# I/O Delays")

            # Group by direction
            input_delays = [d for d in io_delays if d['direction'] == 'input']
            output_delays = [d for d in io_delays if d['direction'] == 'output']

            if input_delays:
                lines.append("# Input Delays")
                for io in input_delays:
                    lines.append(f"set_input_delay {io['delay_ns']:.3f} "
                               f"-clock {io['clock_ref']} "
                               f"[get_ports {io['port_name']}]")

            lines.append("")

            if output_delays:
                lines.append("# Output Delays")
                for io in output_delays:
                    lines.append(f"set_output_delay {io['delay_ns']:.3f} "
                               f"-clock {io['clock_ref']} "
                               f"[get_ports {io['port_name']}]")

            lines.append("")

        # Timing exceptions
        if timing_exceptions:
            lines.append("# Timing Exceptions")

            for exc in timing_exceptions:
                if exc['exception_type'] == 'false_path':
                    lines.append(f"set_false_path "
                               f"-from [get_pins {exc['from_pin']}] "
                               f"-to [get_pins {exc['to_pin']}]")
                elif exc['exception_type'] == 'multicycle':
                    lines.append(f"set_multicycle_path {exc['value']} "
                               f"-from [get_pins {exc['from_pin']}] "
                               f"-to [get_pins {exc['to_pin']}]")
                elif exc['exception_type'] == 'max_delay':
                    lines.append(f"set_max_delay {exc['value']:.3f} "
                               f"-from [get_pins {exc['from_pin']}] "
                               f"-to [get_pins {exc['to_pin']}]")

            lines.append("")

        # Load constraints (capacitance)
        lines.append("# Load Constraints")
        lines.append("set_load 0.01 [all_outputs]")
        lines.append("")

        # Drive constraints
        lines.append("# Drive Constraints")
        lines.append("set_driving_cell -lib_cell BUF_X1 [all_inputs]")
        lines.append("")

        return "\n".join(lines)

    def _validate_constraints(
        self,
        sdc_content: str,
        netlist_file: str,
        lib_files: List[str],
        design_name: str
    ) -> Dict[str, Any]:
        """
        Validate constraints with OpenSTA.

        Returns:
            Validation result dict
        """
        result = {
            'sta_pass': False,
            'wns': None,
            'tns': None,
            'errors': []
        }

        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdc', delete=False) as f:
            f.write(sdc_content)
            sdc_path = f.name

        # Create OpenSTA script
        sta_script = f"""# OpenSTA validation script
"""
        for lib in lib_files:
            sta_script += f"read_liberty {lib}\n"

        sta_script += f"""
read_verilog {netlist_file}
link_design {design_name}
read_sdc {sdc_path}

# Report timing
report_checks -path_delay max -format full_clock_expanded
report_worst_slack -max
report_tns -digits 3
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tcl', delete=False) as f:
            f.write(sta_script)
            script_path = f.name

        try:
            # Run OpenSTA
            cmd = [self.opensta_binary, script_path]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            output = proc.stdout + proc.stderr

            # Parse output for timing results
            wns = self._extract_wns(output)
            tns = self._extract_tns(output)

            result['wns'] = wns
            result['tns'] = tns

            # Success if no negative slack
            result['sta_pass'] = (wns is not None and wns >= 0)

            # Extract errors
            for line in output.split('\n'):
                if 'ERROR' in line.upper():
                    result['errors'].append(line.strip())

        except subprocess.TimeoutExpired:
            result['errors'].append("OpenSTA validation timed out")
        except FileNotFoundError:
            result['errors'].append("OpenSTA not found")
            result['sta_pass'] = None  # Can't validate
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        finally:
            # Clean up
            Path(sdc_path).unlink(missing_ok=True)
            Path(script_path).unlink(missing_ok=True)

        return result

    def _extract_wns(self, output: str) -> Optional[float]:
        """Extract WNS from OpenSTA output"""
        import re
        match = re.search(r'worst slack\s+([-\d.]+)', output, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _extract_tns(self, output: str) -> Optional[float]:
        """Extract TNS from OpenSTA output"""
        import re
        match = re.search(r'total negative slack\s+([-\d.]+)', output, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input against design_intent schema"""
        if 'module_name' not in input_data and 'design_name' not in input_data:
            logger.error("Missing module_name or design_name")
            return False
        return True

    def get_schema(self) -> Dict[str, Any]:
        """Return input schema for A3"""
        schema_path = Path(__file__).parent.parent / 'schemas' / 'design_intent.json'
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {}
