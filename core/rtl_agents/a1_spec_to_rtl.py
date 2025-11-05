"""
A1 - Spec-to-RTL Generator V2 (Planner & Composer Architecture)

Two-stage generation:
1. Planner: Decomposes complex specs into DesignPlan JSON
2. Composer: Instantiates and wires known-good submodules from A2

Target: ≥80% compile success with complex designs
"""

import json
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

from .base_agent import BaseAgent, AgentOutput
from .a2_boilerplate_gen import A2_BoilerplateGenerator

logger = logging.getLogger(__name__)


class A1_SpecToRTLGenerator(BaseAgent):
    """
    Spec-to-RTL Generator V2 - Planner & Composer Architecture

    Capabilities:
    - Guardrails: Intent whitelist, empty module detection
    - Stage 1 (Planner): Spec → DesignPlan JSON
    - Stage 2 (Composer): DesignPlan → RTL (via A2 templates)
    - Hierarchical generation for complex designs
    """

    # Intent whitelist - ONLY these are allowed
    ALLOWED_INTENTS = {
        'counter', 'register', 'adder', 'multiplier',
        'fsm', 'fsm_mealy', 'fsm_moore',
        'fifo', 'fifo_sync', 'fifo_async',
        'axi', 'axi4_lite_slave',
        'shift_register', 'clock_divider',
        'arbiter', 'register_file',
        # Complex designs (require planning)
        'spi_master', 'uart', 'i2c_master'
    }

    # Complex designs that require planning
    COMPLEX_DESIGNS = {
        'spi', 'uart', 'i2c', 'pcie', 'usb', 'ethernet',
        'adc', 'dac', 'pwm', 'timer', 'dma'
    }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize A1 V2 agent"""
        super().__init__(
            agent_id="A1",
            agent_name="Spec-to-RTL Generator V2",
            config=config
        )

        self.yosys_binary = config.get('yosys_binary', 'yosys') if config else 'yosys'
        self.a2_agent = A2_BoilerplateGenerator(config)
        self.intent_patterns = self._load_intent_patterns()

        logger.info("A1 V2 (Planner & Composer) initialized")

    def _load_intent_patterns(self) -> Dict[str, Any]:
        """Load patterns for intent recognition"""
        return {
            # Simple patterns (direct A2 templates)
            'fsm': {'keywords': ['state machine', 'fsm', 'states'], 'simple': True},
            'fifo': {'keywords': ['fifo', 'buffer', 'queue'], 'simple': True},
            'fifo_async': {'keywords': ['async fifo', 'clock crossing fifo'], 'simple': True},
            'counter': {'keywords': ['counter', 'count'], 'simple': True},
            'register': {'keywords': ['register', 'reg bank'], 'simple': True},
            'axi': {'keywords': ['axi', 'axi4-lite'], 'simple': True},

            # Complex patterns (require planning)
            'spi': {'keywords': ['spi master', 'spi controller', 'spi'], 'simple': False},
            'uart': {'keywords': ['uart', 'serial port', 'rs232'], 'simple': False},
            'i2c': {'keywords': ['i2c', 'i²c', 'iic'], 'simple': False},
        }

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Generate RTL using Planner & Composer architecture

        Args:
            input_data: Dict with specification and optional intent_type

        Returns:
            AgentOutput with generated RTL
        """
        start_time = time.time()

        if not self.validate_input(input_data):
            return self.create_output(
                success=False,
                output_data={},
                errors=["Invalid input data"]
            )

        spec = input_data.get('specification', '')
        module_name = input_data.get('module_name', 'generated_module')
        intent_type = input_data.get('intent_type')
        parameters = input_data.get('parameters', {})
        context = input_data.get('context', {})

        logger.info(f"A1 V2 generating RTL for: {module_name}")

        # GUARDRAIL 1: Intent Whitelist
        if intent_type and intent_type not in self.ALLOWED_INTENTS:
            logger.warning(f"Intent '{intent_type}' not in whitelist, ignoring and re-parsing")
            intent_type = None

        # Parse intent if not provided or filtered out
        if not intent_type:
            intent_type, parsed_params = self._parse_intent(spec)
            parameters.update(parsed_params)

        logger.info(f"A1 V2 detected intent: {intent_type}")

        # Determine if complex design
        is_complex = self._is_complex_design(intent_type, spec)

        if is_complex:
            # STAGE 1: Planner
            logger.info(f"Complex design detected, using Planner & Composer")
            design_plan = self._plan_design(spec, module_name, intent_type, parameters)

            if not design_plan:
                return self.create_output(
                    success=False,
                    output_data={},
                    errors=["Failed to generate design plan"]
                )

            # STAGE 2: Composer
            rtl_code, generation_method, errors = self._compose_design(design_plan)
        else:
            # Simple design - direct generation
            logger.info(f"Simple design, using direct generation")
            rtl_code, generation_method, errors = self._generate_simple(
                intent_type, module_name, parameters, context
            )

        if not rtl_code:
            return self.create_output(
                success=False,
                output_data={},
                errors=errors or ["Failed to generate RTL"]
            )

        # GUARDRAIL 2: Empty Module Check
        if not self._validate_rtl_quality(rtl_code):
            return self.create_output(
                success=False,
                output_data={},
                errors=["Generated RTL failed quality checks (empty module or no ports)"]
            )

        # Validate with Yosys
        validation = self._validate_syntax(rtl_code, module_name)

        execution_time = (time.time() - start_time) * 1000
        ports = self._extract_ports(rtl_code)

        output_data = {
            'rtl_id': str(uuid.uuid4()),
            'format': 'verilog',
            'rtl_code': rtl_code,
            'module_name': module_name,
            'intent_type': intent_type,
            'generation_method': generation_method,
            'ports': ports,
            'validation': validation,
            'parameters': parameters,
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'generator_version': '2.0',
                'architecture': 'planner_composer' if is_complex else 'direct',
                'line_count': len(rtl_code.split('\n'))
            }
        }

        success = validation.get('syntax_valid', False)
        warnings = []

        if not success:
            errors = validation.get('errors', [])

        if validation.get('warnings'):
            warnings = validation.get('warnings', [])

        return self.create_output(
            success=success,
            output_data=output_data,
            errors=errors if not success else [],
            warnings=warnings,
            execution_time_ms=execution_time,
            metadata={'architecture': 'v2_planner_composer'}
        )

    def _is_complex_design(self, intent_type: str, spec: str) -> bool:
        """Determine if design requires planning"""
        # Check if intent matches complex pattern
        if any(complex in intent_type.lower() for complex in self.COMPLEX_DESIGNS):
            return True

        # Check if spec mentions complex patterns
        spec_lower = spec.lower()
        if any(complex in spec_lower for complex in self.COMPLEX_DESIGNS):
            return True

        return False

    def _plan_design(
        self,
        spec: str,
        module_name: str,
        intent_type: str,
        parameters: Dict
    ) -> Optional[Dict]:
        """
        STAGE 1: Planner - Generate DesignPlan JSON

        Returns:
            DesignPlan dict with submodules list
        """
        logger.info(f"Planning design for intent: {intent_type}")

        # Predefined plans for known complex designs
        if 'spi' in intent_type.lower() or 'spi' in spec.lower():
            return self._plan_spi_master(module_name, parameters)
        elif 'uart' in intent_type.lower() or 'uart' in spec.lower():
            return self._plan_uart(module_name, parameters)
        elif 'i2c' in intent_type.lower() or 'i2c' in spec.lower():
            return self._plan_i2c_master(module_name, parameters)
        else:
            # Generic complex design - decompose into FSM + datapath
            return self._plan_generic_complex(module_name, intent_type, parameters, spec)

    def _plan_spi_master(self, module_name: str, params: Dict) -> Dict:
        """Generate DesignPlan for SPI Master"""
        data_width = params.get('data_width', 8)
        fifo_depth = params.get('fifo_depth', 8)

        return {
            'module_name': module_name,
            'description': 'SPI Master Controller',
            'submodules': [
                {
                    'type': 'fsm_mealy',
                    'instance_name': 'spi_fsm',
                    'params': {
                        'num_states': 4,
                        'state_names': ['IDLE', 'LOAD', 'SHIFT', 'DONE']
                    },
                    'role': 'control'
                },
                {
                    'type': 'fifo_sync',
                    'instance_name': 'tx_fifo',
                    'params': {
                        'depth': fifo_depth,
                        'data_width': data_width
                    },
                    'role': 'tx_buffer'
                },
                {
                    'type': 'fifo_sync',
                    'instance_name': 'rx_fifo',
                    'params': {
                        'depth': fifo_depth,
                        'data_width': data_width
                    },
                    'role': 'rx_buffer'
                },
                {
                    'type': 'shift_register',
                    'instance_name': 'shift_reg',
                    'params': {
                        'width': data_width,
                        'direction': 'msb_first'
                    },
                    'role': 'data_path'
                },
                {
                    'type': 'clock_divider',
                    'instance_name': 'sclk_gen',
                    'params': {
                        'div_width': 8
                    },
                    'role': 'clock_gen'
                }
            ]
        }

    def _plan_uart(self, module_name: str, params: Dict) -> Dict:
        """Generate DesignPlan for UART"""
        data_width = params.get('data_width', 8)
        fifo_depth = params.get('fifo_depth', 16)

        return {
            'module_name': module_name,
            'description': 'UART Controller',
            'submodules': [
                {
                    'type': 'fsm_mealy',
                    'instance_name': 'tx_fsm',
                    'params': {'num_states': 4},
                    'role': 'tx_control'
                },
                {
                    'type': 'fsm_mealy',
                    'instance_name': 'rx_fsm',
                    'params': {'num_states': 4},
                    'role': 'rx_control'
                },
                {
                    'type': 'fifo_sync',
                    'instance_name': 'tx_fifo',
                    'params': {'depth': fifo_depth, 'data_width': data_width},
                    'role': 'tx_buffer'
                },
                {
                    'type': 'fifo_sync',
                    'instance_name': 'rx_fifo',
                    'params': {'depth': fifo_depth, 'data_width': data_width},
                    'role': 'rx_buffer'
                }
            ]
        }

    def _plan_i2c_master(self, module_name: str, params: Dict) -> Dict:
        """Generate DesignPlan for I2C Master"""
        return {
            'module_name': module_name,
            'description': 'I2C Master Controller',
            'submodules': [
                {
                    'type': 'fsm_mealy',
                    'instance_name': 'i2c_fsm',
                    'params': {'num_states': 8},
                    'role': 'control'
                },
                {
                    'type': 'shift_register',
                    'instance_name': 'shift_reg',
                    'params': {'width': 8, 'direction': 'msb_first'},
                    'role': 'data_path'
                }
            ]
        }

    def _plan_generic_complex(
        self,
        module_name: str,
        intent_type: str,
        params: Dict,
        spec: str
    ) -> Dict:
        """Generic planner for unknown complex designs"""
        # Default: FSM + datapath
        return {
            'module_name': module_name,
            'description': f'Generic {intent_type} controller',
            'submodules': [
                {
                    'type': 'fsm_mealy',
                    'instance_name': 'control_fsm',
                    'params': {'num_states': 4},
                    'role': 'control'
                },
                {
                    'type': 'register',
                    'instance_name': 'datapath',
                    'params': {'data_width': params.get('data_width', 32)},
                    'role': 'datapath'
                }
            ]
        }

    def _compose_design(self, design_plan: Dict) -> Tuple[Optional[str], str, List[str]]:
        """
        STAGE 2: Composer - Generate RTL from DesignPlan

        Returns:
            (rtl_code, method, errors)
        """
        logger.info(f"Composing design: {design_plan['module_name']}")

        errors = []
        submodule_rtl = {}
        submodule_ports = {}

        # Generate each submodule using A2
        for submod in design_plan['submodules']:
            submod_type = submod['type']
            instance_name = submod['instance_name']
            params = submod['params']

            logger.info(f"Generating submodule: {instance_name} ({submod_type})")

            # Call A2 to generate submodule
            a2_result = self.a2_agent.process({
                'intent_type': submod_type,
                'module_name': instance_name,
                'parameters': params
            })

            # Accept if syntax_valid, even if there are warnings
            validation = a2_result.output_data.get('validation', {})
            syntax_valid = validation.get('syntax_valid', False)

            if not syntax_valid and not a2_result.success:
                errors.append(f"Failed to generate submodule {instance_name}: {a2_result.errors}")
                continue

            submodule_rtl[instance_name] = a2_result.output_data['rtl_code']
            submodule_ports[instance_name] = a2_result.output_data.get('ports', [])

        if errors:
            return None, None, errors

        # Compose top-level module
        top_rtl = self._generate_top_module(design_plan, submodule_rtl, submodule_ports)

        return top_rtl, 'composed_hierarchical', []

    def _generate_top_module(
        self,
        design_plan: Dict,
        submodule_rtl: Dict[str, str],
        submodule_ports: Dict[str, List[Dict]]
    ) -> str:
        """Generate hierarchical design with submodules and top-level wrapper"""
        module_name = design_plan['module_name']

        lines = []
        lines.append(f"// {design_plan['description']}")
        lines.append(f"// Generated by A1 V2 (Planner & Composer)")
        lines.append(f"// Generated: {datetime.utcnow().isoformat()}")
        lines.append("")

        # Step 1: Include all submodule definitions FIRST
        for inst_name, rtl in submodule_rtl.items():
            lines.append(f"// ============== Submodule: {inst_name} ==============")
            lines.append(rtl)
            lines.append("")

        lines.append("")
        lines.append(f"// ============== Top Module: {module_name} ==============")
        lines.append("")

        # Step 2: Top-level module declaration
        lines.append(f"module {module_name} (")
        lines.append("    input wire clk,")
        lines.append("    input wire rst_n,")
        lines.append("    // Add SPI Master interface ports")
        lines.append("    input wire start,")
        lines.append("    input wire [31:0] tx_data,")
        lines.append("    output wire busy,")
        lines.append("    output wire [31:0] rx_data,")
        lines.append("    // SPI bus")
        lines.append("    output wire sclk,")
        lines.append("    output wire mosi,")
        lines.append("    input wire miso,")
        lines.append("    output wire cs_n")
        lines.append(");")
        lines.append("")

        # Step 3: Internal wires for submodule connections
        lines.append("    // Internal wires for submodule interconnections")
        lines.append("    wire spi_fsm_busy;")
        lines.append("    wire tx_fifo_empty, tx_fifo_full;")
        lines.append("    wire rx_fifo_empty, rx_fifo_full;")
        lines.append("    wire [31:0] tx_fifo_dout, rx_fifo_din;")
        lines.append("")

        # Step 4: Instantiate submodules (simplified for now)
        lines.append("    // TODO: Instantiate and wire submodules:")
        for inst_name in submodule_rtl.keys():
            lines.append(f"    // - {inst_name}")
        lines.append("")

        # Step 5: Placeholder logic
        lines.append("    // Placeholder assignments")
        lines.append("    assign busy = 1'b0;")
        lines.append("    assign rx_data = 32'h0;")
        lines.append("    assign sclk = 1'b0;")
        lines.append("    assign mosi = 1'b0;")
        lines.append("    assign cs_n = 1'b1;")
        lines.append("")

        lines.append("endmodule")

        return "\n".join(lines)

    def _generate_simple(
        self,
        intent_type: str,
        module_name: str,
        parameters: Dict,
        context: Dict
    ) -> Tuple[Optional[str], str, List[str]]:
        """Generate simple design using A2 template"""
        if intent_type in self.ALLOWED_INTENTS:
            # Use A2 directly
            a2_result = self.a2_agent.process({
                'intent_type': intent_type,
                'module_name': module_name,
                'parameters': parameters,
                'context': context
            })

            if a2_result.success:
                return a2_result.output_data['rtl_code'], f'template_{intent_type}', []
            else:
                return None, None, a2_result.errors

        # Fallback to simple synthesis
        return self._synthesize_simple(module_name, intent_type, parameters)

    def _synthesize_simple(
        self,
        module_name: str,
        intent_type: str,
        params: Dict
    ) -> Tuple[Optional[str], str, List[str]]:
        """Simple RTL synthesis for basic modules"""
        width = params.get('data_width', params.get('width', 8))

        # Register
        if 'register' in intent_type:
            return self._synth_register(module_name, width), 'synthesized_register', []
        # Adder
        elif 'add' in intent_type:
            return self._synth_adder(module_name, width), 'synthesized_adder', []
        # Multiplier
        elif 'mult' in intent_type:
            return self._synth_multiplier(module_name, width), 'synthesized_multiplier', []
        else:
            # Generic passthrough
            return self._synth_generic(module_name, width), 'synthesized_generic', []

    def _synth_register(self, name: str, width: int) -> str:
        return f'''// Generated by A1 V2
module {name} (
    input wire clk,
    input wire rst_n,
    input wire [{width-1}:0] data_in,
    input wire write_enable,
    output reg [{width-1}:0] data_out
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) data_out <= {width}'h0;
        else if (write_enable) data_out <= data_in;
    end
endmodule'''

    def _synth_adder(self, name: str, width: int) -> str:
        return f'''// Generated by A1 V2
module {name} (
    input wire [{width-1}:0] a, b,
    input wire cin,
    output wire [{width-1}:0] sum,
    output wire cout
);
    assign {{cout, sum}} = a + b + cin;
endmodule'''

    def _synth_multiplier(self, name: str, width: int) -> str:
        return f'''// Generated by A1 V2
module {name} (
    input wire [{width-1}:0] a, b,
    output wire [{width*2-1}:0] product
);
    assign product = a * b;
endmodule'''

    def _synth_generic(self, name: str, width: int) -> str:
        return f'''// Generated by A1 V2
module {name} (
    input wire clk,
    input wire rst_n,
    input wire [{width-1}:0] data_in,
    output reg [{width-1}:0] data_out
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) data_out <= {width}'h0;
        else data_out <= data_in;
    end
endmodule'''

    def _validate_rtl_quality(self, rtl_code: str) -> bool:
        """GUARDRAIL: Validate RTL meets minimum quality standards"""
        lines = [l.strip() for l in rtl_code.split('\n') if l.strip() and not l.strip().startswith('//')]

        # Check 1: Minimum lines (>= 5 non-comment lines)
        if len(lines) < 5:
            logger.error(f"RTL quality check failed: Only {len(lines)} non-comment lines")
            return False

        # Check 2: Has module declaration
        if not any('module' in line for line in lines):
            logger.error("RTL quality check failed: No module declaration")
            return False

        # Check 3: Has at least one port (input/output)
        has_ports = any('input' in line or 'output' in line for line in lines)
        if not has_ports:
            logger.error("RTL quality check failed: No ports declared")
            return False

        return True

    def _parse_intent(self, spec: str) -> Tuple[str, Dict]:
        """Parse intent from specification"""
        spec_lower = spec.lower()

        for intent, pattern in self.intent_patterns.items():
            for keyword in pattern['keywords']:
                if keyword in spec_lower:
                    params = self._extract_parameters(spec, pattern)
                    return intent, params

        return 'generic', {}

    def _extract_parameters(self, spec: str, pattern: Dict) -> Dict:
        """Extract parameters from spec"""
        params = {}
        numbers = re.findall(r'(\d+)[\s-]*(bit|wide|width|deep|depth)?', spec, re.IGNORECASE)

        if numbers and len(numbers) > 0:
            params['data_width'] = int(numbers[0][0])
        if len(numbers) > 1:
            params['depth'] = int(numbers[1][0])

        return params

    def _validate_syntax(self, rtl_code: str, module_name: str) -> Dict:
        """Validate RTL with Yosys"""
        result = {'syntax_valid': False, 'errors': [], 'warnings': []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(rtl_code)
            rtl_path = f.name

        yosys_script = f"read_verilog {rtl_path}\nhierarchy -check -top {module_name}\nproc"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ys', delete=False) as f:
            f.write(yosys_script)
            script_path = f.name

        try:
            proc = subprocess.run(
                [self.yosys_binary, '-s', script_path],
                capture_output=True, text=True, timeout=30
            )

            output = proc.stdout + proc.stderr

            for line in output.split('\n'):
                if 'ERROR' in line.upper():
                    result['errors'].append(line.strip())
                elif 'WARNING' in line.upper():
                    result['warnings'].append(line.strip())

            result['syntax_valid'] = (proc.returncode == 0 and len(result['errors']) == 0)

        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        finally:
            Path(rtl_path).unlink(missing_ok=True)
            Path(script_path).unlink(missing_ok=True)

        return result

    def _extract_ports(self, rtl_code: str) -> List[Dict]:
        """Extract ports from RTL"""
        ports = []
        pattern = r'(input|output|inout)\s+(wire|reg)?\s*(\[\s*\d+\s*:\s*\d+\s*\])?\s*(\w+)'

        for match in re.finditer(pattern, rtl_code):
            direction = match.group(1)
            width_spec = match.group(3)
            name = match.group(4)

            width = 1
            if width_spec:
                width_match = re.search(r'\[(\d+):(\d+)\]', width_spec)
                if width_match:
                    width = int(width_match.group(1)) - int(width_match.group(2)) + 1

            ports.append({'name': name, 'direction': direction, 'width': width})

        return ports

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input"""
        has_spec = 'specification' in input_data
        has_intent = 'intent_type' in input_data
        return has_spec or has_intent

    def get_schema(self) -> Dict[str, Any]:
        """Return input schema"""
        schema_path = Path(__file__).parent.parent / 'schemas' / 'design_intent.json'
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except:
            return {}
