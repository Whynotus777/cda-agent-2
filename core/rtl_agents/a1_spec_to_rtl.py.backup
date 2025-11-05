"""
A1 - Spec-to-RTL Generator

Generates RTL from natural language specifications or design intent.
Target: ≥80% compile success on first attempt.
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
    Spec-to-RTL Generator - Converts design intent to RTL code.

    Capabilities:
    - Parse natural language specifications
    - Generate RTL modules from intent
    - Infer interfaces and ports
    - Synthesize behavioral logic
    - Validate syntax with Yosys
    - Integrate with A2 templates
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize A1 agent.

        Args:
            config: Configuration dict
        """
        super().__init__(
            agent_id="A1",
            agent_name="Spec-to-RTL Generator",
            config=config
        )

        self.yosys_binary = config.get('yosys_binary', 'yosys') if config else 'yosys'

        # Initialize A2 for template generation
        self.a2_agent = A2_BoilerplateGenerator(config)

        # Load intent patterns
        self.intent_patterns = self._load_intent_patterns()

        logger.info("A1 Spec-to-RTL Generator initialized")

    def _load_intent_patterns(self) -> Dict[str, Any]:
        """Load patterns for intent recognition"""
        return {
            # FSM patterns
            'fsm': {
                'keywords': ['state machine', 'fsm', 'states', 'transitions'],
                'template': 'fsm_mealy',
                'parameters': ['num_states', 'state_bits']
            },

            # FIFO patterns
            'fifo': {
                'keywords': ['fifo', 'buffer', 'queue'],
                'template': 'fifo_sync',
                'parameters': ['depth', 'data_width']
            },
            'fifo_async': {
                'keywords': ['async fifo', 'asynchronous fifo', 'clock crossing fifo'],
                'template': 'fifo_async',
                'parameters': ['depth', 'data_width']
            },

            # Counter patterns
            'counter': {
                'keywords': ['counter', 'count'],
                'template': 'counter',
                'parameters': ['width', 'max_count']
            },

            # Register patterns
            'register': {
                'keywords': ['register', 'reg bank', 'configuration'],
                'template': 'register_file',
                'parameters': ['num_registers', 'data_width']
            },

            # AXI patterns
            'axi': {
                'keywords': ['axi', 'axi4-lite', 'axi4lite'],
                'template': 'axi4_lite_slave',
                'parameters': ['addr_width', 'data_width', 'num_registers']
            },

            # Arithmetic patterns
            'adder': {
                'keywords': ['adder', 'add', 'sum'],
                'template': 'arithmetic',
                'operation': 'add'
            },
            'multiplier': {
                'keywords': ['multiplier', 'multiply', 'mult'],
                'template': 'arithmetic',
                'operation': 'multiply'
            },

            # Control patterns
            'arbiter': {
                'keywords': ['arbiter', 'arbitration'],
                'template': 'arbiter',
                'parameters': ['num_requesters']
            }
        }

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Generate RTL from design specification.

        Args:
            input_data: Dict conforming to design_intent schema

        Returns:
            AgentOutput with generated RTL and validation results
        """
        start_time = time.time()

        if not self.validate_input(input_data):
            return self.create_output(
                success=False,
                output_data={},
                errors=["Invalid input data"]
            )

        # Extract specification
        spec = input_data.get('specification', '')
        module_name = input_data.get('module_name', 'generated_module')
        intent_type = input_data.get('intent_type')
        parameters = input_data.get('parameters', {})
        context = input_data.get('context', {})

        logger.info(f"A1 generating RTL for: {module_name}")

        # Parse intent if not explicitly provided
        if not intent_type:
            intent_type, parsed_params = self._parse_intent(spec)
            parameters.update(parsed_params)

        logger.info(f"A1 detected intent: {intent_type}")

        # Generate RTL
        rtl_code = None
        generation_method = None
        errors = []
        warnings = []

        # Try template-based generation first (A2)
        if intent_type in self.intent_patterns:
            pattern = self.intent_patterns[intent_type]
            template_type = pattern.get('template')

            if template_type:
                rtl_code, generation_method, template_errors = self._generate_from_template(
                    template_type, module_name, parameters, context
                )
                if template_errors:
                    warnings.extend(template_errors)

        # Fallback: synthesize from specification
        if not rtl_code:
            rtl_code, generation_method, synth_errors = self._synthesize_rtl(
                spec, module_name, intent_type, parameters, context
            )
            if synth_errors:
                errors.extend(synth_errors)

        if not rtl_code:
            return self.create_output(
                success=False,
                output_data={},
                errors=errors or ["Failed to generate RTL"]
            )

        # Validate with Yosys
        validation = self._validate_syntax(rtl_code, module_name)

        execution_time = (time.time() - start_time) * 1000

        # Extract ports from generated RTL
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
                'generator_version': '1.0',
                'line_count': len(rtl_code.split('\n'))
            }
        }

        # Success if syntax-valid
        success = validation.get('syntax_valid', False)

        if not success:
            errors.extend(validation.get('errors', []))

        if validation.get('warnings'):
            warnings.extend(validation.get('warnings', []))

        return self.create_output(
            success=success,
            output_data=output_data,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time,
            metadata={
                'generation_method': generation_method,
                'line_count': len(rtl_code.split('\n'))
            }
        )

    def _parse_intent(self, specification: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse natural language specification to detect intent.

        Returns:
            (intent_type, parameters)
        """
        spec_lower = specification.lower()
        parameters = {}

        # Match against patterns
        for intent_type, pattern in self.intent_patterns.items():
            keywords = pattern.get('keywords', [])
            for keyword in keywords:
                if keyword in spec_lower:
                    # Extract parameters from spec
                    parameters = self._extract_parameters(specification, pattern)
                    return intent_type, parameters

        # Default: generic module
        return 'generic', parameters

    def _extract_parameters(self, spec: str, pattern: Dict) -> Dict[str, Any]:
        """Extract parameters from specification text"""
        parameters = {}

        # Extract numbers (for width, depth, count, etc.)
        numbers = re.findall(r'(\d+)[\s-]*(bit|wide|width|deep|depth|entry|entries)?', spec, re.IGNORECASE)

        if numbers:
            if 'data_width' in pattern.get('parameters', []):
                # First number is often data width
                parameters['data_width'] = int(numbers[0][0])

            if 'depth' in pattern.get('parameters', []):
                # Look for depth/entries
                for num, unit in numbers:
                    if unit.lower() in ['deep', 'depth', 'entry', 'entries']:
                        parameters['depth'] = int(num)
                        break
                else:
                    # Second number might be depth
                    if len(numbers) > 1:
                        parameters['depth'] = int(numbers[1][0])

            if 'num_states' in pattern.get('parameters', []):
                # Look for state count
                state_match = re.search(r'(\d+)[\s-]*states?', spec, re.IGNORECASE)
                if state_match:
                    parameters['num_states'] = int(state_match.group(1))

        return parameters

    def _generate_from_template(
        self,
        template_type: str,
        module_name: str,
        parameters: Dict,
        context: Dict
    ) -> Tuple[Optional[str], str, List[str]]:
        """
        Generate RTL using A2 template.

        Returns:
            (rtl_code, method, errors)
        """
        errors = []

        try:
            # Call A2 agent
            a2_input = {
                'intent_type': template_type,
                'module_name': module_name,
                'parameters': parameters,
                'context': context
            }

            result = self.a2_agent.process(a2_input)

            if result.success:
                rtl_code = result.output_data.get('rtl_code')
                return rtl_code, f'template_{template_type}', []
            else:
                errors = result.errors

        except Exception as e:
            errors.append(f"Template generation failed: {str(e)}")

        return None, None, errors

    def _synthesize_rtl(
        self,
        spec: str,
        module_name: str,
        intent_type: str,
        parameters: Dict,
        context: Dict
    ) -> Tuple[Optional[str], str, List[str]]:
        """
        Synthesize RTL from specification (fallback method).

        Returns:
            (rtl_code, method, errors)
        """
        errors = []

        try:
            # Simple RTL synthesis based on intent
            if intent_type == 'register':
                rtl_code = self._synthesize_register(module_name, parameters)
                return rtl_code, 'synthesized_register', []

            elif intent_type == 'adder':
                rtl_code = self._synthesize_adder(module_name, parameters)
                return rtl_code, 'synthesized_adder', []

            elif intent_type == 'multiplier':
                rtl_code = self._synthesize_multiplier(module_name, parameters)
                return rtl_code, 'synthesized_multiplier', []

            else:
                # Generic module with basic structure
                rtl_code = self._synthesize_generic(module_name, spec, parameters, context)
                return rtl_code, 'synthesized_generic', []

        except Exception as e:
            errors.append(f"RTL synthesis failed: {str(e)}")

        return None, None, errors

    def _synthesize_register(self, module_name: str, params: Dict) -> str:
        """Synthesize a simple register module"""
        width = params.get('data_width', 8)

        return f'''// Generated by A1 Spec-to-RTL Generator
module {module_name} (
    input wire clk,
    input wire rst_n,
    input wire [{width-1}:0] data_in,
    input wire write_enable,
    output reg [{width-1}:0] data_out
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= {width}'h0;
        end else if (write_enable) begin
            data_out <= data_in;
        end
    end

endmodule
'''

    def _synthesize_adder(self, module_name: str, params: Dict) -> str:
        """Synthesize an adder module"""
        width = params.get('data_width', 8)

        return f'''// Generated by A1 Spec-to-RTL Generator
module {module_name} (
    input wire [{width-1}:0] a,
    input wire [{width-1}:0] b,
    input wire cin,
    output wire [{width-1}:0] sum,
    output wire cout
);

    assign {{cout, sum}} = a + b + cin;

endmodule
'''

    def _synthesize_multiplier(self, module_name: str, params: Dict) -> str:
        """Synthesize a multiplier module"""
        width = params.get('data_width', 8)

        return f'''// Generated by A1 Spec-to-RTL Generator
module {module_name} (
    input wire [{width-1}:0] a,
    input wire [{width-1}:0] b,
    output wire [{width*2-1}:0] product
);

    assign product = a * b;

endmodule
'''

    def _synthesize_generic(
        self,
        module_name: str,
        spec: str,
        params: Dict,
        context: Dict
    ) -> str:
        """Synthesize a generic module with basic structure"""
        width = params.get('data_width', 8)

        # Infer if clocked or combinational
        is_clocked = any(keyword in spec.lower() for keyword in ['register', 'sequential', 'state', 'clock'])

        if is_clocked:
            return f'''// Generated by A1 Spec-to-RTL Generator
// Specification: {spec[:100]}...
module {module_name} (
    input wire clk,
    input wire rst_n,
    input wire [{width-1}:0] data_in,
    output reg [{width-1}:0] data_out
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= {width}'h0;
        end else begin
            data_out <= data_in;
        end
    end

endmodule
'''
        else:
            return f'''// Generated by A1 Spec-to-RTL Generator
// Specification: {spec[:100]}...
module {module_name} (
    input wire [{width-1}:0] data_in,
    output wire [{width-1}:0] data_out
);

    assign data_out = data_in;

endmodule
'''

    def _validate_syntax(self, rtl_code: str, module_name: str) -> Dict[str, Any]:
        """
        Validate RTL syntax with Yosys.

        Returns:
            Validation result dict
        """
        result = {
            'syntax_valid': False,
            'errors': [],
            'warnings': []
        }

        # Write RTL to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(rtl_code)
            rtl_path = f.name

        # Create Yosys script
        yosys_script = f"""
read_verilog {rtl_path}
hierarchy -check -top {module_name}
proc
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ys', delete=False) as f:
            f.write(yosys_script)
            script_path = f.name

        try:
            # Run Yosys
            cmd = [self.yosys_binary, '-s', script_path]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = proc.stdout + proc.stderr

            # Parse output for errors/warnings
            for line in output.split('\n'):
                if 'ERROR' in line.upper():
                    result['errors'].append(line.strip())
                elif 'WARNING' in line.upper():
                    result['warnings'].append(line.strip())

            # Success if no errors and return code 0
            result['syntax_valid'] = (proc.returncode == 0 and len(result['errors']) == 0)

        except subprocess.TimeoutExpired:
            result['errors'].append("Yosys validation timed out")
        except FileNotFoundError:
            result['errors'].append("Yosys not found")
            result['syntax_valid'] = None  # Can't validate
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        finally:
            # Clean up
            Path(rtl_path).unlink(missing_ok=True)
            Path(script_path).unlink(missing_ok=True)

        return result

    def _extract_ports(self, rtl_code: str) -> List[Dict[str, Any]]:
        """Extract port information from RTL code"""
        ports = []

        # Simple regex for port extraction
        port_pattern = r'(input|output|inout)\s+(wire|reg)?\s*(\[\s*\d+\s*:\s*\d+\s*\])?\s*(\w+)'

        for match in re.finditer(port_pattern, rtl_code):
            direction = match.group(1)
            port_type = match.group(2) or 'wire'
            width_spec = match.group(3)
            name = match.group(4)

            # Parse width
            width = 1
            if width_spec:
                width_match = re.search(r'\[(\d+):(\d+)\]', width_spec)
                if width_match:
                    msb = int(width_match.group(1))
                    lsb = int(width_match.group(2))
                    width = msb - lsb + 1

            ports.append({
                'name': name,
                'direction': direction,
                'type': port_type,
                'width': width
            })

        return ports

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # Need either specification or intent_type
        has_spec = 'specification' in input_data and input_data['specification']
        has_intent = 'intent_type' in input_data

        if not (has_spec or has_intent):
            logger.error("No specification or intent_type provided")
            return False

        return True

    def get_schema(self) -> Dict[str, Any]:
        """Return input schema for A1"""
        schema_path = Path(__file__).parent.parent / 'schemas' / 'design_intent.json'
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {}
