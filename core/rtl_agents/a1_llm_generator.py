"""
A1 LLM Generator - Fine-tuned Mixtral RTL Generator
Wrapper for using fine-tuned LLM models (V3, V4, V5) in the pipeline
"""

import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .base_agent import BaseAgent, AgentOutput


class A1_LLMGenerator(BaseAgent):
    """
    A1 LLM-based RTL Generator

    Uses fine-tuned Mixtral-8x7B models for spec-to-RTL generation.
    Supports loading different model versions (V3, V4, V5).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM-based generator

        Args:
            config: Dict with:
                - model_path: Path to fine-tuned model/adapter
                - max_new_tokens: Maximum tokens to generate (default: 4096)
                - temperature: Sampling temperature (default: 0.7)
                - top_p: Nucleus sampling parameter (default: 0.95)
        """
        super().__init__(
            agent_id="A1_LLM",
            agent_name="LLM Spec-to-RTL Generator",
            config=config
        )

        self.adapter_path = Path(config.get('model_path',
            'models/mixtral_rtl/run_pure_20251030_121523/final_model'))
        self.base_model_path = Path(config.get('base_model_path',
            'models/mixtral_base/Mixtral-8x7B-Instruct-v0.1'))
        self.max_new_tokens = config.get('max_new_tokens', 4096)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.95)
        self.use_rag = bool(config.get('use_rag', True))
        self.rag_top_k = int(config.get('rag_top_k', 4))
        self.rag_max_context = int(config.get('rag_max_context', 2000))
        self.rag = None
        self._rag_error: Optional[str] = None

        self.model = None
        self.tokenizer = None
        self._load_model()
        self._init_rag()

    def _ensure_model_cached(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model path not found: {path}. "
                f"Run ./scripts/cache_models.sh to download the weights."
            )

    def _load_model(self):
        """Load fine-tuned model and tokenizer"""
        print("\n🔧 Loading A1 LLM Generator from local cache")
        self._ensure_model_cached(self.base_model_path)
        self._ensure_model_cached(self.adapter_path)

        # BitsAndBytes config for 4-bit inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Check if adapter or full model
        adapter_config = self.adapter_path / "adapter_config.json"

        if adapter_config.exists():
            # Load as adapter on base model
            print("   Loading base Mixtral-8x7B...")
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )

            print("   Loading fine-tuned adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
                local_files_only=True
            )
        else:
            # Load as full model
            print("   Loading full fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )

        # Load tokenizer
        print("   Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.adapter_path),
                local_files_only=True
            )
        except:
            # Fallback to base model tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.base_model_path),
                local_files_only=True
            )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("   ✅ Model loaded successfully\n")

    def _init_rag(self) -> None:
        """Initialise RAG retriever if available."""
        if not self.use_rag:
            self._rag_error = "RAG integration disabled via configuration"
            return

        try:
            from core.rag import RAGRetriever  # Local import to avoid heavy dependency when unused
            self.rag = RAGRetriever()
            print("   ✅ RAG retriever initialised")
        except Exception as exc:  # pylint: disable=broad-except
            self._rag_error = f"Failed to initialise RAG retriever: {exc}"
            print(f"   ⚠️  {self._rag_error}")

    def _retrieve_rag_context(self, specification: str, parameters: Dict[str, Any]) -> str:
        """Retrieve contextual knowledge snippets for the given specification."""
        if not self.rag:
            return ""

        query_parts = [specification.strip()]
        if parameters:
            param_summary = ", ".join(f"{k}={v}" for k, v in parameters.items())
            query_parts.append(f"Parameters: {param_summary}")

        query = " | ".join(filter(None, query_parts))

        try:
            context = self.rag.retrieve_and_format(
                query=query,
                top_k=self.rag_top_k,
                max_context_length=self.rag_max_context
            )
            return context.strip()
        except Exception as exc:  # pylint: disable=broad-except
            if not self._rag_error:
                self._rag_error = f"RAG retrieval failed: {exc}"
                print(f"   ⚠️  {self._rag_error}")
            return ""

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input specification"""
        required = ['module_name', 'specification']
        return all(key in input_data for key in required)

    def get_schema(self) -> Dict[str, Any]:
        """Return minimal input schema for the LLM generator."""
        return {
            "type": "object",
            "required": ["module_name", "specification"],
            "properties": {
                "module_name": {"type": "string"},
                "specification": {"type": "string"},
                "parameters": {"type": "object"},
                "intent_type": {"type": ["string", "null"]},
            },
        }

    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Generate RTL using fine-tuned LLM

        Args:
            input_data: Dict with:
                - module_name: Module name
                - specification: Natural language description
                - parameters: Dict of design parameters (optional)
                - intent_type: Design intent type (optional)

        Returns:
            AgentOutput with generated RTL
        """
        start_time = time.time()

        if not self.validate_input(input_data):
            return self.create_output(
                success=False,
                output_data={},
                errors=["Invalid input: missing module_name or specification"]
            )

        module_name = input_data.get('module_name', 'generated_module')
        specification = input_data.get('specification', '')
        parameters = input_data.get('parameters', {})

        rag_context = self._retrieve_rag_context(specification, parameters)

        # Create prompt
        param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
        context_block = f"Context from knowledge base:\n{rag_context}\n\n" if rag_context else ""

        prompt = f"""[INST] {context_block}Generate synthesizable SystemVerilog (IEEE 1800) for a module named '{module_name}'.

Specification: {specification}

Parameters: {param_str}

Requirements:
- Use `.sv` SystemVerilog syntax (e.g., logic types, always_ff/always_comb).
- Include all module ports with explicit directions and widths.
- Implement the complete functionality, including FIFOs or clock dividers if required.
- Do not emit testbenches, assertions, or simulation-only constructs.

Generate only the SystemVerilog module code. [/INST]"""

        print(f"\n📝 Generating RTL with A1 LLM...")
        print(f"   Module: {module_name}")
        print(f"   Prompt length: {len(prompt)} chars")
        print(f"   Max tokens: {self.max_new_tokens}")

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # Generate
            gen_start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generation_time = time.time() - gen_start

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract Verilog code (after [/INST])
            if '[/INST]' in generated_text:
                rtl_code = generated_text.split('[/INST]')[1].strip()
            else:
                rtl_code = generated_text.strip()

            print(f"   ✅ Generation complete ({generation_time:.2f}s)")
            print(f"   Generated {len(rtl_code)} characters, {len(rtl_code.split('\\n'))} lines")

            # Validate syntax with Yosys if available
            validation = self._validate_syntax(rtl_code, module_name)

            # Extract ports (basic extraction)
            ports = self._extract_ports(rtl_code)

            execution_time = (time.time() - start_time) * 1000

            return self.create_output(
                success=True,
                output_data={
                    'rtl_code': rtl_code,
                    'ports': ports,
                    'generation_method': 'llm_mixtral',
                    'model_path': str(self.adapter_path),
                    'generation_time_s': generation_time,
                    'tokens_generated': len(outputs[0]) - len(inputs['input_ids'][0]),
                    'validation': validation,
                    'generated_at': datetime.utcnow().isoformat(),
                    'rag_context_chars': len(rag_context),
                    'rag_enabled': bool(self.rag)
                },
                confidence=0.8 if validation.get('syntax_valid', False) else 0.5,
                execution_time_ms=execution_time
            )

        except Exception as e:
            return self.create_output(
                success=False,
                output_data={},
                errors=[f"LLM generation failed: {str(e)}"],
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def _validate_syntax(self, rtl_code: str, module_name: str) -> Dict[str, Any]:
        """Validate RTL syntax using Yosys"""
        import subprocess
        import tempfile

        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
                f.write(rtl_code)
                temp_file = f.name

            # Run Yosys
            result = subprocess.run(
                ['yosys', '-p', f'read_verilog {temp_file}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            syntax_valid = result.returncode == 0
            errors = []
            warnings = []

            if not syntax_valid:
                for line in result.stderr.split('\\n'):
                    if 'ERROR' in line.upper():
                        errors.append(line.strip())
                    elif 'WARNING' in line.upper():
                        warnings.append(line.strip())

            Path(temp_file).unlink(missing_ok=True)

            return {
                'syntax_valid': syntax_valid,
                'errors': errors,
                'warnings': warnings
            }

        except Exception as e:
            return {
                'syntax_valid': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _extract_ports(self, rtl_code: str) -> list:
        """Extract port information from RTL code"""
        ports = []

        try:
            # Simple regex-based extraction
            import re

            # Find input/output declarations
            input_pattern = r'input\s+(?:wire\s+)?(?:\[.*?\]\s+)?(\w+)'
            output_pattern = r'output\s+(?:reg\s+|wire\s+)?(?:\[.*?\]\s+)?(\w+)'

            for match in re.finditer(input_pattern, rtl_code):
                ports.append({
                    'name': match.group(1),
                    'direction': 'input',
                    'width': 1
                })

            for match in re.finditer(output_pattern, rtl_code):
                ports.append({
                    'name': match.group(1),
                    'direction': 'output',
                    'width': 1
                })

        except Exception:
            pass

        return ports
