"""
Model Router - Hierarchical LLM Selection

Routes queries to appropriate model size based on complexity.
Always consults 70B orchestrator for learning and routing decisions.
"""

import logging
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DesignPhase(Enum):
    """Chip design phases - each can have specialized models"""
    SPECIFICATION = "specification"
    RTL_DESIGN = "rtl_design"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    FLOORPLANNING = "floorplanning"
    PLACEMENT = "placement"
    CTS = "cts"  # Clock Tree Synthesis
    ROUTING = "routing"
    TIMING_ANALYSIS = "timing_analysis"
    POWER_ANALYSIS = "power_analysis"
    DRC_LVS = "drc_lvs"
    SIGNOFF = "signoff"
    GENERAL = "general"


class ModelSize(Enum):
    """Available model sizes"""
    SMALL = "3b"
    MEDIUM = "8b"
    LARGE = "70b"


class ModelRouter:
    """
    Intelligent router that:
    1. Always consults 70B orchestrator first
    2. Routes to appropriate specialized model
    3. Collects training data for fine-tuning
    """

    def __init__(self, llm_interface):
        """
        Initialize model router.

        Args:
            llm_interface: Base LLM interface
        """
        self.llm = llm_interface

        # Model naming convention: {base_model}:{phase}
        # e.g., "llama3:8b-synthesis", "llama3:3b-placement"
        self.specialist_models = self._init_specialist_models()

        # Track what models exist
        self.available_specialists = self._check_available_specialists()

        # Routing thresholds (characters)
        self.size_thresholds = {
            ModelSize.SMALL: 500,   # < 500 chars → 3B
            ModelSize.MEDIUM: 2000,  # < 2000 chars → 8B
            ModelSize.LARGE: float('inf')  # >= 2000 chars → 70B
        }

        # Training data collection
        self.training_buffer = []

        logger.info("Initialized ModelRouter with hierarchical routing")

    def _init_specialist_models(self) -> Dict[DesignPhase, Dict[ModelSize, str]]:
        """
        Define specialist model names for each phase and size.

        Returns:
            Dict mapping phase -> size -> model_name
        """
        models = {}

        for phase in DesignPhase:
            models[phase] = {
                ModelSize.SMALL: f"llama3.2:3b-{phase.value}",
                ModelSize.MEDIUM: f"llama3:8b-{phase.value}",
                ModelSize.LARGE: f"llama3:70b-{phase.value}"
            }

        # Generic models (fallback if specialist doesn't exist)
        models[DesignPhase.GENERAL] = {
            ModelSize.SMALL: "llama3.2:3b",
            ModelSize.MEDIUM: "llama3:8b",
            ModelSize.LARGE: "llama3:70b"
        }

        return models

    def _check_available_specialists(self) -> Dict[DesignPhase, Dict[ModelSize, bool]]:
        """
        Check which specialist models are actually available in Ollama.

        Returns:
            Dict mapping phase -> size -> is_available
        """
        import subprocess

        available = {}

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )

            model_list = result.stdout.lower()

            for phase in DesignPhase:
                available[phase] = {}
                for size in ModelSize:
                    model_name = self.specialist_models[phase][size]
                    available[phase][size] = model_name.lower() in model_list

        except Exception as e:
            logger.warning(f"Could not check available models: {e}")
            # Assume only base models exist
            for phase in DesignPhase:
                available[phase] = {size: False for size in ModelSize}

        return available

    def route_query(
        self,
        user_input: str,
        current_phase: Optional[DesignPhase] = None,
        context: Optional[Dict] = None
    ) -> Tuple[str, str, Dict]:
        """
        Route query through hierarchical system.

        Process:
        1. Always consult 70B orchestrator first (for learning + routing)
        2. Orchestrator decides: phase, complexity, which specialist
        3. Route to specialist model
        4. Collect training data

        Args:
            user_input: User's query
            current_phase: Current design phase (if known)
            context: Conversation context

        Returns:
            (response, model_used, routing_info)
        """
        # Step 1: Consult orchestrator (70B)
        orchestrator_decision = self._consult_orchestrator(
            user_input, current_phase, context
        )

        phase = orchestrator_decision['phase']
        suggested_size = orchestrator_decision['suggested_size']
        reasoning = orchestrator_decision['reasoning']

        logger.info(
            f"Orchestrator routed to: {phase.value} / {suggested_size.value} "
            f"(Reason: {reasoning})"
        )

        # Step 2: Determine actual model to use
        model_name = self._select_model(phase, suggested_size, user_input)

        # Step 3: Query specialist model
        specialist_response = self._query_specialist(
            model_name=model_name,
            user_input=user_input,
            phase=phase,
            context=context
        )

        # Step 4: Collect training data
        self._collect_training_data(
            user_input=user_input,
            phase=phase,
            model_size=suggested_size,
            response=specialist_response,
            context=context
        )

        routing_info = {
            'phase': phase.value,
            'model_size': suggested_size.value,
            'model_name': model_name,
            'orchestrator_reasoning': reasoning,
            'is_specialist': self.available_specialists.get(phase, {}).get(suggested_size, False)
        }

        return specialist_response, model_name, routing_info

    def _consult_orchestrator(
        self,
        user_input: str,
        current_phase: Optional[DesignPhase],
        context: Optional[Dict]
    ) -> Dict:
        """
        Always consult 70B orchestrator for routing decision.

        This ensures:
        - 70B learns from all queries
        - Intelligent routing based on content, not just length
        - Can override size suggestions for complex queries
        """
        orchestrator_prompt = f"""You are the orchestrator for a chip design AI system.

Analyze this user query and determine:
1. Which design phase this belongs to: {[p.value for p in DesignPhase]}
2. What model size is needed: 3b (simple), 8b (moderate), 70b (complex)
3. Brief reasoning

Current context: {context or 'None'}
Current phase: {current_phase.value if current_phase else 'Unknown'}

User query: {user_input}

Respond in JSON format:
{{
    "phase": "phase_name",
    "suggested_size": "3b|8b|70b",
    "reasoning": "brief explanation",
    "complexity_score": 0.0-1.0
}}"""

        try:
            # Force use of 70B orchestrator
            original_model = self.llm.model_name
            self.llm.model_name = "llama3:70b"

            response = self.llm.query(orchestrator_prompt)

            # Restore original model
            self.llm.model_name = original_model

            # Parse JSON response
            import json
            decision = json.loads(response)

            # Convert to enums
            phase = DesignPhase(decision.get('phase', 'general'))
            size_str = decision.get('suggested_size', '8b')

            size_map = {'3b': ModelSize.SMALL, '8b': ModelSize.MEDIUM, '70b': ModelSize.LARGE}
            suggested_size = size_map.get(size_str, ModelSize.MEDIUM)

            return {
                'phase': phase,
                'suggested_size': suggested_size,
                'reasoning': decision.get('reasoning', 'N/A'),
                'complexity_score': decision.get('complexity_score', 0.5)
            }

        except Exception as e:
            logger.warning(f"Orchestrator call failed, using fallback: {e}")

            # Fallback: simple heuristic
            return self._fallback_routing(user_input, current_phase)

    def _fallback_routing(
        self,
        user_input: str,
        current_phase: Optional[DesignPhase]
    ) -> Dict:
        """
        Fallback routing based on simple heuristics.
        """
        # Length-based size selection
        length = len(user_input)

        if length < self.size_thresholds[ModelSize.SMALL]:
            suggested_size = ModelSize.SMALL
        elif length < self.size_thresholds[ModelSize.MEDIUM]:
            suggested_size = ModelSize.MEDIUM
        else:
            suggested_size = ModelSize.LARGE

        # Phase detection from keywords
        phase_keywords = {
            DesignPhase.SPECIFICATION: ['requirement', 'spec', 'architecture', 'define'],
            DesignPhase.RTL_DESIGN: ['verilog', 'rtl', 'hdl', 'systemverilog'],
            DesignPhase.SYNTHESIS: ['synthesis', 'synth', 'yosys', 'gate-level'],
            DesignPhase.PLACEMENT: ['place', 'placement', 'dreamplace'],
            DesignPhase.ROUTING: ['route', 'routing', 'wire'],
            DesignPhase.TIMING_ANALYSIS: ['timing', 'slack', 'sta', 'setup', 'hold'],
            DesignPhase.POWER_ANALYSIS: ['power', 'leakage', 'dynamic'],
        }

        detected_phase = current_phase or DesignPhase.GENERAL
        user_lower = user_input.lower()

        for phase, keywords in phase_keywords.items():
            if any(kw in user_lower for kw in keywords):
                detected_phase = phase
                break

        return {
            'phase': detected_phase,
            'suggested_size': suggested_size,
            'reasoning': f'Fallback routing (length={length})',
            'complexity_score': min(length / 2000, 1.0)
        }

    def _select_model(
        self,
        phase: DesignPhase,
        size: ModelSize,
        user_input: str
    ) -> str:
        """
        Select actual model name to use.

        Prefers specialist if available, falls back to general.
        """
        # Check if specialist exists
        if self.available_specialists.get(phase, {}).get(size, False):
            model_name = self.specialist_models[phase][size]
            logger.info(f"Using specialist: {model_name}")
            return model_name

        # Fallback to general model
        general_model = self.specialist_models[DesignPhase.GENERAL][size]
        logger.info(f"Specialist not found, using general: {general_model}")
        return general_model

    def _query_specialist(
        self,
        model_name: str,
        user_input: str,
        phase: DesignPhase,
        context: Optional[Dict]
    ) -> str:
        """
        Query the selected specialist model.
        """
        # Construct phase-specific system prompt
        system_prompt = self._get_phase_system_prompt(phase)

        # Temporarily switch model
        original_model = self.llm.model_name
        self.llm.model_name = model_name

        try:
            response = self.llm.query(
                user_input,
                system_prompt=system_prompt
            )
        finally:
            # Restore original model
            self.llm.model_name = original_model

        return response

    def _get_phase_system_prompt(self, phase: DesignPhase) -> str:
        """
        Get specialized system prompt for each phase.
        """
        prompts = {
            DesignPhase.SPECIFICATION: "You are an expert chip architect specializing in SoC specifications and requirements.",
            DesignPhase.RTL_DESIGN: "You are an expert RTL designer specializing in Verilog/SystemVerilog.",
            DesignPhase.SYNTHESIS: "You are an expert in logic synthesis and gate-level optimization.",
            DesignPhase.PLACEMENT: "You are an expert in physical design and cell placement optimization.",
            DesignPhase.ROUTING: "You are an expert in chip routing and interconnect design.",
            DesignPhase.TIMING_ANALYSIS: "You are an expert in static timing analysis and timing closure.",
            DesignPhase.POWER_ANALYSIS: "You are an expert in power analysis and low-power design.",
            DesignPhase.GENERAL: "You are an expert EDA assistant helping with chip design."
        }

        return prompts.get(phase, prompts[DesignPhase.GENERAL])

    def _collect_training_data(
        self,
        user_input: str,
        phase: DesignPhase,
        model_size: ModelSize,
        response: str,
        context: Optional[Dict]
    ):
        """
        Collect training data for fine-tuning specialist models.

        Format: JSONL with prompt, response, phase, metadata
        """
        training_entry = {
            'prompt': user_input,
            'response': response,
            'phase': phase.value,
            'model_size': model_size.value,
            'context': context,
            'timestamp': self._get_timestamp()
        }

        self.training_buffer.append(training_entry)

        # Flush to disk periodically
        if len(self.training_buffer) >= 10:
            self._flush_training_data()

    def _flush_training_data(self):
        """
        Write training data to disk for later fine-tuning.
        """
        import json
        import os
        from datetime import datetime

        if not self.training_buffer:
            return

        # Create training data directory
        training_dir = "./data/training"
        os.makedirs(training_dir, exist_ok=True)

        # Write to JSONL file (one JSON object per line)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{training_dir}/training_data_{timestamp}.jsonl"

        try:
            with open(filename, 'a') as f:
                for entry in self.training_buffer:
                    f.write(json.dumps(entry) + '\n')

            logger.info(f"Flushed {len(self.training_buffer)} training samples to {filename}")
            self.training_buffer = []

        except Exception as e:
            logger.error(f"Failed to write training data: {e}")

    def _get_timestamp(self) -> str:
        """Get ISO timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_routing_stats(self) -> Dict:
        """Get statistics on routing decisions"""
        return {
            'total_specialists': sum(
                sum(1 for available in phase_dict.values() if available)
                for phase_dict in self.available_specialists.values()
            ),
            'training_samples_buffered': len(self.training_buffer),
            'available_specialists': self.available_specialists
        }
