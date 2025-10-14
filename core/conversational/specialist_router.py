"""
Specialist Model Router

Routes queries to phase-specific expert models when available,
falls back to general models when specialists aren't trained yet.

Specialist Models:
- triage_specialist:8b - Conversational routing and intent understanding
- rtl_design_specialist:8b - Verilog/SystemVerilog code generation
- synthesis_specialist:8b - Yosys synthesis expertise
- placement_specialist:8b - DREAMPlace placement optimization
- routing_specialist:8b - TritonRoute routing expertise
- timing_specialist:8b - OpenSTA timing analysis
- power_specialist:8b - Power optimization strategies
"""

import logging
from typing import Dict, Optional
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


class SpecialistRouter:
    """
    Routes queries to phase-specific specialist models.

    Falls back to general models if specialists aren't available.
    """

    def __init__(self, llm_interface):
        """
        Initialize specialist router.

        Args:
            llm_interface: LLM interface for querying models
        """
        self.llm = llm_interface

        # Specialist model names
        self.specialists = {
            'triage': 'triage_specialist:8b',
            'rtl_design': 'rtl_design_specialist:8b',
            'synthesis': 'synthesis_specialist:8b',
            'placement': 'placement_specialist:8b',
            'routing': 'routing_specialist:8b',
            'timing': 'timing_specialist:8b',
            'power': 'power_specialist:8b',
        }

        # Fallback models if specialists aren't available
        self.fallbacks = {
            'triage': 'llama3:8b',
            'rtl_design': 'llama3:8b',
            'synthesis': 'llama3:8b',
            'placement': 'llama3:8b',
            'routing': 'llama3:8b',
            'timing': 'llama3:8b',
            'power': 'llama3:8b',
        }

        # Check which specialists are available
        self.available_specialists = self._check_available_models()

        logger.info(f"Initialized SpecialistRouter")
        logger.info(f"Available specialists: {list(self.available_specialists.keys())}")

    def _check_available_models(self) -> Dict[str, bool]:
        """
        Check which specialist models are actually available.

        Returns:
            Dict mapping phase to availability bool
        """
        available = {}

        try:
            # List available Ollama models
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                model_list = result.stdout.lower()

                for phase, model_name in self.specialists.items():
                    # Check if specialist model exists
                    model_key = model_name.split(':')[0]  # e.g., "triage_specialist"
                    available[phase] = model_key in model_list

                    if available[phase]:
                        logger.info(f"✓ {phase} specialist available: {model_name}")
                    else:
                        logger.info(f"⊙ {phase} specialist not found, will use fallback")
            else:
                logger.warning("Could not check Ollama models, assuming no specialists")
                available = {phase: False for phase in self.specialists.keys()}

        except Exception as e:
            logger.warning(f"Error checking models: {e}, assuming no specialists")
            available = {phase: False for phase in self.specialists.keys()}

        return available

    def route_to_specialist(
        self,
        query: str,
        phase: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Route query to appropriate specialist or fallback model.

        Args:
            query: User query
            phase: Design phase (triage, synthesis, placement, etc.)
            context: Optional context dictionary

        Returns:
            Model response string
        """
        # Determine which model to use
        if self.available_specialists.get(phase, False):
            model = self.specialists[phase]
            logger.info(f"Routing to specialist: {model}")
        else:
            model = self.fallbacks[phase]
            logger.info(f"Routing to fallback: {model} (specialist not available)")

        # Build prompt with phase context
        system_prompt = self._get_system_prompt(phase)

        # Call model
        try:
            response = self.llm.query(
                model=model,
                prompt=query,
                system=system_prompt,
                context=context
            )

            return response

        except Exception as e:
            logger.error(f"Error querying {model}: {e}")

            # Try fallback if specialist failed
            if model != self.fallbacks[phase]:
                logger.info(f"Falling back to {self.fallbacks[phase]}")
                try:
                    response = self.llm.query(
                        model=self.fallbacks[phase],
                        prompt=query,
                        system=system_prompt,
                        context=context
                    )
                    return response
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    return f"Error: Unable to process query. {e2}"
            else:
                return f"Error: Unable to process query. {e}"

    def _get_system_prompt(self, phase: str) -> str:
        """Get phase-specific system prompt"""
        prompts = {
            'triage': """You are an expert at understanding user intent in chip design conversations.
Your role is to route queries to the appropriate specialist and ask clarifying questions when needed.
Focus on: intent classification, query clarification, routing decisions.""",

            'rtl_design': """You are an expert RTL designer specializing in Verilog and SystemVerilog.
You help write, debug, and optimize hardware description language code.
Focus on: RTL coding best practices, design patterns, parameterization, modularity.""",

            'synthesis': """You are an expert in logic synthesis and gate-level optimization.
You help optimize RTL designs for area, timing, and power using synthesis tools like Yosys.
Focus on: synthesis constraints, optimization techniques, technology mapping.""",

            'placement': """You are an expert in physical design and cell placement.
You help optimize chip layout for wirelength, density, and routability using tools like DREAMPlace.
Focus on: floorplanning, placement strategies, congestion analysis.""",

            'routing': """You are an expert in chip routing and interconnect design.
You help create optimal wire connections while meeting design rules using tools like TritonRoute.
Focus on: routing strategies, DRC compliance, timing-driven routing.""",

            'timing': """You are an expert in static timing analysis and timing closure.
You help analyze and fix timing violations using tools like OpenSTA.
Focus on: critical paths, slack analysis, timing optimization.""",

            'power': """You are an expert in power analysis and low-power design techniques.
You help analyze and reduce power consumption in chip designs.
Focus on: power estimation, clock gating, multi-VT optimization, power domains.""",
        }

        return prompts.get(phase, "You are an expert chip design assistant.")

    def classify_phase(self, query: str) -> str:
        """
        Classify which phase a query belongs to.

        Args:
            query: User query

        Returns:
            Phase name (triage, synthesis, placement, etc.)
        """
        query_lower = query.lower()

        # Phase keywords
        phase_keywords = {
            'rtl_design': [
                'verilog', 'systemverilog', 'rtl', 'module', 'always',
                'assign', 'reg', 'wire', 'code', 'hdl'
            ],
            'synthesis': [
                'synthesis', 'synthesize', 'yosys', 'gate', 'netlist',
                'technology mapping', 'abc', 'optimize logic'
            ],
            'placement': [
                'placement', 'place', 'dreamplace', 'floorplan', 'wirelength',
                'hpwl', 'density', 'cell location', 'congestion'
            ],
            'routing': [
                'routing', 'route', 'tritonroute', 'wire', 'via',
                'interconnect', 'drc', 'metal layer', 'track'
            ],
            'timing': [
                'timing', 'sta', 'static timing', 'opensta', 'setup',
                'hold', 'slack', 'critical path', 'wns', 'tns', 'delay'
            ],
            'power': [
                'power', 'leakage', 'dynamic power', 'switching',
                'clock gating', 'low power', 'power domain', 'voltage'
            ],
        }

        # Score each phase
        scores = {}
        for phase, keywords in phase_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[phase] = score

        # Return phase with highest score, default to triage
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'triage'

    def route_query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Auto-classify and route query to appropriate specialist.

        Args:
            query: User query
            context: Optional context

        Returns:
            {
                'phase': str,
                'specialist_used': str,
                'response': str
            }
        """
        # Classify phase
        phase = self.classify_phase(query)

        # Route to specialist
        response = self.route_to_specialist(query, phase, context)

        # Determine which model was actually used
        model_used = (
            self.specialists[phase]
            if self.available_specialists.get(phase, False)
            else self.fallbacks[phase]
        )

        return {
            'phase': phase,
            'specialist_used': model_used,
            'response': response
        }

    def reload_specialists(self):
        """
        Reload specialist availability.

        Call this after training new specialists.
        """
        logger.info("Reloading specialist model availability...")
        self.available_specialists = self._check_available_models()
        logger.info(f"Available specialists: {list(self.available_specialists.keys())}")
