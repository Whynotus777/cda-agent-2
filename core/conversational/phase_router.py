"""
Phase-Aware Router - Routes to specialized 8B models by design phase

Architecture:
- All models are 8B (no weak 3B layer)
- Each phase has a fine-tuned 8B specialist:
  * Synthesis: llama3:8b-synthesis
  * Placement: llama3:8b-placement
  * Routing: llama3:8b-routing
  * Timing: llama3:8b-timing
  * Power: llama3:8b-power
  * General: llama3:8b-chipdesign (fallback)
- 70B supervisor only intervenes when 8B struggles
- 70B always learning in background
"""

import logging
from typing import Dict, Optional, List
from enum import Enum
import re

logger = logging.getLogger(__name__)


class DesignPhase(Enum):
    """Chip design phases - each has a specialized 8B model"""
    SYNTHESIS = "synthesis"
    PLACEMENT = "placement"
    ROUTING = "routing"
    TIMING = "timing"
    POWER = "power"
    VERIFICATION = "verification"
    FLOORPLAN = "floorplan"
    GENERAL = "general"  # Cross-cutting or unclear


class PhaseRouter:
    """
    Routes queries to phase-specific 8B specialists.

    No 3B layer - goes straight to trained 8B experts.
    70B only intervenes when 8B shows signs of struggle.
    """

    def __init__(self, llm_interface):
        """
        Initialize phase router.

        Args:
            llm_interface: LLM interface for querying models
        """
        self.llm = llm_interface

        # Phase specialists - all 8B models
        self.specialists = {
            DesignPhase.SYNTHESIS: "llama3:8b-synthesis",
            DesignPhase.PLACEMENT: "llama3:8b-placement",
            DesignPhase.ROUTING: "llama3:8b-routing",
            DesignPhase.TIMING: "llama3:8b-timing",
            DesignPhase.POWER: "llama3:8b-power",
            DesignPhase.VERIFICATION: "llama3:8b-verification",
            DesignPhase.FLOORPLAN: "llama3:8b-floorplan",
            DesignPhase.GENERAL: "llama3:8b"  # Default 8B
        }

        # Fallback to base 8B if specialist not available
        self.base_model = "llama3:8b"
        self.supervisor_model = "llama3:70b"

        # Struggle tracking - when to escalate to 70B
        self.struggle_counter = 0
        self.struggle_threshold = 2

        # Phase detection patterns
        self.phase_patterns = {
            DesignPhase.SYNTHESIS: [
                r'\bsynth(?:esis|esize)?\b',
                r'\byosys\b',
                r'\brtl\b',
                r'\bgate[- ]level\b',
                r'\bnetlist\b',
                r'\blogic\s+optimization\b',
                r'\btechnology\s+mapping\b'
            ],
            DesignPhase.PLACEMENT: [
                r'\bplace(?:ment)?\b',
                r'\bcell\s+placement\b',
                r'\bstandard\s+cells?\b',
                r'\bdreamplace\b',
                r'\blegaliz(?:e|ation)\b',
                r'\bwirelength\b'
            ],
            DesignPhase.ROUTING: [
                r'\brout(?:e|ing)\b',
                r'\bwires?\b',
                r'\binterconnect\b',
                r'\btriton\s*route\b',
                r'\bvia\b',
                r'\bmetal\s+layers?\b',
                r'\bglobal\s+routing\b',
                r'\bdetailed\s+routing\b'
            ],
            DesignPhase.TIMING: [
                r'\btiming\b',
                r'\bsetup\b',
                r'\bhold\b',
                r'\bslack\b',
                r'\bclock\b',
                r'\bsta\b',
                r'\bstatic\s+timing\b',
                r'\bpath\s+delay\b',
                r'\bcritical\s+path\b'
            ],
            DesignPhase.POWER: [
                r'\bpower\b',
                r'\bleakage\b',
                r'\bdynamic\s+power\b',
                r'\bvoltage\b',
                r'\bclock\s+gating\b',
                r'\bpower\s+grid\b'
            ],
            DesignPhase.VERIFICATION: [
                r'\bverif(?:y|ication)\b',
                r'\bsimulat(?:e|ion)\b',
                r'\bformal\b',
                r'\bdrc\b',
                r'\blvs\b',
                r'\bdesign\s+rule\b'
            ],
            DesignPhase.FLOORPLAN: [
                r'\bfloor\s*plan\b',
                r'\bdie\s+size\b',
                r'\baspect\s+ratio\b',
                r'\bcore\s+area\b',
                r'\bpin\s+placement\b'
            ]
        }

        # Struggle indicators
        self.struggle_patterns = [
            'not sure',
            'unclear',
            'not certain',
            'difficult to say',
            'more information needed',
            'need more context',
            "don't have enough",
            'insufficient information'
        ]

        logger.info("Initialized PhaseRouter with 8B specialists")

    def route(
        self,
        user_input: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Route query to appropriate 8B specialist.

        Returns:
            {
                'response': str,
                'phase': DesignPhase,
                'model_used': str,
                'escalated_to_70b': bool
            }
        """
        # Detect phase
        detected_phase = self._detect_phase(user_input, context)
        logger.info(f"Detected phase: {detected_phase.value}")

        # Check if we should bypass 8B and go straight to 70B
        if self.struggle_counter >= self.struggle_threshold:
            logger.info(f"Specialist struggled {self.struggle_counter} times - escalating to 70B supervisor")
            response = self._supervisor_intervention(user_input, context, detected_phase)
            self.struggle_counter = 0  # Reset after supervisor helps

            return {
                'response': response,
                'phase': detected_phase,
                'model_used': self.supervisor_model,
                'escalated_to_70b': True
            }

        # Get specialist model for this phase
        specialist_model = self._get_specialist_model(detected_phase)

        # Query specialist
        response = self._query_specialist(
            user_input=user_input,
            phase=detected_phase,
            model=specialist_model,
            context=context
        )

        # Check if specialist is struggling
        is_struggling = self._detect_struggle(response)

        if is_struggling:
            self.struggle_counter += 1
            logger.warning(
                f"Specialist shows signs of struggle. "
                f"Counter: {self.struggle_counter}/{self.struggle_threshold}"
            )

            # Escalate immediately to 70B
            logger.info("Escalating to 70B supervisor due to specialist struggle")
            response = self._supervisor_intervention(user_input, context, detected_phase)

            return {
                'response': response,
                'phase': detected_phase,
                'model_used': self.supervisor_model,
                'escalated_to_70b': True
            }
        else:
            # Specialist handled it well - reset counter
            self.struggle_counter = 0

        return {
            'response': response,
            'phase': detected_phase,
            'model_used': specialist_model,
            'escalated_to_70b': False
        }

    def _detect_phase(
        self,
        user_input: str,
        context: Optional[Dict]
    ) -> DesignPhase:
        """
        Detect which design phase this query is about.

        Uses pattern matching + context.
        """
        user_lower = user_input.lower()

        # Check context first - if we're in a specific stage, bias toward that
        current_stage = None
        if context and 'current_stage' in context:
            current_stage = context['current_stage']

        # Score each phase
        phase_scores = {phase: 0 for phase in DesignPhase}

        # Pattern matching
        for phase, patterns in self.phase_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, user_lower, re.IGNORECASE))
                phase_scores[phase] += matches * 2  # Weight pattern matches heavily

        # Context bias
        if current_stage:
            stage_to_phase = {
                'synthesis': DesignPhase.SYNTHESIS,
                'placement': DesignPhase.PLACEMENT,
                'routing': DesignPhase.ROUTING,
                'analysis': DesignPhase.TIMING,
                'optimization': DesignPhase.POWER
            }

            if current_stage in stage_to_phase:
                phase_scores[stage_to_phase[current_stage]] += 1

        # Find highest scoring phase
        detected_phase = max(phase_scores, key=phase_scores.get)

        # If no strong signal, default to GENERAL
        if phase_scores[detected_phase] == 0:
            detected_phase = DesignPhase.GENERAL

        return detected_phase

    def _get_specialist_model(self, phase: DesignPhase) -> str:
        """
        Get the specialist model name for this phase.

        Falls back to base 8B if specialist not available.
        """
        specialist = self.specialists.get(phase, self.base_model)

        # TODO: Check if specialist model actually exists via ollama list
        # For now, just return the name

        return specialist

    def _query_specialist(
        self,
        user_input: str,
        phase: DesignPhase,
        model: str,
        context: Optional[Dict]
    ) -> str:
        """
        Query the phase specialist with domain-specific prompt.
        """
        # Format recent conversation
        recent_conv = ""
        if context and 'recent_conversation' in context and context['recent_conversation']:
            recent_conv = "\nRecent conversation:\n"
            for turn in context['recent_conversation'][-4:]:  # Last 2 exchanges
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                if len(content) > 200:
                    content = content[:197] + "..."
                recent_conv += f"{role.capitalize()}: {content}\n"

        # Build specialist prompt
        phase_context = {
            DesignPhase.SYNTHESIS: "You are an expert in RTL synthesis and logic optimization using Yosys.",
            DesignPhase.PLACEMENT: "You are an expert in cell placement and physical design using DREAMPlace/OpenROAD.",
            DesignPhase.ROUTING: "You are an expert in routing, interconnect design, and TritonRoute.",
            DesignPhase.TIMING: "You are an expert in static timing analysis, clock design, and OpenSTA.",
            DesignPhase.POWER: "You are an expert in power optimization, leakage reduction, and power grid design.",
            DesignPhase.VERIFICATION: "You are an expert in verification, simulation, DRC, and LVS.",
            DesignPhase.FLOORPLAN: "You are an expert in floorplanning, die sizing, and pin placement.",
            DesignPhase.GENERAL: "You are an expert chip design assistant with broad EDA knowledge."
        }

        system_prompt = phase_context.get(phase, phase_context[DesignPhase.GENERAL])

        prompt = f"""{system_prompt}

{recent_conv}
User query: {user_input}

Provide a detailed, technical answer based on your expertise in {phase.value}.
Use conversation context to understand follow-up questions.
Be comprehensive but practical."""

        # Query with specialist model
        original_model = self.llm.model_name
        self.llm.model_name = model

        try:
            response = self.llm.query(
                prompt,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.3  # Lower temp for technical accuracy
            )
            return response
        except Exception as e:
            logger.error(f"Specialist query failed: {e}")
            # Fallback to base model
            self.llm.model_name = self.base_model
            response = self.llm.query(prompt, system_prompt=system_prompt)
            return response
        finally:
            self.llm.model_name = original_model

    def _detect_struggle(self, response: str) -> bool:
        """
        Detect if specialist is struggling with the query.

        Returns True if response shows uncertainty or asks for clarification.
        """
        response_lower = response.lower()

        for pattern in self.struggle_patterns:
            if pattern in response_lower:
                logger.debug(f"Struggle pattern detected: '{pattern}'")
                return True

        return False

    def _supervisor_intervention(
        self,
        user_input: str,
        context: Optional[Dict],
        phase: DesignPhase
    ) -> str:
        """
        70B supervisor intervenes when 8B specialist struggles.

        70B has cross-cutting knowledge and can handle complex queries.
        """
        # Format recent conversation
        recent_conv = ""
        if context and 'recent_conversation' in context and context['recent_conversation']:
            recent_conv = "\nRecent conversation:\n"
            for turn in context['recent_conversation']:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                if len(content) > 300:
                    content = content[:297] + "..."
                recent_conv += f"{role.capitalize()}: {content}\n"

        supervisor_prompt = f"""You are a senior chip design expert providing guidance. The 8B specialist for {phase.value} was struggling with this query.

{recent_conv}
User query: {user_input}

Provide comprehensive, expert-level guidance. This query requires:
1. Deep technical knowledge across multiple phases
2. Trade-off analysis and architectural insight
3. Clear, structured explanation

Draw on your broad expertise to give a thorough answer that addresses the complexity of this question."""

        # Use 70B supervisor
        original_model = self.llm.model_name
        self.llm.model_name = self.supervisor_model

        try:
            response = self.llm.query(
                supervisor_prompt,
                system_prompt="You are a senior chip design architect with expertise across all design phases.",
                max_tokens=2048,
                temperature=0.4
            )
            logger.info("70B supervisor provided intervention guidance")
            return response
        finally:
            self.llm.model_name = original_model

    def get_available_specialists(self) -> List[str]:
        """
        Get list of available specialist models.

        TODO: Actually check ollama for which models exist.
        """
        return list(self.specialists.values())

    def reset_struggle_tracking(self):
        """Reset struggle counter for new conversation."""
        self.struggle_counter = 0
