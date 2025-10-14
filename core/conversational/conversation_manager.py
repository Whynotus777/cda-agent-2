"""
Conversation Manager Module

Orchestrates the conversational flow between the user and the agent.
Maintains conversation state, context, and history.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .intent_parser import IntentParser, ParsedIntent
from .llm_interface import LLMInterface
from .triage_router import TriageRouter
from .phase_router import PhaseRouter

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Maintains context of the current conversation and design session"""
    project_name: Optional[str] = None
    process_node: Optional[str] = None
    design_file: Optional[str] = None
    current_stage: str = "initialization"  # initialization, synthesis, placement, routing, optimization
    design_goals: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    conversation_history: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class ConversationManager:
    """
    Manages the interactive conversation flow with the chip designer.

    Handles:
    - Parsing user messages
    - Maintaining conversation context
    - Generating appropriate responses
    - Coordinating with other agent modules
    """

    def __init__(self, llm_interface: LLMInterface, intent_parser: IntentParser):
        """
        Initialize conversation manager.

        Args:
            llm_interface: LLM interface for generating responses
            intent_parser: Parser for extracting structured intents
        """
        self.llm = llm_interface
        self.parser = intent_parser
        self.context = ConversationContext()
        self.session_start = datetime.now()

        # Initialize routers
        self.triage_router = TriageRouter(llm_interface)
        self.phase_router = PhaseRouter(llm_interface)

        # Check routing configuration
        self.phase_routing_enabled = False
        self.triage_enabled = False

        try:
            from utils import load_config
            config = load_config()

            # Phase routing (preferred)
            self.phase_routing_enabled = config.get('llm', {}).get('phase_routing', {}).get('enable', False)

            # Legacy triage routing (fallback)
            self.triage_enabled = config.get('llm', {}).get('triage', {}).get('enable', False)

            if self.phase_routing_enabled:
                logger.info("Phase-based specialist routing: ENABLED ✓")
                logger.info("Using 8B specialists + 70B supervisor")
            elif self.triage_enabled:
                logger.info("Triage routing: ENABLED (legacy mode)")
            else:
                logger.info("Direct routing: No triage or phase routing")

        except Exception as e:
            logger.debug(f"Could not load routing config: {e}")

    def process_message(self, user_message: str) -> Dict:
        """
        Process user message and generate response.

        Args:
            user_message: Natural language input from user

        Returns:
            Dict containing:
                - 'response': Text response to user
                - 'intent': Parsed intent object
                - 'actions': List of actions to execute
                - 'requires_input': Whether further input is needed
        """
        # Add to conversation history
        self.context.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'role': 'user',
            'content': user_message
        })

        # Use phase router if enabled (preferred - 8B specialists)
        if self.phase_routing_enabled:
            logger.info(f"Using phase-based specialist routing: {user_message[:50]}...")
            return self._process_with_phase_router(user_message)

        # Use triage router if enabled (legacy - 3B→8B→70B)
        if self.triage_enabled:
            logger.info(f"Using legacy triage routing: {user_message[:50]}...")
            return self._process_with_triage(user_message)

        # Fall back to traditional intent parsing
        logger.info(f"Using traditional intent parsing (no routing)")
        return self._process_with_intent_parser(user_message)

    def _process_with_phase_router(self, user_message: str) -> Dict:
        """
        Process message using phase-aware specialist routing.

        Flow:
        1. Detect design phase (synthesis, placement, routing, etc.)
        2. Route to appropriate 8B specialist
        3. If specialist struggles → 70B supervisor intervenes
        4. Parse intent and determine actions
        """
        try:
            # Route to phase specialist
            phase_result = self.phase_router.route(
                user_input=user_message,
                context=self._get_context_dict()
            )

            response_text = phase_result['response']

            # Parse intent
            intent = self.parser.parse(
                user_message,
                context=self._get_context_dict()
            )

            # Update context based on intent
            self._update_context(intent)

            # Determine actions
            actions = self._determine_actions(intent)

            # Add response to history
            self.context.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'role': 'assistant',
                'content': response_text,
                'metadata': {
                    'phase': phase_result['phase'].value,
                    'model_used': phase_result['model_used'],
                    'escalated_to_70b': phase_result['escalated_to_70b']
                }
            })

            return {
                'response': response_text,
                'intent': intent,
                'actions': actions,
                'requires_input': False,
                'context': self.context,
                'routing_info': {
                    'phase': phase_result['phase'].value,
                    'model': phase_result['model_used'],
                    'escalated': phase_result['escalated_to_70b']
                }
            }

        except Exception as e:
            logger.error(f"Phase routing failed: {e}", exc_info=True)
            # Fall back to traditional processing
            return self._process_with_intent_parser(user_message)

    def _process_with_triage(self, user_message: str) -> Dict:
        """
        Process message using triage router for fast initial response.

        Flow:
        1. 3B always responds first (1-2 sec) - user gets immediate feedback
        2. Parse intent from response
        3. If escalated, append refined response
        4. Determine actions from intent
        """
        try:
            # Route through triage system
            triage_result = self.triage_router.route_streaming(
                user_message,
                context=self._get_context_dict()
            )

            # Build response from immediate + refined
            response_text = triage_result['immediate_response']

            # Add refined response if escalated
            if triage_result.get('refined_response'):
                response_text += "\n\n" + triage_result['refined_response']

            # Parse intent from the full response context
            # The triage already did complexity analysis, now extract structured intent
            intent = self.parser.parse(
                user_message,
                context=self._get_context_dict()
            )

            # Update context based on intent
            self._update_context(intent)

            # Determine actions
            actions = self._determine_actions(intent)

            # Add response to history
            self.context.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'role': 'assistant',
                'content': response_text,
                'metadata': {
                    'escalation_level': triage_result['escalation_level'].name,
                    'needs_escalation': triage_result['needs_escalation'],
                    'conversation_depth': triage_result['conversation_depth']
                }
            })

            return {
                'response': response_text,
                'intent': intent,
                'actions': actions,
                'requires_input': False,
                'context': self.context,
                'triage_info': {
                    'complexity': triage_result['escalation_level'].name,
                    'escalated': triage_result['needs_escalation']
                }
            }

        except Exception as e:
            logger.error(f"Triage routing failed: {e}", exc_info=True)
            # Fall back to traditional processing
            return self._process_with_intent_parser(user_message)

    def _process_with_intent_parser(self, user_message: str) -> Dict:
        """
        Traditional intent-first processing (fallback when triage disabled).
        """
        # Parse intent
        intent = self.parser.parse(
            user_message,
            context=self._get_context_dict()
        )

        # Validate intent
        if not self.parser.validate_intent(intent):
            return self._handle_invalid_intent(intent)

        # Update context based on intent
        self._update_context(intent)

        # Generate response
        response = self._generate_response(intent)

        # If we had an invalid or low-confidence parse earlier, ask a fast clarifier via triage
        if intent.confidence < 0.3:
            try:
                clarifier = self.llm.query(
                    prompt=(
                        "Given this user request, ask one concise clarifying question to disambiguate "
                        "process node, goals (power/performance/area), and any accelerator preference (NPU/VPU/GPU).\n"
                        f"User: {user_message}"
                    ),
                    system_prompt="You are a concise, fast triage assistant for SoC design.",
                    max_tokens=128,
                    temperature=0.2,
                    triage=True,
                    context_role="conversation"
                )
                response = {'text': clarifier, 'requires_input': True}
            except Exception:
                pass

        # Determine required actions
        actions = self._determine_actions(intent) if intent.confidence >= 0.3 else []

        # Add response to history
        self.context.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'role': 'assistant',
            'content': response['text']
        })

        return {
            'response': response['text'],
            'intent': intent,
            'actions': actions,
            'requires_input': response.get('requires_input', False),
            'context': self.context
        }

    def _get_context_dict(self) -> Dict:
        """Get current context as dictionary for intent parsing"""
        # Include recent conversation history (last 6 messages, 3 exchanges)
        recent_conversation = []
        if len(self.context.conversation_history) > 0:
            # Get last 6 messages (3 exchanges)
            recent_conversation = self.context.conversation_history[-6:]

        return {
            'project_name': self.context.project_name,
            'process_node': self.context.process_node,
            'current_stage': self.context.current_stage,
            'design_goals': self.context.design_goals,
            'last_action': self.context.last_action,
            'metrics': self.context.metrics,
            'recent_conversation': recent_conversation
        }

    def _update_context(self, intent: ParsedIntent):
        """Update conversation context based on parsed intent"""
        # Update project parameters
        if 'process_node' in intent.parameters:
            self.context.process_node = intent.parameters['process_node']

        if 'project_name' in intent.parameters:
            self.context.project_name = intent.parameters['project_name']

        if 'design_file' in intent.parameters:
            self.context.design_file = intent.parameters['design_file']

        # Update design goals
        if intent.goals:
            self.context.design_goals = [g.value for g in intent.goals]

        # Update stage based on action
        stage_map = {
            'create_project': 'initialization',
            'load_design': 'initialization',
            'synthesize': 'synthesis',
            'place': 'placement',
            'route': 'routing',
            'optimize': 'optimization',
            'analyze_timing': 'analysis',
            'analyze_power': 'analysis',
        }

        if intent.action.value in stage_map:
            self.context.current_stage = stage_map[intent.action.value]

        self.context.last_action = intent.action.value

    def _generate_response(self, intent: ParsedIntent) -> Dict:
        """
        Generate natural language response to user based on intent.

        Returns:
            Dict with 'text' and optionally 'requires_input'
        """
        # Build response based on action type
        action = intent.action.value

        if action == 'query':
            return self._response_query(intent)
        elif action == 'create_project':
            return self._response_create_project(intent)
        elif action == 'load_design':
            return self._response_load_design(intent)
        elif action == 'synthesize':
            return self._response_synthesize(intent)
        elif action in ['place', 'route', 'optimize']:
            return self._response_optimization(intent)
        elif action in ['analyze_timing', 'analyze_power']:
            return self._response_analysis(intent)
        elif action == 'adjust_floorplan':
            return self._response_floorplan_adjustment(intent)
        else:
            return {'text': "I understand. Let me work on that."}

    def _response_query(self, intent: ParsedIntent) -> Dict:
        """Generate response for informational queries with RAG support"""
        query = intent.raw_text

        # Try to use RAG for enhanced responses
        try:
            # Check if RAG is available
            rag_context = ""
            try:
                from core.rag import RAGRetriever

                rag = RAGRetriever()
                stats = rag.get_stats()

                # Only use RAG if we have indexed documents
                if stats['document_count'] > 0:
                    logger.info(f"Using RAG for query: {query[:50]}...")
                    rag_context = rag.retrieve_and_format(query, top_k=3, max_context_length=3000)

                    if rag_context:
                        logger.debug(f"Retrieved {len(rag_context)} chars of context")
                else:
                    logger.debug("RAG index empty, using LLM knowledge only")

            except Exception as rag_error:
                logger.debug(f"RAG not available: {rag_error}")

            # Build prompt with or without RAG context
            if rag_context:
                answer_prompt = f"""You are an expert chip design assistant. Use the provided documentation to answer this question accurately.

{rag_context}

Question: {query}

Provide a detailed, technical answer based on the documentation above. Include:
- Clear explanation of concepts
- Relevant technical details from the docs
- Practical considerations
- Examples if helpful

If the documentation doesn't fully answer the question, supplement with your knowledge but indicate what comes from docs vs. general knowledge."""
            else:
                answer_prompt = f"""You are an expert chip design assistant. Answer this question clearly and comprehensively:

Question: {query}

Provide a detailed, technical answer suitable for a chip designer. Include:
- Clear explanation of concepts
- Relevant technical details
- Practical considerations
- Examples if helpful

Keep the answer focused and well-structured."""

            response_text = self.llm.query(
                answer_prompt,
                system_prompt="You are an expert EDA assistant specializing in chip design.",
                max_tokens=1024,
                temperature=0.7
            )

            return {'text': response_text}

        except Exception as e:
            logger.error(f"Failed to generate query response: {e}")
            return {'text': "I'd be happy to answer that, but I'm having trouble formulating a response right now. Could you rephrase your question?"}

    def _response_create_project(self, intent: ParsedIntent) -> Dict:
        """Generate response for project creation"""
        process_node = intent.parameters.get('process_node', 'specified')
        goals = ', '.join([g.value.replace('_', ' ') for g in intent.goals])

        text = (
            f"Understood. Creating a new project with {process_node} process technology. "
            f"Primary goal: {goals}. "
            f"I'll load the appropriate technology libraries and prepare the design environment."
        )

        return {'text': text}

    def _response_load_design(self, intent: ParsedIntent) -> Dict:
        """Generate response for design loading"""
        design_file = intent.parameters.get('file_path', 'the design file')

        text = (
            f"Loading design from {design_file}. "
            f"I'll parse the Verilog/SystemVerilog and prepare it for synthesis."
        )

        return {'text': text}

    def _response_synthesize(self, intent: ParsedIntent) -> Dict:
        """Generate response for synthesis action"""
        text = (
            f"Beginning synthesis using Yosys. "
            f"I'll convert your RTL design into a gate-level netlist optimized for "
            f"{', '.join([g.value.replace('_', ' ') for g in intent.goals])}."
        )

        return {'text': text}

    def _response_optimization(self, intent: ParsedIntent) -> Dict:
        """Generate response for optimization actions"""
        action_name = intent.action.value.replace('_', ' ')

        text = (
            f"Starting {action_name} phase. "
            f"The RL optimization core will iteratively improve the design "
            f"to meet your goals. This may take some time."
        )

        return {'text': text}

    def _response_analysis(self, intent: ParsedIntent) -> Dict:
        """Generate response for analysis actions"""
        analysis_type = intent.action.value.replace('analyze_', '')

        text = (
            f"Running {analysis_type} analysis. "
            f"I'll provide detailed metrics and identify any issues."
        )

        return {'text': text}

    def _response_floorplan_adjustment(self, intent: ParsedIntent) -> Dict:
        """Generate response for floorplan adjustments"""
        adjustment = intent.parameters.get('adjustment', 'the requested changes')

        text = (
            f"Applying floorplan adjustment: {adjustment}. "
            f"I'll update the design and re-run placement to incorporate this change."
        )

        return {'text': text}

    def _handle_invalid_intent(self, intent: ParsedIntent) -> Dict:
        """Handle cases where intent parsing failed or was invalid"""
        logger.warning(f"Invalid intent parsed: {intent}")

        # Use LLM to generate clarification request
        clarification_prompt = (
            f"The user said: '{intent.raw_text}'\n"
            f"I couldn't fully understand this request in the context of chip design. "
            f"Generate a polite clarification question."
        )

        response_text = self.llm.query(
            clarification_prompt,
            system_prompt="You are a helpful EDA assistant. Ask for clarification when needed."
        )

        return {
            'response': response_text,
            'intent': intent,
            'actions': [],
            'requires_input': True
        }

    def _determine_actions(self, intent: ParsedIntent) -> List[Dict]:
        """
        Determine what backend actions need to be executed based on intent.

        Returns:
            List of action dictionaries for the orchestrator to execute
        """
        actions = []

        action_type = intent.action.value

        # QUERY actions don't trigger backend execution - just conversational
        if action_type == 'query':
            return []

        if action_type == 'create_project':
            actions.append({
                'module': 'world_model',
                'function': 'initialize_project',
                'params': intent.parameters
            })

        elif action_type == 'load_design':
            actions.append({
                'module': 'world_model',
                'function': 'load_design',
                'params': {'file_path': intent.parameters.get('file_path')}
            })

        elif action_type == 'synthesize':
            actions.append({
                'module': 'simulation_engine',
                'function': 'run_synthesis',
                'params': {'goals': [g.value for g in intent.goals]}
            })

        elif action_type == 'place':
            actions.append({
                'module': 'simulation_engine',
                'function': 'run_placement',
                'params': intent.parameters
            })

        elif action_type == 'route':
            actions.append({
                'module': 'simulation_engine',
                'function': 'run_routing',
                'params': intent.parameters
            })

        elif action_type == 'optimize':
            actions.append({
                'module': 'rl_optimizer',
                'function': 'optimize_design',
                'params': {
                    'target_metric': intent.parameters.get('target_metric'),
                    'goals': [g.value for g in intent.goals]
                }
            })

        elif action_type == 'analyze_timing':
            actions.append({
                'module': 'simulation_engine',
                'function': 'analyze_timing',
                'params': {}
            })

        elif action_type == 'analyze_power':
            actions.append({
                'module': 'simulation_engine',
                'function': 'analyze_power',
                'params': {}
            })

        return actions

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation and current project state"""
        summary_prompt = f"""
Current project state:
- Project: {self.context.project_name or 'Unnamed'}
- Process Node: {self.context.process_node or 'Not specified'}
- Stage: {self.context.current_stage}
- Goals: {', '.join(self.context.design_goals) or 'Not specified'}
- Recent actions: {self.context.last_action or 'None'}

Generate a brief, technical summary of the design session progress.
"""

        return self.llm.query(summary_prompt, system_prompt="You are an EDA assistant.")

    def reset_context(self):
        """Reset conversation context (start new project)"""
        self.context = ConversationContext()
        self.llm.clear_history()
        self.session_start = datetime.now()
