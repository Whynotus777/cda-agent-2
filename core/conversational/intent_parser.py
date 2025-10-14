"""
Intent Parser Module

Translates natural language commands into structured goals and actions.
Uses LLM to extract:
- Project parameters (process node, design goals)
- Design constraints (power, performance, area priorities)
- Specific actions (synthesis, placement, routing, analysis)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class DesignGoal(Enum):
    """Primary design optimization goals"""
    MINIMIZE_POWER = "minimize_power"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MINIMIZE_AREA = "minimize_area"
    BALANCED = "balanced"


class ActionType(Enum):
    """Types of actions the agent can perform"""
    QUERY = "query"  # Informational questions (explain, what is, tell me about)
    CREATE_PROJECT = "create_project"
    LOAD_DESIGN = "load_design"
    SYNTHESIZE = "synthesize"
    PLACE = "place"
    ROUTE = "route"
    ANALYZE_TIMING = "analyze_timing"
    ANALYZE_POWER = "analyze_power"
    OPTIMIZE = "optimize"
    ADJUST_FLOORPLAN = "adjust_floorplan"
    EXPORT_GDSII = "export_gdsii"


@dataclass
class ParsedIntent:
    """Structured representation of user intent"""
    action: ActionType
    parameters: Dict[str, any]
    goals: List[DesignGoal]
    confidence: float
    raw_text: str


class IntentParser:
    """
    Parses natural language input into structured intents using LLM.

    Example:
        "Let's start a new 7nm design for a low-power microcontroller"
        -> ParsedIntent(
            action=CREATE_PROJECT,
            parameters={'process_node': '7nm', 'design_type': 'microcontroller'},
            goals=[MINIMIZE_POWER],
            confidence=0.95
        )
    """

    def __init__(self, llm_interface):
        """
        Initialize the intent parser.

        Args:
            llm_interface: Interface to the LLM (Ollama)
        """
        self.llm = llm_interface
        self._init_prompts()

    def _init_prompts(self):
        """Initialize system prompts for intent parsing"""
        self.system_prompt = """You are an expert EDA (Electronic Design Automation) assistant.
Your job is to parse chip designer requests into structured commands.

Extract the following information:
1. ACTION: What does the user want to do?
   - query: Informational questions (explain, what is, tell me about, describe)
   - create_project: Start a new chip design project
   - synthesize, place, route, analyze: Execute design phases
2. PARAMETERS: Specific values like process node (7nm, 12nm), file paths, design type
3. GOALS: What should be optimized? (power, performance, area, or balanced)
4. CONSTRAINTS: Any specific requirements or limitations

IMPORTANT: If the user is asking for information/explanation, use action="query" instead of create_project.

Respond in JSON format with high accuracy."""

    def parse(self, user_input: str, context: Optional[Dict] = None) -> ParsedIntent:
        """
        Parse user input into structured intent.

        Args:
            user_input: Natural language command from user
            context: Optional context from conversation history

        Returns:
            ParsedIntent object with structured information
        """
        # Heuristic fast-path: obvious informational queries → QUERY with high confidence
        text_l = (user_input or "").strip().lower()

        # Common typo normalization
        typo_map = {
            "palcement": "placement",
            "overviwe": "overview",
            "sumary": "summary",
            "explian": "explain",
        }
        for typo, correct in typo_map.items():
            if typo in text_l:
                text_l = text_l.replace(typo, correct)
                user_input = user_input.replace(typo, correct)

        # Expanded query keywords
        query_keywords = [
            "explain", "what is", "what's", "tell me", "describe", "overview",
            "walk me through", "how does", "how do", "stages", "steps", "everything",
            "all of it", "what are", "help me understand", "summary", "give me",
            "show me", "teach me", "learn about", "understand", "info about",
            "information", "details about"
        ]

        # Also check if query starts with question word
        question_starters = ["what", "how", "why", "when", "where", "who", "which", "explain", "describe", "tell"]
        starts_with_question = any(text_l.startswith(q) for q in question_starters)

        # Check if it's a short query without design action words
        design_actions = ["create", "build", "synthesize", "place", "route", "optimize", "run", "execute", "start"]
        has_design_action = any(action in text_l for action in design_actions)

        # If it matches query patterns and doesn't have design actions, treat as QUERY
        if (any(k in text_l for k in query_keywords) or starts_with_question) and not has_design_action:
            return ParsedIntent(
                action=ActionType.QUERY,
                parameters={},
                goals=[DesignGoal.BALANCED],
                confidence=0.95,
                raw_text=user_input
            )

        # Construct prompt with context
        prompt = self._build_parse_prompt(user_input, context)

        # Get LLM response with graceful fallback on failure/timeouts
        try:
            llm_response = self.llm.query(prompt, system_prompt=self.system_prompt)
            # Parse LLM response into structured format
            parsed = self._parse_llm_response(llm_response, user_input)
            return parsed
        except Exception:
            # Fallback: return a low-confidence, non-committal intent to trigger clarification
            return ParsedIntent(
                action=ActionType.CREATE_PROJECT,
                parameters={},  # intentionally empty so validation fails and asks for clarification
                goals=[DesignGoal.BALANCED],
                confidence=0.0,
                raw_text=user_input
            )

    def _build_parse_prompt(self, user_input: str, context: Optional[Dict]) -> str:
        """Build the full prompt for LLM"""
        prompt_parts = []

        # Format recent conversation for context understanding
        if context and 'recent_conversation' in context and context['recent_conversation']:
            prompt_parts.append("Recent conversation:")
            for turn in context['recent_conversation']:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                if len(content) > 150:
                    content = content[:147] + "..."
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("\nUse this conversation context to understand follow-up questions like 'both', 'yes', 'my options'.\n")

        if context:
            # Only include non-conversation parts of context
            context_filtered = {k: v for k, v in context.items() if k != 'recent_conversation'}
            if context_filtered:
                prompt_parts.append(f"Current project context: {context_filtered}")

        prompt_parts.append(f"User request: {user_input}")
        prompt_parts.append("\nParse this into structured format (JSON):")

        return "\n".join(prompt_parts)

    def _parse_llm_response(self, llm_response: str, raw_input: str) -> ParsedIntent:
        """
        Convert LLM JSON response into ParsedIntent object.

        TODO: Implement robust JSON parsing with error handling
        """
        import json

        try:
            data = json.loads(llm_response)

            return ParsedIntent(
                action=ActionType(data.get('action', 'create_project')),
                parameters=data.get('parameters', {}),
                goals=[DesignGoal(g) for g in data.get('goals', ['balanced'])],
                confidence=data.get('confidence', 0.0),
                raw_text=raw_input
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to safe defaults
            return ParsedIntent(
                action=ActionType.CREATE_PROJECT,
                parameters={},
                goals=[DesignGoal.BALANCED],
                confidence=0.0,
                raw_text=raw_input
            )

    def validate_intent(self, intent: ParsedIntent) -> bool:
        """
        Validate that the parsed intent is valid and executable.

        Returns:
            True if intent is valid, False otherwise
        """
        # Check if required parameters are present for the action
        required_params = self._get_required_params(intent.action)

        for param in required_params:
            if param not in intent.parameters:
                return False

        return intent.confidence > 0.3  # Confidence threshold

    def _get_required_params(self, action: ActionType) -> List[str]:
        """Get required parameters for a given action"""
        param_map = {
            ActionType.QUERY: [],  # No params required for informational queries
            ActionType.CREATE_PROJECT: ['process_node'],
            ActionType.LOAD_DESIGN: ['file_path'],
            ActionType.SYNTHESIZE: [],
            ActionType.PLACE: [],
            ActionType.ROUTE: [],
            ActionType.ANALYZE_TIMING: [],
            ActionType.ANALYZE_POWER: [],
            ActionType.OPTIMIZE: ['target_metric'],
            ActionType.ADJUST_FLOORPLAN: ['adjustment'],
            ActionType.EXPORT_GDSII: ['output_path'],
        }

        return param_map.get(action, [])
