"""
Triage Router - Streaming Multi-Layer Architecture

Layer 1 (3B): ALWAYS responds first (~1-2 sec)
  - Gives immediate feedback to user
  - Asks clarifying questions if needed
  - Analyzes complexity and decides routing
  - Can handle simple queries end-to-end

Layer 2 (8B): For moderate complexity (background)
  - Deeper technical responses
  - Multi-step reasoning
  - Phase-specific expertise

Layer 3 (70B): For complex reasoning (background)
  - Architecture decisions
  - Trade-off analysis
  - Novel problem solving
"""

import logging
from typing import Dict, Optional, Tuple
from enum import IntEnum
import json

logger = logging.getLogger(__name__)


class ComplexityLevel(IntEnum):
    """Query complexity levels"""
    SIMPLE = 1      # 3B can handle
    MODERATE = 2    # Need 8B
    COMPLEX = 3     # Need 70B


class TriageRouter:
    """
    Intelligent routing system that always starts with fastest model.

    Flow:
    1. 3B responds immediately (1-2 sec)
    2. 3B decides if escalation needed
    3. If needed, larger model refines in background
    4. User always gets fast initial response
    """

    def __init__(self, llm_interface):
        """
        Initialize triage router.

        Args:
            llm_interface: LLM interface for querying models
        """
        self.llm = llm_interface

        # Conversation depth tracking
        self.conversation_depth = 0
        self.conversation_history_summary = []

        # Confusion tracking - when 3B doesn't understand, escalate
        self.consecutive_unclear_exchanges = 0
        self.confusion_threshold = 2  # Escalate after 2 unclear exchanges

        # Background model state
        self.background_model_ready = False
        self.background_guidance = None

        # Complexity indicators
        self.complexity_signals = {
            'keywords': {
                ComplexityLevel.SIMPLE: [
                    'what', 'define', 'explain', 'basic', 'simple',
                    'overview', 'introduction', 'quick'
                ],
                ComplexityLevel.MODERATE: [
                    'how', 'optimize', 'implement', 'design', 'analyze',
                    'compare', 'trade-off', 'best practice'
                ],
                ComplexityLevel.COMPLEX: [
                    'architecture', 'strategy', 'multiple', 'complex',
                    'novel', 'research', 'why choose', 'evaluate options',
                    'comprehensive', 'detailed analysis'
                ]
            },
            # Confusion signals - when 3B says these, it doesn't understand
            'confusion_patterns': [
                'what do you mean',
                'could you clarify',
                'more context',
                'more details',
                'provide more',
                'not clear',
                'unclear',
                'specify',
                'which one',
                'what are you referring to'
            ]
        }

        logger.info("Initialized TriageRouter with intelligent confusion-based escalation")

    def route_streaming(
        self,
        user_input: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Route query with streaming responses.

        Smart escalation: If 3B shows confusion, immediately go to larger model.

        Returns:
            {
                'immediate_response': str,  # From 3B (fast)
                'needs_escalation': bool,
                'escalation_level': ComplexityLevel,
                'routing_reasoning': str,
                'refined_response': str or None  # From 8B/70B (if escalated)
            }
        """
        self.conversation_depth += 1

        # Check if we should bypass 3B due to repeated confusion
        force_escalate = self.consecutive_unclear_exchanges >= self.confusion_threshold

        if force_escalate:
            logger.info(
                f"Bypassing 3B - {self.consecutive_unclear_exchanges} consecutive unclear exchanges. "
                f"Going directly to 8B for clarity."
            )
            # Go straight to 8B with guidance request
            refined = self._escalate_with_guidance(
                user_input=user_input,
                context=context,
                reason="repeated_confusion"
            )

            # Reset confusion counter since larger model is handling it
            self.consecutive_unclear_exchanges = 0

            return {
                'immediate_response': refined,
                'needs_escalation': True,
                'escalation_level': ComplexityLevel.MODERATE,
                'routing_reasoning': 'Bypassed 3B due to conversation confusion',
                'refined_response': None,
                'conversation_depth': self.conversation_depth
            }

        # LAYER 1: Call 3B first for immediate response
        triage_result = self._triage_layer(user_input, context)

        # CONFUSION DETECTION: Check if 3B is confused
        is_confused = self._detect_confusion(triage_result['quick_response'])

        if is_confused:
            self.consecutive_unclear_exchanges += 1
            logger.warning(
                f"3B shows confusion (pattern detected). "
                f"Consecutive unclear: {self.consecutive_unclear_exchanges}/{self.confusion_threshold}"
            )
            # Force escalation when confusion is detected
            triage_result['escalate'] = True
            triage_result['complexity'] = ComplexityLevel.MODERATE
            triage_result['reasoning'] = '3B confused - escalating to 8B'
        else:
            # Reset confusion counter on successful response
            self.consecutive_unclear_exchanges = 0

        result = {
            'immediate_response': triage_result['quick_response'],
            'needs_escalation': triage_result['escalate'],
            'escalation_level': triage_result['complexity'],
            'routing_reasoning': triage_result['reasoning'],
            'refined_response': None,
            'conversation_depth': self.conversation_depth
        }

        # LAYER 2/3: Escalate if needed
        if triage_result['escalate']:
            logger.info(
                f"Escalating to {triage_result['complexity'].name} "
                f"(Reason: {triage_result['reasoning']})"
            )

            refined = self._escalate_query(
                user_input=user_input,
                complexity=triage_result['complexity'],
                triage_response=triage_result['quick_response'],
                context=context
            )

            result['refined_response'] = refined

        # Track conversation evolution
        self._update_conversation_tracking(user_input, triage_result)

        return result

    def _triage_layer(
        self,
        user_input: str,
        context: Optional[Dict]
    ) -> Dict:
        """
        Layer 1: Fast 3B triage.

        Responsibilities:
        1. Give immediate response (even if just clarifying question)
        2. Analyze complexity
        3. Decide if escalation needed
        """
        # Format recent conversation for context
        recent_conv = ""
        if context and 'recent_conversation' in context and context['recent_conversation']:
            recent_conv = "\nRecent conversation:\n"
            for turn in context['recent_conversation']:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                # Truncate long messages
                if len(content) > 200:
                    content = content[:197] + "..."
                recent_conv += f"{role.capitalize()}: {content}\n"

        triage_prompt = f"""You are a fast triage assistant for a chip design AI system.

Analyze this query and provide:
1. A QUICK initial response (can be a clarifying question if query is vague)
2. Complexity assessment: SIMPLE, MODERATE, or COMPLEX
3. Whether to escalate to larger model
4. Brief reasoning

Conversation depth: {self.conversation_depth}
{recent_conv}
Current user query: {user_input}

Respond in JSON:
{{
    "quick_response": "your immediate response or clarifying question",
    "complexity": "SIMPLE|MODERATE|COMPLEX",
    "escalate": true/false,
    "reasoning": "why this complexity level"
}}

IMPORTANT:
- USE THE RECENT CONVERSATION CONTEXT to understand follow-up questions like "both", "yes", "tell me more", "my options"
- If query is vague/unclear AND no conversation context: ask clarifying question, set complexity=SIMPLE, escalate=false
- If query references previous conversation: use that context to answer, decide escalation based on complexity
- If you can fully answer: give answer, escalate=false
- If needs deeper expertise: give initial thoughts, escalate=true
- Simple definitions: complexity=SIMPLE, escalate=false
- Technical "how-to": complexity=MODERATE, escalate=true (to 8B)
- Architecture decisions: complexity=COMPLEX, escalate=true (to 70B)
"""

        # Force use of 3B for triage
        original_model = self.llm.model_name
        self.llm.model_name = "llama3.2:3b"

        try:
            response = self.llm.query(triage_prompt)
            self.llm.model_name = original_model

            # Parse JSON
            result = json.loads(response)

            complexity_map = {
                'SIMPLE': ComplexityLevel.SIMPLE,
                'MODERATE': ComplexityLevel.MODERATE,
                'COMPLEX': ComplexityLevel.COMPLEX
            }

            return {
                'quick_response': result.get('quick_response', 'Let me help you with that.'),
                'complexity': complexity_map.get(result.get('complexity', 'SIMPLE'), ComplexityLevel.SIMPLE),
                'escalate': result.get('escalate', False),
                'reasoning': result.get('reasoning', 'Triage assessment')
            }

        except Exception as e:
            logger.warning(f"Triage parsing failed: {e}")
            self.llm.model_name = original_model

            # Fallback: use heuristics
            return self._fallback_triage(user_input, context)

    def _fallback_triage(
        self,
        user_input: str,
        context: Optional[Dict]
    ) -> Dict:
        """
        Fallback triage using heuristics when JSON parsing fails.

        This fallback actually tries to answer the query directly instead of
        just giving generic "provide more details" responses.
        """
        user_lower = user_input.lower()

        # Analyze complexity signals
        complexity_scores = {
            ComplexityLevel.SIMPLE: 0,
            ComplexityLevel.MODERATE: 0,
            ComplexityLevel.COMPLEX: 0
        }

        for level, keywords in self.complexity_signals['keywords'].items():
            for keyword in keywords:
                if keyword in user_lower:
                    complexity_scores[level] += 1

        # Additional signals
        word_count = len(user_input.split())
        question_marks = user_input.count('?')
        has_multiple_questions = question_marks > 1

        # Adjust scores
        if word_count > 100:
            complexity_scores[ComplexityLevel.COMPLEX] += 2
        elif word_count > 50:
            complexity_scores[ComplexityLevel.MODERATE] += 1

        if has_multiple_questions:
            complexity_scores[ComplexityLevel.COMPLEX] += 1

        # Conversation depth increases complexity
        if self.conversation_depth > 5:
            complexity_scores[ComplexityLevel.MODERATE] += 1
        if self.conversation_depth > 10:
            complexity_scores[ComplexityLevel.COMPLEX] += 1

        # Determine complexity
        detected_complexity = max(complexity_scores, key=complexity_scores.get)

        # Decide escalation based on complexity
        escalate = detected_complexity > ComplexityLevel.SIMPLE

        # Format recent conversation for context
        recent_conv = ""
        if context and 'recent_conversation' in context and context['recent_conversation']:
            recent_conv = "\nRecent conversation:\n"
            for turn in context['recent_conversation']:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                if len(content) > 150:
                    content = content[:147] + "..."
                recent_conv += f"{role.capitalize()}: {content}\n"
            recent_conv += "\n"

        # Actually try to answer using 3B model (non-JSON mode)
        try:
            simple_prompt = f"""You are a helpful chip design assistant. Answer this question concisely:

{recent_conv}Current user query: {user_input}

Use the conversation context above to understand follow-up questions like "both", "yes", "tell me more", "my options".
Give a brief, helpful answer (2-3 sentences). If the question needs more detail, say so and offer to provide more information."""

            quick_response = self.llm.query(
                simple_prompt,
                system_prompt="You are a concise, helpful chip design assistant.",
                max_tokens=256,
                temperature=0.7
            )

            # If the response looks too generic, escalate
            generic_phrases = ["more details", "more context", "more information", "could you provide"]
            if any(phrase in quick_response.lower() for phrase in generic_phrases):
                escalate = True

        except Exception as e:
            logger.warning(f"Fallback query also failed: {e}")
            # Last resort: simple pattern-based response
            if any(word in user_lower for word in ['explain', 'what', 'define']):
                quick_response = "Let me explain that. I'll provide detailed information..."
                escalate = True
            elif any(word in user_lower for word in ['how', 'implement', 'create']):
                quick_response = "Here's how to do that. Let me provide implementation details..."
                escalate = True
            else:
                quick_response = "I understand. Let me help you with that..."
                escalate = True

        return {
            'quick_response': quick_response,
            'complexity': detected_complexity,
            'escalate': escalate,
            'reasoning': f'Fallback: depth={self.conversation_depth}, words={word_count}, complexity={detected_complexity.name}'
        }

    def _detect_confusion(self, response: str) -> bool:
        """
        Detect if the response shows confusion/uncertainty.

        Returns True if response contains confusion patterns like:
        - "What do you mean?"
        - "Could you clarify?"
        - "More context needed"
        """
        response_lower = response.lower()

        for pattern in self.complexity_signals['confusion_patterns']:
            if pattern in response_lower:
                logger.debug(f"Confusion pattern detected: '{pattern}'")
                return True

        return False

    def _escalate_with_guidance(
        self,
        user_input: str,
        context: Optional[Dict],
        reason: str
    ) -> str:
        """
        Escalate directly to 8B with request for conversation guidance.

        When 3B is repeatedly confused, 8B should:
        1. Understand the conversation trajectory
        2. Provide helpful structure/guidance
        3. Orient the conversation toward what user wants
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
            recent_conv += "\n"

        guidance_prompt = f"""You are an expert chip design assistant. The conversation has become unclear - provide clear guidance.

{recent_conv}Current user query: {user_input}

IMPORTANT: The small model couldn't understand this conversation flow. You need to:
1. Analyze what the user is trying to accomplish from the conversation history
2. Provide a clear, structured response that addresses their likely intent
3. Offer specific options or next steps to orient the conversation

Don't ask for more clarification - use the context to infer what they want and provide helpful guidance.
If they said "both", "tell me more", "my options" - look at the conversation history to understand what they're referring to.

Provide a comprehensive answer that gets the conversation back on track."""

        # Use 8B for guidance
        original_model = self.llm.model_name
        self.llm.model_name = "llama3:8b"

        try:
            guidance = self.llm.query(guidance_prompt)
            logger.info("8B provided conversation guidance after confusion")
            return guidance
        finally:
            self.llm.model_name = original_model

    def _escalate_query(
        self,
        user_input: str,
        complexity: ComplexityLevel,
        triage_response: str,
        context: Optional[Dict]
    ) -> str:
        """
        Escalate to appropriate larger model for refined response.
        """
        # Select model based on complexity
        if complexity == ComplexityLevel.MODERATE:
            model = "llama3:8b"
        else:  # COMPLEX
            model = "llama3:70b"

        # Format recent conversation for context
        recent_conv = ""
        if context and 'recent_conversation' in context and context['recent_conversation']:
            recent_conv = "\nRecent conversation:\n"
            for turn in context['recent_conversation']:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                # Keep more context for larger models
                if len(content) > 300:
                    content = content[:297] + "..."
                recent_conv += f"{role.capitalize()}: {content}\n"
            recent_conv += "\n"

        # Build escalation prompt
        escalation_prompt = f"""You are an expert chip design assistant providing a detailed response.

The user was initially told: "{triage_response}"

Now provide a comprehensive, technical response.

Conversation depth: {self.conversation_depth}
{recent_conv}Current user query: {user_input}

IMPORTANT: Use the conversation context above to understand follow-up questions and references to previous discussion.
If the user said vague things like "both", "tell me more", "my options" - use the conversation history to understand what they mean.

Provide detailed, actionable guidance with specific examples and recommendations."""

        # Query larger model
        original_model = self.llm.model_name
        self.llm.model_name = model

        try:
            refined_response = self.llm.query(escalation_prompt)
            return refined_response
        finally:
            self.llm.model_name = original_model

    def _update_conversation_tracking(self, user_input: str, triage_result: Dict):
        """
        Track conversation evolution to inform future routing.
        """
        summary = {
            'turn': self.conversation_depth,
            'complexity': triage_result['complexity'].name,
            'escalated': triage_result['escalate'],
            'query_length': len(user_input)
        }

        self.conversation_history_summary.append(summary)

        # Keep last 20 turns
        if len(self.conversation_history_summary) > 20:
            self.conversation_history_summary.pop(0)

    def get_routing_stats(self) -> Dict:
        """
        Get statistics on routing decisions.
        """
        if not self.conversation_history_summary:
            return {
                'total_turns': 0,
                'escalations': 0,
                'avg_complexity': 0
            }

        escalations = sum(1 for t in self.conversation_history_summary if t['escalated'])

        complexity_values = {
            'SIMPLE': 1,
            'MODERATE': 2,
            'COMPLEX': 3
        }

        avg_complexity = sum(
            complexity_values[t['complexity']]
            for t in self.conversation_history_summary
        ) / len(self.conversation_history_summary)

        return {
            'total_turns': self.conversation_depth,
            'escalations': escalations,
            'escalation_rate': escalations / max(self.conversation_depth, 1),
            'avg_complexity': avg_complexity,
            'conversation_history': self.conversation_history_summary[-5:]  # Last 5 turns
        }

    def should_proactively_escalate(self) -> bool:
        """
        Determine if conversation has naturally progressed to need larger model.

        Triggers:
        - 3+ consecutive moderate complexity queries
        - Conversation depth > 10
        - Pattern of increasing complexity
        """
        if self.conversation_depth < 5:
            return False

        recent = self.conversation_history_summary[-5:]

        # Check for sustained moderate/complex queries
        moderate_or_complex = sum(
            1 for t in recent
            if t['complexity'] in ['MODERATE', 'COMPLEX']
        )

        if moderate_or_complex >= 3:
            logger.info("Proactive escalation: sustained complex conversation")
            return True

        # Check for depth
        if self.conversation_depth > 10:
            logger.info("Proactive escalation: deep conversation")
            return True

        # Check for increasing complexity trend
        if len(recent) >= 3:
            complexity_values = {'SIMPLE': 1, 'MODERATE': 2, 'COMPLEX': 3}
            trend = [complexity_values[t['complexity']] for t in recent[-3:]]

            if trend == sorted(trend):  # Increasing
                logger.info("Proactive escalation: increasing complexity trend")
                return True

        return False

    def reset_conversation(self):
        """Reset conversation tracking for new session."""
        self.conversation_depth = 0
        self.conversation_history_summary = []
        logger.info("Conversation context reset")
