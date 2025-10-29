"""
Base Agent Class

Abstract base class for all RTL multi-agents.
Provides common interface and utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """Standardized agent output format"""
    agent_id: str
    agent_name: str
    success: bool
    confidence: float  # 0.0-1.0
    output_data: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class BaseAgent(ABC):
    """
    Abstract base class for all RTL agents.

    All agents (A1-A6) must inherit from this class and implement
    the abstract methods.
    """

    def __init__(self, agent_id: str, agent_name: str, config: Optional[Dict] = None):
        """
        Initialize base agent.

        Args:
            agent_id: Unique identifier (e.g., "A6", "A1")
            agent_name: Human-readable name
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config = config or {}
        self.run_id = str(uuid.uuid4())

        logger.info(f"Initialized {self.agent_name} (ID: {self.agent_id})")

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Main processing method - must be implemented by subclasses.

        Args:
            input_data: Input dictionary conforming to agent's schema

        Returns:
            AgentOutput with results
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against agent's schema.

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for this agent's input.

        Returns:
            JSON schema dictionary
        """
        pass

    def log_to_run_db(self, run_data: Dict[str, Any]):
        """
        Log execution data to Run DB.

        Args:
            run_data: Data to log
        """
        # TODO: Implement Run DB logging
        logger.debug(f"[{self.agent_id}] Run logged: {run_data.get('request_id', 'N/A')}")

    def calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for output.

        Args:
            result: Result data

        Returns:
            Confidence score 0.0-1.0
        """
        # Default implementation - can be overridden
        if not result:
            return 0.0

        success = result.get('success', False)
        errors = len(result.get('errors', []))
        warnings = len(result.get('warnings', []))

        base_confidence = 1.0 if success else 0.3
        penalty = (errors * 0.15) + (warnings * 0.05)

        return max(0.0, min(1.0, base_confidence - penalty))

    def create_output(
        self,
        success: bool,
        output_data: Dict[str, Any],
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        execution_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Helper to create standardized AgentOutput.

        Args:
            success: Whether operation succeeded
            output_data: Main output data
            errors: List of error messages
            warnings: List of warning messages
            execution_time_ms: Execution time in milliseconds
            metadata: Additional metadata

        Returns:
            AgentOutput object
        """
        result = {
            'success': success,
            'errors': errors or [],
            'warnings': warnings or []
        }

        confidence = self.calculate_confidence(result)

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            success=success,
            confidence=confidence,
            output_data=output_data,
            errors=errors or [],
            warnings=warnings or [],
            execution_time_ms=execution_time_ms,
            metadata=metadata or {}
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.agent_name}>"
