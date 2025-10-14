"""
World Model & Knowledge Base

This module contains the agent's understanding of chip design "physics":
- Technology libraries (standard cells, timing, power)
- Design parsers (Verilog, LEF/DEF, SDC)
- Design Rule Checking (DRC) rules
- Physical and electrical constraints
"""

from .tech_library import TechLibrary
from .design_parser import DesignParser
from .rule_engine import RuleEngine
from .design_state import DesignState


class WorldModel:
    """
    Unified interface to all world model components.

    Provides easy access to tech libraries, design parsers, and rule engines.
    """

    def __init__(self, process_node: str = "7nm"):
        """
        Initialize world model.

        Args:
            process_node: Technology process node (e.g., "7nm", "14nm")
        """
        self.process_node = process_node
        self.tech_library = TechLibrary(process_node=process_node)
        self.design_parser = DesignParser()
        self.rule_engine = RuleEngine(process_node=process_node)


__all__ = ['WorldModel', 'TechLibrary', 'DesignParser', 'RuleEngine', 'DesignState']
