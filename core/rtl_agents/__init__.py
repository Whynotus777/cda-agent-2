"""
RTL Multi-Agent System

Front-end agents for RTL design, verification, and optimization.
"""

from .base_agent import BaseAgent, AgentOutput
from .a1_spec_to_rtl import A1_SpecToRTLGenerator
from .a6_eda_command import A6_EDACommandCopilot
from .a4_lint_cdc import A4_LintCDCAssistant
from .a2_boilerplate_gen import A2_BoilerplateGenerator
from .a3_constraint_synth import A3_ConstraintSynthesizer
from .a5_style_review import A5_StyleReviewCopilot

__all__ = [
    'BaseAgent',
    'AgentOutput',
    'A1_SpecToRTLGenerator',
    'A6_EDACommandCopilot',
    'A4_LintCDCAssistant',
    'A2_BoilerplateGenerator',
    'A3_ConstraintSynthesizer',
    'A5_StyleReviewCopilot',
]
