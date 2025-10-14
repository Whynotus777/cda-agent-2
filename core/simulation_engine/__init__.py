"""
Simulation & Analysis Engine

Orchestrates specialized EDA tools for chip design tasks:
- Synthesis (Yosys)
- Placement (DREAMPlace)
- Routing (TritonRoute)
- Timing Analysis (OpenSTA)
- Power Analysis
"""

from .synthesis import SynthesisEngine
from .placement import PlacementEngine
from .routing import RoutingEngine
from .timing_analysis import TimingAnalyzer
from .power_analysis import PowerAnalyzer


class SimulationEngine:
    """
    Unified interface to all EDA simulation tools.

    Provides easy access to synthesis, placement, routing, timing, and power analysis.
    """

    def __init__(self):
        """Initialize all simulation engines"""
        self.synthesis = SynthesisEngine()
        self.placement = PlacementEngine()
        self.routing = RoutingEngine()
        self.timing = TimingAnalyzer()
        self.power = PowerAnalyzer()


__all__ = [
    'SimulationEngine',
    'SynthesisEngine',
    'PlacementEngine',
    'RoutingEngine',
    'TimingAnalyzer',
    'PowerAnalyzer'
]
