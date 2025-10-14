"""
Design State Module

Maintains the current state of the chip design throughout the flow.
Tracks metrics, status, and results from each stage.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DesignStage(Enum):
    """Stages in the design flow"""
    UNINITIALIZED = "uninitialized"
    RTL_LOADED = "rtl_loaded"
    SYNTHESIZED = "synthesized"
    PLACED = "placed"
    CTS_DONE = "cts_done"  # Clock Tree Synthesis
    ROUTED = "routed"
    OPTIMIZED = "optimized"
    VERIFIED = "verified"
    SIGNOFF_READY = "signoff_ready"


@dataclass
class TimingMetrics:
    """Timing analysis results"""
    clock_period: Optional[float] = None  # Target clock period (ns)
    worst_negative_slack: Optional[float] = None  # WNS (ns)
    total_negative_slack: Optional[float] = None  # TNS (ns)
    worst_path_delay: Optional[float] = None  # Critical path delay (ns)
    setup_violations: int = 0
    hold_violations: int = 0
    max_frequency: Optional[float] = None  # MHz


@dataclass
class PowerMetrics:
    """Power analysis results"""
    total_power: Optional[float] = None  # Total power (mW)
    dynamic_power: Optional[float] = None  # Dynamic/switching power (mW)
    static_power: Optional[float] = None  # Leakage power (mW)
    internal_power: Optional[float] = None  # Cell internal power (mW)
    switching_power: Optional[float] = None  # Net switching power (mW)


@dataclass
class AreaMetrics:
    """Area metrics"""
    total_area: Optional[float] = None  # Total die area (um^2)
    cell_area: Optional[float] = None  # Standard cell area (um^2)
    macro_area: Optional[float] = None  # Hard macro area (um^2)
    utilization: Optional[float] = None  # Area utilization (0.0-1.0)
    aspect_ratio: Optional[float] = None  # Width/Height ratio


@dataclass
class RoutingMetrics:
    """Routing quality metrics"""
    total_wirelength: Optional[float] = None  # Total wire length (um)
    via_count: Dict[str, int] = field(default_factory=dict)  # Via counts per layer
    congestion_overflow: Optional[float] = None  # Routing congestion
    drc_violations: int = 0  # Design rule violations
    avg_wire_density: Optional[float] = None  # Average wire density


@dataclass
class QualityMetrics:
    """Overall quality metrics (PPA - Power, Performance, Area)"""
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    power: PowerMetrics = field(default_factory=PowerMetrics)
    area: AreaMetrics = field(default_factory=AreaMetrics)
    routing: RoutingMetrics = field(default_factory=RoutingMetrics)

    # Composite scores (normalized 0-1, higher is better)
    timing_score: Optional[float] = None
    power_score: Optional[float] = None
    area_score: Optional[float] = None
    overall_score: Optional[float] = None


class DesignState:
    """
    Maintains complete state of the chip design.

    This is the "world model" that the RL agent observes and acts upon.
    """

    def __init__(self, project_name: str):
        """
        Initialize design state.

        Args:
            project_name: Name of the design project
        """
        self.project_name = project_name
        self.stage = DesignStage.UNINITIALIZED
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        # Design files
        self.rtl_files: List[str] = []
        self.constraint_files: List[str] = []
        self.netlist_file: Optional[str] = None
        self.def_file: Optional[str] = None
        self.gds_file: Optional[str] = None

        # Design parameters
        self.process_node: Optional[str] = None
        self.top_module: Optional[str] = None
        self.clock_period: Optional[float] = None

        # Design statistics
        self.cell_count: int = 0
        self.net_count: int = 0
        self.pin_count: int = 0
        self.macro_count: int = 0

        # Quality metrics
        self.metrics = QualityMetrics()

        # Optimization history
        self.optimization_history: List[Dict] = []

        # Warnings and errors
        self.warnings: List[str] = []
        self.errors: List[str] = []

        logger.info(f"Initialized DesignState for project: {project_name}")

    def update_stage(self, new_stage: DesignStage):
        """Update design stage"""
        logger.info(f"Design stage: {self.stage.value} -> {new_stage.value}")
        self.stage = new_stage
        self.last_updated = datetime.now()

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update design metrics from analysis results.

        Args:
            metrics: Dictionary of metric updates
        """
        self.last_updated = datetime.now()

        # Update timing metrics
        if 'timing' in metrics:
            timing = metrics['timing']
            if 'wns' in timing:
                self.metrics.timing.worst_negative_slack = timing['wns']
            if 'tns' in timing:
                self.metrics.timing.total_negative_slack = timing['tns']
            if 'worst_path_delay' in timing:
                self.metrics.timing.worst_path_delay = timing['worst_path_delay']
            if 'setup_violations' in timing:
                self.metrics.timing.setup_violations = timing['setup_violations']
            if 'hold_violations' in timing:
                self.metrics.timing.hold_violations = timing['hold_violations']

        # Update power metrics
        if 'power' in metrics:
            power = metrics['power']
            if 'total_power' in power:
                self.metrics.power.total_power = power['total_power']
            if 'dynamic_power' in power:
                self.metrics.power.dynamic_power = power['dynamic_power']
            if 'static_power' in power:
                self.metrics.power.static_power = power['static_power']

        # Update area metrics
        if 'area' in metrics:
            area = metrics['area']
            if 'total_area' in area:
                self.metrics.area.total_area = area['total_area']
            if 'cell_area' in area:
                self.metrics.area.cell_area = area['cell_area']
            if 'utilization' in area:
                self.metrics.area.utilization = area['utilization']

        # Update routing metrics
        if 'routing' in metrics:
            routing = metrics['routing']
            if 'total_wirelength' in routing:
                self.metrics.routing.total_wirelength = routing['total_wirelength']
            if 'via_count' in routing:
                self.metrics.routing.via_count = routing['via_count']
            if 'drc_violations' in routing:
                self.metrics.routing.drc_violations = routing['drc_violations']

        # Recalculate composite scores
        self._calculate_scores()

        logger.info(f"Updated metrics: {self.get_metrics_summary()}")

    def _calculate_scores(self):
        """
        Calculate normalized quality scores (0-1, higher is better).

        These scores are used by the RL agent as rewards.
        """
        # Timing score (based on slack)
        if self.metrics.timing.worst_negative_slack is not None:
            if self.metrics.timing.worst_negative_slack >= 0:
                # Positive slack is good
                self.metrics.timing_score = min(1.0, 0.8 + self.metrics.timing.worst_negative_slack / 10)
            else:
                # Negative slack is bad
                self.metrics.timing_score = max(0.0, 0.5 + self.metrics.timing.worst_negative_slack / 10)

        # Power score (lower power is better, normalize to 0-1)
        if self.metrics.power.total_power is not None:
            # Assume target power budget (would come from design goals)
            target_power = 1000.0  # mW
            if self.metrics.power.total_power <= target_power:
                self.metrics.power_score = 1.0 - (self.metrics.power.total_power / target_power) * 0.5
            else:
                # Exceeds budget
                self.metrics.power_score = 0.5 * (target_power / self.metrics.power.total_power)

        # Area score (based on utilization)
        if self.metrics.area.utilization is not None:
            # Target utilization: 0.6-0.8 is optimal
            util = self.metrics.area.utilization
            if 0.6 <= util <= 0.8:
                self.metrics.area_score = 1.0
            elif util < 0.6:
                # Underutilized (wasted area)
                self.metrics.area_score = util / 0.6
            else:
                # Overutilized (congestion)
                self.metrics.area_score = max(0.0, 1.0 - (util - 0.8) * 2)

        # Overall score (weighted combination)
        scores = []
        if self.metrics.timing_score is not None:
            scores.append(self.metrics.timing_score * 0.4)  # Timing most important
        if self.metrics.power_score is not None:
            scores.append(self.metrics.power_score * 0.35)  # Power second
        if self.metrics.area_score is not None:
            scores.append(self.metrics.area_score * 0.25)  # Area third

        if scores:
            self.metrics.overall_score = sum(scores)

    def log_optimization_step(self, iteration: int, action: str, metrics: Dict):
        """
        Log an optimization step for history tracking.

        Args:
            iteration: Optimization iteration number
            action: Action taken by the RL agent
            metrics: Resulting metrics
        """
        self.optimization_history.append({
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'metrics': metrics.copy(),
            'overall_score': self.metrics.overall_score
        })

    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(f"[{datetime.now().isoformat()}] {message}")
        logger.warning(message)

    def add_error(self, message: str):
        """Add an error message"""
        self.errors.append(f"[{datetime.now().isoformat()}] {message}")
        logger.error(message)

    def is_timing_clean(self) -> bool:
        """Check if design meets timing requirements"""
        return (
            self.metrics.timing.worst_negative_slack is not None
            and self.metrics.timing.worst_negative_slack >= 0
            and self.metrics.timing.setup_violations == 0
            and self.metrics.timing.hold_violations == 0
        )

    def is_drc_clean(self) -> bool:
        """Check if design has no DRC violations"""
        return self.metrics.routing.drc_violations == 0

    def is_signoff_ready(self) -> bool:
        """Check if design is ready for signoff"""
        return (
            self.stage in [DesignStage.VERIFIED, DesignStage.SIGNOFF_READY]
            and self.is_timing_clean()
            and self.is_drc_clean()
            and len(self.errors) == 0
        )

    def get_metrics_summary(self) -> Dict:
        """Get summary of current metrics"""
        return {
            'stage': self.stage.value,
            'timing': {
                'wns': self.metrics.timing.worst_negative_slack,
                'tns': self.metrics.timing.total_negative_slack,
                'violations': (
                    self.metrics.timing.setup_violations +
                    self.metrics.timing.hold_violations
                )
            },
            'power': {
                'total': self.metrics.power.total_power,
                'dynamic': self.metrics.power.dynamic_power,
                'static': self.metrics.power.static_power
            },
            'area': {
                'total': self.metrics.area.total_area,
                'utilization': self.metrics.area.utilization
            },
            'routing': {
                'wirelength': self.metrics.routing.total_wirelength,
                'drc_violations': self.metrics.routing.drc_violations
            },
            'scores': {
                'timing': self.metrics.timing_score,
                'power': self.metrics.power_score,
                'area': self.metrics.area_score,
                'overall': self.metrics.overall_score
            }
        }

    def get_state_vector(self) -> List[float]:
        """
        Get design state as a vector for RL agent observation.

        Returns:
            Normalized state vector
        """
        state = []

        # Timing features (normalized)
        state.append(self.metrics.timing.worst_negative_slack or 0.0)
        state.append(float(self.metrics.timing.setup_violations))
        state.append(float(self.metrics.timing.hold_violations))

        # Power features (normalized to mW)
        state.append((self.metrics.power.total_power or 0.0) / 1000.0)
        state.append((self.metrics.power.dynamic_power or 0.0) / 1000.0)

        # Area features
        state.append(self.metrics.area.utilization or 0.0)
        state.append((self.metrics.area.total_area or 0.0) / 1e6)  # um^2 to mm^2

        # Routing features
        state.append((self.metrics.routing.total_wirelength or 0.0) / 1000.0)  # um to mm
        state.append(float(self.metrics.routing.drc_violations))

        # Composite scores
        state.append(self.metrics.timing_score or 0.0)
        state.append(self.metrics.power_score or 0.0)
        state.append(self.metrics.area_score or 0.0)

        return state

    def export_state(self) -> Dict:
        """Export complete state as dictionary (for saving/checkpointing)"""
        return {
            'project_name': self.project_name,
            'stage': self.stage.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'process_node': self.process_node,
            'top_module': self.top_module,
            'metrics': self.get_metrics_summary(),
            'optimization_history': self.optimization_history,
            'warnings': self.warnings,
            'errors': self.errors
        }
