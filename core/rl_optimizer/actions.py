"""
Action Space Definition

Defines all possible actions the RL agent can take to optimize the design.
"""

from typing import Dict, List, Callable
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Enumeration of all possible actions"""
    # Placement density adjustments
    INCREASE_DENSITY = 0
    DECREASE_DENSITY = 1

    # DREAMPlace optimization focus
    OPTIMIZE_WIRELENGTH = 2
    OPTIMIZE_ROUTABILITY = 3
    OPTIMIZE_DENSITY_BALANCE = 4

    # Cell sizing (swap with different drive strengths)
    UPSIZE_CRITICAL_CELLS = 5
    DOWNSIZE_NON_CRITICAL_CELLS = 6

    # Timing optimization
    BUFFER_CRITICAL_PATHS = 7
    OPTIMIZE_CLOCK_TREE = 8

    # Power optimization
    REDUCE_SWITCHING_POWER = 9
    USE_LOW_POWER_CELLS = 10

    # Re-run tools with different parameters
    RERUN_PLACEMENT = 11
    RERUN_ROUTING = 12
    INCREMENTAL_OPTIMIZATION = 13

    # Floorplan adjustments
    ADJUST_ASPECT_RATIO = 14
    MOVE_MACROS = 15

    # Do nothing (useful for learning when current state is good)
    NO_OP = 16


class ActionSpace:
    """
    Manages the action space for chip design optimization.

    Maps action indices to actual design transformations.
    """

    def __init__(self, simulation_engine, design_state, world_model):
        """
        Initialize action space.

        Args:
            simulation_engine: Simulation engine with EDA tools
            design_state: Current design state
            world_model: World model with tech libraries
        """
        self.simulation_engine = simulation_engine
        self.design_state = design_state
        self.world_model = world_model

        # Map actions to execution functions
        self.action_map: Dict[int, Callable] = {
            Action.INCREASE_DENSITY: self._increase_density,
            Action.DECREASE_DENSITY: self._decrease_density,
            Action.OPTIMIZE_WIRELENGTH: self._optimize_wirelength,
            Action.OPTIMIZE_ROUTABILITY: self._optimize_routability,
            Action.OPTIMIZE_DENSITY_BALANCE: self._optimize_density,
            Action.UPSIZE_CRITICAL_CELLS: self._upsize_critical_cells,
            Action.DOWNSIZE_NON_CRITICAL_CELLS: self._downsize_non_critical,
            Action.BUFFER_CRITICAL_PATHS: self._buffer_critical_paths,
            Action.OPTIMIZE_CLOCK_TREE: self._optimize_clock_tree,
            Action.REDUCE_SWITCHING_POWER: self._reduce_switching_power,
            Action.USE_LOW_POWER_CELLS: self._use_low_power_cells,
            Action.RERUN_PLACEMENT: self._rerun_placement,
            Action.RERUN_ROUTING: self._rerun_routing,
            Action.INCREMENTAL_OPTIMIZATION: self._incremental_optimization,
            Action.ADJUST_ASPECT_RATIO: self._adjust_aspect_ratio,
            Action.MOVE_MACROS: self._move_macros,
            Action.NO_OP: self._no_op,
        }

        logger.info(f"Initialized ActionSpace with {len(self.action_map)} actions")

    @staticmethod
    def get_action_count() -> int:
        """Get total number of actions"""
        return len(Action)

    @staticmethod
    def get_action_name(action_idx: int) -> str:
        """Get human-readable name for action"""
        return Action(action_idx).name

    def execute_action(self, action_idx: int) -> Dict:
        """
        Execute the specified action.

        Args:
            action_idx: Index of action to execute

        Returns:
            Dictionary with:
                - 'success': Whether action executed successfully
                - 'metrics_delta': Change in metrics
                - 'info': Additional information
        """
        action_name = self.get_action_name(action_idx)
        logger.info(f"Executing action: {action_name}")

        # Get action function
        action_func = self.action_map.get(action_idx)

        if action_func is None:
            logger.error(f"Unknown action: {action_idx}")
            return {'success': False, 'metrics_delta': {}, 'info': 'Unknown action'}

        # Record metrics before action
        metrics_before = self.design_state.get_metrics_summary()

        # Execute action
        try:
            result = action_func()

            # Record metrics after action
            metrics_after = self.design_state.get_metrics_summary()

            # Calculate delta
            metrics_delta = self._calculate_metrics_delta(metrics_before, metrics_after)

            result['metrics_delta'] = metrics_delta
            result['success'] = True

            return result

        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            return {
                'success': False,
                'metrics_delta': {},
                'info': f'Action failed: {str(e)}'
            }

    def _calculate_metrics_delta(self, before: Dict, after: Dict) -> Dict:
        """Calculate change in metrics"""
        delta = {}

        # Timing delta
        if before['timing']['wns'] and after['timing']['wns']:
            delta['wns_delta'] = after['timing']['wns'] - before['timing']['wns']

        # Power delta
        if before['power']['total'] and after['power']['total']:
            delta['power_delta'] = after['power']['total'] - before['power']['total']

        # Area delta
        if before['area']['utilization'] and after['area']['utilization']:
            delta['util_delta'] = after['area']['utilization'] - before['area']['utilization']

        # Score delta
        if before['scores']['overall'] and after['scores']['overall']:
            delta['score_delta'] = after['scores']['overall'] - before['scores']['overall']

        return delta

    # Action implementations

    def _increase_density(self) -> Dict:
        """Increase placement density"""
        logger.info("Increasing placement density by 5%")
        # TODO: Implement actual density adjustment
        return {'info': 'Increased density'}

    def _decrease_density(self) -> Dict:
        """Decrease placement density"""
        logger.info("Decreasing placement density by 5%")
        return {'info': 'Decreased density'}

    def _optimize_wirelength(self) -> Dict:
        """Run placement optimizing for wirelength"""
        logger.info("Optimizing for wirelength")

        # Re-run DREAMPlace with wirelength focus
        # TODO: Get actual file paths from design state
        placement_result = self.simulation_engine.placement.optimize_placement(
            current_def="/tmp/current.def",
            output_def="/tmp/optimized.def",
            optimization_focus="wirelength"
        )

        # Update design state with new metrics
        if placement_result.get('success'):
            self.design_state.update_metrics({
                'routing': {
                    'total_wirelength': placement_result.get('hpwl', 0)
                }
            })

        return {'info': 'Optimized wirelength', 'result': placement_result}

    def _optimize_routability(self) -> Dict:
        """Run placement optimizing for routability"""
        logger.info("Optimizing for routability")
        return {'info': 'Optimized routability'}

    def _optimize_density(self) -> Dict:
        """Balance density across chip"""
        logger.info("Optimizing density balance")
        return {'info': 'Optimized density'}

    def _upsize_critical_cells(self) -> Dict:
        """Replace critical cells with higher drive strength versions"""
        logger.info("Upsizing critical path cells")

        # TODO: Identify critical cells from timing analysis
        # TODO: Find higher drive strength equivalents from tech library
        # TODO: Replace cells in netlist

        return {'info': 'Upsized critical cells'}

    def _downsize_non_critical(self) -> Dict:
        """Replace non-critical cells with lower power versions"""
        logger.info("Downsizing non-critical cells for power savings")
        return {'info': 'Downsized non-critical cells'}

    def _buffer_critical_paths(self) -> Dict:
        """Insert buffers on critical timing paths"""
        logger.info("Buffering critical paths")
        return {'info': 'Buffered critical paths'}

    def _optimize_clock_tree(self) -> Dict:
        """Optimize clock tree synthesis"""
        logger.info("Optimizing clock tree")
        return {'info': 'Optimized clock tree'}

    def _reduce_switching_power(self) -> Dict:
        """Apply power-reduction techniques"""
        logger.info("Reducing switching power")
        return {'info': 'Reduced switching power'}

    def _use_low_power_cells(self) -> Dict:
        """Swap to low-power cell variants"""
        logger.info("Using low-power cell variants")
        return {'info': 'Used low-power cells'}

    def _rerun_placement(self) -> Dict:
        """Re-run placement with current settings"""
        logger.info("Re-running placement")
        return {'info': 'Re-ran placement'}

    def _rerun_routing(self) -> Dict:
        """Re-run routing"""
        logger.info("Re-running routing")
        return {'info': 'Re-ran routing'}

    def _incremental_optimization(self) -> Dict:
        """Run incremental optimization pass"""
        logger.info("Running incremental optimization")
        return {'info': 'Ran incremental optimization'}

    def _adjust_aspect_ratio(self) -> Dict:
        """Adjust chip aspect ratio"""
        logger.info("Adjusting aspect ratio")
        return {'info': 'Adjusted aspect ratio'}

    def _move_macros(self) -> Dict:
        """Adjust macro placement"""
        logger.info("Moving macros")
        return {'info': 'Moved macros'}

    def _no_op(self) -> Dict:
        """Do nothing (current state is good)"""
        logger.info("No operation")
        return {'info': 'No operation'}

    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions in current state.

        Some actions may not be valid depending on design stage.

        Returns:
            List of valid action indices
        """
        valid_actions = []

        stage = self.design_state.stage.value

        # All actions valid in most stages
        if stage in ['placed', 'routed', 'optimized']:
            valid_actions = list(range(len(Action)))
        elif stage == 'synthesized':
            # Can only run placement-related actions
            valid_actions = [
                Action.INCREASE_DENSITY,
                Action.DECREASE_DENSITY,
                Action.RERUN_PLACEMENT,
                Action.NO_OP
            ]
        else:
            # Only no-op valid
            valid_actions = [Action.NO_OP]

        return valid_actions

    def get_action_description(self, action_idx: int) -> str:
        """Get detailed description of an action"""
        descriptions = {
            Action.INCREASE_DENSITY: "Increase placement density by 5% to reduce area",
            Action.DECREASE_DENSITY: "Decrease placement density by 5% to improve routability",
            Action.OPTIMIZE_WIRELENGTH: "Re-run placement optimizing for minimal wirelength",
            Action.OPTIMIZE_ROUTABILITY: "Re-run placement optimizing for routing congestion",
            Action.OPTIMIZE_DENSITY_BALANCE: "Balance cell density across the chip",
            Action.UPSIZE_CRITICAL_CELLS: "Replace critical path cells with higher drive strength",
            Action.DOWNSIZE_NON_CRITICAL_CELLS: "Replace non-critical cells with lower power versions",
            Action.BUFFER_CRITICAL_PATHS: "Insert buffers to reduce critical path delay",
            Action.OPTIMIZE_CLOCK_TREE: "Re-synthesize clock tree for better skew",
            Action.REDUCE_SWITCHING_POWER: "Apply clock gating and other power techniques",
            Action.USE_LOW_POWER_CELLS: "Swap to low-VT or low-power cell variants",
            Action.RERUN_PLACEMENT: "Re-run placement with current parameters",
            Action.RERUN_ROUTING: "Re-run detailed routing",
            Action.INCREMENTAL_OPTIMIZATION: "Run incremental timing/power optimization",
            Action.ADJUST_ASPECT_RATIO: "Change chip aspect ratio (width/height)",
            Action.MOVE_MACROS: "Adjust placement of large macros",
            Action.NO_OP: "Do nothing - current design is satisfactory"
        }

        return descriptions.get(action_idx, "Unknown action")
