"""
Reward Calculator

Calculates rewards for the RL agent based on design improvements.
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Calculates rewards based on changes in design metrics.

    Rewards guide the RL agent toward better designs (PPA optimization).
    """

    def __init__(self, design_goals: Dict[str, float]):
        """
        Initialize reward calculator.

        Args:
            design_goals: Target goals with weights
                         e.g., {'performance': 1.0, 'power': 0.8, 'area': 0.6}
        """
        self.design_goals = design_goals

        # Normalize goal weights
        total_weight = sum(design_goals.values())
        self.normalized_goals = {
            k: v / total_weight for k, v in design_goals.items()
        }

        logger.info(f"Initialized RewardCalculator with goals: {self.normalized_goals}")

    def calculate_reward(
        self,
        current_metrics: Dict,
        action_result: Dict
    ) -> float:
        """
        Calculate reward based on metric changes.

        Args:
            current_metrics: Current design metrics
            action_result: Result from action execution

        Returns:
            Reward value (positive = improvement, negative = degradation)
        """
        if not action_result.get('success', False):
            # Failed action gets negative reward
            return -1.0

        metrics_delta = action_result.get('metrics_delta', {})

        # Calculate component rewards
        timing_reward = self._timing_reward(metrics_delta, current_metrics)
        power_reward = self._power_reward(metrics_delta, current_metrics)
        area_reward = self._area_reward(metrics_delta, current_metrics)
        routing_reward = self._routing_reward(metrics_delta, current_metrics)

        # Weighted combination based on design goals
        performance_weight = self.normalized_goals.get('performance', 0.4)
        power_weight = self.normalized_goals.get('power', 0.3)
        area_weight = self.normalized_goals.get('area', 0.2)
        routing_weight = 0.1  # Always some weight on routability

        total_reward = (
            timing_reward * performance_weight +
            power_reward * power_weight +
            area_reward * area_weight +
            routing_reward * routing_weight
        )

        # Bonus for meeting all constraints
        if self._check_constraints_met(current_metrics):
            total_reward += 5.0

        # Penalty for violations
        violation_penalty = self._calculate_violation_penalty(current_metrics)
        total_reward -= violation_penalty

        logger.debug(
            f"Reward breakdown: timing={timing_reward:.2f}, "
            f"power={power_reward:.2f}, area={area_reward:.2f}, "
            f"routing={routing_reward:.2f}, total={total_reward:.2f}"
        )

        return total_reward

    def _timing_reward(self, delta: Dict, metrics: Dict) -> float:
        """
        Calculate reward for timing improvements.

        Positive reward for:
        - Improved slack (less negative or more positive)
        - Reduced violations

        Args:
            delta: Metrics changes
            metrics: Current metrics

        Returns:
            Timing reward
        """
        reward = 0.0

        # Slack improvement (most important)
        if 'wns_delta' in delta:
            wns_delta = delta['wns_delta']

            if wns_delta > 0:
                # Improved slack (WNS became less negative or more positive)
                # Scale reward: 1ns improvement = 10 reward points
                reward += wns_delta * 10.0
            else:
                # Degraded slack - penalty
                reward += wns_delta * 5.0  # Smaller penalty than reward

        # Violation reduction
        violations = metrics.get('timing', {}).get('violations', 0)
        if violations == 0:
            reward += 2.0  # Bonus for clean timing

        # Check if timing is met
        wns = metrics.get('timing', {}).get('wns', -999)
        if wns >= 0:
            reward += 5.0  # Large bonus for meeting timing

        return reward

    def _power_reward(self, delta: Dict, metrics: Dict) -> float:
        """
        Calculate reward for power improvements.

        Positive reward for reducing power consumption.

        Args:
            delta: Metrics changes
            metrics: Current metrics

        Returns:
            Power reward
        """
        reward = 0.0

        # Power reduction
        if 'power_delta' in delta:
            power_delta = delta['power_delta']

            if power_delta < 0:
                # Power reduced (negative delta is good)
                # Scale: 10mW reduction = 1 reward point
                reward += abs(power_delta) / 10.0
            else:
                # Power increased (penalty)
                reward -= power_delta / 20.0

        # Bonus for being under power budget
        total_power = metrics.get('power', {}).get('total', 0)
        power_budget = 1000.0  # mW (TODO: get from design goals)

        if total_power and total_power < power_budget:
            reward += 1.0

        return reward

    def _area_reward(self, delta: Dict, metrics: Dict) -> float:
        """
        Calculate reward for area improvements.

        Reward optimal utilization (60-80%).

        Args:
            delta: Metrics changes
            metrics: Current metrics

        Returns:
            Area reward
        """
        reward = 0.0

        utilization = metrics.get('area', {}).get('utilization', 0)

        if utilization:
            # Optimal utilization range: 0.6 - 0.8
            if 0.6 <= utilization <= 0.8:
                reward += 2.0
            elif utilization < 0.6:
                # Underutilized (wasted area)
                reward -= (0.6 - utilization) * 5.0
            else:
                # Over-utilized (congestion)
                reward -= (utilization - 0.8) * 10.0

        # Utilization improvement
        if 'util_delta' in delta:
            util_delta = delta['util_delta']

            # Moving toward optimal range is good
            if utilization:
                if utilization < 0.6 and util_delta > 0:
                    reward += util_delta * 5.0
                elif utilization > 0.8 and util_delta < 0:
                    reward += abs(util_delta) * 5.0

        return reward

    def _routing_reward(self, delta: Dict, metrics: Dict) -> float:
        """
        Calculate reward for routing improvements.

        Positive reward for:
        - Reduced wirelength
        - Fewer DRC violations
        - Lower congestion

        Args:
            delta: Metrics changes
            metrics: Current metrics

        Returns:
            Routing reward
        """
        reward = 0.0

        # DRC violations
        drc_violations = metrics.get('routing', {}).get('drc_violations', 0)

        if drc_violations == 0:
            reward += 3.0  # Bonus for DRC clean
        else:
            reward -= drc_violations * 0.1  # Penalty per violation

        # Wirelength (shorter is better)
        # Rewarded indirectly through timing and power

        return reward

    def _check_constraints_met(self, metrics: Dict) -> bool:
        """
        Check if all design constraints are met.

        Args:
            metrics: Current metrics

        Returns:
            True if all constraints satisfied
        """
        # Timing constraints
        wns = metrics.get('timing', {}).get('wns')
        if wns is None or wns < 0:
            return False

        violations = metrics.get('timing', {}).get('violations', 0)
        if violations > 0:
            return False

        # DRC constraints
        drc_violations = metrics.get('routing', {}).get('drc_violations', 0)
        if drc_violations > 0:
            return False

        return True

    def _calculate_violation_penalty(self, metrics: Dict) -> float:
        """
        Calculate penalty for design rule violations.

        Args:
            metrics: Current metrics

        Returns:
            Penalty value (positive)
        """
        penalty = 0.0

        # Timing violations
        setup_violations = metrics.get('timing', {}).get('violations', 0)
        penalty += setup_violations * 0.5

        # DRC violations
        drc_violations = metrics.get('routing', {}).get('drc_violations', 0)
        penalty += drc_violations * 0.2

        return penalty

    def get_reward_breakdown(
        self,
        current_metrics: Dict,
        action_result: Dict
    ) -> Dict:
        """
        Get detailed breakdown of reward components.

        Useful for debugging and understanding agent behavior.

        Args:
            current_metrics: Current metrics
            action_result: Action result

        Returns:
            Dictionary with reward breakdown
        """
        if not action_result.get('success', False):
            return {'total': -1.0, 'reason': 'Action failed'}

        metrics_delta = action_result.get('metrics_delta', {})

        timing_reward = self._timing_reward(metrics_delta, current_metrics)
        power_reward = self._power_reward(metrics_delta, current_metrics)
        area_reward = self._area_reward(metrics_delta, current_metrics)
        routing_reward = self._routing_reward(metrics_delta, current_metrics)

        constraints_bonus = 5.0 if self._check_constraints_met(current_metrics) else 0.0
        violation_penalty = self._calculate_violation_penalty(current_metrics)

        total = (
            timing_reward * self.normalized_goals.get('performance', 0.4) +
            power_reward * self.normalized_goals.get('power', 0.3) +
            area_reward * self.normalized_goals.get('area', 0.2) +
            routing_reward * 0.1 +
            constraints_bonus -
            violation_penalty
        )

        return {
            'total': total,
            'timing': timing_reward,
            'power': power_reward,
            'area': area_reward,
            'routing': routing_reward,
            'constraints_bonus': constraints_bonus,
            'violation_penalty': violation_penalty,
            'weights': self.normalized_goals
        }


class AdaptiveRewardCalculator(RewardCalculator):
    """
    Reward calculator that adapts weights based on progress.

    Initially focuses on meeting constraints, then shifts to
    optimizing PPA metrics.

    TODO: Implement curriculum learning for reward shaping
    """

    def __init__(self, design_goals: Dict[str, float]):
        """Initialize adaptive reward calculator"""
        super().__init__(design_goals)
        self.phase = "constraint_satisfaction"  # or "optimization"

    def calculate_reward(self, current_metrics: Dict, action_result: Dict) -> float:
        """Calculate reward with adaptive weighting"""
        # TODO: Implement phase-based reward adaptation
        return super().calculate_reward(current_metrics, action_result)
