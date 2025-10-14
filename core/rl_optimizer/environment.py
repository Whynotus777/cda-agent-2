"""
Chip Design Environment

Defines the RL environment for chip design optimization.
Implements standard gym-like interface.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ChipDesignEnv:
    """
    RL Environment for chip design optimization.

    State: Current design metrics (timing, power, area, routing)
    Actions: Design transformations (adjust density, swap cells, re-run tools)
    Rewards: Improvement in PPA (Power, Performance, Area) metrics
    """

    def __init__(
        self,
        design_state,
        simulation_engine,
        world_model,
        design_goals: Dict[str, float]
    ):
        """
        Initialize chip design environment.

        Args:
            design_state: Current design state object
            simulation_engine: Simulation engine with all EDA tools
            world_model: World model with tech libraries and rules
            design_goals: Target goals (e.g., {'power': 0.8, 'performance': 1.0})
        """
        self.design_state = design_state
        self.simulation_engine = simulation_engine
        self.world_model = world_model
        self.design_goals = design_goals

        # Environment configuration
        self.max_steps = 50  # Maximum optimization iterations
        self.current_step = 0

        # Track history
        self.history = []

        logger.info("Initialized ChipDesignEnv")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.history = []

        # Get initial state vector
        state = self.design_state.get_state_vector()

        logger.info("Environment reset")
        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in the environment.

        Args:
            action: Action index to execute

        Returns:
            (next_state, reward, done, info)
        """
        self.current_step += 1

        # Execute action
        action_result = self._execute_action(action)

        # Get new state
        next_state = self.design_state.get_state_vector()

        # Calculate reward
        reward = self._calculate_reward(action_result)

        # Check if done
        done = self._check_done()

        # Additional info
        info = {
            'action': action,
            'step': self.current_step,
            'metrics': self.design_state.get_metrics_summary(),
            'action_result': action_result
        }

        # Record history
        self.history.append({
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'state': next_state.copy(),
            'metrics': info['metrics'].copy()
        })

        return np.array(next_state, dtype=np.float32), reward, done, info

    def _execute_action(self, action: int) -> Dict:
        """
        Execute the specified action.

        Args:
            action: Action index

        Returns:
            Dictionary with action results
        """
        from .actions import ActionSpace

        action_space = ActionSpace(
            simulation_engine=self.simulation_engine,
            design_state=self.design_state,
            world_model=self.world_model
        )

        # Map action index to actual action
        result = action_space.execute_action(action)

        return result

    def _calculate_reward(self, action_result: Dict) -> float:
        """
        Calculate reward based on action outcome.

        Args:
            action_result: Results from action execution

        Returns:
            Reward value (higher is better)
        """
        from .reward import RewardCalculator

        reward_calc = RewardCalculator(design_goals=self.design_goals)

        reward = reward_calc.calculate_reward(
            current_metrics=self.design_state.get_metrics_summary(),
            action_result=action_result
        )

        return reward

    def _check_done(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if done, False otherwise
        """
        # Done if max steps reached
        if self.current_step >= self.max_steps:
            return True

        # Done if design meets all goals
        if self.design_state.is_signoff_ready():
            logger.info("Design is signoff ready - episode complete")
            return True

        # Done if no improvement for many steps
        if len(self.history) >= 10:
            recent_rewards = [h['reward'] for h in self.history[-10:]]
            if all(r <= 0 for r in recent_rewards):
                logger.info("No improvement in last 10 steps - episode complete")
                return True

        return False

    def get_state_dim(self) -> int:
        """Get dimension of state vector"""
        return len(self.design_state.get_state_vector())

    def get_action_dim(self) -> int:
        """Get number of possible actions"""
        from .actions import ActionSpace
        return ActionSpace.get_action_count()

    def render(self):
        """
        Render current state (for debugging/visualization).

        Prints current metrics and progress.
        """
        metrics = self.design_state.get_metrics_summary()

        print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
        print(f"Stage: {metrics['stage']}")
        print(f"Timing WNS: {metrics['timing']['wns']:.3f} ns")
        print(f"Power: {metrics['power']['total']:.2f} mW")
        print(f"Area Util: {metrics['area']['utilization']:.2%}")
        print(f"Overall Score: {metrics['scores']['overall']:.3f}")
        print()

    def get_episode_summary(self) -> Dict:
        """
        Get summary of current episode.

        Returns:
            Dictionary with episode statistics
        """
        if not self.history:
            return {}

        rewards = [h['reward'] for h in self.history]
        scores = [h['metrics']['scores']['overall'] for h in self.history]

        return {
            'total_steps': self.current_step,
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'max_reward': max(rewards),
            'final_score': scores[-1] if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'improvement': scores[-1] - scores[0] if len(scores) >= 2 else 0.0
        }

    def get_best_design(self) -> Dict:
        """
        Get the design state with the best overall score.

        Returns:
            Best design metrics and state
        """
        if not self.history:
            return {}

        best_step = max(
            self.history,
            key=lambda h: h['metrics']['scores']['overall']
        )

        return {
            'step': best_step['step'],
            'action': best_step['action'],
            'metrics': best_step['metrics'],
            'state': best_step['state']
        }


class MultiObjectiveEnv(ChipDesignEnv):
    """
    Extension of ChipDesignEnv for multi-objective optimization.

    Handles Pareto-optimal solutions for PPA trade-offs.

    TODO: Implement multi-objective RL algorithms
    """

    def __init__(self, *args, **kwargs):
        """Initialize multi-objective environment"""
        super().__init__(*args, **kwargs)
        self.pareto_front = []

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Step with multi-dimensional reward.

        Returns:
            (next_state, reward_vector, done, info)
        """
        state, scalar_reward, done, info = super().step(action)

        # Decompose reward into components
        reward_vector = self._get_reward_vector()

        # Update Pareto front
        self._update_pareto_front(reward_vector, state)

        return state, reward_vector, done, info

    def _get_reward_vector(self) -> np.ndarray:
        """Get reward as vector [timing, power, area]"""
        metrics = self.design_state.metrics

        return np.array([
            metrics.timing_score or 0.0,
            metrics.power_score or 0.0,
            metrics.area_score or 0.0
        ], dtype=np.float32)

    def _update_pareto_front(self, reward: np.ndarray, state: np.ndarray):
        """Update Pareto-optimal solutions"""
        # Check if this solution is dominated by any existing solution
        is_dominated = False

        for existing_reward, _ in self.pareto_front:
            if np.all(existing_reward >= reward) and np.any(existing_reward > reward):
                is_dominated = True
                break

        if not is_dominated:
            # Remove solutions dominated by this one
            self.pareto_front = [
                (r, s) for r, s in self.pareto_front
                if not (np.all(reward >= r) and np.any(reward > r))
            ]

            # Add this solution
            self.pareto_front.append((reward.copy(), state.copy()))
