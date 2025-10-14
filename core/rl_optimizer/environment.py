"""
Chip Design Environment

Defines the RL environment for chip design optimization.
Implements standard gym-like interface.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class ChipDesignEnv(gym.Env):
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
        super().__init__()

        self.design_state = design_state
        self.simulation_engine = simulation_engine
        self.world_model = world_model
        self.design_goals = design_goals

        # Environment configuration
        self.max_steps = 50  # Maximum optimization iterations
        self.current_step = 0

        # Track history
        self.history = []

        # Define action and observation space for Gym compatibility
        from .actions import ActionSpace
        self.num_actions = ActionSpace.get_action_count()
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: normalized design metrics
        # [timing_score, power_score, area_score, routing_score, stage_progress,
        #  wns, tns, power_total, area_util, wirelength, overflow, density]
        state_dim = len(design_state.get_state_vector())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        logger.info(f"Initialized ChipDesignEnv with {self.num_actions} actions and state_dim={state_dim}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            (initial_state, info)
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.history = []

        # Get initial state vector
        state = self.design_state.get_state_vector()

        info = {
            'initial_metrics': self.design_state.get_metrics_summary()
        }

        logger.info("Environment reset")
        return np.array(state, dtype=np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action in the environment.

        Args:
            action: Action index to execute

        Returns:
            (next_state, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Execute action
        action_result = self._execute_action(action)

        # Get new state
        next_state = self.design_state.get_state_vector()

        # Calculate reward
        reward = self._calculate_reward(action_result)

        # Check termination conditions
        terminated, truncated = self._check_termination()

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

        return np.array(next_state, dtype=np.float32), reward, terminated, truncated, info

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

    def _check_termination(self) -> Tuple[bool, bool]:
        """
        Check if episode should terminate.

        Returns:
            (terminated, truncated) where:
            - terminated: episode ended naturally (goal reached or failure)
            - truncated: episode ended due to time limit
        """
        terminated = False
        truncated = False

        # Truncated if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            return terminated, truncated

        # Terminated if design meets all goals
        if self.design_state.is_signoff_ready():
            logger.info("Design is signoff ready - episode complete")
            terminated = True
            return terminated, truncated

        # Terminated if no improvement for many steps
        if len(self.history) >= 10:
            recent_rewards = [h['reward'] for h in self.history[-10:]]
            if all(r <= 0 for r in recent_rewards):
                logger.info("No improvement in last 10 steps - episode complete")
                terminated = True
                return terminated, truncated

        return terminated, truncated

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

        wns = metrics['timing'].get('wns')
        if wns is not None:
            print(f"Timing WNS: {wns:.3f} ns")
        else:
            print(f"Timing WNS: N/A")

        power = metrics['power'].get('total')
        if power is not None:
            print(f"Power: {power:.2f} mW")
        else:
            print(f"Power: N/A")

        util = metrics['area'].get('utilization')
        if util is not None:
            print(f"Area Util: {util:.2%}")
        else:
            print(f"Area Util: N/A")

        score = metrics['scores'].get('overall')
        if score is not None:
            print(f"Overall Score: {score:.3f}")
        else:
            print(f"Overall Score: N/A")

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
