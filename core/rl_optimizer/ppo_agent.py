"""
PPO Agent using Stable-Baselines3

Implements a Proximal Policy Optimization agent for chip design optimization.
This is the recommended agent for the RL loop.
"""

import logging
from typing import Optional, Dict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode info when available
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

                if self.verbose > 0:
                    logger.info(
                        f"Episode {len(self.episode_rewards)}: "
                        f"Reward={info['episode']['r']:.2f}, "
                        f"Length={info['episode']['l']}"
                    )

        return True

    def get_stats(self) -> Dict:
        """Get training statistics"""
        if not self.episode_rewards:
            return {}

        return {
            'num_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'mean_length': np.mean(self.episode_lengths[-100:]),
            'total_reward': np.sum(self.episode_rewards),
        }


class PPOAgent:
    """
    PPO Agent for chip design optimization using Stable-Baselines3.

    This agent learns to optimize chip designs by trying different actions
    (placement parameters, cell sizing, etc.) and receiving rewards based
    on PPA (Power, Performance, Area) improvements.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Initialize PPO agent.

        Args:
            env: Gym environment (ChipDesignEnv)
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum value for gradient clipping
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            tensorboard_log: Path for TensorBoard logs
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.env = env

        # Wrap environment for compatibility
        self.wrapped_env = Monitor(env)

        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=self.wrapped_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=device
        )

        # Training callback
        self.training_callback = TrainingCallback(verbose=verbose)

        logger.info(f"Initialized PPO agent")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Steps per update: {n_steps}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Device: {device}")

    def learn(
        self,
        total_timesteps: int,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 10000,
        eval_env=None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5
    ):
        """
        Train the agent.

        Args:
            total_timesteps: Total number of timesteps to train
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Frequency of checkpoint saving
            eval_env: Environment for evaluation
            eval_freq: Frequency of evaluation
            n_eval_episodes: Number of episodes for evaluation
        """
        callbacks = [self.training_callback]

        # Add checkpoint callback if directory specified
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix='ppo_chip_design'
            )
            callbacks.append(checkpoint_callback)

        # Add evaluation callback if eval env specified
        if eval_env:
            eval_callback = EvalCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        logger.info(f"Starting training for {total_timesteps} timesteps...")

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )

        logger.info("Training complete!")

    def predict(self, state: np.ndarray, deterministic: bool = True):
        """
        Predict action given state.

        Args:
            state: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            (action, state) tuple
        """
        action, _states = self.model.predict(state, deterministic=deterministic)
        return action

    def save(self, path: str):
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        self.model = PPO.load(path, env=self.wrapped_env)
        logger.info(f"Model loaded from {path}")

    def get_training_stats(self) -> Dict:
        """
        Get training statistics.

        Returns:
            Dictionary with training stats
        """
        return self.training_callback.get_stats()

    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Dict:
        """
        Evaluate the agent.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment

        Returns:
            Evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = self.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                if render:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            logger.info(f"Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
