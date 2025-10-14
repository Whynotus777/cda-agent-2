"""
RL Agent Module

Implements the reinforcement learning agent that learns to optimize chip designs.
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:  # ImportError and CUDA init issues
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class QNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Deep Q-Network for action-value estimation.

    Takes design state as input, outputs Q-values for each action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RL features. Install torch to use RLAgent.")
        super(QNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers = []
        prev_dim = state_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass"""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for training stability.

    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        """Current buffer size"""
        return len(self.buffer)


class RLAgent:
    """
    Reinforcement Learning Agent for chip design optimization.

    Uses Deep Q-Learning (DQN) to learn which actions improve design metrics.

    Actions include:
    - Adjusting placement density
    - Changing cell drive strengths
    - Re-running placement/routing with different parameters
    - Modifying floorplan
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1000
    ):
        """
        Initialize RL agent.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Number of steps for epsilon decay
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Please install torch to enable RLAgent.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device/config
        from utils import load_config
        cfg = load_config().get('rl_agent', {})
        device_str = cfg.get('device', 'cuda' if torch and torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        self.enable_amp = bool(cfg.get('enable_amp', True)) and (self.device.type == 'cuda')
        self.enable_compile = bool(cfg.get('enable_torch_compile', False)) and hasattr(torch, 'compile')

        # Q-networks (main and target for stability)
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        if self.enable_compile:
            try:
                self.q_network = torch.compile(self.q_network)
                self.target_network = torch.compile(self.target_network)
            except Exception:
                pass

        # Optimizer and AMP scaler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Move to device
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        # Training stats
        self.steps_done = 0
        self.training_losses = []

        logger.info(f"Initialized RLAgent with state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"Using device: {self.device}")

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state vector
            explore: Whether to use exploration (epsilon-greedy)

        Returns:
            Selected action index
        """
        # Epsilon decay
        if explore:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          np.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1

            # Exploration: random action
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)

        # Exploitation: best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """
        Perform one training step.

        Args:
            batch_size: Mini-batch size

        Returns:
            Training loss, or None if not enough data
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Forward and loss (with AMP)
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        loss_value = loss.item()
        self.training_losses.append(loss_value)

        return loss_value

    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, path: str):
        """
        Save agent checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'training_losses': self.training_losses
        }, path)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """
        Load agent checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.training_losses = checkpoint['training_losses']

        logger.info(f"Loaded checkpoint from {path}")

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'avg_loss_last_100': (
                np.mean(self.training_losses[-100:])
                if len(self.training_losses) >= 100
                else 0.0
            ),
            'total_training_steps': len(self.training_losses)
        }


class PolicyGradientAgent:
    """
    Alternative RL agent using Policy Gradient (PPO-style) approach.

    Can be used instead of DQN for continuous action spaces.

    TODO: Implement PPO for more advanced optimization
    """

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize policy gradient agent"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        logger.info("PolicyGradientAgent initialized (placeholder)")

    def select_action(self, state: np.ndarray) -> int:
        """Select action using policy network"""
        # TODO: Implement policy network
        return random.randrange(self.action_dim)

    def train_step(self) -> float:
        """Train policy network"""
        # TODO: Implement PPO training
        return 0.0
