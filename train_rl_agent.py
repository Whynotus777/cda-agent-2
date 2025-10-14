#!/usr/bin/env python3
"""
Train RL Agent for Chip Design Optimization

This script demonstrates how to train a PPO agent to optimize chip designs.
The agent learns to make decisions about placement parameters, cell sizing,
and other design choices to optimize PPA (Power, Performance, Area) metrics.
"""

import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from core.rl_optimizer.environment import ChipDesignEnv
from core.rl_optimizer.ppo_agent import PPOAgent
from core.simulation_engine import SimulationEngine
from core.world_model import WorldModel
from core.world_model.design_state import DesignState, DesignStage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_design(design_name: str = "simple_counter"):
    """
    Create a test design for training.

    Args:
        design_name: Name of the design

    Returns:
        Path to RTL file
    """
    if design_name == "simple_counter":
        verilog_code = """
module simple_counter (
    input wire clk,
    input wire reset,
    output reg [3:0] count
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= 4'b0000;
        else
            count <= count + 1;
    end
endmodule
"""
    elif design_name == "alu":
        verilog_code = """
module alu (
    input wire [7:0] a,
    input wire [7:0] b,
    input wire [1:0] op,
    output reg [7:0] result
);
    always @(*) begin
        case (op)
            2'b00: result = a + b;
            2'b01: result = a - b;
            2'b10: result = a & b;
            2'b11: result = a | b;
        endcase
    end
endmodule
"""
    else:
        raise ValueError(f"Unknown design: {design_name}")

    rtl_file = f"/tmp/{design_name}_train.v"
    with open(rtl_file, 'w') as f:
        f.write(verilog_code)

    logger.info(f"Created test RTL: {rtl_file}")
    return rtl_file


def setup_environment(rtl_file: str, top_module: str):
    """
    Setup the RL environment for training.

    Args:
        rtl_file: Path to RTL file
        top_module: Top module name

    Returns:
        Configured ChipDesignEnv
    """
    logger.info("Setting up environment...")

    # Initialize components
    simulation_engine = SimulationEngine()
    world_model = WorldModel()
    design_state = DesignState(project_name=f"rl_train_{top_module}")

    # Setup design state
    design_state.rtl_files = [rtl_file]
    design_state.top_module = top_module
    design_state.process_node = "7nm"
    design_state.clock_period = 10.0  # 10ns = 100MHz

    # Run synthesis to get netlist
    logger.info("Running initial synthesis...")
    output_netlist = f"/tmp/{top_module}_synth.v"

    synth_result = simulation_engine.synthesis.synthesize(
        rtl_files=[rtl_file],
        top_module=top_module,
        output_netlist=output_netlist,
        optimization_goal="balanced"
    )

    if synth_result and synth_result.get('cell_count', 0) > 0:
        logger.info(f"Synthesis complete: {synth_result['cell_count']} cells, {synth_result.get('flip_flops', 0)} flip-flops")
        design_state.netlist_file = output_netlist
        design_state.update_stage(DesignStage.SYNTHESIZED)
    else:
        raise RuntimeError(f"Synthesis failed: {synth_result}")

    # Define design goals
    design_goals = {
        'performance': 1.0,  # High priority on timing
        'power': 0.5,        # Medium priority on power
        'area': 0.7          # Medium-high priority on area
    }

    # Create RL environment
    env = ChipDesignEnv(
        design_state=design_state,
        simulation_engine=simulation_engine,
        world_model=world_model,
        design_goals=design_goals
    )

    logger.info("Environment setup complete")
    logger.info(f"  - Action space: {env.action_space}")
    logger.info(f"  - Observation space: {env.observation_space}")

    return env


def train(
    design_name: str = "simple_counter",
    total_timesteps: int = 10000,
    checkpoint_dir: str = "./checkpoints",
    tensorboard_log: str = "./tensorboard_logs",
    eval_freq: int = 1000,
    save_path: str = "./models/ppo_chip_design.zip"
):
    """
    Train the RL agent.

    Args:
        design_name: Name of design to use for training
        total_timesteps: Total training timesteps
        checkpoint_dir: Directory for checkpoints
        tensorboard_log: Directory for TensorBoard logs
        eval_freq: Frequency of evaluation
        save_path: Path to save final model
    """
    print("="*70)
    print("TRAINING RL AGENT FOR CHIP DESIGN OPTIMIZATION")
    print("="*70)
    print()

    # Create design
    rtl_file = create_test_design(design_name)

    # Setup environment
    env = setup_environment(rtl_file, design_name)

    # Create PPO agent
    print("\nCreating PPO agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=128,  # Smaller n_steps for EDA tasks (each step is expensive)
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    print("PPO agent created successfully!")
    print()

    # Train
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"TensorBoard logs: {tensorboard_log}")
    print()

    try:
        agent.learn(
            total_timesteps=total_timesteps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=max(1000, total_timesteps // 10),
            eval_freq=eval_freq
        )

        # Save final model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(save_path)

        print()
        print("="*70)
        print("TRAINING COMPLETE!")
        print("="*70)

        # Get training stats
        stats = agent.get_training_stats()
        print("\nTraining Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Evaluate
        print("\n" + "="*70)
        print("EVALUATING AGENT")
        print("="*70)

        eval_stats = agent.evaluate(n_episodes=5, render=False)
        print("\nEvaluation Results:")
        for key, value in eval_stats.items():
            print(f"  {key}: {value:.2f}")

        print()
        print(f"Model saved to: {save_path}")
        print()
        print("Next Steps:")
        print("  - View training progress: tensorboard --logdir=" + tensorboard_log)
        print("  - Load and evaluate model: use load_and_evaluate.py")
        print("  - Train on larger designs for better optimization")
        print()

        return agent

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        checkpoint_path = Path(checkpoint_dir) / "interrupted_model.zip"
        agent.save(str(checkpoint_path))
        print(f"Checkpoint saved to: {checkpoint_path}")
        return agent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train RL agent for chip design optimization")
    parser.add_argument('--design', type=str, default='simple_counter',
                        choices=['simple_counter', 'alu'],
                        help='Design to use for training')
    parser.add_argument('--timesteps', type=int, default=10000,
                        help='Total training timesteps')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--tensorboard-log', type=str, default='./tensorboard_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save-path', type=str, default='./models/ppo_chip_design.zip',
                        help='Path to save final model')
    parser.add_argument('--eval-freq', type=int, default=1000,
                        help='Frequency of evaluation')

    args = parser.parse_args()

    train(
        design_name=args.design,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_log=args.tensorboard_log,
        eval_freq=args.eval_freq,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
