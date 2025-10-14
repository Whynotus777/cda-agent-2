#!/usr/bin/env python3
"""
Quick test to verify PPO agent can be instantiated and used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.rl_optimizer.environment import ChipDesignEnv
from core.rl_optimizer.ppo_agent import PPOAgent
from core.simulation_engine import SimulationEngine
from core.world_model import WorldModel
from core.world_model.design_state import DesignState, DesignStage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ppo_agent():
    """Test PPO agent instantiation and basic operations"""
    print("="*70)
    print("TESTING PPO AGENT")
    print("="*70)
    print()

    # Create a simple counter design
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

    rtl_file = "/tmp/test_ppo_counter.v"
    with open(rtl_file, 'w') as f:
        f.write(verilog_code)

    # Setup environment
    simulation_engine = SimulationEngine()
    world_model = WorldModel()
    design_state = DesignState(project_name="test_ppo")

    design_state.rtl_files = [rtl_file]
    design_state.top_module = "simple_counter"
    design_state.process_node = "7nm"
    design_state.clock_period = 10.0

    # Run synthesis
    output_netlist = "/tmp/test_ppo_counter_synth.v"
    synth_result = simulation_engine.synthesis.synthesize(
        rtl_files=[rtl_file],
        top_module="simple_counter",
        output_netlist=output_netlist,
        optimization_goal="balanced"
    )

    if synth_result and synth_result.get('cell_count', 0) > 0:
        design_state.netlist_file = output_netlist
        design_state.update_stage(DesignStage.SYNTHESIZED)
    else:
        print("✗ Synthesis failed")
        return False

    # Create environment
    design_goals = {'performance': 1.0, 'power': 0.5, 'area': 0.7}
    env = ChipDesignEnv(
        design_state=design_state,
        simulation_engine=simulation_engine,
        world_model=world_model,
        design_goals=design_goals
    )

    print("✓ Environment created")

    # Create PPO agent
    try:
        agent = PPOAgent(
            env=env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            verbose=1,
            tensorboard_log=None
        )
        print("✓ PPO agent created successfully")
    except Exception as e:
        print(f"✗ Failed to create PPO agent: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test prediction
    try:
        state, info = env.reset()
        action = agent.predict(state, deterministic=True)
        print(f"✓ Agent can predict actions: action={action}")
    except Exception as e:
        print(f"✗ Failed to predict action: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test a few training steps (very short)
    try:
        print("\nTesting short training run (100 steps)...")
        agent.learn(total_timesteps=100)
        print("✓ Training executed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test save/load
    try:
        model_path = "/tmp/test_ppo_model.zip"
        agent.save(model_path)
        print(f"✓ Model saved to {model_path}")

        agent.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"✗ Save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("="*70)
    print("PPO AGENT TEST PASSED!")
    print("="*70)
    print()
    print("✓ All tests passed:")
    print("  - Environment creation")
    print("  - PPO agent instantiation")
    print("  - Action prediction")
    print("  - Training execution")
    print("  - Model save/load")
    print()
    print("Ready for full training with train_rl_agent.py!")
    print()

    return True


if __name__ == "__main__":
    success = test_ppo_agent()
    sys.exit(0 if success else 1)
