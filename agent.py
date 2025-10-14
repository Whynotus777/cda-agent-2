"""
CDA Agent - Main Orchestrator

The central agent that coordinates all modules:
- Conversational interface
- World model
- Simulation engine
- RL optimizer
"""

import logging
from typing import Dict, Optional

from core.conversational import LLMInterface, IntentParser, ConversationManager
from core.world_model import TechLibrary, DesignParser, RuleEngine, DesignState
from core.simulation_engine import (
    SynthesisEngine, PlacementEngine, RoutingEngine,
    TimingAnalyzer, PowerAnalyzer
)
from core.rl_optimizer import RLAgent, ChipDesignEnv, ActionSpace
from utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class CDAAgent:
    """
    Main CDA (Chip Design Automation) Agent.

    Integrates all components to provide an interactive, AI-powered
    chip design assistant.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CDA Agent.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        setup_logging(self.config.get('logging', {}))

        logger.info("=" * 60)
        logger.info("Initializing CDA Agent")
        logger.info("=" * 60)

        # Initialize modules
        self._init_conversational_layer()
        self._init_world_model()
        self._init_simulation_engine()
        # Defer RL optimizer initialization until first use
        self.rl_agent = None

        # Current design state
        self.current_design: Optional[DesignState] = None

        logger.info("CDA Agent initialization complete")

    def _init_conversational_layer(self):
        """Initialize conversational interface"""
        logger.info("Initializing conversational layer...")

        llm_config = self.config.get('llm', {})

        self.llm_interface = LLMInterface(
            model_name=llm_config.get('model_name', 'llama3:70b'),
            ollama_host=llm_config.get('ollama_host', 'http://localhost:11434'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 2048)
        )

        self.intent_parser = IntentParser(self.llm_interface)
        self.conversation_manager = ConversationManager(
            self.llm_interface,
            self.intent_parser
        )

        logger.info("Conversational layer initialized")

    def _init_world_model(self):
        """Initialize world model components"""
        logger.info("Initializing world model...")

        tech_config = self.config.get('technology', {})

        self.tech_library = TechLibrary(
            process_node=tech_config.get('default_process_node', '7nm')
        )

        self.design_parser = DesignParser()

        self.rule_engine = RuleEngine(
            process_node=tech_config.get('default_process_node', '7nm')
        )

        logger.info("World model initialized")

    def _init_simulation_engine(self):
        """Initialize simulation and analysis engines"""
        logger.info("Initializing simulation engine...")

        tools_config = self.config.get('tools', {})

        self.synthesis_engine = SynthesisEngine()

        self.placement_engine = PlacementEngine(
            dreamplace_path=tools_config.get('dreamplace_path')
        )

        self.routing_engine = RoutingEngine(
            tritonroute_path=tools_config.get('tritonroute_binary')
        )

        self.timing_analyzer = TimingAnalyzer(
            opensta_path=tools_config.get('opensta_binary')
        )

        self.power_analyzer = PowerAnalyzer(tech_library=self.tech_library)

        # Create simulation engine container
        class SimulationEngine:
            def __init__(self, synthesis, placement, routing, timing, power):
                self.synthesis = synthesis
                self.placement = placement
                self.routing = routing
                self.timing = timing
                self.power = power

        self.simulation_engine = SimulationEngine(
            self.synthesis_engine,
            self.placement_engine,
            self.routing_engine,
            self.timing_analyzer,
            self.power_analyzer
        )

        logger.info("Simulation engine initialized")

    def _init_rl_optimizer(self):
        """Initialize RL optimization core"""
        logger.info("RL optimizer is deferred and will initialize on first use")
        self.rl_env: Optional[ChipDesignEnv] = None

    def chat(self, user_message: str) -> str:
        """
        Process user message and return response.

        Args:
            user_message: User's natural language input

        Returns:
            Agent's response
        """
        logger.info(f"User: {user_message}")

        # Process message through conversation manager
        result = self.conversation_manager.process_message(user_message)

        response_text = result['response']
        intent = result['intent']
        actions = result['actions']

        # Execute backend actions if needed
        if actions:
            self._execute_backend_actions(actions)

        logger.info(f"Agent: {response_text}")
        return response_text

    def _execute_backend_actions(self, actions: list):
        """Execute backend actions (synthesis, placement, etc.)"""
        for action in actions:
            module = action.get('module')
            function = action.get('function')
            params = action.get('params', {})

            logger.info(f"Executing: {module}.{function}")

            try:
                if module == 'world_model':
                    self._execute_world_model_action(function, params)
                elif module == 'simulation_engine':
                    self._execute_simulation_action(function, params)
                elif module == 'rl_optimizer':
                    self._execute_rl_action(function, params)
            except Exception as e:
                logger.error(f"Action execution failed: {e}")

    def _execute_world_model_action(self, function: str, params: Dict):
        """Execute world model actions"""
        if function == 'initialize_project':
            project_name = params.get('project_name', 'new_design')
            self.current_design = DesignState(project_name)
            self.current_design.process_node = params.get('process_node', '7nm')

        elif function == 'load_design':
            file_path = params.get('file_path')
            self.design_parser.parse_verilog(file_path)

    def _execute_simulation_action(self, function: str, params: Dict):
        """Execute simulation engine actions"""
        if function == 'run_synthesis':
            # Run synthesis
            pass
        elif function == 'run_placement':
            # Run placement
            pass
        elif function == 'run_routing':
            # Run routing
            pass
        elif function == 'analyze_timing':
            # Run timing analysis
            pass
        elif function == 'analyze_power':
            # Run power analysis
            pass

    def _execute_rl_action(self, function: str, params: Dict):
        """Execute RL optimizer actions"""
        if function == 'optimize_design':
            self.run_rl_optimization(params)

    def run_rl_optimization(self, params: Dict):
        """
        Run RL-based design optimization.

        Args:
            params: Optimization parameters
        """
        if not self.current_design:
            logger.error("No design loaded for optimization")
            return

        logger.info("Starting RL-based optimization")

        # Create environment
        design_goals = params.get('goals', self.config.get('design_goals', {}))

        # Initialize RL agent on first use
        if self.rl_agent is None:
            rl_config = self.config.get('rl_agent', {})
            try:
                self.rl_agent = RLAgent(
                    state_dim=rl_config.get('state_dim', 12),
                    action_dim=rl_config.get('action_dim', 17),
                    learning_rate=rl_config.get('learning_rate', 1e-4),
                    gamma=rl_config.get('gamma', 0.99),
                    epsilon_start=rl_config.get('epsilon_start', 1.0),
                    epsilon_end=rl_config.get('epsilon_end', 0.01),
                    epsilon_decay=rl_config.get('epsilon_decay', 1000)
                )
            except Exception as e:
                logger.error(f"RL initialization failed: {e}")
                return

        self.rl_env = ChipDesignEnv(
            design_state=self.current_design,
            simulation_engine=self.simulation_engine,
            world_model=self.tech_library,
            design_goals=design_goals
        )

        # Training loop
        training_config = self.config.get('training', {})
        max_episodes = training_config.get('max_episodes', 100)
        max_steps = training_config.get('max_steps_per_episode', 50)

        for episode in range(max_episodes):
            state = self.rl_env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Select action
                action = self.rl_agent.select_action(state)

                # Take step
                next_state, reward, done, info = self.rl_env.step(action)

                # Store transition
                self.rl_agent.store_transition(state, action, reward, next_state, done)

                # Train
                loss = self.rl_agent.train_step()

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Update target network periodically
            if episode % training_config.get('target_network_update_freq', 10) == 0:
                self.rl_agent.update_target_network()

            # Save checkpoint
            if episode % training_config.get('save_checkpoint_every', 10) == 0:
                checkpoint_path = f"./data/checkpoints/agent_episode_{episode}.pt"
                self.rl_agent.save_checkpoint(checkpoint_path)

            logger.info(
                f"Episode {episode}/{max_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Steps={step+1}"
            )

        logger.info("RL optimization complete")

    def get_design_summary(self) -> Dict:
        """Get summary of current design"""
        if not self.current_design:
            return {'status': 'No design loaded'}

        return self.current_design.get_metrics_summary()

    def save_design(self, output_path: str):
        """
        Save current design state.

        Args:
            output_path: Path to save design
        """
        if not self.current_design:
            logger.error("No design to save")
            return

        import json

        design_data = self.current_design.export_state()

        with open(output_path, 'w') as f:
            json.dump(design_data, f, indent=2)

        logger.info(f"Design saved to {output_path}")

    def load_design(self, design_path: str):
        """
        Load design state from file.

        Args:
            design_path: Path to saved design
        """
        import json

        with open(design_path, 'r') as f:
            design_data = json.load(f)

        # Reconstruct design state
        project_name = design_data.get('project_name', 'loaded_design')
        self.current_design = DesignState(project_name)

        # Restore state
        # TODO: Implement full state restoration

        logger.info(f"Design loaded from {design_path}")


def main():
    """Main entry point for interactive mode"""
    import sys

    # Create agent
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    agent = CDAAgent(config_path)

    print("\n" + "=" * 60)
    print("CDA Agent - AI Chip Design Assistant")
    print("=" * 60)
    print("\nType your commands or questions. Type 'exit' to quit.\n")

    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            # Process message
            response = agent.chat(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
