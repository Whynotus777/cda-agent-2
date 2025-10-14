"""
Action Executor Module

Executes actions based on parsed intents by calling backend functions.
Connects the conversational layer to the functional backend (RL, EDA tools, etc.)
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path

from .intent_parser import ActionType, ParsedIntent
from ..simulation_engine import SimulationEngine
from ..world_model import WorldModel, DesignState
from ..world_model.design_state import DesignStage
from ..rl_optimizer.environment import ChipDesignEnv
from ..rl_optimizer.ppo_agent import PPOAgent
from ..rag import RAGRetriever

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Executes actions parsed from user intents.

    Connects natural language commands to backend functions:
    - "start a new design" → create design project
    - "set goal to minimize wirelength" → configure design goals
    - "run optimization" → start RL training loop
    - "what is placement?" → query RAG system
    """

    def __init__(self):
        """Initialize action executor with backend components"""
        self.simulation_engine = None
        self.world_model = None
        self.design_state = None
        self.rl_env = None
        self.rl_agent = None
        self.rag = RAGRetriever()

        # Current session state
        self.active_project = None
        self.design_goals = {
            'performance': 1.0,
            'power': 0.5,
            'area': 0.7
        }

        logger.info("ActionExecutor initialized")

    def execute(self, intent: ParsedIntent) -> Dict[str, Any]:
        """
        Execute an action based on parsed intent.

        Args:
            intent: Parsed intent from IntentParser

        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing action: {intent.action.value}")

        # Route to appropriate handler
        handlers = {
            ActionType.QUERY: self._handle_query,
            ActionType.CREATE_PROJECT: self._handle_create_project,
            ActionType.LOAD_DESIGN: self._handle_load_design,
            ActionType.SYNTHESIZE: self._handle_synthesize,
            ActionType.PLACE: self._handle_place,
            ActionType.ROUTE: self._handle_route,
            ActionType.ANALYZE_TIMING: self._handle_analyze_timing,
            ActionType.ANALYZE_POWER: self._handle_analyze_power,
            ActionType.OPTIMIZE: self._handle_optimize,
            ActionType.ADJUST_FLOORPLAN: self._handle_adjust_floorplan,
            ActionType.EXPORT_GDSII: self._handle_export_gdsii,
        }

        handler = handlers.get(intent.action)
        if not handler:
            return {
                'success': False,
                'message': f"Unknown action: {intent.action}",
                'data': None
            }

        try:
            result = handler(intent)
            return result
        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'data': None
            }

    def _handle_query(self, intent: ParsedIntent) -> Dict:
        """Handle informational queries using RAG"""
        query = intent.raw_text

        logger.info(f"RAG query: {query}")

        # Retrieve relevant documentation
        results = self.rag.retrieve(query, top_k=3)

        if not results:
            return {
                'success': True,
                'message': "I don't have specific documentation on that topic yet.",
                'data': {'query': query, 'results': []}
            }

        # Format context from results
        context = self.rag.retrieve_and_format(query, top_k=3)

        return {
            'success': True,
            'message': f"Retrieved {len(results)} relevant documents",
            'data': {
                'query': query,
                'results': results,
                'context': context
            }
        }

    def _handle_create_project(self, intent: ParsedIntent) -> Dict:
        """Create a new chip design project"""
        params = intent.parameters

        # Extract parameters
        process_node = params.get('process_node', '7nm')
        project_name = params.get('project_name', 'new_design')
        design_type = params.get('design_type', 'general')

        logger.info(f"Creating project: {project_name} ({process_node})")

        # Initialize backend components
        self.simulation_engine = SimulationEngine()
        self.world_model = WorldModel(process_node=process_node)
        self.design_state = DesignState(project_name=project_name)

        # Set process parameters
        self.design_state.process_node = process_node
        self.design_state.clock_period = params.get('clock_period', 10.0)  # 100MHz default

        # Extract design goals from intent
        for goal in intent.goals:
            if goal.value == 'minimize_power':
                self.design_goals['power'] = 1.0
                self.design_goals['performance'] = 0.5
            elif goal.value == 'maximize_performance':
                self.design_goals['performance'] = 1.0
                self.design_goals['power'] = 0.3
            elif goal.value == 'minimize_area':
                self.design_goals['area'] = 1.0
                self.design_goals['performance'] = 0.5

        self.active_project = project_name

        return {
            'success': True,
            'message': f"Created {process_node} design project '{project_name}'",
            'data': {
                'project_name': project_name,
                'process_node': process_node,
                'design_type': design_type,
                'goals': self.design_goals
            }
        }

    def _handle_load_design(self, intent: ParsedIntent) -> Dict:
        """Load an existing design from file"""
        file_path = intent.parameters.get('file_path')

        if not file_path:
            return {
                'success': False,
                'message': "No file path specified",
                'data': None
            }

        if not Path(file_path).exists():
            return {
                'success': False,
                'message': f"File not found: {file_path}",
                'data': None
            }

        # Ensure we have a project
        if not self.design_state:
            self._handle_create_project(intent)

        # Load RTL
        self.design_state.rtl_files = [file_path]
        self.design_state.top_module = intent.parameters.get('top_module', Path(file_path).stem)

        return {
            'success': True,
            'message': f"Loaded design from {file_path}",
            'data': {
                'file_path': file_path,
                'top_module': self.design_state.top_module
            }
        }

    def _handle_synthesize(self, intent: ParsedIntent) -> Dict:
        """Run synthesis on the design"""
        if not self.design_state or not self.design_state.rtl_files:
            return {
                'success': False,
                'message': "No design loaded. Please load a design first.",
                'data': None
            }

        logger.info("Running synthesis...")

        # Run synthesis
        output_netlist = f"/tmp/{self.design_state.project_name}_synth.v"

        result = self.simulation_engine.synthesis.synthesize(
            rtl_files=self.design_state.rtl_files,
            top_module=self.design_state.top_module,
            output_netlist=output_netlist,
            optimization_goal=intent.parameters.get('goal', 'balanced')
        )

        if result and result.get('cell_count', 0) > 0:
            self.design_state.netlist_file = output_netlist
            self.design_state.update_stage(DesignStage.SYNTHESIZED)

            return {
                'success': True,
                'message': f"Synthesis complete: {result['cell_count']} cells, {result.get('flip_flops', 0)} flip-flops",
                'data': result
            }
        else:
            return {
                'success': False,
                'message': "Synthesis failed",
                'data': result
            }

    def _handle_place(self, intent: ParsedIntent) -> Dict:
        """Run placement on the design"""
        if not self.design_state or not self.design_state.netlist_file:
            return {
                'success': False,
                'message': "No synthesized netlist. Please run synthesis first.",
                'data': None
            }

        logger.info("Running placement...")

        # Extract placement parameters from intent
        params = {
            'target_density': intent.parameters.get('density', 0.7),
            'wirelength_weight': 1.0 if 'wirelength' in intent.raw_text.lower() else 0.5,
            'routability_weight': 0.5,
        }

        output_def = f"/tmp/{self.design_state.project_name}_placed.def"

        result = self.simulation_engine.placement.place(
            netlist_file=self.design_state.netlist_file,
            def_file=intent.parameters.get('def_file', '/tmp/floorplan.def'),
            output_def=output_def,
            placement_params=params
        )

        if result.get('success'):
            self.design_state.def_file = output_def
            self.design_state.update_stage(DesignStage.PLACED)

            return {
                'success': True,
                'message': f"Placement complete: HPWL={result.get('hpwl', 0):.2f}",
                'data': result
            }
        else:
            return {
                'success': False,
                'message': "Placement failed",
                'data': result
            }

    def _handle_route(self, intent: ParsedIntent) -> Dict:
        """Run routing on the design"""
        return {
            'success': False,
            'message': "Routing not yet implemented",
            'data': None
        }

    def _handle_analyze_timing(self, intent: ParsedIntent) -> Dict:
        """Analyze timing of the design"""
        return {
            'success': False,
            'message': "Timing analysis not yet fully implemented",
            'data': None
        }

    def _handle_analyze_power(self, intent: ParsedIntent) -> Dict:
        """Analyze power of the design"""
        return {
            'success': False,
            'message': "Power analysis not yet fully implemented",
            'data': None
        }

    def _handle_optimize(self, intent: ParsedIntent) -> Dict:
        """Run RL optimization loop"""
        if not self.design_state:
            return {
                'success': False,
                'message': "No design loaded. Please create a project first.",
                'data': None
            }

        # Ensure design is synthesized
        if self.design_state.stage.value not in ['synthesized', 'placed', 'routed']:
            return {
                'success': False,
                'message': "Design must be synthesized before optimization. Please run synthesis first.",
                'data': None
            }

        logger.info("Starting RL optimization...")

        # Extract target metric from intent
        target_metric = intent.parameters.get('target_metric', 'balanced')

        # Update design goals based on target
        if 'wirelength' in target_metric.lower():
            self.design_goals = {'performance': 0.3, 'power': 0.2, 'area': 0.5}
        elif 'power' in target_metric.lower():
            self.design_goals = {'performance': 0.3, 'power': 1.0, 'area': 0.3}
        elif 'performance' in target_metric.lower() or 'timing' in target_metric.lower():
            self.design_goals = {'performance': 1.0, 'power': 0.3, 'area': 0.3}

        # Create RL environment
        self.rl_env = ChipDesignEnv(
            design_state=self.design_state,
            simulation_engine=self.simulation_engine,
            world_model=self.world_model,
            design_goals=self.design_goals
        )

        # Create PPO agent
        self.rl_agent = PPOAgent(
            env=self.rl_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            verbose=1
        )

        # Train for specified timesteps
        timesteps = intent.parameters.get('timesteps', 1000)

        logger.info(f"Training agent for {timesteps} timesteps...")
        self.rl_agent.learn(total_timesteps=timesteps)

        # Get training stats
        stats = self.rl_agent.get_training_stats()

        return {
            'success': True,
            'message': f"Optimization complete after {timesteps} timesteps",
            'data': {
                'timesteps': timesteps,
                'target_metric': target_metric,
                'stats': stats
            }
        }

    def _handle_adjust_floorplan(self, intent: ParsedIntent) -> Dict:
        """Adjust floorplan parameters"""
        return {
            'success': False,
            'message': "Floorplan adjustment not yet implemented",
            'data': None
        }

    def _handle_export_gdsii(self, intent: ParsedIntent) -> Dict:
        """Export design to GDSII format"""
        return {
            'success': False,
            'message': "GDSII export not yet implemented",
            'data': None
        }

    def get_status(self) -> Dict:
        """Get current status of the design"""
        if not self.design_state:
            return {
                'active': False,
                'message': "No active project"
            }

        return {
            'active': True,
            'project_name': self.design_state.project_name,
            'stage': self.design_state.stage.value,
            'process_node': self.design_state.process_node,
            'goals': self.design_goals,
            'rtl_files': self.design_state.rtl_files,
            'netlist_file': self.design_state.netlist_file,
            'def_file': self.design_state.def_file
        }
