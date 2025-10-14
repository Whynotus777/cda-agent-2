#!/usr/bin/env python3
"""
CDA Agent - "Insanely Great" Demo UI

A magical, user-friendly interface that makes chip design optimization
as simple as: Upload → Describe Goal → Click Go → See Results

Built with Streamlit for maximum polish and minimal code.
"""

import streamlit as st
import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.simulation_engine import SimulationEngine
from core.world_model import DesignState, TechLibrary
from core.world_model.design_state import DesignStage
from core.rl_optimizer.actions import ActionSpace
from demo.optimization_tracker import OptimizationTracker
from demo.report_generator import ReportGenerator


# Page config
st.set_page_config(
    page_title="CDA Agent - AI Chip Design",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #10b981;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background: #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


class DemoApp:
    """Main demo application"""

    def __init__(self):
        """Initialize demo app"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.design_uploaded = False
            st.session_state.optimization_running = False
            st.session_state.optimization_complete = False
            st.session_state.tracker = None
            st.session_state.report = None

    def render_header(self):
        """Render beautiful header"""
        st.markdown('<h1 class="main-header">🔬 CDA Agent</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="subtitle">AI-Powered Chip Design Optimization</p>',
            unsafe_allow_html=True
        )
        st.markdown("---")

    def render_upload_section(self):
        """Render file upload and goal input"""
        st.header("📁 Step 1: Upload Your Design")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload Verilog File (.v)",
                type=['v', 'sv'],
                help="Upload your RTL design file"
            )

            if uploaded_file:
                # Save uploaded file
                design_path = Path(f"/tmp/{uploaded_file.name}")
                design_path.write_bytes(uploaded_file.read())
                st.session_state.design_path = str(design_path)
                st.session_state.design_name = uploaded_file.name
                st.session_state.design_uploaded = True

                st.success(f"✓ Uploaded: {uploaded_file.name}")

                # Show preview
                with st.expander("📄 Preview Design"):
                    content = design_path.read_text()
                    st.code(content[:1000] + "..." if len(content) > 1000 else content, language="verilog")

        with col2:
            st.metric(
                "Design Status",
                "Ready" if st.session_state.design_uploaded else "Waiting",
                "File uploaded" if st.session_state.design_uploaded else "No file"
            )

    def render_goal_section(self):
        """Render natural language goal input"""
        st.header("🎯 Step 2: Describe Your Goal")

        goal_presets = {
            "Highest Clock Speed": "Optimize for maximum clock frequency with minimal setup/hold violations",
            "Lowest Power": "Minimize power consumption while maintaining acceptable performance",
            "Smallest Area": "Reduce chip area and cell count for cost optimization",
            "Balanced PPA": "Balance performance, power, and area equally",
            "Custom": ""
        }

        col1, col2 = st.columns([3, 1])

        with col1:
            preset = st.selectbox(
                "Quick Presets",
                list(goal_presets.keys()),
                help="Choose a preset or write your own goal"
            )

            if preset == "Custom":
                goal_text = st.text_area(
                    "Describe your optimization goal in plain English",
                    placeholder="Example: I want the highest possible clock speed with power under 100mW",
                    height=100
                )
            else:
                goal_text = st.text_area(
                    "Optimization Goal",
                    value=goal_presets[preset],
                    height=100
                )

            st.session_state.optimization_goal = goal_text

        with col2:
            st.metric("Episodes", 50, "iterations")
            st.metric("Time Est.", "~10 min", "approximate")

    def render_go_button(self):
        """Render the magical 'Go' button"""
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            ready = st.session_state.design_uploaded and st.session_state.optimization_goal

            if not ready:
                st.warning("⚠️ Please upload a design and set your goal")

            if st.button(
                "🚀 Optimize My Design",
                disabled=not ready or st.session_state.optimization_running,
                key="go_button"
            ):
                st.session_state.optimization_running = True
                st.rerun()

    def render_optimization_progress(self):
        """Render real-time optimization progress"""
        st.header("⚙️ Optimization in Progress")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Metrics display
        col1, col2, col3, col4 = st.columns(4)

        metric_area = col1.empty()
        metric_power = col2.empty()
        metric_timing = col3.empty()
        metric_wire = col4.empty()

        # Live chart
        chart_placeholder = st.empty()

        # Run optimization with updates
        tracker = OptimizationTracker()
        st.session_state.tracker = tracker

        try:
            # Initialize components
            simulation_engine = SimulationEngine()
            design_state = DesignState(project_name="demo")
            tech_library = TechLibrary(process_node="7nm")

            # Parse goal
            goal_weights = self._parse_goal(st.session_state.optimization_goal)

            # Run synthesis first
            status_text.text("🔧 Running synthesis...")
            synthesis_result = simulation_engine.synthesis.synthesize(
                rtl_files=[st.session_state.design_path],
                top_module=self._extract_top_module(st.session_state.design_path),
                output_netlist="/tmp/demo_synth.v",
                optimization_goal=goal_weights.get('optimization_focus', 'balanced')
            )

            if not synthesis_result or synthesis_result.get('cell_count', 0) == 0:
                st.error("❌ Synthesis failed. Please check your Verilog file.")
                st.session_state.optimization_running = False
                return

            design_state.netlist_file = "/tmp/demo_synth.v"
            design_state.update_stage(DesignStage.SYNTHESIZED)

            status_text.text(f"✓ Synthesis complete: {synthesis_result['cell_count']} cells")
            time.sleep(1)

            # Initialize RL action space
            action_space = ActionSpace(
                simulation_engine=simulation_engine,
                design_state=design_state,
                world_model=tech_library
            )

            # Run optimization episodes
            max_episodes = 50

            for episode in range(max_episodes):
                progress = (episode + 1) / max_episodes
                progress_bar.progress(progress)
                status_text.text(f"🔄 Episode {episode + 1}/{max_episodes}: Exploring actions...")

                # Select action (simplified for demo - random exploration)
                import random
                action_idx = random.randint(0, action_space.get_action_count() - 1)
                action_name = action_space.get_action_name(action_idx)

                # Execute action
                result = action_space.execute_action(action_idx)

                # Track episode
                tracker.log_episode(
                    episode=episode,
                    action_idx=action_idx,
                    action_name=action_name,
                    metrics_before=result.get('metrics_before', {}),
                    metrics_after=result.get('metrics_after', {}),
                    reward=result.get('reward', 0.0),
                    success=result.get('success', True)
                )

                # Update metrics display
                current_metrics = tracker.get_current_metrics()

                metric_area.metric(
                    "Area Utilization",
                    f"{current_metrics.get('area_util', 0):.1f}%",
                    f"{current_metrics.get('area_delta', 0):+.1f}%"
                )
                metric_power.metric(
                    "Power",
                    f"{current_metrics.get('power', 0):.1f}mW",
                    f"{current_metrics.get('power_delta', 0):+.1f}%"
                )
                metric_timing.metric(
                    "WNS",
                    f"{current_metrics.get('wns', 0):.0f}ps",
                    f"{current_metrics.get('wns_delta', 0):+.0f}ps"
                )
                metric_wire.metric(
                    "Wirelength",
                    f"{current_metrics.get('wirelength', 0):.0f}μm",
                    f"{current_metrics.get('wire_delta', 0):+.1f}%"
                )

                # Update live chart
                if episode > 0:
                    chart_placeholder.plotly_chart(
                        tracker.create_progress_chart(),
                        use_container_width=True
                    )

                # Small delay for visual effect
                time.sleep(0.1)

            # Generate final report
            st.session_state.report = ReportGenerator(tracker).generate()
            st.session_state.optimization_complete = True
            st.session_state.optimization_running = False

            status_text.text("✨ Optimization Complete!")
            progress_bar.progress(1.0)

        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            st.session_state.optimization_running = False
            import traceback
            st.code(traceback.format_exc())

    def render_results_story(self):
        """Render the beautiful results story"""
        st.header("📊 Optimization Results Story")

        report = st.session_state.report
        tracker = st.session_state.tracker

        # Hero metrics
        st.subheader("🎯 What I Achieved")

        col1, col2, col3, col4 = st.columns(4)

        improvements = report['improvements']

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{improvements['wirelength']['delta']:+.1f}%</h3>
                <p>Wirelength</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{improvements['timing']['delta']:+.1f}%</h3>
                <p>Timing Slack</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{improvements['power']['delta']:+.1f}%</h3>
                <p>Power</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{improvements['area']['delta']:+.1f}%</h3>
                <p>Area</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # The Story
        st.subheader("📖 Optimization Journey")

        narrative = report['narrative']
        st.markdown(f"""
        <div class="info-box">
        <h4>Here's what happened:</h4>
        <p><strong>Episodes Run:</strong> {narrative['total_episodes']}</p>
        <p><strong>Actions Taken:</strong> {narrative['total_actions']}</p>
        <p><strong>Success Rate:</strong> {narrative['success_rate']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("**Key Insights:**")
        for insight in narrative['key_insights']:
            st.write(f"- {insight}")

        st.markdown("---")

        # Visual comparisons
        st.subheader("📈 Progress Over Time")

        tab1, tab2, tab3 = st.tabs(["Metrics Trend", "Action Distribution", "Best Actions"])

        with tab1:
            st.plotly_chart(
                tracker.create_progress_chart(),
                use_container_width=True
            )

        with tab2:
            st.plotly_chart(
                tracker.create_action_distribution_chart(),
                use_container_width=True
            )

        with tab3:
            best_actions_df = pd.DataFrame(report['best_actions'])
            st.dataframe(best_actions_df, use_container_width=True)

        st.markdown("---")

        # Interactive follow-up
        st.subheader("💬 What's Next?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>Are you satisfied with these results?</h4>
            <p>I optimized your design based on your goal. If you'd like different tradeoffs,
            I can run another optimization focusing on different metrics.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if st.button("🔄 Try Different Goal"):
                st.session_state.optimization_complete = False
                st.rerun()

            if st.button("💾 Export Report"):
                self._export_report(report)

            if st.button("🏁 Start New Design"):
                self._reset_demo()
                st.rerun()

    def _parse_goal(self, goal_text: str) -> Dict:
        """Parse natural language goal into weights"""
        goal_lower = goal_text.lower()

        weights = {
            'performance': 0.33,
            'power': 0.33,
            'area': 0.33,
            'optimization_focus': 'balanced'
        }

        # Simple keyword matching
        if any(word in goal_lower for word in ['speed', 'frequency', 'clock', 'fast', 'performance']):
            weights['performance'] = 0.7
            weights['power'] = 0.15
            weights['area'] = 0.15
            weights['optimization_focus'] = 'speed'
        elif any(word in goal_lower for word in ['power', 'energy', 'low power', 'battery']):
            weights['power'] = 0.7
            weights['performance'] = 0.15
            weights['area'] = 0.15
            weights['optimization_focus'] = 'area'
        elif any(word in goal_lower for word in ['area', 'size', 'small', 'compact', 'cost']):
            weights['area'] = 0.7
            weights['power'] = 0.15
            weights['performance'] = 0.15
            weights['optimization_focus'] = 'area'

        return weights

    def _extract_top_module(self, verilog_path: str) -> str:
        """Extract top module name from Verilog"""
        content = Path(verilog_path).read_text()
        import re
        match = re.search(r'module\s+(\w+)', content)
        return match.group(1) if match else "top"

    def _export_report(self, report: Dict):
        """Export report as JSON"""
        report_path = Path("/tmp/optimization_report.json")
        report_path.write_text(json.dumps(report, indent=2))
        st.success(f"✓ Report exported to {report_path}")

    def _reset_demo(self):
        """Reset demo to initial state"""
        st.session_state.design_uploaded = False
        st.session_state.optimization_running = False
        st.session_state.optimization_complete = False
        st.session_state.tracker = None
        st.session_state.report = None

    def run(self):
        """Main app entry point"""
        self.render_header()

        if not st.session_state.optimization_complete:
            if not st.session_state.optimization_running:
                # Upload and goal sections
                self.render_upload_section()
                st.markdown("---")
                self.render_goal_section()
                self.render_go_button()
            else:
                # Optimization in progress
                self.render_optimization_progress()
        else:
            # Results story
            self.render_results_story()


def main():
    """Run the demo app"""
    app = DemoApp()
    app.run()


if __name__ == "__main__":
    main()
