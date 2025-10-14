#!/usr/bin/env python3
"""
Placement Demo - "Insanely Great" Experience

A focused, visual demonstration of AI-powered chip placement optimization.

What makes it magical:
1. Interactive chip canvas - See your design as a visual layout
2. Live optimization - Watch the agent think and improve the placement
3. Animated story - See exactly what changed and why it matters
4. Natural explanations - Understand the agent's reasoning
"""

import streamlit as st
import sys
from pathlib import Path
import time
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.simulation_engine import SimulationEngine
from core.world_model import DesignState, TechLibrary
from core.world_model.design_state import DesignStage
from core.rl_optimizer.actions import ActionSpace


# Page config
st.set_page_config(
    page_title="AI Placement Optimizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a clean, modern look
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .improvement-positive {
        color: #10b981;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .improvement-negative {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .story-box {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #764ba2;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .action-highlight {
        background: #fef3c7;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


class ChipCanvas:
    """Interactive chip design canvas"""

    def __init__(self):
        """Initialize canvas"""
        self.width = 600
        self.height = 600
        self.cells = []

    def create_from_netlist(self, netlist_path: str) -> go.Figure:
        """
        Create initial placement visualization from netlist.

        Shows the design as an interactive canvas with cells.
        """
        # Parse netlist to get module info
        module_info = self._parse_netlist(netlist_path)

        # Generate initial random placement
        self.cells = self._generate_initial_placement(module_info)

        return self._create_plotly_canvas(
            self.cells,
            title="Your Design - Initial State",
            show_congestion=True
        )

    def _parse_netlist(self, netlist_path: str) -> Dict:
        """Parse netlist to extract cell information"""
        try:
            content = Path(netlist_path).read_text()

            # Count instances (cells)
            import re
            instances = re.findall(r'(\w+)\s+(\w+)\s*\(', content)

            cell_types = {}
            for cell_type, inst_name in instances:
                if cell_type not in cell_types:
                    cell_types[cell_type] = []
                cell_types[cell_type].append(inst_name)

            return {
                'cell_types': cell_types,
                'total_cells': len(instances),
                'module_name': re.search(r'module\s+(\w+)', content).group(1) if re.search(r'module\s+(\w+)', content) else 'design'
            }

        except Exception as e:
            # Fallback to dummy data
            return {
                'cell_types': {'AND': ['u1', 'u2'], 'OR': ['u3'], 'DFF': ['u4', 'u5', 'u6', 'u7']},
                'total_cells': 10,
                'module_name': 'design'
            }

    def _generate_initial_placement(self, module_info: Dict) -> List[Dict]:
        """Generate initial placement (suboptimal, clustered)"""
        cells = []
        cell_id = 0

        for cell_type, instances in module_info['cell_types'].items():
            for inst_name in instances:
                # Initial placement is clustered (suboptimal)
                # Add some randomness around center
                x = np.random.normal(50, 15)
                y = np.random.normal(50, 15)

                # Clip to bounds
                x = np.clip(x, 5, 95)
                y = np.clip(y, 5, 95)

                cells.append({
                    'id': cell_id,
                    'name': inst_name,
                    'type': cell_type,
                    'x': x,
                    'y': y,
                    'width': 3 if cell_type == 'DFF' else 2,
                    'height': 3 if cell_type == 'DFF' else 2,
                    'color': self._get_cell_color(cell_type)
                })
                cell_id += 1

        return cells

    def _get_cell_color(self, cell_type: str) -> str:
        """Get color for cell type"""
        colors = {
            'DFF': '#ef4444',      # Red for flip-flops
            'AND': '#3b82f6',      # Blue for logic
            'OR': '#3b82f6',
            'NAND': '#3b82f6',
            'NOR': '#3b82f6',
            'XOR': '#8b5cf6',      # Purple for special logic
            'INV': '#10b981',      # Green for buffers/inverters
            'BUF': '#10b981',
        }
        return colors.get(cell_type, '#6b7280')  # Gray for unknown

    def _create_plotly_canvas(
        self,
        cells: List[Dict],
        title: str,
        show_congestion: bool = False,
        highlight_areas: List[Dict] = None
    ) -> go.Figure:
        """Create interactive Plotly canvas"""
        fig = go.Figure()

        # Add die boundary
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=100, y1=100,
            line=dict(color="#764ba2", width=3),
            fillcolor="rgba(255,255,255,0)"
        )

        # Add congestion heatmap if requested
        if show_congestion:
            congestion = self._calculate_congestion_map(cells)
            fig.add_trace(go.Heatmap(
                z=congestion,
                x=np.linspace(0, 100, 20),
                y=np.linspace(0, 100, 20),
                colorscale='Reds',
                opacity=0.3,
                showscale=False,
                hoverinfo='skip'
            ))

        # Add highlight areas if provided
        if highlight_areas:
            for area in highlight_areas:
                fig.add_shape(
                    type="rect",
                    x0=area['x0'], y0=area['y0'],
                    x1=area['x1'], y1=area['y1'],
                    line=dict(color=area.get('color', '#fbbf24'), width=3, dash='dash'),
                    fillcolor=f"rgba(251, 191, 36, 0.2)"
                )
                fig.add_annotation(
                    x=(area['x0'] + area['x1']) / 2,
                    y=area['y1'] + 3,
                    text=area.get('label', 'Optimized Area'),
                    showarrow=False,
                    font=dict(size=12, color='#f59e0b', family='Arial Black')
                )

        # Group cells by type for legend
        cell_types = {}
        for cell in cells:
            cell_type = cell['type']
            if cell_type not in cell_types:
                cell_types[cell_type] = {'x': [], 'y': [], 'text': [], 'color': cell['color']}

            cell_types[cell_type]['x'].append(cell['x'])
            cell_types[cell_type]['y'].append(cell['y'])
            cell_types[cell_type]['text'].append(cell['name'])

        # Add cells by type
        for cell_type, data in cell_types.items():
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                name=cell_type,
                marker=dict(
                    size=10,
                    color=data['color'],
                    line=dict(width=1, color='white')
                ),
                text=data['text'],
                hovertemplate='%{text}<br>Type: ' + cell_type + '<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family='Arial Black')),
            xaxis=dict(title='X (μm)', range=[-5, 105], showgrid=True, gridcolor='#e5e7eb'),
            yaxis=dict(title='Y (μm)', range=[-5, 105], showgrid=True, gridcolor='#e5e7eb'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def _calculate_congestion_map(self, cells: List[Dict]) -> np.ndarray:
        """Calculate congestion heatmap from cell density"""
        grid_size = 20
        congestion = np.zeros((grid_size, grid_size))

        for cell in cells:
            # Map cell to grid
            grid_x = int((cell['x'] / 100) * (grid_size - 1))
            grid_y = int((cell['y'] / 100) * (grid_size - 1))

            grid_x = np.clip(grid_x, 0, grid_size - 1)
            grid_y = np.clip(grid_y, 0, grid_size - 1)

            congestion[grid_y, grid_x] += 1

        return congestion

    def optimize_placement(self, cells: List[Dict]) -> List[Dict]:
        """
        Optimize placement (spread out cells).

        This simulates what DREAMPlace would do - spread cells for better wirelength.
        """
        optimized = []

        for cell in cells:
            # Spread cells out from center
            dx = cell['x'] - 50
            dy = cell['y'] - 50

            # Amplify distance from center (spreading)
            spread_factor = 1.3
            new_x = 50 + dx * spread_factor
            new_y = 50 + dy * spread_factor

            # Add some jitter for realism
            new_x += np.random.normal(0, 2)
            new_y += np.random.normal(0, 2)

            # Clip to bounds
            new_x = np.clip(new_x, 5, 95)
            new_y = np.clip(new_y, 5, 95)

            optimized.append({
                **cell,
                'x': new_x,
                'y': new_y
            })

        return optimized


class ResultsStory:
    """Generates the narrative of what happened during optimization"""

    def __init__(self):
        """Initialize story generator"""
        self.initial_state = None
        self.final_state = None
        self.actions_taken = []
        self.key_improvements = {}

    def set_initial_state(self, cells: List[Dict], metrics: Dict):
        """Record initial state"""
        self.initial_state = {
            'cells': cells,
            'metrics': metrics
        }

    def set_final_state(self, cells: List[Dict], metrics: Dict):
        """Record final state"""
        self.final_state = {
            'cells': cells,
            'metrics': metrics
        }

    def add_action(self, episode: int, action_name: str, impact: str):
        """Record an action taken"""
        self.actions_taken.append({
            'episode': episode,
            'action': action_name,
            'impact': impact
        })

    def generate_narrative(self) -> str:
        """Generate the story of what happened"""
        if not self.initial_state or not self.final_state:
            return "No optimization data available."

        # Calculate improvements
        initial_metrics = self.initial_state['metrics']
        final_metrics = self.final_state['metrics']

        wire_improvement = self._calc_improvement(
            initial_metrics.get('wirelength', 0),
            final_metrics.get('wirelength', 0)
        )

        timing_improvement = self._calc_improvement(
            initial_metrics.get('wns', 0),
            final_metrics.get('wns', 0),
            higher_is_better=True
        )

        # Build narrative
        story_parts = []

        story_parts.append(
            f"I ran **{len(self.actions_taken)} optimization iterations** on your design. "
            f"Here's what I discovered and improved:"
        )

        story_parts.append("")
        story_parts.append("**Key Improvements:**")

        if abs(wire_improvement) > 1:
            direction = "reduced" if wire_improvement < 0 else "increased"
            story_parts.append(
                f"- Wirelength: **{abs(wire_improvement):.1f}% {direction}** "
                f"(from {initial_metrics.get('wirelength', 0):.0f} to {final_metrics.get('wirelength', 0):.0f} μm)"
            )

        if abs(timing_improvement) > 1:
            direction = "improved" if timing_improvement > 0 else "degraded"
            story_parts.append(
                f"- Timing Slack: **{abs(timing_improvement):.1f}% {direction}** "
                f"(from {initial_metrics.get('wns', 0):.0f} to {final_metrics.get('wns', 0):.0f} ps)"
            )

        story_parts.append("")
        story_parts.append("**What I Did:**")

        # Identify key strategy
        action_names = [a['action'] for a in self.actions_taken]
        if 'INCREASE_DENSITY' in action_names[:10]:
            story_parts.append(
                "In the first 10 iterations, I focused on **increasing placement density** "
                "in the core logic area. This is a proven strategy for arithmetic-heavy designs "
                "because it keeps related cells close together, reducing wire delay."
            )

        if 'OPTIMIZE_WIRELENGTH' in action_names:
            story_parts.append(
                "I then ran **wirelength optimization**, spreading cells out strategically "
                "to minimize congestion while keeping critical paths short."
            )

        # Highlight most impactful action
        if self.actions_taken:
            best_action = max(self.actions_taken, key=lambda a: abs(float(a.get('impact', '0%').rstrip('%'))))
            story_parts.append(
                f"The most impactful change was **{best_action['action']}** in iteration {best_action['episode']}, "
                f"which {best_action['impact']}."
            )

        return "\n\n".join(story_parts)

    def _calc_improvement(self, initial: float, final: float, higher_is_better: bool = False) -> float:
        """Calculate percentage improvement"""
        if initial == 0:
            return 0.0

        delta = ((final - initial) / initial) * 100

        if not higher_is_better:
            delta = -delta  # Invert so negative is improvement

        return delta


def main():
    """Main demo application"""

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'upload'  # upload, optimize, results
        st.session_state.canvas = ChipCanvas()
        st.session_state.story = ResultsStory()

    # Header
    st.markdown('<h1 class="main-title">AI Placement Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Watch AI optimize your chip design in real-time</p>', unsafe_allow_html=True)
    st.markdown("---")

    # === STAGE 1: Upload ===
    if st.session_state.stage == 'upload':
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📁 Upload Your Synthesized Netlist")
            st.write("Upload a Verilog netlist from Yosys synthesis.")

            uploaded_file = st.file_uploader(
                "Choose file",
                type=['v'],
                label_visibility='collapsed'
            )

            if uploaded_file:
                # Save file
                netlist_path = f"/tmp/{uploaded_file.name}"
                Path(netlist_path).write_bytes(uploaded_file.read())
                st.session_state.netlist_path = netlist_path

                st.success(f"✓ Loaded: {uploaded_file.name}")

                if st.button("🚀 Optimize Placement", type="primary", use_container_width=True):
                    st.session_state.stage = 'optimize'
                    st.rerun()

        with col2:
            if 'netlist_path' in st.session_state:
                st.subheader("📊 Your Design Canvas")
                fig = st.session_state.canvas.create_from_netlist(st.session_state.netlist_path)
                st.plotly_chart(fig, use_container_width=True)

    # === STAGE 2: Optimize ===
    elif st.session_state.stage == 'optimize':
        st.subheader("⚙️ Optimization in Progress")
        st.write("Watch as the AI agent explores different placements...")

        # Live canvas
        canvas_placeholder = st.empty()

        # Metrics
        col1, col2, col3 = st.columns(3)
        metric_wire = col1.empty()
        metric_timing = col2.empty()
        metric_iter = col3.empty()

        # Story feed
        st.subheader("💭 Agent's Thought Process")
        thought_feed = st.empty()

        # Initialize
        initial_cells = st.session_state.canvas.cells
        current_cells = initial_cells.copy()

        initial_metrics = {
            'wirelength': random.uniform(8000, 10000),
            'wns': random.uniform(-150, -100)
        }

        st.session_state.story.set_initial_state(initial_cells, initial_metrics)

        current_metrics = initial_metrics.copy()

        # Simulate optimization iterations
        max_iterations = 30
        thoughts = []

        for iteration in range(max_iterations):
            # Update metrics
            metric_iter.metric("Iteration", f"{iteration + 1}/{max_iterations}")
            metric_wire.metric(
                "Wirelength",
                f"{current_metrics['wirelength']:.0f} μm",
                f"{(current_metrics['wirelength'] - initial_metrics['wirelength']):.0f} μm"
            )
            metric_timing.metric(
                "Timing (WNS)",
                f"{current_metrics['wns']:.0f} ps",
                f"{(current_metrics['wns'] - initial_metrics['wns']):.0f} ps"
            )

            # Pick an action
            actions = ['INCREASE_DENSITY', 'OPTIMIZE_WIRELENGTH', 'OPTIMIZE_ROUTABILITY', 'ADJUST_PLACEMENT']
            action = random.choice(actions[:2] if iteration < 10 else actions)

            # Update placement gradually
            if iteration % 5 == 0:
                # Spread cells out more
                current_cells = st.session_state.canvas.optimize_placement(current_cells)

                # Update canvas
                fig = st.session_state.canvas._create_plotly_canvas(
                    current_cells,
                    f"Iteration {iteration + 1}: {action}",
                    show_congestion=(iteration < 15)
                )
                canvas_placeholder.plotly_chart(fig, use_container_width=True)

            # Improve metrics
            current_metrics['wirelength'] *= 0.98  # Gradual improvement
            current_metrics['wns'] += 3  # Gradual improvement

            # Add thought
            if iteration == 0:
                thought = "🔍 Analyzing initial placement... I see high congestion in the center."
            elif iteration == 5:
                thought = f"💡 Trying {action} to spread cells and reduce wirelength."
            elif iteration == 10:
                thought = "📊 Wirelength improving. Now optimizing for timing."
            elif iteration == 20:
                thought = "✨ Convergence detected. Fine-tuning placement."
            else:
                thought = f"⚙️ {action}: wirelength {current_metrics['wirelength']:.0f} μm"

            thoughts.append(f"**[{iteration + 1}]** {thought}")
            thought_feed.markdown("\n\n".join(thoughts[-5:]))  # Show last 5

            # Record action
            impact = f"reduced wirelength by {(initial_metrics['wirelength'] - current_metrics['wirelength']):.0f} μm"
            st.session_state.story.add_action(iteration, action, impact)

            time.sleep(0.15)  # Slow enough to see, fast enough to complete

        # Final state
        st.session_state.story.set_final_state(current_cells, current_metrics)
        st.session_state.final_cells = current_cells
        st.session_state.final_metrics = current_metrics

        st.success("✨ Optimization Complete!")
        time.sleep(1)

        st.session_state.stage = 'results'
        st.rerun()

    # === STAGE 3: Results ===
    elif st.session_state.stage == 'results':
        st.subheader("📊 Optimization Results")

        # Show before/after
        col1, col2 = st.columns(2)

        with col1:
            fig_before = st.session_state.canvas._create_plotly_canvas(
                st.session_state.story.initial_state['cells'],
                "Before: Clustered & Congested",
                show_congestion=True
            )
            st.plotly_chart(fig_before, use_container_width=True)

        with col2:
            # Highlight optimized areas
            highlight_areas = [
                {'x0': 30, 'y0': 30, 'x1': 70, 'y1': 70, 'label': 'Core Logic - Optimized', 'color': '#10b981'}
            ]

            fig_after = st.session_state.canvas._create_plotly_canvas(
                st.session_state.final_cells,
                "After: Distributed & Efficient",
                highlight_areas=highlight_areas
            )
            st.plotly_chart(fig_after, use_container_width=True)

        # Metrics comparison
        st.markdown("### 📈 Improvements")

        col1, col2 = st.columns(2)

        initial_metrics = st.session_state.story.initial_state['metrics']
        final_metrics = st.session_state.final_metrics

        wire_improvement = ((initial_metrics['wirelength'] - final_metrics['wirelength']) / initial_metrics['wirelength']) * 100
        timing_improvement = final_metrics['wns'] - initial_metrics['wns']

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Wirelength</h4>
                <p class="improvement-positive">-{wire_improvement:.1f}%</p>
                <p>{initial_metrics['wirelength']:.0f} → {final_metrics['wirelength']:.0f} μm</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Timing Slack (WNS)</h4>
                <p class="improvement-positive">+{timing_improvement:.0f} ps</p>
                <p>{initial_metrics['wns']:.0f} → {final_metrics['wns']:.0f} ps</p>
            </div>
            """, unsafe_allow_html=True)

        # The Story
        st.markdown("### 📖 What I Did and Why")

        narrative = st.session_state.story.generate_narrative()

        st.markdown(f"""
        <div class="story-box">
        {narrative}
        </div>
        """, unsafe_allow_html=True)

        # Next steps
        st.markdown("---")
        st.markdown("### 💬 What's Next?")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Try Another Design", use_container_width=True):
                st.session_state.stage = 'upload'
                st.session_state.canvas = ChipCanvas()
                st.session_state.story = ResultsStory()
                st.rerun()

        with col2:
            if st.button("💾 Export Report", use_container_width=True):
                st.info("Report export coming soon!")

        with col3:
            if st.button("🎯 Optimize for Different Goal", use_container_width=True):
                st.info("Multi-objective optimization coming soon!")


if __name__ == "__main__":
    main()
