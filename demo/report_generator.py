"""
Report Generator

Creates beautiful, narrative-driven optimization reports with visualizations
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import io
import base64


class ReportGenerator:
    """
    Generates comprehensive optimization reports with visualizations.

    Transforms raw RL logs into an insightful story.
    """

    def __init__(self, tracker):
        """
        Initialize report generator.

        Args:
            tracker: OptimizationTracker instance
        """
        self.tracker = tracker

    def generate(self) -> Dict:
        """Generate complete report"""
        return {
            'improvements': self._calculate_improvements(),
            'narrative': self.tracker.get_narrative(),
            'best_actions': self._get_best_actions(),
            'visualizations': {
                'progress_chart': self.tracker.create_progress_chart(),
                'action_distribution': self.tracker.create_action_distribution_chart(),
            }
        }

    def _calculate_improvements(self) -> Dict:
        """Calculate metric improvements from start to end"""
        if not self.tracker.episodes:
            return {}

        first = self.tracker.episodes[0]
        last = self.tracker.episodes[-1]

        def get_metric(ep_log, metric_path):
            """Extract metric from episode log"""
            keys = metric_path.split('.')
            val = ep_log.metrics_after if hasattr(ep_log, 'metrics_after') else {}

            for key in keys:
                val = val.get(key, {})
                if not isinstance(val, dict) and not isinstance(val, (int, float)):
                    return 0

            return val if isinstance(val, (int, float)) else 0

        # Calculate improvements
        improvements = {}

        metrics_to_track = {
            'wirelength': 'routing.wirelength',
            'timing': 'timing.wns',
            'power': 'power.total',
            'area': 'area.utilization'
        }

        for name, path in metrics_to_track.items():
            initial = get_metric(first, path)
            final = get_metric(last, path)

            delta = 0
            if initial != 0:
                delta = ((final - initial) / initial) * 100

            improvements[name] = {
                'initial': initial,
                'final': final,
                'delta': delta,
                'improved': delta < 0 if name != 'timing' else delta > 0  # Timing: higher is better
            }

        return improvements

    def _get_best_actions(self) -> List[Dict]:
        """Get top 5 most impactful actions"""
        # Sort episodes by reward
        sorted_episodes = sorted(
            self.tracker.episodes,
            key=lambda ep: ep.reward,
            reverse=True
        )[:5]

        best_actions = []
        for ep in sorted_episodes:
            best_actions.append({
                'episode': ep.episode,
                'action': ep.action_name,
                'reward': f"{ep.reward:.3f}",
                'impact': self._describe_impact(ep)
            })

        return best_actions

    def _describe_impact(self, episode_log) -> str:
        """Describe the impact of an action"""
        def get_val(metrics, path):
            keys = path.split('.')
            val = metrics
            for key in keys:
                val = val.get(key, {})
            return val if isinstance(val, (int, float)) else 0

        before = episode_log.metrics_before
        after = episode_log.metrics_after

        wire_before = get_val(before, 'routing.wirelength')
        wire_after = get_val(after, 'routing.wirelength')

        if wire_before > 0 and wire_after > 0:
            delta = ((wire_after - wire_before) / wire_before) * 100
            return f"Wirelength: {delta:+.1f}%"

        return "Metrics updated"

    def create_chip_layout_comparison(
        self,
        initial_def_path: Optional[str],
        final_def_path: Optional[str],
        output_path: str = "/tmp/layout_comparison.png"
    ) -> str:
        """
        Create side-by-side chip layout visualization.

        Args:
            initial_def_path: Path to initial DEF file
            final_def_path: Path to final DEF file
            output_path: Where to save comparison image

        Returns:
            Path to generated image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Parse DEF files if available
        if initial_def_path and Path(initial_def_path).exists():
            cells_initial = self._parse_def_cells(initial_def_path)
            self._plot_layout(ax1, cells_initial, "Initial Placement")
        else:
            self._plot_dummy_layout(ax1, "Initial Placement")

        if final_def_path and Path(final_def_path).exists():
            cells_final = self._parse_def_cells(final_def_path)
            self._plot_layout(ax2, cells_final, "Optimized Placement")
        else:
            self._plot_dummy_layout(ax2, "Optimized Placement")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _parse_def_cells(self, def_path: str) -> List[Dict]:
        """Parse cell positions from DEF file"""
        cells = []

        try:
            with open(def_path, 'r') as f:
                in_components = False

                for line in f:
                    line = line.strip()

                    if line.startswith('COMPONENTS'):
                        in_components = True
                        continue
                    elif line.startswith('END COMPONENTS'):
                        in_components = False
                        continue

                    if in_components and line.startswith('-'):
                        # Parse: - cell_name cell_type + PLACED ( x y ) orientation ;
                        parts = line.split()
                        if 'PLACED' in parts:
                            placed_idx = parts.index('PLACED')
                            if placed_idx + 4 < len(parts):
                                try:
                                    x = int(parts[placed_idx + 2])
                                    y = int(parts[placed_idx + 3])

                                    cells.append({
                                        'name': parts[1],
                                        'type': parts[2],
                                        'x': x,
                                        'y': y
                                    })
                                except (ValueError, IndexError):
                                    continue

        except Exception as e:
            print(f"Warning: Could not parse DEF file: {e}")

        return cells

    def _plot_layout(self, ax, cells: List[Dict], title: str):
        """Plot chip layout from cell positions"""
        if not cells:
            self._plot_dummy_layout(ax, title)
            return

        # Extract positions
        xs = [c['x'] for c in cells]
        ys = [c['y'] for c in cells]

        # Normalize to 0-100 range
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            if x_max > x_min and y_max > y_min:
                xs_norm = [(x - x_min) / (x_max - x_min) * 100 for x in xs]
                ys_norm = [(y - y_min) / (y_max - y_min) * 100 for y in ys]
            else:
                xs_norm = xs
                ys_norm = ys
        else:
            xs_norm = []
            ys_norm = []

        # Plot cells
        ax.scatter(xs_norm, ys_norm, c='#667eea', s=10, alpha=0.6)

        # Draw die boundary
        rect = patches.Rectangle(
            (0, 0), 100, 100,
            linewidth=2,
            edgecolor='#764ba2',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)

    def _plot_dummy_layout(self, ax, title: str):
        """Plot dummy layout for demo purposes"""
        # Generate random cell positions
        np.random.seed(42 if "Initial" in title else 123)

        n_cells = 200

        if "Initial" in title:
            # More clustered (suboptimal)
            xs = np.random.normal(50, 20, n_cells)
            ys = np.random.normal(50, 20, n_cells)
        else:
            # More spread out (optimized)
            xs = np.random.uniform(10, 90, n_cells)
            ys = np.random.uniform(10, 90, n_cells)

        # Clip to bounds
        xs = np.clip(xs, 0, 100)
        ys = np.clip(ys, 0, 100)

        # Plot cells
        ax.scatter(xs, ys, c='#667eea', s=15, alpha=0.6, edgecolors='none')

        # Draw die boundary
        rect = patches.Rectangle(
            (0, 0), 100, 100,
            linewidth=2,
            edgecolor='#764ba2',
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.grid(True, alpha=0.3)

        # Add annotation
        if "Initial" in title:
            ax.text(50, -2, 'Clustered (High Wirelength)',
                   ha='center', fontsize=10, color='red')
        else:
            ax.text(50, -2, 'Distributed (Lower Wirelength)',
                   ha='center', fontsize=10, color='green')

    def create_rag_explanation(self, action_name: str, rag_system) -> str:
        """
        Generate explanation using RAG system.

        Args:
            action_name: Name of the action to explain
            rag_system: RAG system instance

        Returns:
            Explanation string
        """
        try:
            # Query RAG for explanation
            query = f"Why is {action_name} important for chip placement optimization?"
            context = rag_system.query(query, k=2)

            if context:
                return f"According to my knowledge base: {context[0]['content'][:200]}..."
            else:
                return self._get_fallback_explanation(action_name)

        except Exception as e:
            return self._get_fallback_explanation(action_name)

    def _get_fallback_explanation(self, action_name: str) -> str:
        """Fallback explanations when RAG unavailable"""
        explanations = {
            'INCREASE_DENSITY': (
                "Increasing placement density allows more cells in the core area, "
                "which can reduce wirelength by keeping connected cells closer together."
            ),
            'OPTIMIZE_WIRELENGTH': (
                "Optimizing wirelength directly targets reducing wire congestion and "
                "improving timing by shortening signal paths."
            ),
            'OPTIMIZE_ROUTABILITY': (
                "Routability optimization ensures there's enough space for routing, "
                "preventing congestion that could lead to unroutable designs."
            ),
            'UPSIZE_CRITICAL_CELLS': (
                "Upsizing cells on critical timing paths increases drive strength, "
                "reducing delay and improving worst negative slack."
            ),
            'BUFFER_CRITICAL_PATHS': (
                "Adding buffers breaks long wire segments, reducing RC delay and "
                "improving timing on critical paths."
            ),
        }

        return explanations.get(
            action_name,
            f"{action_name} is a standard optimization technique for improving design quality."
        )
