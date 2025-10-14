"""
Optimization Tracker

Logs every episode of RL optimization to create the "Results Story"
"""

from typing import Dict, List, Optional
import json
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict


@dataclass
class EpisodeLog:
    """Log entry for a single RL episode"""
    episode: int
    action_idx: int
    action_name: str
    metrics_before: Dict
    metrics_after: Dict
    reward: float
    success: bool
    timestamp: float


class OptimizationTracker:
    """
    Tracks optimization progress for storytelling.

    Logs every action, metric change, and builds narrative.
    """

    def __init__(self):
        """Initialize tracker"""
        self.episodes: List[EpisodeLog] = []
        self.action_counts = defaultdict(int)
        self.best_episode = None
        self.best_reward = float('-inf')

    def log_episode(
        self,
        episode: int,
        action_idx: int,
        action_name: str,
        metrics_before: Dict,
        metrics_after: Dict,
        reward: float,
        success: bool
    ):
        """Log a single RL episode"""
        import time

        log = EpisodeLog(
            episode=episode,
            action_idx=action_idx,
            action_name=action_name,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            reward=reward,
            success=success,
            timestamp=time.time()
        )

        self.episodes.append(log)
        self.action_counts[action_name] += 1

        # Track best episode
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_episode = log

    def get_current_metrics(self) -> Dict:
        """Get latest metrics with deltas"""
        if not self.episodes:
            return {}

        latest = self.episodes[-1]
        first = self.episodes[0]

        def safe_get(d, *keys, default=0):
            """Safely get nested dict value"""
            for key in keys:
                if isinstance(d, dict):
                    d = d.get(key, {})
                else:
                    return default
            return d if isinstance(d, (int, float)) else default

        current = {
            'area_util': safe_get(latest.metrics_after, 'area', 'utilization') or 0,
            'power': safe_get(latest.metrics_after, 'power', 'total') or 0,
            'wns': safe_get(latest.metrics_after, 'timing', 'wns') or 0,
            'wirelength': safe_get(latest.metrics_after, 'routing', 'wirelength') or 0,
        }

        initial = {
            'area_util': safe_get(first.metrics_before, 'area', 'utilization') or 0,
            'power': safe_get(first.metrics_before, 'power', 'total') or 0,
            'wns': safe_get(first.metrics_before, 'timing', 'wns') or 0,
            'wirelength': safe_get(first.metrics_before, 'routing', 'wirelength') or 0,
        }

        # Calculate deltas
        def calc_delta(current_val, initial_val):
            if initial_val == 0:
                return 0
            return ((current_val - initial_val) / initial_val) * 100

        current['area_delta'] = calc_delta(current['area_util'], initial['area_util'])
        current['power_delta'] = calc_delta(current['power'], initial['power'])
        current['wns_delta'] = current['wns'] - initial['wns']  # Absolute for timing
        current['wire_delta'] = calc_delta(current['wirelength'], initial['wirelength'])

        return current

    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric"""
        values = []

        for ep in self.episodes:
            if metric_name == 'wirelength':
                val = ep.metrics_after.get('routing', {}).get('wirelength', 0)
            elif metric_name == 'wns':
                val = ep.metrics_after.get('timing', {}).get('wns', 0)
            elif metric_name == 'power':
                val = ep.metrics_after.get('power', {}).get('total', 0)
            elif metric_name == 'area':
                val = ep.metrics_after.get('area', {}).get('utilization', 0)
            else:
                val = 0

            values.append(val or 0)

        return values

    def create_progress_chart(self) -> go.Figure:
        """Create interactive progress chart"""
        episodes = [ep.episode for ep in self.episodes]

        fig = go.Figure()

        # Wirelength
        fig.add_trace(go.Scatter(
            x=episodes,
            y=self.get_metric_history('wirelength'),
            mode='lines+markers',
            name='Wirelength (μm)',
            line=dict(color='#667eea', width=2)
        ))

        # Timing (WNS)
        fig.add_trace(go.Scatter(
            x=episodes,
            y=self.get_metric_history('wns'),
            mode='lines+markers',
            name='WNS (ps)',
            line=dict(color='#764ba2', width=2),
            yaxis='y2'
        ))

        fig.update_layout(
            title='Optimization Progress',
            xaxis_title='Episode',
            yaxis_title='Wirelength (μm)',
            yaxis2=dict(
                title='Timing Slack (ps)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        return fig

    def create_action_distribution_chart(self) -> go.Figure:
        """Create action distribution pie chart"""
        actions = list(self.action_counts.keys())
        counts = list(self.action_counts.values())

        fig = go.Figure(data=[go.Pie(
            labels=actions,
            values=counts,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])

        fig.update_layout(
            title='Actions Taken During Optimization',
            template='plotly_white',
            height=400
        )

        return fig

    def get_narrative(self) -> Dict:
        """Generate narrative summary"""
        if not self.episodes:
            return {}

        total_episodes = len(self.episodes)
        successful_episodes = sum(1 for ep in self.episodes if ep.success)

        # Find key turning points
        key_insights = self._identify_insights()

        return {
            'total_episodes': total_episodes,
            'total_actions': len(self.action_counts),
            'success_rate': (successful_episodes / total_episodes) * 100,
            'best_episode': self.best_episode.episode if self.best_episode else 0,
            'best_reward': self.best_reward,
            'key_insights': key_insights,
            'most_used_action': max(self.action_counts, key=self.action_counts.get) if self.action_counts else "None"
        }

    def _identify_insights(self) -> List[str]:
        """Identify key insights from optimization"""
        insights = []

        if not self.episodes:
            return insights

        # Analyze first 10 episodes
        first_10_actions = [ep.action_name for ep in self.episodes[:10]]
        if 'INCREASE_DENSITY' in first_10_actions:
            insights.append(
                "I focused on increasing placement density in the early episodes, "
                "which is a common strategy for improving wirelength."
            )

        if 'OPTIMIZE_WIRELENGTH' in first_10_actions:
            insights.append(
                "I prioritized wirelength optimization early, which typically helps "
                "with both timing and power consumption."
            )

        # Check for convergence
        if len(self.episodes) > 20:
            late_rewards = [ep.reward for ep in self.episodes[-10:]]
            avg_late_reward = sum(late_rewards) / len(late_rewards)

            if avg_late_reward > 0:
                insights.append(
                    f"The optimization converged successfully with average reward "
                    f"{avg_late_reward:.2f} in the final 10 episodes."
                )

        # Most effective action
        if self.best_episode:
            insights.append(
                f"The most effective action was {self.best_episode.action_name} "
                f"in episode {self.best_episode.episode}, achieving reward {self.best_reward:.2f}."
            )

        return insights

    def export_json(self, filepath: str):
        """Export tracking data to JSON"""
        data = {
            'episodes': [asdict(ep) for ep in self.episodes],
            'narrative': self.get_narrative(),
            'action_counts': dict(self.action_counts)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
