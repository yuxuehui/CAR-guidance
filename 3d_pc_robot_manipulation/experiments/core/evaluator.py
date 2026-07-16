import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

class Evaluator:

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics = self.config.get('metrics', ['success_rate', 'smoothness'])

    def compute_metrics(self,
                       trajectories: List[np.ndarray],
                       success_flags: List[bool],
                       execution_times: Optional[List[float]] = None) -> Dict[str, float]:
        metrics = {}

        if 'success_rate' in self.metrics:
            metrics['success_rate'] = self.compute_success_rate(success_flags)

        if 'smoothness' in self.metrics:
            metrics['smoothness'] = self.compute_smoothness(trajectories)

        if 'execution_time' in self.metrics and execution_times:
            metrics['execution_time'] = np.mean(execution_times)
            metrics['execution_time_std'] = np.std(execution_times)

        return metrics

    def compute_success_rate(self, success_flags: List[bool]) -> float:
        if len(success_flags) == 0:
            return 0.0
        return np.mean(success_flags).item()

    def compute_smoothness(self, trajectories: List[np.ndarray]) -> float:
        if len(trajectories) == 0:
            return 0.0

        smoothness_values = []
        for traj in trajectories:
            if traj.shape[0] < 3:
                continue

            accel = np.diff(traj, n=2, axis=0)

            jerk = np.diff(accel, n=1, axis=0)

            smoothness = np.mean(np.linalg.norm(jerk, axis=-1))
            smoothness_values.append(smoothness)

        if len(smoothness_values) == 0:
            return 0.0

        return np.mean(smoothness_values).item()

    def compute_obstacle_distance(self,
                                 trajectories: List[np.ndarray],
                                 obstacles: List[np.ndarray]) -> float:
        if len(trajectories) == 0 or len(obstacles) == 0:
            return 0.0

        min_distances = []
        for traj in trajectories:
            traj_pos = traj[:, :3]

            for obstacle in obstacles:

                distances = np.linalg.norm(traj_pos - obstacle, axis=-1)
                min_distances.append(np.min(distances))

        if len(min_distances) == 0:
            return 0.0

        return np.mean(min_distances).item()

    def generate_report(self,
                       all_metrics: Dict[str, Dict[str, float]],
                       save_path: Optional[str] = None) -> str:
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("评估报告")
        report_lines.append("=" * 60)
        report_lines.append("")

        experiments = list(all_metrics.keys())
        metrics = set()
        for exp_metrics in all_metrics.values():
            metrics.update(exp_metrics.keys())
        metrics = sorted(list(metrics))

        header = f"{'指标':<20}"
        for exp_name in experiments:
            header += f"{exp_name:<20}"
        report_lines.append(header)
        report_lines.append("-" * 60)

        for metric in metrics:
            row = f"{metric:<20}"
            for exp_name in experiments:
                value = all_metrics[exp_name].get(metric, 0.0)
                row += f"{value:<20.4f}"
            report_lines.append(row)

        report_lines.append("")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

            json_path = save_path.replace('.txt', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        return report
