import numpy as np
from typing import List, Dict

def compute_success_rate(success_flags: List[bool]) -> float:
    if len(success_flags) == 0:
        return 0.0
    return np.mean(success_flags).item()

def compute_smoothness_metric(trajectories: List[np.ndarray]) -> float:
    from .trajectory_utils import compute_smoothness

    smoothness_values = []
    for traj in trajectories:
        smoothness = compute_smoothness(traj)
        smoothness_values.append(smoothness)

    if len(smoothness_values) == 0:
        return 0.0

    return np.mean(smoothness_values).item()

def compute_obstacle_distance(trajectories: List[np.ndarray],
                              obstacles: List[np.ndarray]) -> float:
    from .trajectory_utils import compute_min_distance_to_points

    min_distances = []
    for traj in trajectories:
        min_dist = compute_min_distance_to_points(traj, np.array(obstacles))
        min_distances.append(min_dist)

    if len(min_distances) == 0:
        return 0.0

    return np.mean(min_distances).item()

def compute_execution_time(execution_times: List[float]) -> Dict[str, float]:
    if len(execution_times) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

    return {
        'mean': np.mean(execution_times).item(),
        'std': np.std(execution_times).item(),
        'min': np.min(execution_times).item(),
        'max': np.max(execution_times).item(),
    }
