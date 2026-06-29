import numpy as np
from typing import List, Optional

def compute_smoothness(trajectory: np.ndarray) -> float:
    if trajectory.shape[0] < 3:
        return 0.0

    accel = np.diff(trajectory, n=2, axis=0)

    jerk = np.diff(accel, n=1, axis=0)

    smoothness = np.mean(np.linalg.norm(jerk, axis=-1))

    return smoothness

def interpolate_trajectory(trajectory: np.ndarray, num_points: int) -> np.ndarray:
    from scipy.interpolate import interp1d

    n_steps, state_dim = trajectory.shape

    if n_steps == num_points:
        return trajectory

    t_original = np.linspace(0, 1, n_steps)
    t_new = np.linspace(0, 1, num_points)

    interpolated = np.zeros((num_points, state_dim))
    for i in range(state_dim):
        f = interp1d(t_original, trajectory[:, i], kind='linear')
        interpolated[:, i] = f(t_new)

    return interpolated

def extract_position(trajectory: np.ndarray) -> np.ndarray:
    return trajectory[:, :3]

def compute_trajectory_length(trajectory: np.ndarray) -> float:
    if trajectory.shape[0] < 2:
        return 0.0

    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=-1)

    length = np.sum(distances)

    return length

def compute_min_distance_to_points(trajectory: np.ndarray, points: np.ndarray) -> float:
    traj_pos = extract_position(trajectory)

    min_distances = []
    for point in points:
        distances = np.linalg.norm(traj_pos - point, axis=-1)
        min_distances.append(np.min(distances))

    if len(min_distances) == 0:
        return 0.0

    return np.min(min_distances)
