import numpy as np
from typing import List

def compute_path_length(trajectory: np.ndarray) -> float:
    if len(trajectory) < 2:
        return 0.0

    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))

def compute_smoothness(trajectory: np.ndarray) -> float:
    if len(trajectory) < 3:
        return 0.0

    directions = np.diff(trajectory, axis=0)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    directions = directions / norms

    angles = []
    for i in range(len(directions) - 1):
        dot = np.clip(np.dot(directions[i], directions[i+1]), -1.0, 1.0)
        angle = np.arccos(dot)
        angles.append(angle)

    if len(angles) == 0:
        return 0.0

    return float(np.std(angles))

def compute_curvature(trajectory: np.ndarray) -> float:
    if len(trajectory) < 3:
        return 0.0

    curvatures = []
    for i in range(1, len(trajectory) - 1):
        p1 = trajectory[i-1]
        p2 = trajectory[i]
        p3 = trajectory[i+1]

        v1 = p2 - p1
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        curvatures.append(angle)

    if len(curvatures) == 0:
        return 0.0

    return float(np.mean(curvatures))
